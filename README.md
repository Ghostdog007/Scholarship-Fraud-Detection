# NIC Scholarship Fraud Detection — v1.4

Unsupervised + weak-supervision pipeline for flagging anomalous Pre-Matric and
Post-Matric scholarship applications submitted through the NIC national portal.

For full project context, dataset facts, architecture rationale, and the
quantitative-claims protocol agents must follow, see **`docs/AGENTS.md`** —
that file is the source of truth for this project and should be read in full
before making any change here.

---

## The problem

The dataset (`data_for_ml_model.csv`, 15,000 rows, 136 columns) is 99.97% valid:
only 4 records carry a historical fraud flag, and even those 4 can't be used as
training signal — their duplicate counterparts live outside this slice, so the
slice is treated as independent and these 4 are assumed clean for training and
evaluation. Standard supervised classifiers fail on this kind of imbalance, so
the system is built to work without reliable fraud labels.

## Data Sources & Lineage

### 2.1 Raw Source
`data_for_ml_model.csv` is a 15,000-row slice of the NIC national scholarship portal's application database (Pre-Matric and Post-Matric), per AGENTS.md Section 1. This slice is treated as independent since the 4 known fraud records' duplicate counterparts exist outside this slice and thus trigger no rules locally.

### 2.2 Transformation Chain (ordered)

1. **`load_and_clean_data(filepath)` in `feature_selection.py`**
   - Drops 100%-null columns: `updated_by`, `delete_record`, `deleted_by`, `delete_on`, `delete_ip_address`, `deleted_by_level`, `c_university_id`, `p_institution_id`, `x_institution_id`, `xii_institution_id`, `competitive_exam_score`, `xii_course_id`, `new_entitled_fee_amount_centre_share`, `sub_category_id`, `updated_by-2`, `updated_on-2`
   - Drops duplicate spatial columns: `state_id-2`, `pfms_state_code`, `state_name-2`, `district_id`, `district_name-2`

2. **`engineer_features(df)` in `feature_selection.py` (and `vae_detection.py`)**
   - Layer 0 — Text-to-boolean identity flags: 
     - `is_applicant_name_eq_father = (df['applicant_name'] == df['father_name']).astype(int)`
     - `is_applicant_name_eq_mother = (df['applicant_name'] == df['mother_name']).astype(int)`
     - `is_father_name_eq_mother = (df['father_name'] == df['mother_name']).astype(int)`
     - `name_similarity_score` via `SequenceMatcher` fuzzy match on `applicant_name` vs `father_name`
   - Layer 1 — Relational aggregates: 
     - `mobile_application_count = df.groupby('mobile_no')['application_id'].transform('count')`
     - `ip_application_count = df.groupby('ip_address')['application_id'].transform('count')`
     - `mobile_unique_names_count`, `mobile_unique_fathers_count` (transform nunique)
     - `institute_application_count` (conditional on `c_institution_id` present)
     - `district_application_count` (conditional on `district_id` present — **note: absent in File 1 path due to `load_and_clean_data()` dropping `district_id` first; see discrepancy note below**)
     - `ip_to_mobile_ratio = ip_application_count / (mobile_application_count + 1)`
   - Layer 2 — Policy boundary flags:
     - `flag_income_below_20000`: income < 20000
     - `flag_income_below_10000`: income <= 10000
     - `flag_income_extreme_low`: income < 1000
     - `flag_prematric_age_over20`: pre_post_matric==1 AND age > 20
     - `flag_postmatric_age_over35`: pre_post_matric==2 AND age > 35
     - `flag_postmatric_age_under13`: pre_post_matric==2 AND age < 13
     - `flag_fee_exceeds_income`: fee_income_ratio > 1.0

3. **MI → Pearson → mRMR selection in `feature_selection.py` main()**
   - MI\_KEEP\_RATIO = 0.8
   - MI\_MAX\_FEATURES = 50
   - MRMR\_MAX\_FEATURES = 40
   - Pearson threshold = 0.90
   - The protected features list bypasses the cutoff

4. **`vae_detection.py` stages:**
   - Stage A: Reads `selected_features.json`, trains VAE on all records (`valid_mask = all True`), produces `vae_reconstruction_prob` per row
   - Stage B: Runs `apply_rules()` on the full dataframe, produces `rule_violation_score` and `rule_codes_fired`
   - Stage C: Trains LightGBM on `[selected_features] + [vae_reconstruction_prob]` with `rule_violation_score > 0` as weak label. Excludes `rule_violation_score` from features (target leakage guard)

5. **SHAP pruning in `evaluate_model.py`**
   - Threshold: 0.001
   - Input: `shap_summary.json` + `selected_features.json`
   - Output: `selected_features_shap_pruned.json`

### 2.3 Known Structural Discrepancy

`vae_detection.py` calls `engineer_features()` directly on the raw CSV without first calling `load_and_clean_data()`. As a result, `district_id` is still present when `engineer_features()` runs in File 2, meaning `district_application_count` is computed and available to `apply_rules()`. However, in File 1's path (`feature_selection.py`), `load_and_clean_data()` drops `district_id` first, so `district_application_count` is never computed and therefore never appears in `selected_features.json`. Since LightGBM only sees features listed in that JSON, `district_application_count` is computable in the rule engine but invisible to the classifier. Verified with an actual run:
```
district_application_count in File 1 path: False
district_application_count in File 2 path: True
district_application_count selected: False
```

### 2.4 Lineage Table

| Output Artifact | Source Data | Producing Script | Producing Function |
|---|---|---|---|
| `selected_features.json` | `datasets/data_for_ml_model.csv` | `feature_selection.py` | `main()` |
| `risk_scores.csv` | `datasets/data_for_ml_model.csv` + `selected_features.json` | `vae_detection.py` | `main()` |
| `selected_features_shap_pruned.json` | `risk_scores.csv` + `selected_features.json` + `shap_summary.json` | `evaluate_model.py` | `main()` |

## How it works — three stages, two files

```
data_for_ml_model.csv
        │
        ▼
feature_selection.py   →  selected_features.json
        │
        ▼
vae_detection.py        →  risk_scores.csv
        │
        ▼
evaluate_model.py        →  console metrics + selected_features_shap_pruned.json
```

**`feature_selection.py`** loads and cleans the raw CSV (drops 100%-null and
duplicate columns), engineers ~20 features across three layers (text-to-boolean
identity flags, cross-row relational aggregates like mobile/IP concentration,
and policy-boundary flags like income/age thresholds), then runs classwise
Mutual Information filtering, Pearson correlation pruning, and mRMR selection
to produce a final feature list. A fixed set of engineered features is
protected from being dropped by these filters regardless of score.

**`vae_detection.py`** runs the actual detection pipeline in three stages: a
PyTorch Variational Autoencoder learns what a normal application looks like and
outputs a reconstruction-probability anomaly score; a rule engine
(`apply_rules()`) evaluates both the original NIC revalidation rules and a set
of engineered "bridge" rules and produces a weak label (`rule_violation_score`);
and a LightGBM classifier trains on the selected features plus the VAE score,
using the weak label as its target, then explains its own output with SHAP.

**`evaluate_model.py`** scores the pipeline's output against the weak labels
(PR-AUC, F1, MCC, Brier score), reports how the 4 known fraud cases scored, and
runs a second SHAP pass that prunes near-zero-importance features into
`selected_features_shap_pruned.json` (never back into `selected_features.json`
— that's a hard contract boundary enforced by a runtime guard).

`main.py` runs all three in sequence.

## Running it

```bash
python main.py --data_path datasets/data_for_ml_model.csv
```

Or run stages individually:

```bash
python feature_selection.py --data_path datasets/data_for_ml_model.csv --output_json selected_features.json
python vae_detection.py --features_json selected_features.json --output_csv risk_scores.csv
python evaluate_model.py --scores_csv risk_scores.csv
```

`vae_detection.py` accepts `--disabled_bridges` (comma-separated rule codes,
e.g. `IP_CONC_ENG,FEE_ENG,FM_ENG`) to turn off specific engineered weak-label
bridges without touching the legacy NIC rules — this is what the ablation
testing in Section 11.5 of `AGENTS.md` uses to isolate each bridge's
contribution.

## Current verified baseline (v1.4, 2026-06-17)

These numbers come from a single canonical run with the VAE seeded for
reproducibility; see `docs/AGENTS.md` Section 6 for the full table and the caveat
attached to each figure.

| Metric | Value |
|---|---|
| Positive weak labels | 1,986 of 15,000 |
| PR-AUC | 0.9906 |
| Best F1-score | 0.9509 |
| MCC | 0.9434 |
| Optimal threshold | 0.7749 |

**Important caveat:** PR-AUC here measures how well LightGBM agrees with the
system's own rule-based weak labels — not with confirmed real fraud. It shows
the classifier learned the rule boundaries correctly; it does not show the
true real-world fraud-catching rate, which remains unmeasured due to the lack
of usable ground truth.

## Synthetic validation

Because real fraud examples are too scarce to test against, the project
maintains a synthetic fraud injection test (Phase D harness, the current
permanent standard — see `AGENTS.md` Section 11.5 for why the earlier v1.x
synthetic categories were retired). It plants five fraud types (income, age,
mother-name collision, fee inflation, IP concentration) into a frozen copy of
the dataset and checks whether the model catches them.

Two things this testing has shown, with the caveats that matter attached:

- The VAE carries genuine, non-zero unsupervised signal across every fraud
  category on its own (ROC-AUC 0.77–0.95), strongest on univariate violations
  (income, age) and weaker on relational fraud (IP clustering, name collisions,
  fee inflation), where its PR-AUC alone is too low (<0.12) to set a safe
  threshold without flooding the system with false alarms.
- The engineered rule bridges (`IP_CONC_ENG`, `FEE_ENG`, `FM_ENG`) convert that
  weaker relational signal into a usable, hard decision boundary for the
  supervised classifier. The ablation deltas table in `AGENTS.md` should be read
  alongside its own caveat — those three bridges are the sole source of
  positive labels for the rows they target, so part of the swing shown there is
  a mechanical consequence of label removal, not independent proof on its own.
  The VAE-alone PR-AUC figures above are the non-circular evidence for why the
  bridges are needed.
- The VAE's training data includes the synthetic anomalies (no held-out split
  exists yet), so its true performance on genuinely novel, never-before-seen
  fraud patterns is untested in either direction.

## Hard constraints

A few of the most consequential, enforced throughout the code — the full list
is in `AGENTS.md` Section 7:

- `sanity`, `application_id`, and `jwt` are never used as model features.
- `rule_violation_score` is the training label and must never be fed back in
  as a LightGBM input feature.
- `selected_features_shap_pruned.json` is the only valid output path for the
  SHAP pruner; `vae_detection.py` raises `RuntimeError` if it detects pruner
  output mixed into `selected_features.json`.
- All 4 known fraud records are treated as clean during training — they are
  not added to the training set under any circumstance, since this slice can't
  independently confirm them and doing so would be target leakage.

## Tech stack

PyTorch (VAE), LightGBM (classifier), scikit-learn (MI / mRMR / metrics), SHAP
(explainability). See `AGENTS.md` Section 5 for version constraints and what's
deliberately excluded from this stack (TensorFlow, XGBoost, SMOTE, AutoML).

## Working on this project

Before making any change: read `AGENTS.md` in full, especially Section 0's
quantitative-claims protocol. The short version — paste raw stdout, never
paraphrase a number; name the exact baseline a comparison is against; don't
resolve an open question in the same turn its supporting number was generated;
if a number conflicts with something said earlier, stop and re-derive it
rather than explaining the conflict away in prose.
