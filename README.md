# NIC Scholarship Fraud Detection — v1.4

Unsupervised + weak-supervision pipeline for flagging anomalous Pre-Matric and
Post-Matric scholarship applications submitted through the NIC national portal.

For full project context, dataset facts, architecture rationale, and the
quantitative-claims protocol agents must follow, see **`docs/AGENTS.md`** —
that file is the source of truth for this project and should be read in full
before making any change here.

## The problem

The dataset (`data_for_ml_model.csv`, 15,000 rows, 136 columns) is 99.97% valid:
only 4 records carry a historical fraud flag. Standard supervised classifiers
fail on imbalance this extreme, so the system is built to work without
relying on those labels at all.

**Those 4 records are treated as valid throughout this project.** They were
originally flagged because of duplicate application records elsewhere in
NIC's national database — but this 15,000-row file is just one independent
slice, and those duplicate records don't exist inside it. To any rule or
model trained only on this slice, the 4 records look like ordinary, clean
applications, because the evidence that made them suspicious literally isn't
present here. So they're never added to training data, never used to
validate model accuracy, and never treated as ground truth — anywhere in this
codebase.

## How it works — three files, in sequence

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
duplicate columns), engineers roughly 20 new features across three layers
(text-to-boolean identity flags, cross-row relational aggregates like
mobile/IP concentration, and policy-boundary flags like income/age
thresholds), then runs classwise Mutual Information filtering, Pearson
correlation pruning, and mRMR selection to produce a final feature list. A
fixed set of engineered features is protected from being dropped by these
filters regardless of score.

**`vae_detection.py`** runs the actual detection pipeline in three stages: a
PyTorch Variational Autoencoder learns what a normal application looks like
and outputs a reconstruction-probability anomaly score; a rule engine
(`apply_rules()`) evaluates both the original NIC revalidation rules and a
set of engineered "bridge" rules and produces a weak label
(`rule_violation_score`); and a LightGBM classifier trains on the selected
features plus the VAE score, using the weak label as its target, then
explains its own output with SHAP.

**`evaluate_model.py`** scores the pipeline's output against the weak labels
(PR-AUC, F1, MCC, Brier score), reports how the 4 known fraud cases scored
(for reference, not validation — see above), and runs a second SHAP pass that
prunes near-zero-importance features into
`selected_features_shap_pruned.json` (never back into `selected_features.json`
— that's a hard contract boundary enforced by a runtime guard).

`main.py` runs all three in sequence.

## Data sources & lineage

#### Raw source

`data_for_ml_model.csv` is the 15,000-row, 136-column slice described above.
Nothing else feeds into this pipeline as raw input.

#### Transformation chain

| Step | Function | What happens |
|---|---|---|
| 1 | `load_and_clean_data()` — `feature_selection.py` | Drops 16 confirmed 100%-null columns and 5 confirmed duplicate columns (full lists in `AGENTS.md` §2.2–2.3) |
| 2 | `engineer_features()` — `feature_selection.py` & `vae_detection.py` | Adds ~20 engineered columns, listed below |
| 3 | MI → Pearson → mRMR selection — `feature_selection.py main()` | Narrows the full column set down to a final selected list |
| 4 | Stage A/B/C — `vae_detection.py main()` | VAE scores anomaly, rule engine builds the weak label, LightGBM trains on both |
| 5 | SHAP second pass — `evaluate_model.py main()` | Drops near-zero-importance features into a separate pruned list |

Engineered features, by layer:

| Layer | Feature | What it checks |
|---|---|---|
| Identity flags | `is_applicant_name_eq_father` | Applicant name == father's name |
| | `is_applicant_name_eq_mother` | Applicant name == mother's name |
| | `is_father_name_eq_mother` | Father's name == mother's name |
| | `name_similarity_score` | Fuzzy-match ratio, applicant vs. father |
| Relational aggregates | `mobile_application_count` | Applications sharing one mobile number |
| | `ip_application_count` | Applications sharing one IP address |
| | `mobile_unique_names_count` | Distinct applicant names per mobile number |
| | `mobile_unique_fathers_count` | Distinct father names per mobile number |
| | `institute_application_count` | Applications per institute (if column present) |
| | `district_application_count` | Applications per district (if column present — see discrepancy below) |
| | `ip_to_mobile_ratio` | IP concentration relative to mobile concentration |
| Policy boundaries | `flag_income_below_20000` / `_10000` / `_extreme_low` | Income under 20k / 10k / 1k |
| | `flag_prematric_age_over20` | Pre-Matric applicant older than 20 |
| | `flag_postmatric_age_over35` / `_under13` | Post-Matric applicant outside the 13–35 range |
| | `flag_fee_exceeds_income` | Claimed fees exceed reported income |

Selection thresholds, read directly from `feature_selection.py`: MI keep
ratio 0.8, MI max 50 features, mRMR max 40 features, Pearson correlation
cutoff 0.90. The SHAP pruning threshold in `evaluate_model.py` is 0.001.

#### A known discrepancy worth knowing

`vae_detection.py` calls `engineer_features()` directly on the raw CSV
without first running it through `load_and_clean_data()`. That means
`district_id` is still present when File 2 engineers features, so
`district_application_count` gets computed there — but File 1 already
dropped `district_id` before engineering anything, so that same feature
never exists in `selected_features.json`. Net effect: the rule engine could
in principle see district-level concentration, but LightGBM never can,
since it only sees what's listed in that JSON. Verified directly rather than
assumed:

```
district_application_count in File 1 path: False
district_application_count in File 2 path: True
district_application_count selected: False
```

#### Lineage table

| Output artifact | Built from | Script | Function |
|---|---|---|---|
| `selected_features.json` | `data_for_ml_model.csv` | `feature_selection.py` | `main()` |
| `risk_scores.csv` | `data_for_ml_model.csv` + `selected_features.json` | `vae_detection.py` | `main()` |
| `selected_features_shap_pruned.json` | `risk_scores.csv` + `selected_features.json` + `shap_summary.json` | `evaluate_model.py` | `main()` |

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
testing in `AGENTS.md` §11.5 uses to isolate each bridge's contribution.

## Current verified baseline (v1.4, 2026-06-17)

From a single canonical run with the VAE seeded for reproducibility; see
`AGENTS.md` §6 for the full table and the caveat attached to each figure.

| Metric | Value |
|---|---|
| Positive weak labels | 1,986 of 15,000 |
| PR-AUC | 0.9906 |
| Best F1-score | 0.9509 |
| MCC | 0.9434 |
| Optimal threshold | 0.7749 |

**Caveat:** this PR-AUC measures agreement with the system's own rule-based
weak labels, not confirmed real fraud. It shows the classifier learned the
rule boundaries correctly — it does not show the true real-world
fraud-catching rate, which remains unmeasured.

## Synthetic validation

Real fraud examples are too scarce to test against, so the project maintains
a synthetic fraud injection test (the Phase D harness — see `AGENTS.md` §11.5
for why earlier versions were retired). It plants five fraud types into a
frozen copy of the dataset and checks whether the model catches them.

What that testing has shown, caveats included: the VAE carries genuine,
non-zero unsupervised signal on every fraud type on its own (ROC-AUC
0.77–0.95), strongest on simple violations like income and age, weaker on
relational fraud like IP clustering or name collisions — weak enough there
(PR-AUC <0.12 alone) that it can't safely set a threshold without flooding
the system with false alarms. The engineered rule bridges close that gap by
turning the VAE's weaker relational signal into a usable decision boundary
for LightGBM. The VAE was trained on data that included the synthetic
anomalies, though, so its real performance on genuinely novel fraud is still
untested in either direction.

## Hard constraints

The full list is in `AGENTS.md` §7 — the most consequential:

- `sanity`, `application_id`, and `jwt` are never used as model features.
- `rule_violation_score` is the training label and never a LightGBM input.
- `selected_features_shap_pruned.json` is the only valid output path for the
  SHAP pruner — `vae_detection.py` raises `RuntimeError` if it detects
  pruner output mixed into `selected_features.json`.
- The 4 known fraud records are never added to training data, for the
  reason explained in "The problem" above.

## Tech stack

PyTorch (VAE), LightGBM (classifier), scikit-learn (MI / mRMR / metrics),
SHAP (explainability). See `AGENTS.md` §5 for version constraints and what's
deliberately excluded (TensorFlow, XGBoost, SMOTE, AutoML).

## Working on this project

Read `AGENTS.md` in full before making any change, especially Section 0's
quantitative-claims protocol. The short version: paste raw stdout, never
paraphrase a number; name the exact baseline a comparison is against; don't
resolve an open question in the same turn its supporting number was
generated; if a number conflicts with something said earlier, stop and
re-derive it rather than explaining the conflict away in prose.
