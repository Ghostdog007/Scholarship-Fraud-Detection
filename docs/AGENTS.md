# NIC Fraud Detection — Agent Context File
<!-- VERSION: 1.2 | OWNER: Project Lead | LAST REVIEWED: 2026-06 -->
<!-- PURPOSE: Agent-facing context. Human-curated. Do NOT auto-regenerate. -->
<!-- SCOPE: Covers project mission, data facts, ML architecture decisions, -->
<!--        coding conventions, and hard constraints. Read fully before acting. -->

---

## 0. How to Use This File (Agent Instructions)

> **Read this section first on every session. It governs how you use everything below.**

- **Do not hallucinate dataset facts.** All column names, null ratios, record counts,
  and rule codes in this file are sourced from executed analysis. Treat them as ground
  truth. If something contradicts your priors, trust this file.
- **Web-search before recommending any ML approach.** This project requires
  research-backed decisions. If you suggest an algorithm or architecture, cite a paper
  or credible source. Speculation is not acceptable.
- **Ask before assuming scope.** If a task could touch either `feature_selection.py`
  or `vae_detection.py`, confirm which file before writing code.
- **One concern per response.** Do not batch unrelated suggestions. Surface one
  well-researched idea at a time.
- **Never modify this file autonomously.** Only the project lead updates AGENTS.md.
  If you think something is outdated, flag it explicitly rather than changing it.

### 0.1 Protocol for Quantitative Claims

1. **Raw output only:** Before any number enters a summary or gets written into AGENTS.md, paste the literal, unedited stdout it came from. If it can't be traced to a specific printed line in the conversation, it doesn't go in the file.
2. **Comparisons name their baseline explicitly:** Any "before vs after" claim must state which specific prior run it is being compared to (commit hash, timestamp, or quoted conversation line).
3. **Seed everything before comparing runs:** Beyond the VAE, `synthetic_anomaly_test.py`'s row sampling, `train_test_split`, and the `random` module must use fixed seeds before runs are comparable.
4. **Row-level counting only, never code-frequency summing:** `rule_codes_fired.explode().value_counts()` counts code occurrences, not unique flagged rows. Use isolated masking to determine a rule's net-new contribution.
5. **No same-turn resolution:** An open question in AGENTS.md doesn't get checked off in the same turn its supporting number was generated. It gets logged as "proposed, pending review".
6. **Conflicting numbers halt:** If a number contradicts something said earlier in the same session, surface the conflict and stop. Do not reconcile it by narrative without re-deriving the number from scratch.

---

## 1. Project Mission

**Domain:** Government scholarship fraud detection — National Informatics Centre (NIC),
India. The system detects anomalous Pre-Matric and Post-Matric scholarship applications
submitted through a national portal.

**Core problem:** The labeled dataset is 99.97% valid (only 4 confirmed fraud records
out of 15,000). Standard supervised binary classifiers fail here due to extreme class
imbalance. The approach must work without reliable fraud labels.

**Goal:** Produce a per-application **anomaly risk score** (0–1 continuous) that:
1. Captures deviations from the statistical distribution of legitimate applications
   (unsupervised signal — VAE reconstruction probability).
2. Encodes known fraud logic from 99 existing static revalidation rules
   (rule-based weak supervision signal).
3. Combines both signals through a gradient-boosted classifier (LightGBM/XGBoost)
   to produce a final interpretable risk score with SHAP explainability.

---

## 2. Dataset Facts (Do Not Hallucinate Beyond These)

### 2.1 Primary Dataset — `data_for_ml_model.csv`

| Property | Value |
|---|---|
| Rows | 15,000 |
| Columns | 136 |
| Fraud-labeled records (`sanity` not null) | 4 (0.027%) |
| All applicants | Fresh applicants only (`fresh_renewal = 'F'`) |
| Pre-Matric applicants (`pre_post_matric = 1`) | 5,073 |
| Post-Matric applicants (`pre_post_matric = 2`) | 9,908 |

### 2.2 Confirmed 100% Null Columns (Drop Before Any Processing)

```
updated_by, delete_record, deleted_by, delete_on, delete_ip_address,
deleted_by_level, c_university_id, p_institution_id, x_institution_id,
xii_institution_id, competitive_exam_score, xii_course_id,
new_entitled_fee_amount_centre_share, sub_category_id,
updated_by-2, updated_on-2
```
> **Agent rule:** Never use these columns as features. Drop them at load time.

### 2.3 Confirmed Duplicate Columns (Keep Only One per Group)

```
# State ID group — all four are identical:
domicile_state_id == state_id == state_id-2 == pfms_state_code

# State name group — identical:
state_name == state_name-2

# District ID group — identical:
permanent_district_id == district_id

# District name group — identical:
district_name == district_name-2
```
> **Agent rule:** In `feature_selection.py`, drop duplicates before computing
> correlation — otherwise they inflate MI scores artificially.

### 2.4 Key High-Nullity Fields (Handle, Don't Drop Blindly)

| Column | Null % | Reason |
|---|---|---|
| `disability_percentage`, `disablity_type` | 99.49% | Disability is rare — expected |
| `orphan_flag` | 99.75% | Rare demographic |
| `gaurdian_name` | 99.77% | Rare demographic |
| `enroll_udid_no` | 99.49% | UDID mostly not provided |
| `ration_card_no`, `ration_card_member_no` | 96.49% | Optional field |
| `district_short_name` | 99.97% | Near-empty — drop |

### 2.5 Confirmed Missing Fields (Cannot Model These Rules)

The following fields appear in revalidation rules but are **entirely absent** from
the CSV. Do not attempt to engineer proxies without explicit instruction:

```
bank_account_no, bank_name, ifsc_code
```
This blocks rules: `K`, `H`, `X2`, `X4`, `X5`, `V`, `W`, `R5`.

### 2.6 Key Confirmed Anomalies in Valid Data

- **3 Post-Matric applicants exceed 35-year age limit** (Rule X7) but are NOT flagged
  in `sanity`. This confirms rule enforcement gaps exist and the ML model must catch
  what the rule system misses.
- **1 IP address submitted 39 applications.** Top 10 IPs each submitted 15–39
  applications. High IP concentration is a valid fraud signal.
- **1 mobile number shared by 6 applicants.** 59 mobiles shared by 2, 3 shared by 3.
- **Family income as low as 5 INR** — likely data entry errors or deliberate fraud.
- **Institute `c_institution_id=10791`** has 151 applications — highest concentration,
  below the 500-application Rule X3 threshold but worth monitoring.

### 2.7 Known Fraud Cases — Ground Truth (4 Records)

These are the only human-confirmed fraud records in the dataset. However, because their duplicate counterparts are not present in this 15,000-record slice, they do not trigger any rules locally. Therefore, we treat this slice as completely independent and assume these 4 records are valid/clean during training and evaluation.

| Application ID | Sanity Flag | v1.2 Risk Score | Rules Fired | Top SHAP Features |
|---|---|---|---|---|
| BR202526000069940 | A2 | 0.2810 | None | x_university_id, state_verify_by, permanent_pincode |
| TN202526000099321 | M | 0.0001 | None | flag_fee_exceeds_income, x_university_id, competitive_exam_year |
| AS202526000130331 | A1A2GM | 0.0026 | None | flag_fee_exceeds_income, x_percentage, entitled_lumpsump_amount |
| BR202526000218140 | A1M | 0.6987 | None | x_university_id, permanent_pincode, state_verify_by |

> **Critical observation:** All 4 known frauds have `Rules Fired: None`. This means
> none of them triggered any rule code — NIC's own rulebook missed them. This is
> the core limitation of the weak supervision approach: LightGBM never saw these
> 4 records as "positive" during training. Their risk scores are driven entirely
> by VAE anomaly signal and SHAP-visible feature patterns. This is an open problem.
> Do not attempt to fix this by adding these 4 records to the training set — that
> is target leakage and will overfit to 4 specific cases.

---

## 3. Static Revalidation Rules — What the Agent Needs to Know

### 3.1 Rules by Evaluability Status

**EVALUABLE from CSV (use in weak label generator in `vae_detection.py`):**

| Rule Code(s) | Logic | CSV Columns |
|---|---|---|
| `A`, `B`, `F`, `G`, `M`, `A1`, `A2`, `R1`–`R6` | Duplicate identity combinations | `applicant_name`, `father_name`, `mother_name`, `date_of_birth`, `mobile_no` |
| `X1` | Pre-Matric age > 20 | `date_of_birth`, `registered_date`, `pre_post_matric` |
| `X7` | Post-Matric age > 35 | same as above |
| `X8` | Post-Matric age < 13 | same as above |
| `UW` | Family income < 20,000 | `annual_family_income` |
| `X13`, `X21` | Family income ≤ 10,000 | `annual_family_income` |
| `YF` | Applicant name == father or mother name | `applicant_name`, `father_name`, `mother_name` |
| `X9`, `X10` | Duplicate Class 10/12 board + roll + year | `x_roll_no`, `x_course_year`, `xii_roll_no`, `xii_course_year` |
| `YK`, `YL` | Same mobile on 11–20 or >20 applications | `mobile_no` (aggregate count) |
| `UN` | Same mobile, different father name | `mobile_no`, `father_name` |

**ENGINEERED FEATURE VIOLATIONS (Layer 0/1/2 → weak label bridge):**

These are not NIC rulebook rules. They are engineered signal → weak label bridges
added in v1.2 to close the feedback loop identified in Section 11.5. They extend
`apply_rules()` so LightGBM's training labels reward the protected engineered features.

| Rule Code | Condition | Engineered Column | Weight |
|---|---|---|---|
| `IP_CONC_ENG` | `ip_application_count >= 15` | `ip_application_count` | `MOBILE_CONCENTRATION` |
| `YF_ENG` | `is_applicant_name_eq_father == 1` | `is_applicant_name_eq_father` | `NAME_MATCH` |
| `YF_MOTHER_ENG` | `is_applicant_name_eq_mother == 1` | `is_applicant_name_eq_mother` | `NAME_MATCH` |
| `FEE_ENG` | `flag_fee_exceeds_income == 1` | `flag_fee_exceeds_income` | `INCOME_VIOLATION` |
| `INC_EXT_ENG` | `flag_income_extreme_low == 1` | `flag_income_extreme_low` | `INCOME_VIOLATION` |
| `UN_FATHER_ENG` | `mobile_unique_fathers_count > 1` | `mobile_unique_fathers_count` | `MOBILE_FATHER_MISMATCH` |
| `X1_ENG` | `flag_prematric_age_over20 == 1` | `flag_prematric_age_over20` | `AGE_VIOLATION` |
| `X7_ENG` | `flag_postmatric_age_over35 == 1` | `flag_postmatric_age_over35` | `AGE_VIOLATION` |

> **Agent rule:** All 8 blocks use `if 'column_name' in df.columns` guards.
> Do not remove guards — the synthetic test pipeline uses a subset of columns.

**NOT EVALUABLE (missing data — skip in code):**
`K`, `H`, `X2`, `X4`, `X5`, `V`, `W`, `R5`, `VA`, `YP`, `UA`–`UI`

---

## 4. ML Architecture Decisions (Locked — Do Not Propose Alternatives Without Research)

### 4.1 Two-File Pipeline Design

The ML system is split into exactly **two Python scripts** with a clean handoff interface.
This separation is validated by a 2021 arXiv paper on anomaly detection with latent-space
feature selection, which shows decoupling feature selection from model training reduces
false positives and enables independent validation of each stage.

```
data_for_ml_model.csv
        │
        ▼
┌─────────────────────────┐
│  feature_selection.py   │  ← File 1
│  (classical pipeline)   │
└────────────┬────────────┘
             │ outputs: selected_features.json
             ▼
┌─────────────────────────┐
│  vae_detection.py       │  ← File 2
│  (VAE + rules + LGBM)   │
└─────────────────────────┘
             │ outputs: risk_scores.csv
             ▼
      Per-application anomaly risk score (0–1)
```

**Why two files, not one:**
- ETH Zurich empirical study (2025) confirms that modular, independently validatable
  pipeline stages outperform monolithic scripts in agent-assisted development.
- Anthropic engineering docs recommend decoupling stages so each can be retested
  without rerunning the full pipeline.
- The AED-LGB paper (PMC 2024) that validates this exact AE + LightGBM architecture
  explicitly separates feature reconstruction from classification.

### 4.2 File 1 — `feature_selection.py`: What It Does

**Algorithm pipeline (in order):**

1. **Load & clean:** Drop 100% null columns. Drop duplicate spatial columns (keep
   `state_id`, `district_id`). Drop `sanity` column from features.
2. **Classwise Mutual Information filtering:** Compute class-weighted MI scores.
   Upweight minority class using `w_c = max(1e-6, mean(y == c))`. Retain top
   features by `MI_KEEP_RATIO` and `MI_MAX_FEATURES`.
3. **Pearson correlation pruning:** For pairs with `|r| >= 0.90`, drop the one
   with lower MI score.
4. **mRMR selection:** Iteratively select features maximising relevance minus
   redundancy. Output `MRMR_MAX_FEATURES` final features.
5. **Save output:** Write `selected_features.json` with feature names and their
   importance scores.

**What to engineer before MI scoring (mandatory):**

```python
# These must exist as columns before feature_selection.py runs MI
age_at_registration     = (registered_date - date_of_birth).days / 365.25
fee_income_ratio        = (admission_fee + tution_fee + misc_fee) / annual_family_income
mobile_occurrence_count = mobile_no.map(mobile_no.value_counts())
ip_occurrence_count     = ip_address.map(ip_address.value_counts())
state_match_flag        = (domicile_state_id == inst_state_id).astype(int)
```

**Known performance warning (non-blocking):**
`feature_selection.py` triggers pandas `PerformanceWarning: DataFrame is highly
fragmented` during Layer 2 flag engineering. This is cosmetic — it does not affect
output correctness. The fix is to batch all new column assignments via `pd.concat`
rather than sequential `df[col] = ...` inserts. Do not fix this unless performance
becomes a bottleneck; it is low priority relative to modelling work.

**Output contract:**
```json
{
  "selected_features": ["age_at_registration", "fee_income_ratio", ...],
  "feature_scores": {"age_at_registration": 0.42, ...},
  "protected_features": [...],
  "n_selected": 56,
  "pipeline_run_timestamp": "2026-06-15T10:00:00"
}
```

### 4.3 File 2 — `vae_detection.py`: What It Does

**Stage A — VAE (Variational Autoencoder):**
- Train ONLY on valid records (records where `sanity` is null).
- Input: features from `selected_features.json` only.
- Architecture: Encoder → μ (mean) and σ (std) → reparameterisation trick →
  Decoder → reconstruction.
- Loss: ELBO = Reconstruction loss (BCE or MSE) + KL divergence.
- Output per record: `vae_reconstruction_prob` — reconstruction probability score.
  Higher score = more normal. Lower score = more anomalous.
- **Why VAE not plain AE:** Reconstruction probability from VAE is a more
  principled anomaly score than MSE because it accounts for distributional
  variability. Source: "Variational Autoencoder based Anomaly Detection using
  Reconstruction Probability" (Semantic Scholar, validated on tabular fraud data).

**Stage B — Rule-Based Weak Label Generator:**
- Run only the EVALUABLE rules from Section 3.1 (both NIC rulebook rules AND
  the engineered feature violation bridges added in v1.2).
- Produce `rule_violation_score` per record: weighted count of rules fired.
- Severity weights (start here, tune later):
  ```
  Identity duplicate rules (A, B, F, G, M): weight = 2.0
  Age boundary violations (X1, X7, X8):     weight = 1.5
  Income threshold rules (UW, X13, X21):    weight = 1.0
  Name match rule (YF):                     weight = 1.5
  Mobile concentration (YK, YL):            weight = 2.0
  ```
- This does NOT require retraining VAE when new rules are added.

**Stage C — LightGBM Classifier (Final Layer):**
- Input feature set for LightGBM:
  ```
  [selected_features from JSON] + [vae_reconstruction_prob]
  ```
  Note: `rule_violation_score` is explicitly excluded from inputs to prevent
  target leakage. It is the label, never a feature.
- Weak labels: binarise `rule_violation_score > 0` as positive class for initial
  training. Tune threshold using PR-AUC, NOT ROC-AUC (ROC-AUC is misleading on
  extreme imbalance).
- Use `scale_pos_weight` in LightGBM to handle imbalance.
- SHAP explainability: run `shap.TreeExplainer` on the trained LightGBM model.
  Output mean absolute SHAP values per feature per application.

**Output contract:**
```csv
application_id, vae_reconstruction_prob, rule_violation_score,
rule_codes_fired, lgbm_risk_score, top_shap_features
```

---

## 5. Tech Stack

### 5.1 Confirmed Libraries

| Purpose | Library | Version Constraint |
|---|---|---|
| Data loading & manipulation | `pandas` | >= 1.5 |
| Numerical ops | `numpy` | >= 1.23 |
| MI & mRMR | `scikit-learn` | >= 1.2 |
| VAE implementation | `torch` (PyTorch) | >= 2.0 |
| Gradient boosting | `lightgbm` | >= 4.0 |
| SHAP explainability | `shap` | >= 0.44 |
| Serialisation | `json`, `pickle` | stdlib |
| Evaluation metrics | `scikit-learn` metrics | PR-AUC, F1, MCC |

### 5.2 NOT in Stack (Do Not Introduce Without Discussion)

- `tensorflow` / `keras` — PyTorch is chosen for VAE, do not switch
- `xgboost` — LightGBM is primary; XGBoost only as fallback if discussed
- QAOA / Qiskit — quantum components explicitly excluded from this pipeline
- SMOTE / oversampling — not used; class imbalance handled via `scale_pos_weight`
  and rule-based weak labels instead
- Any autoML library (e.g. `AutoGluon`, `H2O`) — not in scope

---

## 6. Evaluation Standards

**Primary metric:** PR-AUC (Precision-Recall AUC).
> Rationale: ROC-AUC is optimistic on imbalanced datasets. PR-AUC reflects
> precision/recall trade-offs at the positive (fraud) class level. Source:
> AED-LGB paper (PMC 2024) and Fraud Detection Handbook.

**Secondary metrics:** F1-score (macro), MCC (Matthews Correlation Coefficient),
Brier Score.

**What NOT to use as primary metric:** Accuracy (misleading — 99.97% accuracy
achieved by predicting everything as valid).

**Threshold tuning:** Sweep `[0.2, 0.8]` in steps of `0.02` on validation set.
Select threshold maximising F1 on the positive class.

| Metric | v1.4 Canonical Run (2026-06-17) |
|---|---|
| PR-AUC | 0.9906 |
| Best F1-Score | 0.9509 |
| MCC | 0.9434 |
| Brier Score | 0.0299 |
| Optimal Threshold | 0.7749 |
| Positive weak labels | 1,986 of 15,000 |
| Features used by LightGBM | 50 of 56 |
| Features pruned by SHAP | 11 |

> Any future change that drops PR-AUC below 0.95 or MCC below 0.90 must be
> investigated before merging. These are the floor values for both baselines.
>
> **Note:** Re-baselined from a single verified run on 2026-06-17; prior figures in this section are superseded.
> **Caveat on PR-AUC:** The 0.9906 PR-AUC measures LightGBM's agreement with the system's own rule-based weak labels, *not* with confirmed real fraud. It proves the classifier successfully learned the rule boundaries, but true real-world fraud PR-AUC remains unmeasured due to lack of ground truth labels.

---

## 7. Hard Constraints (Agent Must Enforce These)

1. **Train the VAE on the entire dataset.** The 4 flagged records are treated as valid/clean because their duplicate counterparts are not present in this independent slice.
2. **Never use `sanity` as a feature.** It is the target leakage column.
3. **Never use `application_id` as a feature.** It is a row identifier.
4. **Never use `jwt` as a feature.** It is a system token, 100% null equivalent.
5. **Do not impute 100% null columns.** Drop them.
6. **Do not impute missing bank fields.** They are structurally absent.
7. **All file paths must be relative** — no hardcoded absolute paths.
8. **`selected_features.json` is the only interface** between File 1 and File 2.
   File 2 must load features from this JSON, never hardcode feature names.
9. **Rule violation scoring must be re-runnable independently** — it must not
   depend on a trained model checkpoint.
10. **SHAP output is mandatory** — every application flagged at high risk must have
    human-readable top-3 SHAP feature explanations.
11. **`rule_violation_score` must never be an input feature to LightGBM.** It is
    the training label. Using it as a feature is target leakage.
12. **`selected_features_shap_pruned.json` is the only valid output path for the
    SHAP pruner.** Writing pruned features back to `selected_features.json` from
    any script other than `feature_selection.py` is a hard contract violation.
    `vae_detection.py` contains a runtime guard that raises `RuntimeError` if it
    detects a `dropped_features` key in `selected_features.json`.

---

## 8. Research References the Agent May Cite

When providing advice, prefer citing from this list. These are the sources
this project's architecture is built on:

| Short Ref | Full Citation |
|---|---|
| `AED-LGB 2024` | "An AutoEncoder enhanced LightGBM method for credit card fraud detection", PMC / NCBI, 2024 |
| `VAE-AnomalyProb` | "Variational Autoencoder based Anomaly Detection using Reconstruction Probability", Semantic Scholar |
| `FraudHandbook` | "Reproducible ML for Credit Card Fraud Detection — Practical Handbook", fraud-detection-handbook.github.io |
| `LatentFeatureSel` | "Anomaly Detection Based on Selection and Weighting in Latent Space", arXiv:2103.04662 |
| `CleverCatch` | "CleverCatch: A Knowledge-Guided Weak Supervision Model for Fraud Detection", arXiv:2510.13205 |
| `SemiSupFraud` | "Deep Semi-Supervised Anomaly Detection for Finding Fraud in the Futures Market", arXiv:2309.00088 |
| `ETHContextStudy` | ETH Zurich empirical study on agent context file performance vs auto-generation, 2025 |
| `AnthropicContextEng` | "Effective context engineering for AI agents", Anthropic Engineering Blog |

---

## 9. What This Agent Should NEVER Do

- **Do not propose SMOTE, ADASYN, or oversampling.** The imbalance strategy
  is `scale_pos_weight` + rule-based weak labels. Oversampling on 4 real fraud
  samples is statistically invalid.
- **Do not propose pure unsupervised Isolation Forest** as the final model.
  Research (arXiv:2309.00088) shows pure unsupervised performance is often
  too low to be useful for fraud. The hybrid VAE + LightGBM is chosen.
- **Do not recommend self-supervised pretraining (SCARF, SAINT)** as a
  standalone approach. Research shows SSL does not reliably improve tabular
  anomaly detection. See prior session research.
- **Do not introduce a third Python file** without explicit user approval.
  The two-file architecture is a deliberate, researched decision.
- **Do not change the output contract** of either file without flagging it.
  Downstream consumers depend on `selected_features.json` and `risk_scores.csv`.
- **Do not add the 4 known fraud cases to the training set.** This is target
  leakage. Their role is evaluation only.

---

## 10. Open Questions (Agent Should Surface These When Relevant)

These are unresolved design decisions. Surface the relevant one when working
in the related area, but do not resolve them autonomously:

- [ ] **VAE architecture depth:** How many encoder/decoder layers? What bottleneck
  dimension? (Depends on final feature count from File 1.)
- [ ] **Rule severity weights:** The weights in Section 4.3 Stage B are starting
  values. Domain expert input needed to finalise.
- [ ] **Threshold for `rule_violation_score`:** What score constitutes a "positive"
  weak label? Binary (>0) or weighted continuous?
- [ ] **Handling of new data batches:** Should `feature_selection.py` re-run on
  new data or use the saved `selected_features.json`?
- [ ] **External data integration:** AISHE/DISE institute enrollment data would
  unlock rules UA–UI. This is out of scope for v1 but should be flagged.
- [ ] **Focal Loss vs Binary Cross-Entropy:** Should LightGBM switch to Focal Loss
  to reduce threshold strictness? Requires testing PR-AUC delta.
- [x] **Known frauds not caught by rules:** All 4 confirmed fraud records have
  `Rules Fired: None`. Their sanity flags (A2, M, A1A2GM, A1M) suggest identity
  duplication, but the rule engine is not firing. *Resolved:* Confirmed that the duplicate partner records are outside this 15,000-record slice. The slice is treated as independent, and these records are assumed valid.
- [ ] **DataFrame fragmentation warning:** `feature_selection.py` emits
  `PerformanceWarning` during Layer 2 flag engineering due to sequential column
  inserts. Non-blocking but should be refactored to `pd.concat` when convenient.
- [ ] **`state_match_flag` — non-functional placeholder (pending AISHE/DISE data):**
  No distinct institution-state column exists in this CSV slice (all `*state*`
  columns are aliases of `domicile_state_id`). The feature is hardcoded to
  constant 1, giving it zero variance and guaranteed SHAP = 0. This is NOT a
  fix — it is a documented placeholder. The real fix requires integrating
  AISHE/DISE institute-location data (see Section 10: External data integration).
  Do not mark as resolved until institution-state data is available.
- [x] **`is_father_name_eq_mother` missing bridge:** This feature was
  left out of the v1.2 weak-label bridge work. Needs a single `add_violation()`
  call in `apply_rules()` in `vae_detection.py`. *Resolved:* Confirmed via v1.4 canonical run code that `FM_ENG` violation is present and executing. (Note: SHAP pruning shows it is still pruned due to low variance, as expected).
- [x] **`min_verify_by` distribution unknown:** Must check
  `df['min_verify_by'].value_counts()` before diagnosing. May be near-constant. *Resolved:* Verified it is 98.5% NaN (sparse/near-constant).

---

## 11. Research-Backed Improvement Decisions (Implemented)
<!-- SOURCE: Researched June 2026 via PLOS 2024, ResearchGate, PoliMi DSAEE, TechScience 2024 -->
<!-- STATUS: ALL IMPLEMENTED as of v1.2 (June 2026) -->

---

### 11.1 Relational Boolean Flags (Target: `feature_selection.py`)
<!-- STATUS: IMPLEMENTED v1.1 -->

**Problem identified:** `feature_selection.py` ran `select_dtypes(include=[np.number])`
which permanently deleted all text and categorical columns before any model saw them.
This made the model completely blind to identity-match fraud.

**Decision:** Added 6 boolean policy flags in `engineer_features()` inside
`feature_selection.py`, BEFORE the `select_dtypes` call. See code for full list.

**Agent rule:** These flags must NEVER be dropped by the MI or mRMR filters.
They are in the `protected_features` list that bypasses the cutoff logic.

---

### 11.2 Loosen Feature Selection Cutoffs (Target: `feature_selection.py`)
<!-- STATUS: IMPLEMENTED v1.1 -->

**Problem identified:** `MI_KEEP_RATIO = 0.5` and `MRMR_MAX_FEATURES = 20` discarded
critical fraud signals.

**Decision:** Updated constants:
```python
MI_KEEP_RATIO     = 0.8
MRMR_MAX_FEATURES = 40
```

---

### 11.3 LightGBM Classifier Tuning (Target: `vae_detection.py`)
<!-- STATUS: IMPLEMENTED v1.1 -->

**Problem identified:** Synthetic anomaly test returned optimal threshold of 0.9935 —
extreme conservatism caused by binary cross-entropy overwhelming the minority class.

**Decision:** Updated `train_lgbm()`:
```python
clf = lgb.LGBMClassifier(
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,
    extra_trees=True,
    min_child_samples=5,
    random_state=42
)
```

---

### 11.4 SHAP-Based Second-Pass Feature Pruning (Target: `evaluate_model.py`)
<!-- STATUS: IMPLEMENTED v1.1 -->

**Decision:** After running the full pipeline once, `evaluate_model.py` computes
mean absolute SHAP values per feature across all 15,000 applications and saves a
pruned feature list to `selected_features_shap_pruned.json`, dropping anything
with mean |SHAP| below 0.001.

**Constraint:** SHAP pruner output MUST be saved to `selected_features_shap_pruned.json`.
Writing to `selected_features.json` from any script other than `feature_selection.py`
is a hard contract violation. A runtime guard in `vae_detection.py` enforces this.

---

### 11.5 Weak Label Feedback Loop Fix (Target: `vae_detection.py`)
<!-- STATUS: IMPLEMENTED v1.2 (June 2026) -->
<!-- SOURCE: Diagnosed from SHAP output showing all 22 protected features at SHAP = 0.0 -->

**Problem identified:** SHAP second-pass pruning revealed all 22 protected engineered
features had mean |SHAP| ≈ 0.0 after the v1.1 run. LightGBM was ignoring every
Layer 0/1/2 feature entirely.

**Root cause:** The weak label generator in `apply_rules()` only fired NIC rulebook
codes (YF, X1, X7, YK, UN etc.). The newer engineered features (like
`is_applicant_name_eq_father`, `ip_application_count`, `flag_fee_exceeds_income`)
had no corresponding `add_violation()` calls. Because `rule_violation_score > 0`
is the training label, any feature that doesn't correlate with existing rule firings
gets SHAP ≈ 0 and is discarded by the model.

**Decision:** Extended `apply_rules()` in `vae_detection.py` with 8 new guarded
`add_violation()` calls covering the engineered feature violations. See Section 3.1
for the full table of new rule codes (IP_CONC_ENG, YF_ENG, YF_MOTHER_ENG, FEE_ENG,
INC_EXT_ENG, UN_FATHER_ENG, X1_ENG, X7_ENG).

**Verification results (v1.4 Canonical Run):**

| Metric | v1.4 Canonical |
|---|---|
| PR-AUC | 0.9906 |
| Best F1 | 0.9509 |
| MCC | 0.9434 |
| Brier Score | 0.0299 |
| Optimal Threshold | 0.7749 |
| Positive weak labels | 1,986 |
| SHAP-pruned features | 11 |

> **Note:** Re-baselined from a single verified run on 2026-06-17; prior figures in this section are superseded.

**Phase D Synthetic Ablation Findings (VAE vs Bridges):**

**v1.4 Permanent Test Harness:** The corrected, properly-aligned synthetic dataset (Phase D) and fixed ablation methodology are the new permanent test harness going forward. They supersede all v1.x synthetic tests, as the old categories failed to correctly exercise the engineered bridges.

Ablation testing on these corrected synthetic fraud injections (IP clustering, mother-name collisions, fee inflation) reveals the structural relationship between the VAE and the rule bridges. Raw stdout receipts:

*Ablation Deltas (Recall at 0.9956 threshold):*
```text
AGE_VIOLATION: full=0.9533  ablated=0.9933  delta=-0.0400
INCOME_VIOLATION: full=1.0000  ablated=1.0000  delta=0.0000
IP_CONCENTRATION: full=1.0000  ablated=0.0000  delta=1.0000
MOTHER_NAME_COLLISION: full=0.9733  ablated=0.0067  delta=0.9667
FEE_INFLATION: full=1.0000  ablated=0.0000  delta=1.0000
```

*VAE Independent Signal (ROC-AUC & PR-AUC):*
```text
INCOME_VIOLATION (n=150) -> ROC-AUC: 0.9465 | PR-AUC: 0.1162
AGE_VIOLATION (n=150) -> ROC-AUC: 0.8737 | PR-AUC: 0.0506
MOTHER_NAME_COLLISION (n=150) -> ROC-AUC: 0.8012 | PR-AUC: 0.0258
FEE_INFLATION (n=150) -> ROC-AUC: 0.7961 | PR-AUC: 0.0264
IP_CONCENTRATION (n=150) -> ROC-AUC: 0.7672 | PR-AUC: 0.0239
```

- **VAE Independent Signal:** The VAE carries genuine, non-zero unsupervised signal across every fraud category (ROC-AUC 0.76 - 0.94 when scored independently). It is strongest on univariate violations and weaker-to-moderate on relational fraud.
- **Methodology Note (Untested Variance):** Because the VAE must be trained on the full dataset, its training data *included* the synthetic anomalies (no held-out test yet). It had a chance to partially adapt to them, meaning we cannot say whether real-world performance on truly novel fraud would be better or worse than what is shown here. The result is untested in either direction.
- **Role of the Bridges:** Because the relational fraud signals are moderate (PR-AUC < 0.10 when relying on VAE alone), the VAE alone cannot draw a clean threshold without unacceptable false alarms. The engineered rule bridges (`IP_CONC_ENG`, `FEE_ENG`, `FM_ENG`) are essential. They do not replace a "failed" VAE; instead, they convert the VAE's weaker relational signal into a usable, hard decision boundary for the supervised LightGBM classifier.

**Confirmed non-zero SHAP after fix:**
- `flag_fee_exceeds_income` — appears in SHAP explanations of 2 of 4 known frauds
- `name_similarity_score` — survived pruning, present in known fraud explanations
- `mobile_application_count` — survived pruning

**Remaining zero-SHAP protected features (11 dropped by pruner) — v1.3 Root-Cause Diagnosis:**

| Feature | Root Cause | Action |
|---|---|---|
| `state_match_flag` | **Placeholder — constant 1.** No institution-state column exists in this CSV slice. Hardcoded to 1 pending AISHE/DISE data integration. Zero variance → SHAP = 0 by Dummy property. This is expected and documented. | **No fix possible on current data.** Dormant guard pending AISHE/DISE data. |
| `is_father_name_eq_mother` | **Missing bridge.** Not wired into any `add_violation()` call in `apply_rules()`. The v1.2 bridge work covered 8 features but missed this one. | **FIX in v1.3:** Add bridge in `vae_detection.py`. |
| `flag_prematric_age_over20` | **Constant 0.** No pre-matric applicant in this slice exceeds age 20. | **No fix possible on current data.** Dormant guard for future slices. |
| `flag_postmatric_age_under13` | **Constant 0.** Minimum post-matric age in this slice is 13.14. | **No fix possible on current data.** Dormant guard for future slices. |
| `is_applicant_name_eq_father` | **Redundant.** Hard 0/1 cutoff of `name_similarity_score` (continuous). SHAP attributes the signal to the continuous version. | **Expected behavior. No action needed.** |
| `is_applicant_name_eq_mother` | **Redundant.** Same mechanism as above. | **Expected behavior. No action needed.** |
| `flag_income_below_10000` | **Redundant.** Trees split directly on raw `annual_family_income` at whatever threshold they want, making the pre-baked binary version superfluous. | **Expected behavior. No action needed.** |
| `flag_income_extreme_low` | **Redundant.** Same mechanism as above. | **Expected behavior. No action needed.** |
| `pre_post_matric` | **Subsumed.** Standalone signal fully captured by AND-conditions that combine it with age flags. | **Expected behavior. No action needed.** |
| `modeofstudy` | **Near-constant.** 99.95% of records share one value (8 of 15,000 differ). Also never wired into any rule. | **No fix possible on current data.** |
| `min_verify_by` | **Sparse / Near-constant.** 98.5% of values are NaN. | **No fix possible on current data.** |

> **v1.3 Execution Plan (approved):**
> 1. Verify a distinct institution-state column exists in the CSV (e.g., `c_state_id`).
> 2. Fix `state_match_flag` definition in `feature_selection.py`.
> 3. Add `is_father_name_eq_mother` bridge in `vae_detection.py` → `apply_rules()`.
> 4. Check `min_verify_by` distribution.
> 5. Re-run pipeline and compare SHAP values + PR-AUC against v1.2 baseline.
>
> Do not remove any of the 11 features from the protected list — the redundant
> and constant ones serve as dormant guards for future data distributions where
> violations may appear.

---
<!-- END OF AGENT CONTEXT FILE -->
<!-- Total sections: 11 | Human-curated: YES -->
<!-- Last updated: 2026-06 (v1.3-planning) | Reason: Zero-SHAP root-cause diagnosis + v1.3 execution plan -->

