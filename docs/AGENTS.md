# NIC Fraud Detection — Agent Context File
<!-- VERSION: 1.0 | OWNER: Project Lead | LAST REVIEWED: 2026-06 -->
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

**Output contract:**
```json
{
  "selected_features": ["age_at_registration", "fee_income_ratio", ...],
  "feature_scores": {"age_at_registration": 0.42, ...},
  "n_selected": 20,
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
- Run only the EVALUABLE rules from Section 3.1.
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
  [selected_features from JSON] + [vae_reconstruction_prob] + [rule_violation_score]
  ```
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

---

## 7. Hard Constraints (Agent Must Enforce These)

1. **Never train the VAE on the 4 flagged records.** Training set = `sanity.isnull()`.
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
- [ ] **SHAP-based second-pass feature pruning:** After the first full pipeline run,
  mean SHAP values can be used to drop near-zero features from selected_features.json
  for a tighter, self-validating feature set.

---

## 11. Research-Backed Improvement Decisions (Approved by Project Lead — v1.1)
<!-- SOURCE: Researched June 2026 via PLOS 2024, ResearchGate, PoliMi DSAEE, TechScience 2024 -->
<!-- STATUS: Pending implementation — do not implement without explicit instruction -->

These are **locked-in next steps** approved after synthetic anomaly testing revealed
specific model blind spots. All agents must be aware of these before touching either file.

---

### 11.1 Relational Boolean Flags (Target: `feature_selection.py`)

**Problem identified:** `feature_selection.py` runs `select_dtypes(include=[np.number])`
which permanently deletes all text and categorical columns before any model sees them.
This made the model completely blind to identity-match fraud (applicant_name == father_name).

**Research backing:** SCIRP 2024 and fraud-detection-handbook.github.io confirm that
"relational aggregate features" are the highest-value feature category in tabular fraud.

**Decision:** Add the following 6 boolean policy flags in `engineer_features()` inside
`feature_selection.py`, BEFORE the `select_dtypes` call:

```python
# File: feature_selection.py — add inside engineer_features()
df['is_name_match_father']  = (df['applicant_name'] == df['father_name']).astype(int)
df['is_name_match_mother']  = (df['applicant_name'] == df['mother_name']).astype(int)
df['is_income_below_10k']   = (pd.to_numeric(df['annual_family_income'], errors='coerce') <= 10000).astype(int)
df['is_income_below_20k']   = (pd.to_numeric(df['annual_family_income'], errors='coerce') <= 20000).astype(int)
df['is_prematric_overage']  = ((df['pre_post_matric'] == 1) & (df['age_at_registration'] > 20)).astype(int)
df['is_postmatric_overage'] = ((df['pre_post_matric'] == 2) & (df['age_at_registration'] > 35)).astype(int)
```

**Agent rule:** These 6 flags must NEVER be dropped by the MI or mRMR filters.
Add them to a `protected_features` list that bypasses the cutoff logic.

---

### 11.2 Loosen Feature Selection Cutoffs (Target: `feature_selection.py`)

**Problem identified:** `MI_KEEP_RATIO = 0.5` and `MRMR_MAX_FEATURES = 20` discard
critical fraud signals that have moderate MI scores but high domain relevance (e.g.
`annual_family_income` which was lost in the synthetic test).

**Research backing:** PoliMi Deep Sparse Autoencoder Ensemble (DSAEE 2024) recommends
keeping features with reconstruction-error relevance, not just MI relevance. A hard cap
of 20 on a 136-column dataset is overly aggressive.

**Decision:** Update the constants in `feature_selection.py`:
```python
# Old values — too aggressive
MI_KEEP_RATIO    = 0.5
MRMR_MAX_FEATURES = 20

# New values — approved v1.1
MI_KEEP_RATIO    = 0.8
MRMR_MAX_FEATURES = 40
```

---

### 11.3 LightGBM Classifier Tuning (Target: `vae_detection.py`)

**Problem identified:** Synthetic anomaly test returned an optimal threshold of `0.9935`.
This extreme conservatism is caused by binary cross-entropy treating all 14,996 clean
records with equal weight, drowning out the 750 fraud signals.

**Research backing:** TechScience 2024 and ResearchGate (Cost-sensitive Focal Loss for
fraud) confirm that `extra_trees=True` + increased `n_estimators` improves minority class
recall without SMOTE. Focal Loss (Facebook AI, Lin et al. 2017) is the gold standard fix.

**Decision:** Update `train_lgbm()` in `vae_detection.py`:
```python
# Old
clf = lgb.LGBMClassifier(
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# New — approved v1.1
clf = lgb.LGBMClassifier(
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,
    extra_trees=True,
    min_child_samples=5,
    random_state=42
)
```

**Future option (flag before implementing):** Switch to XGBoost with
`eval_metric='aucpr'` and native Focal Loss for a further threshold improvement.

---

### 11.4 SHAP-Based Second-Pass Feature Pruning (Target: `evaluate_model.py`)

**Research backing:** PoliMi DSAEE 2024 — using the downstream classifier's own
attention (via SHAP) as a filter to remove features that the model never actually uses.

**Decision:** After running the full pipeline once, `evaluate_model.py` should compute
mean absolute SHAP values per feature across all 15,000 applications and output a
pruned feature list, dropping anything with mean |SHAP| below a threshold (e.g., 0.001).
This creates a self-validating, data-driven feature set for the next training cycle.

**Status:** Not yet implemented. Flag this as an open question before implementing.

---
<!-- END OF AGENT CONTEXT FILE -->
<!-- Total sections: 11 | Estimated tokens: ~2,400 | Human-curated: YES -->
<!-- Last updated: 2026-06 | Reason: Post synthetic anomaly test findings + research -->

