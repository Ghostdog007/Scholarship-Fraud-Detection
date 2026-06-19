# NIC Fraud Detection — v2 Rule-Free Architecture

> **National Informatics Centre (NIC) | Pre-Matric & Post-Matric Scholarship Portal**
> Anomaly detection for 15,000 fresh scholarship applications.
> Version: 2.0-draft | Last reviewed: June 2026

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Why v2 Exists — The Rule Ceiling Problem](#2-why-v2-exists--the-rule-ceiling-problem)
3. [Architecture at a Glance](#3-architecture-at-a-glance)
4. [Module Reference](#4-module-reference)
5. [Data Facts](#5-data-facts)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Evaluation Standards](#7-evaluation-standards)
8. [Tech Stack](#8-tech-stack)
9. [Hard Constraints](#9-hard-constraints)
10. [Research Foundations](#10-research-foundations)
11. [Open Questions](#11-open-questions)
12. [v1.5 vs v2 — Architecture Choice Log](#12-v15-vs-v2--architecture-choice-log)

---

## 1. What This System Does

This system produces a per-application **anomaly risk score (0–1)** for scholarship applications submitted through the NIC national portal. The dataset is 99.97% valid — only 4 human-confirmed fraud records exist across 15,000 applications — making standard supervised classification impossible.

v2 detects fraud by learning what *normal* looks like, from two independent perspectives:

- **Tabular normality** — what does a statistically typical application look like across 42+ engineered features? Applications that reconstruct poorly under a trained VAE are anomalous.
- **Relational normality** — what does a typical applicant's identity neighborhood look like in a graph of shared mobile numbers, IP addresses, and names? Nodes that embed far from the normal cluster, or whose neighborhoods reconstruct poorly, are anomalous.

Both signals are combined through a LightGBM fusion classifier whose labels are derived entirely from statistical tail detection (EVT) and self-training — no human-authored rules.

---

## 2. Why v2 Exists — The Rule Ceiling Problem

v1 used 99 NIC revalidation rules as weak supervision labels. This worked well for known fraud patterns — the v1.4 canonical run reached PR-AUC 0.9906 against its own rule labels. But that metric measured how well the model learned the rule boundaries, not how well it detected real fraud.

**The structural limitation:** any fraud pattern a human had not already written a rule for could never become a positive training example. The model's ceiling was the rule author's knowledge.

v2 removes this ceiling entirely. The only domain knowledge that enters the system is:

1. **Which relationships are structurally meaningful** — shared mobile, IP, name, pincode. This defines *what a relationship is*, not what constitutes fraud.
2. **What anomalous configurations look like during pretraining** — synthetic examples constructed programmatically from known fraud archetypes (IP clustering, name collisions, fee inflation). These shape the model's latent geometry in Stage 1 only; Stage 2 discovers freely.
3. **A statistical definition of "extreme"** — a Generalized Pareto Distribution fit to the model's own score distributions. One human-set parameter: the acceptable false-positive rate `q`.

Everything else — what normal looks like, where the fraud boundary sits, which novel patterns cross it — is learned from data.

---

## 3. Architecture at a Glance

```
application records (data_for_ml_model.csv)
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
tabular_feature_engine_v2.py              graph_builder_v2.py
(structural/statistical features only)    (typed identity graph)
        │                                             │
        ▼                                             ▼
tabular_vae_v2.py                         graph_autoencoder_v2.py
Stage 1: synthetic anomaly exposure       DOMINANT dual-decoder
Stage 2: free reconstruction              + DeepSVDD hypersphere
        │                                             │
        │  vae_anomaly_score                          │  graph_anomaly_score
        │  recon_error_vector                         │  attr_recon_error
        │                                             │  struct_recon_error
        └──────────────────┬──────────────────────────┘
                           ▼
                   evt_scorer.py
           GPD fit via Peak Over Threshold
           one parameter: q (false-positive rate)
                           │
                           ▼
               self_training_loop_v2.py
          Round 0: EVT mutual tail only
          Round 1+: EVT + classifier agreement
                           │
                           ▼
               fusion_classifier_v2.py
           LightGBM on scalar inputs only
           labels: EVT-confirmed + pseudo-positives
                           │
                           ▼
                  xai_layer_v2.py
        SHAP + GNNExplainer/PGExplainer
        + per-feature reconstruction decomposition
                           │
                           ▼
          risk_scores_v2.csv + explanation_cards_v2.json
```

**Score direction convention:** `vae_anomaly_score` and `graph_anomaly_score` are both defined as **higher = more anomalous** throughout the entire pipeline. This is the inverse of v1's `vae_reconstruction_prob`. All modules enforce this convention.

---

## 4. Module Reference

### 4.1 `tabular_feature_engine_v2.py`

Produces all tabular node features. No rule flags. No policy-boundary thresholds. Raw structural and statistical features only.

**Reads:** `data_for_ml_model.csv`
**Writes:** `engineered_features_v2.csv`, `v2_feature_schema.json`

Key engineered features:

| Feature | Definition |
|---|---|
| `age_at_registration` | `(registered_date - date_of_birth).days / 365.25` |
| `fee_income_ratio` | `(admission_fee + tution_fee + misc_fee) / annual_family_income` |
| `name_similarity_score` | `SequenceMatcher` ratio between `applicant_name` and `father_name` |
| `mobile_application_count` | Count of applications sharing the same `mobile_no` |
| `ip_application_count` | Count of applications sharing the same `ip_address` |
| `mobile_unique_names` | `nunique(applicant_name)` per `mobile_no` |
| `mobile_unique_fathers` | `nunique(father_name)` per `mobile_no` |
| `institute_application_count` | Count per `c_institution_id` |
| `ip_to_mobile_ratio` | `ip_application_count / (mobile_application_count + 1)` |
| `is_applicant_name_eq_father` | `int(applicant_name == father_name)` |
| `is_applicant_name_eq_mother` | `int(applicant_name == mother_name)` |
| `is_father_name_eq_mother` | `int(father_name == mother_name)` |

**What is explicitly absent vs v1:** no `flag_income_below_10000`, no `flag_fee_exceeds_income`, no `flag_prematric_age_over20`. The raw continuous signals are present; the model learns the boundary, not us.

---

### 4.2 `graph_builder_v2.py`

Builds the typed identity graph from application records.

**Reads:** `engineered_features_v2.csv`
**Writes:** `identity_graph.pt` — PyG `HeteroData` object

Edge types: `shares_mobile`, `shares_ip`, `shares_father_name`, `shares_mother_name`, `shares_pincode`

Implementation note: build with `networkx` first for visual inspection of cluster sizes, convert to PyG last. 15,000 nodes — no mini-batching needed.

---

### 4.3 `tabular_vae_v2.py`

Learns what normal looks like in the tabular feature space.

**Reads:** `v2_feature_schema.json`, `engineered_features_v2.csv`, `synthetic_exposure_set.pt`
**Writes:** `vae_v2_scores.csv` (`application_id`, `vae_anomaly_score`, `recon_error_vector`)

Training curriculum (Outlier Exposure + LOE, ICLR 2019 + ICML 2022):

```
Stage 1 — Synthetic anomaly exposure pretraining
  L = L_reconstruction(normal_data)
      + λ(t) · L_exposure(synthetic_anomaly_set)
  λ(t) decays to 0 over Stage 1.
  Pushes synthetic anomaly embeddings away from the normal cluster centroid.

Stage 2 — Free reconstruction discovery
  λ(t) = 0. Train purely on reconstruction across all 15,000 records.
  Anomaly score = exp(-MSE), inverted so higher = more anomalous.
  Stage 1 auxiliary head is frozen and not used for scoring.
```

---

### 4.4 `graph_autoencoder_v2.py`

Learns what normal looks like in the identity graph. **No raw node embeddings leave this module** — only scalar scores and decomposed reconstruction errors.

**Reads:** `identity_graph.pt`, `synthetic_exposure_set.pt`
**Writes:** `graph_v2_scores.csv` (`application_id`, `graph_anomaly_score`, `attr_recon_error`, `struct_recon_error`)

Architecture (DOMINANT + DeepSVDD, SDM 2019 + ICML 2018):

- **DOMINANT dual-decoder:** GCN encoder → node embeddings → attribute decoder (reconstructs features) + structure decoder (dot-product adjacency reconstruction). Anomaly score: weighted sum of both reconstruction errors.
- **DeepSVDD hypersphere complement:** secondary objective minimizes the hypersphere enclosing normal node embeddings. Distance from centroid catches anomalies that reconstruct well but embed far from the normal cluster — a failure mode DOMINANT alone misses.
- Same Stage 1 / Stage 2 curriculum as §4.3.

---

### 4.5 `evt_scorer.py`

Derives statistically-grounded anomaly thresholds from the score distributions themselves — no domain rules.

**Reads:** score series from `tabular_vae_v2.py` or `graph_autoencoder_v2.py`
**Writes:** `evt_thresholds_v2.json`

```json
{
  "vae_anomaly_score":   {"threshold": 0.74, "q": 0.002, "method": "POT-GPD"},
  "graph_anomaly_score": {"threshold": 0.68, "q": 0.002, "method": "POT-GPD"}
}
```

Method (SPOT, KDD 2017): Peak Over Threshold with GPD fit via `scipy.stats.genpareto`. The only human-set value is `q` — the acceptable false-positive rate.

---

### 4.6 `self_training_loop_v2.py`

Grows the positive label set beyond the EVT-tail cold start, round by round.

**Reads:** `vae_v2_scores.csv`, `graph_v2_scores.csv`, `evt_thresholds_v2.json`, current classifier predictions
**Writes:** `pseudo_labels_v2.json`

Promotion rule (ASTRA-style, NAACL 2021): a record is promoted to pseudo-positive only if **all three** hold:

1. `vae_anomaly_score` exceeds its EVT threshold
2. `graph_anomaly_score` exceeds its EVT threshold
3. The fusion classifier assigns probability ≥ its EVT-derived threshold *(waived at round 0 — code-enforced, not just documented)*

**Rounds are not automatic.** Each round requires a Phase D PR-AUC check before the label set is used for the next cycle.

---

### 4.7 `fusion_classifier_v2.py`

Combines all scalar anomaly signals into a final risk score.

**Reads:** tabular features, `vae_anomaly_score`, `recon_error_vector` columns, `graph_anomaly_score`, `attr_recon_error`, `struct_recon_error`, `pseudo_labels_v2.json`
**Writes:** `risk_scores_v2.csv`

```csv
application_id, vae_anomaly_score, graph_anomaly_score,
lgbm_risk_score_v2, label_source, top_shap_features
```

`label_source` values: `evt_cold_start`, `self_training_round_N`, `negative`. This field is metadata for audit only — never a feature.

---

### 4.8 `xai_layer_v2.py`

Produces a human-readable explanation card for every flagged application across all three signal channels.

**Reads:** trained fusion classifier, trained graph AE, `recon_error_vector`
**Writes:** `explanation_cards_v2.json`

Explanation mechanisms by signal type:

| Signal | Method |
|---|---|
| Classifier-level | `shap.TreeExplainer` |
| Graph-level | `GNNExplainer` per-case; `PGExplainer` at volume scale |
| Reconstruction-level | Per-feature MSE from `recon_error_vector` |
| Hypersphere-level | Distance-to-centroid decomposition from DeepSVDD component |

The `narrative` field in each card is templated from the top signals — it replaces the `rule_codes_fired` column from v1 with a structural explanation (e.g., *"This application shares an IP address with 12 others and reconstructs poorly on name-similarity features."*).

---

## 5. Data Facts

### 5.1 Primary Dataset

| Property | Value |
|---|---|
| File | `data_for_ml_model.csv` |
| Rows | 15,000 |
| Columns | 136 |
| Fraud-labeled records | 4 (0.027%) — confirmed valid in this slice (duplicate partners outside boundary) |
| Applicant type | Fresh only (`fresh_renewal = 'F'`) |
| Pre-Matric (`pre_post_matric = 1`) | 5,073 |
| Post-Matric (`pre_post_matric = 2`) | 9,908 |

### 5.2 Columns to Drop at Load Time

Drop before any processing — confirmed 100% null:

```
updated_by, delete_record, deleted_by, delete_on, delete_ip_address,
deleted_by_level, c_university_id, p_institution_id, x_institution_id,
xii_institution_id, competitive_exam_score, xii_course_id,
new_entitled_fee_amount_centre_share, sub_category_id,
updated_by-2, updated_on-2
```

Drop confirmed duplicate columns, keeping one per group:

```
Keep: domicile_state_id, state_name, permanent_district_id, district_name
Drop: state_id, state_id-2, pfms_state_code, state_name-2,
      district_id, district_name-2
```

Also exclude from features: `sanity`, `application_id`, `jwt`.

### 5.3 Confirmed Missing Fields

Absent from the CSV entirely — do not proxy without instruction:

```
bank_account_no, bank_name, ifsc_code
```

### 5.4 Confirmed Data Anomalies (Graph Sanity Reference)

- 1 IP address submitted 39 applications. Top 10 IPs submitted 15–39 each.
- 1 mobile number shared by 6 applicants. 59 mobiles shared by 2.
- Family income as low as 5 INR.
- 3 Post-Matric applicants exceed the 35-year age limit but are not flagged in `sanity`.
- Institute `c_institution_id=10791` has 151 applications — highest concentration.

---

## 6. Running the Pipeline

Run modules in dependency order. Each step must complete before the next begins.

```bash
# Step 1 — Engineer tabular features
python tabular_feature_engine_v2.py
# Outputs: engineered_features_v2.csv, v2_feature_schema.json

# Step 2 — Build identity graph
python graph_builder_v2.py
# Outputs: identity_graph.pt

# Step 3 — Train tabular VAE (Stage 1 + Stage 2)
python tabular_vae_v2.py
# Outputs: vae_v2_scores.csv

# Step 4 — Train graph autoencoder (Stage 1 + Stage 2)
python graph_autoencoder_v2.py
# Outputs: graph_v2_scores.csv

# Step 5 — Fit EVT thresholds
python evt_scorer.py
# Outputs: evt_thresholds_v2.json

# Step 6 — Self-training round 0 (EVT mutual tail only)
python self_training_loop_v2.py --round 0
# Outputs: pseudo_labels_v2.json (round 0)
# *** Check Phase D PR-AUC before proceeding to round 1 ***

# Step 7 — Train fusion classifier
python fusion_classifier_v2.py
# Outputs: risk_scores_v2.csv

# Step 8 — Generate explanation cards
python xai_layer_v2.py
# Outputs: explanation_cards_v2.json
```

**Self-training rounds 1+:** re-run Steps 6→7→8 for each additional round. Each round requires a Phase D PR-AUC check before its pseudo-labels are used for the next cycle. Do not automate this loop.

**Synthetic exposure set:** `synthetic_exposure_set.pt` must be constructed programmatically before Steps 3 and 4 using the Phase D archetypes (IP clustering, name collisions, fee inflation, age violations). Do not generate it using CTGAN, TVAE, or any tabular GAN — see §10 for why.

---

## 7. Evaluation Standards

### 7.1 Primary Gate — Phase D Synthetic Harness

150 synthetic anomalies per category are injected into the real dataset. The v1 standalone-VAE figures below are the floor v2 must beat on every category:

| Category | What Is Injected | v1 VAE-Alone ROC-AUC | v1 VAE-Alone PR-AUC |
|---|---|---|---|
| INCOME_VIOLATION | Income < 1,000 INR | 0.9465 | 0.1162 |
| AGE_VIOLATION | Pre-matric age > 20 or post-matric age > 35 | 0.8737 | 0.0506 |
| MOTHER_NAME_COLLISION | `father_name == mother_name` | 0.8012 | 0.0258 |
| FEE_INFLATION | `fee_income_ratio > 1.0`, income > 20,000 | 0.7961 | 0.0264 |
| IP_CONCENTRATION | Same IP across 15+ rows | 0.7672 | 0.0239 |

v2's combined tabular VAE + graph AE must beat the relational category PR-AUC figures (IP_CONCENTRATION, MOTHER_NAME_COLLISION, FEE_INFLATION) before any production discussion.

### 7.2 Mandatory Ablations

Before claiming any component helps, report both sides of each ablation:

| Ablation | Question |
|---|---|
| Stage 1 vs no Stage 1 (`λ(t) ≡ 0`) | Does synthetic exposure pretraining actually improve Stage 2 PR-AUC? |
| DeepSVDD vs DOMINANT-only | Does the hypersphere loss add independent signal over reconstruction alone? |
| Self-training round N vs round 0 | Track Phase D PR-AUC across rounds. A declining trend is a stop signal. |

### 7.3 Explainer Faithfulness

For Phase D cases with a known injection mechanism, confirm GNNExplainer/PGExplainer's identified subgraph contains the planted relationship. Report as a hit rate, not just "explainer ran successfully."

---

## 8. Tech Stack

| Purpose | Library | Version | New in v2? |
|---|---|---|---|
| Data manipulation | `pandas` | >= 1.5 | No |
| Numerical ops | `numpy` | >= 1.23 | No |
| VAE / training | `torch` (PyTorch) | >= 2.0 | No |
| Fusion classifier | `lightgbm` | >= 4.0 | No |
| SHAP explainability | `shap` | >= 0.44 | No |
| Evaluation metrics | `scikit-learn` | >= 1.2 | No |
| Graph neural networks | `torch_geometric` (PyG) | >= 2.4 | **Yes** |
| Graph construction / inspection | `networkx` | >= 3.0 | **Yes** |
| EVT / GPD fitting | `scipy.stats.genpareto` | >= 1.11 | **Yes** |
| Graph explainability | `GNNExplainer`, `PGExplainer` (via PyG) | via torch_geometric | **Yes** |
| DeepSVDD | Custom (PyTorch) | — | **Yes** |

**Do not introduce:** `tensorflow`, `keras`, any tabular GAN (`CTGAN`, `TVAE`, `GaussianCopula`), SMOTE/oversampling, any AutoML library, `xgboost` without discussion.

---

## 9. Hard Constraints

1. **Never use `sanity`, `application_id`, or `jwt` as features.** These are leakage, identifier, and null-equivalent columns respectively.
2. **Drop all 16 confirmed 100% null columns at load time** (§5.2). Do not impute them.
3. **No raw GNN node embeddings leave `graph_autoencoder_v2.py`.** Only `graph_anomaly_score`, `attr_recon_error`, and `struct_recon_error` are valid exports. Violating this breaks the explainability guarantee.
4. **No rule codes, rule scores, or hand-set domain thresholds anywhere in this pipeline.** If you find yourself writing `ip_application_count >= 15` as a threshold or referencing a rule code like X1 or YF, stop — you are re-introducing rule dependency.
5. **`vae_anomaly_score` direction is higher = more anomalous throughout.** Any module that inverts this must document the inversion explicitly at the point of inversion.
6. **Self-training rounds are not automatic.** Each round's label set must clear a Phase D PR-AUC check before promotion.
7. **Round 0 EVT-only condition is code-enforced**, not just documented. The classifier-agreement condition (condition 3 in §4.6) must be code-gated off at round 0 and on from round 1.
8. **Synthetic exposure set is programmatically constructed.** A GAN-generated exposure set is a known failure mode on fraud data (arXiv:2604.13125 — composite degradation ratios of 24x+ on behavioral fraud patterns).
9. **GNNExplainer/PGExplainer outputs are not trusted by default.** Validate against Phase D cases before treating explanations as accurate.
10. **All file paths must be relative.** No hardcoded absolute paths.
11. **The 4 originally-flagged records are confirmed valid in this slice** and are not used as an evaluation target anywhere in v2.

---

## 10. Research Foundations

| Reference | Citation | Grounds |
|---|---|---|
| `OutlierExposure` | Hendrycks, Mazeika, Dietterich. "Deep Anomaly Detection with Outlier Exposure." ICLR 2019. arXiv:1812.04606 | Synthetic anomaly exposure curriculum |
| `LOE` | Qiu et al. "Latent Outlier Exposure for Anomaly Detection with Contaminated Data." ICML 2022. arXiv:2202.08088 | Cold-start joint label inference; no pre-labeled anomalies needed |
| `DOMINANT` | Ding, Li, Bhanushali, Liu. "Deep Anomaly Detection on Attributed Networks." SDM 2019 | Dual-decoder graph AE architecture |
| `DeepSVDD` | Ruff et al. "Deep One-Class Classification." ICML 2018 | Hypersphere anomaly boundary; score = distance from normal centroid |
| `EVT-SPOT` | Siffer, Fouque, Termier, Largouet. "Anomaly Detection in Streams with Extreme Value Theory." KDD 2017 | Statistically-derived thresholds; replaces all hand-set numeric cutoffs |
| `ASTRA-SelfTrain` | Karamanolakis et al. "Self-Training with Weak Supervision." NAACL 2021. arXiv:2104.05514 | Self-training promotion mechanism |
| `R-GCN` | Schlichtkrull et al. "Modeling Relational Data with Graph Convolutional Networks." ESWC 2018 | Typed-edge encoder for heterogeneous graph |
| `GNNExplainer` | Ying et al. "GNNExplainer: Generating Explanations for Graph Neural Networks." NeurIPS 2019. arXiv:1903.03894 | Per-case graph explanations |
| `PGExplainer` | Luo et al. "Parameterized Explainer for Graph Neural Network." NeurIPS 2020. arXiv:2011.04573 | Volume-scale upgrade from GNNExplainer |
| `TabSynthFraud` | arXiv:2604.13125 (2026) | Tabular GAN failure on fraud — grounds the programmatic construction constraint |

---

## 11. Open Questions

These are unresolved design decisions. Do not resolve autonomously — surface when working in the relevant module.

- [ ] **Synthetic exposure set size:** how many examples per archetype? What perturbation strategy preserves structural validity without becoming an implicit rule?
- [ ] **λ(t) annealing schedule:** linear, step, or cosine decay in Stage 1? Needs ablation before locking in.
- [ ] **DOMINANT attribute/structure weight:** equal weighting is the starting point. Tuning needs Phase D validation.
- [ ] **DeepSVDD centroid initialization:** mean of normal samples (standard) or mean of Stage 1 synthetic-normal embeddings?
- [ ] **Self-training stability:** at what round does Phase D PR-AUC typically stabilize on this dataset? Unknown until first full run.
- [ ] **Appeals framing:** a flag with no rule-code anchor has different standing in a government appeals process. This is a policy decision, not a code decision — must be resolved before deployment.
- [ ] **`state_match_flag` placeholder:** still absent pending AISHE/DISE institution-location data integration.

---

## 12. v1.5 vs v2 — Architecture Choice Log

Two architectures were evaluated before v2 was chosen. This comparison is preserved so the project lead can revisit if Phase D results warrant it.

**v1.5 (Rule-Informed Distillation):** kept rule concepts but replaced numeric cutoffs with EVT-derived thresholds; used v1's trained LightGBM as a distillation teacher for Stage 1; added the graph AE as an additional channel alongside v1's rule layer.

**v2 (This file):** drops v1's model, all rule codes, all rule-derived labels, and all hand-set numeric thresholds entirely.

| Dimension | v1.5 | v2 |
|---|---|---|
| Rule dependency | Reduced but present | Eliminated |
| Cold-start confidence | High — 1,986 rule-confirmed positives seed round 0 | Lower — EVT mutual tail only at round 0 |
| Novel pattern detection ceiling | Lower — Stage 1 geometry anchored by v1's rule-bounded learning | Higher — no rule ceiling anywhere |
| Explainability at launch | Stronger — rule codes provide named justification | Weaker at launch — structural explanations require investigator training |
| Auditability in government context | More straightforward | More complex |
| Risk of v1 blind spots persisting | Real — distilling v1 bakes its limitations into Stage 1 | None |
| Implementation risk | Lower — incremental additions to validated codebase | Higher — new pipeline, no validated round-0 baseline |

Neither approach has been empirically validated on this dataset's real fraud distribution. The only honest next step before treating either as definitively better is running both against the Phase D harness and comparing relational category PR-AUC. Whether the ceiling (v2) or the floor (v1.5) matters more is a domain decision, not a technical one.

---

*Human-curated. Do not auto-regenerate. For agent instructions, see `AGENTS_v2_rulefree.md`.*
