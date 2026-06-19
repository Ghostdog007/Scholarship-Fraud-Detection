# NIC Fraud Detection — Rule-Free Architecture: Agent Context File (v2)
<!-- VERSION: 2.0-draft | OWNER: Project Lead | LAST REVIEWED: 2026-06 -->
<!-- CHANGE FROM thinking-phase v2: v1 model dropped entirely. All rule-based supervision -->
<!--   removed. No rule_violation_score, no rule_codes_fired, no NIC rulebook -->
<!--   codes, no engineered bridges. Architecture is fully unsupervised at its -->
<!--   core, seeded only by synthetic anomaly exposure and EVT tails. -->
<!-- RELATIONSHIP TO v1: Independent. v1 is reference-only -->
<!--   for dataset facts. No code, no model outputs, no rule logic carries -->
<!--   forward into v2. -->
<!-- PURPOSE: Agent-facing context for the fully rule-free, graph-aware, -->
<!--   self-training NIC fraud detection redesign. Human-curated. -->
<!--   Do NOT auto-regenerate. -->

---

## 0. How to Use This File (Agent Instructions)

> **Read this section first on every session. It governs everything below.**

- **This is the primary reference for all v2 work.** Read this file first and in
  full. Consult v1's AGENTS.md for exactly two things only: (a) dataset column
  facts (null ratios, duplicate groups, shape — v1 §2.1–2.4), and (b) the Phase D
  synthetic harness definition (v1 §11.5) which is inherited as the evaluation
  benchmark. Nothing else from v1 or v2 carries forward.
- **No rules. No exceptions.** There is no `rule_violation_score`, no
  `rule_codes_fired`, no NIC rulebook codes (X1, X7, YF, etc.), no engineered
  bridges (IP_CONC_ENG, FEE_ENG, etc.) anywhere in this architecture. If you find
  yourself writing code that checks a named rule code or a hand-set numeric
  threshold against a domain concept, stop — you are re-introducing rule
  dependency. The only numeric thresholds allowed are statistically derived from
  the data itself (EVT, §5.2) or learned from synthetic anomaly exposure (§5.3,
  §5.4).
- **No v1 model outputs.** `lgbm_risk_score`, the v1 LightGBM model file, and the
  v1 VAE checkpoint are not used here in any capacity — not as teachers, not as
  warm-starts, not as round-0 stand-ins. They do not exist in this pipeline.
- **Do not hallucinate research claims.** Every architectural decision cites a
  source in Section 6. Propose new components only with a citation.
- **Multi-agent design.** Seven modules, strict file-based contracts. Find your
  module in Section 9 and stay inside it. If a task spans two modules, stop and
  confirm scope before writing code.
- **One module, one concern, per response.**
- **Never modify this file autonomously.** Flag outdated content explicitly.
- **Carry v1's quantitative-claims protocol forward in full** (v1 AGENTS.md §0.1):
  raw stdout only, baselines named explicitly, everything seeded, row-level
  counting only, no same-turn resolution, conflicting numbers halt.

---

## 1. Why Rule-Free (The Fundamental Shift)

v1 and v2 both had a structural ceiling: positive training labels came from rules
a human wrote. A fraud pattern no human had documented could not become a positive
example, regardless of how anomalous the detectors found it.

v2 removes this ceiling entirely. The system learns what "normal" looks like
directly from the data distribution and from the geometry of the identity graph.
Anything that deviates from either — whether or not a human has written a rule
for it — surfaces as an anomaly candidate. The only domain knowledge that enters
the system is:

1. **Which relationships between applicants are worth representing as graph
   edges** — shared mobile, IP, name, pincode. This is structural domain
   knowledge about *what constitutes a relationship*, not a rule about fraud.
2. **What "anomalous structure" looks like, used during pretraining only** —
   synthetic examples generated from known fraud archetypes (IP clustering,
   name collisions, fee inflation) used as outlier exposure. These archetypes
   do not become hard rules; they shape the model's latent geometry during
   Stage 1, then Step back so Stage 2 discovers freely.
3. **A statistical definition of "extreme"** — EVT tail fitting on the model's
   own score distributions, with one human-set false-positive rate parameter.
   This replaces hand-set numeric thresholds with a principled statistical
   derivation.

**What the model is responsible for that no human has pre-defined:**
- What "normal" looks like across all 15,000 applications.
- What constitutes an anomalous relational neighborhood in the identity graph.
- Which novel patterns (not in the synthetic exposure set) cross the EVT
  anomaly threshold and get promoted to the self-training label set.
- The continuous risk score that combines all signals.

---

## 2. Inherited Ground Truth (Do Not Re-Derive)

> All data facts below are sourced from executed analysis on the primary dataset
> and confirmed in v1 AGENTS.md. They are reproduced here in full so that an
> agent working on v2 never needs to open v1.

### 2.1 Primary Dataset — `data_for_ml_model.csv`

| Property | Value |
|---|---|
| Rows | 15,000 |
| Columns | 136 |
| Fraud-labeled records (`sanity` not null) | 4 (0.027%) — confirmed valid in this slice; see §2.7 |
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
> **Agent rule:** Never use these columns as features. Drop them at load time
> in `tabular_feature_engine_v2.py`.

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
> **Agent rule:** In `tabular_feature_engine_v2.py`, drop duplicates before
> computing aggregations — otherwise they inflate variance artificially.
>
> **v1 discrepancy note (verified):** v1's `load_and_clean_data()` drops
> `district_id` as a duplicate, but `engineer_features()` uses `district_id`
> for `district_application_count`. In v2, decide explicitly whether to keep
> `district_id` for the graph builder's aggregation or drop it as a duplicate.

### 2.4 Key High-Nullity Fields (Handle, Don't Drop Blindly)

| Column | Null % | Reason |
|---|---|---|
| `disability_percentage`, `disablity_type` | 99.49% | Disability is rare — expected |
| `orphan_flag` | 99.75% | Rare demographic |
| `gaurdian_name` | 99.77% | Rare demographic |
| `enroll_udid_no` | 99.49% | UDID mostly not provided |
| `ration_card_no`, `ration_card_member_no` | 96.49% | Optional field |
| `district_short_name` | 99.97% | Near-empty — drop |

### 2.5 Confirmed Missing Fields

The following fields appear in NIC revalidation rules but are **entirely absent**
from the CSV. Do not attempt to engineer proxies without explicit instruction:

```
bank_account_no, bank_name, ifsc_code
```

### 2.6 Confirmed Data Anomalies (Ground Truth for Graph Sanity Checks)

These are verified statistical anomalies in the valid data. Use them to
sanity-check graph cluster sizes and VAE reconstruction patterns:

- **1 IP address submitted 39 applications.** Top 10 IPs each submitted 15–39
  applications. High IP concentration is a valid anomaly signal for the graph AE.
- **1 mobile number shared by 6 applicants.** 59 mobiles shared by 2, 3 shared
  by 3. These form natural clusters in the `shares_mobile` edge type.
- **Family income as low as 5 INR** — likely data entry errors or deliberate fraud.
- **3 Post-Matric applicants exceed 35-year age limit** but are NOT flagged
  in `sanity`. This confirms enforcement gaps exist in the source system.
- **Institute `c_institution_id=10791`** has 151 applications — highest
  concentration. Below the 500-application threshold but worth monitoring.

### 2.7 Note on the 4 Originally-Flagged Records

> These records are confirmed valid in this dataset slice — their duplicate
> counterparts exist outside the 15,000-record boundary. They are **not** used
> as an evaluation target anywhere in v2. Do not treat them as ground truth for
> scoring, benchmarking, or explanation validation. The only evaluation ground
> truth in v2 is the Phase D synthetic harness (§8.1).

### 2.8 v1.4 Baselines to Beat (Phase D VAE-Alone Figures)

These are v1's standalone-VAE PR-AUC figures (no LightGBM, no rules). They
represent the floor v2's graph + tabular VAE combination must clear on every
category before any production discussion:

| Category | n | v1 VAE-Alone ROC-AUC | v1 VAE-Alone PR-AUC |
|---|---|---|---|
| INCOME_VIOLATION | 150 | 0.9465 | 0.1162 |
| AGE_VIOLATION | 150 | 0.8737 | 0.0506 |
| MOTHER_NAME_COLLISION | 150 | 0.8012 | 0.0258 |
| FEE_INFLATION | 150 | 0.7961 | 0.0264 |
| IP_CONCENTRATION | 150 | 0.7672 | 0.0239 |

> **Source:** v1 AGENTS.md §11.5, raw stdout receipts from v1.4 canonical run.
> These are the numbers v2 §8 references as the "floor."

### 2.9 What Is NOT Inherited From v1

The following v1 concepts do not exist in v2. If code references them, it is a
regression to rule dependency:

- `rule_violation_score` — no equivalent exists
- `rule_codes_fired` — no equivalent exists
- `apply_rules()` — no equivalent exists
- `selected_features.json` — v2 uses its own feature contract (§5.1 output)
- Any named rule code (X1, X7, YF, UW, YK, etc.)
- Any engineered bridge (IP_CONC_ENG, FEE_ENG, FM_ENG, etc.)
- v1's `lgbm_risk_score` or any v1 model checkpoint

---

## 3. Architecture Overview

```
                    ┌──────────────────────────┐
                    │    application records    │
                    └────────────┬─────────────┘
             ┌───────────────────┴───────────────────┐
             ▼                                        ▼
┌────────────────────────┐              ┌──────────────────────────┐
│   tabular feature       │              │   identity graph builder  │
│   engine (v2)            │              │   typed edges: mobile,    │
│   no rule flags          │              │   IP, names, pincode       │
└────────────┬────────────┘              └────────────┬──────────────┘
             │                                         │
             ▼                                         ▼
┌────────────────────────┐              ┌──────────────────────────┐
│  tabular VAE v2         │              │   graph autoencoder v2    │
│                         │              │   (DOMINANT + DeepSVDD    │
│  Stage 1: synthetic     │              │    hypersphere on graph   │
│    anomaly exposure      │              │    embeddings)            │
│  Stage 2: free           │              │                           │
│    reconstruction        │              │   Stage 1: synthetic      │
│                         │              │     anomaly exposure       │
│                         │              │   Stage 2: free            │
│                         │              │     reconstruction         │
└────────────┬────────────┘              └────────────┬──────────────┘
             │  vae_anomaly_score                      │  graph_anomaly_score
             │  per-feature recon error                │  attr/structure error split
             └────────────────────┬────────────────────┘
                                  ▼
             ┌────────────────────────────────────────┐
             │   EVT tail scorer (§5.2)                │
             │   fits GPD to each score distribution   │◄──┐
             │   one q parameter, no domain thresholds │    │ promotes
             └────────────────────┬───────────────────┘    │ extreme
                                  ▼                         │ mutual
             ┌────────────────────────────────────────┐    │ agreements
             │   self-training loop (§5.5)              │────┘
             │   ASTRA-style: EVT tail + classifier     │
             │   agreement → pseudo-positive labels     │
             └────────────────────┬───────────────────┘
                                  ▼
             ┌────────────────────────────────────────┐
             │   fusion classifier (LightGBM)           │
             │   scalar inputs only                     │
             │   labels: EVT-confirmed + pseudo-pos     │
             └────────────────────┬───────────────────┘
                                  ▼
             ┌────────────────────────────────────────┐
             │   XAI layer                              │
             │   SHAP + GNNExplainer/PGExplainer        │
             │   + reconstruction decomposition         │
             └────────────────────┬───────────────────┘
                                  ▼
                    risk score + explanation card
```

---

## 4. The Cold-Start Problem and How It Is Solved

Without rules, the self-training loop has no seed positive set on day one.
This is solved by **Latent Outlier Exposure** (LOE, ICML 2022 — see §6.1):
a strategy for training anomaly detectors in the presence of unlabeled data by
<using a combination of two losses sharing parameters — one for normal data,
one for anomalous data — and iteratively updating both model parameters and
the inferred binary labels jointly>, without requiring any pre-labeled anomalies.

The v2 cold start proceeds in this order:

**Step 1 — Synthetic exposure pretraining (Stage 1):**
Both detectors pretrain on synthetic anomaly archetypes generated from your
known fraud categories (IP clustering, name collisions, fee inflation, age
violations). These come from the Phase D harness categories — they are not
rules, they are *example anomalous configurations* used to shape latent
geometry. The model is never told "IP count >= 15 is fraud"; it is shown
"here are examples of clustered-IP applications, learn that this region is
anomalous." After Stage 1, λ(t) → 0 and Stage 2 runs purely on reconstruction
across the real dataset — the synthetic examples are exposure scaffolding, not
a permanent label set.

**Step 2 — EVT tail scoring (Stage 2 output):**
After Stage 2 reconstruction training, fit a Generalized Pareto Distribution
to each score distribution (vae_anomaly_score, graph_anomaly_score) using
Peak Over Threshold. Records in the joint extreme tail of both scores — above
the EVT-derived threshold for a given false-positive rate parameter q — form
the initial pseudo-positive set for the self-training loop's round 0. No human
sets the threshold value; only q (the acceptable false-positive rate) is set.

**Step 3 — Self-training expansion (§5.5):**
Round 0 positives are EVT-tail records only. Each subsequent round expands
the set with records the fusion classifier confidently agrees are positive AND
that sit in both EVT tails. The loop continues until Phase D PR-AUC stabilizes.

---

## 5. Module Specifications

Each module has a strict input/output contract. An agent working on one module
should never need to read another module's implementation — only its contract.

---

### 5.1 `tabular_feature_engine_v2.py`

**Responsibility:** engineer tabular node features for both the VAE and the graph
builder. No rule flags. No policy boundary violations. Structural and statistical
features only.

**Input contract:** `data_for_ml_model.csv`

**Load-time cleaning (required before engineering — mirrors §2.2–2.3):**
- Drop all 16 confirmed 100% null columns listed in §2.2
- Drop duplicate columns listed in §2.3 (keep `domicile_state_id`,
  `state_name`, `permanent_district_id`, `district_name`)
- Exclude `sanity`, `application_id`, `jwt` from the feature set
- Handle high-nullity fields per §2.4 (do not drop blindly)

**Output contract:** `engineered_features_v2.csv` — the full dataset with
engineered columns appended. Also writes `v2_feature_schema.json`:
```json
{
  "features": ["age_at_registration", "fee_income_ratio", ...],
  "aggregation_features": ["mobile_application_count", "ip_application_count", ...],
  "excluded": ["application_id", "sanity", "jwt"],
  "n_features": 42,
  "timestamp": "..."
}
```

**Engineered features (all structural/statistical — no rule logic):**

```
# Derived scalar features
age_at_registration       = (registered_date - date_of_birth).days / 365.25
fee_income_ratio          = (admission_fee + tution_fee + misc_fee) / annual_family_income
name_similarity_score     = SequenceMatcher ratio(applicant_name, father_name)

# Cross-row aggregates (factual counts, not rule triggers)
mobile_application_count  = count of applications sharing same mobile_no
ip_application_count      = count of applications sharing same ip_address
mobile_unique_names       = nunique(applicant_name) per mobile_no
mobile_unique_fathers     = nunique(father_name) per mobile_no
institute_application_count = count per c_institution_id
ip_to_mobile_ratio        = ip_application_count / (mobile_application_count + 1)

# Boolean identity matches (factual, not rule-coded)
is_applicant_name_eq_father  = int(applicant_name == father_name)
is_applicant_name_eq_mother  = int(applicant_name == mother_name)
is_father_name_eq_mother     = int(father_name == mother_name)
```

**What is explicitly absent (compared to v1):**
- No `flag_income_below_10000`, no `flag_fee_exceeds_income`, no
  `flag_prematric_age_over20` — these are policy-boundary rule flags.
  The raw continuous features (income, fee_income_ratio, age_at_registration)
  are present; the model learns where the boundary is, not us.
- No `state_match_flag` — documented in v1 as a non-functional placeholder.
  Still absent until AISHE/DISE data is available.

**Do not:**
- Do not add any feature whose definition encodes a specific threshold or
  comparison against a policy value. The VAE and graph AE discover thresholds;
  this module only provides the raw signal.

---

### 5.2 `evt_scorer.py`

**Responsibility:** derive statistically-grounded anomaly thresholds from the
model score distributions themselves — not from domain rules.

**Input contract:** any score series from `tabular_vae_v2.py` or
`graph_autoencoder_v2.py` (`vae_anomaly_score`, `graph_anomaly_score`, etc.)

**Output contract:** `evt_thresholds_v2.json`:
```json
{
  "vae_anomaly_score":   {"threshold": 0.74, "q": 0.002, "method": "POT-GPD"},
  "graph_anomaly_score": {"threshold": 0.68, "q": 0.002, "method": "POT-GPD"}
}
```

**Implementation (per `EVT-SPOT`, §6.1):**
- Peak Over Threshold (POT) method: set a lower initial threshold u at the
  ~95th percentile, collect exceedances, fit a Generalized Pareto Distribution
  (GPD) via MLE using `scipy.stats.genpareto`.
- The final anomaly threshold is derived from the fitted GPD at the target
  quantile `1 - q`, where `q` is the false-positive rate parameter set by the
  project lead.
- `q` is the only human-set value in this module. Changing it requires the
  same documentation discipline as any other tunable parameter (§0.1 protocol).

**Do not:**
- Do not set thresholds based on domain concepts (income levels, age cutoffs).
  This module operates purely on score distributions, not feature values.
- Do not expose raw percentile cutoffs as the threshold — the GPD fit is what
  makes this principled rather than just "top N%."

---

### 5.3 `tabular_vae_v2.py`

**Responsibility:** learn what "normal" looks like across the tabular feature
space and produce a per-application anomaly score + per-feature error vector
for the XAI layer.

**Input contract:** `v2_feature_schema.json` (§5.1), `engineered_features_v2.csv`,
and a `synthetic_exposure_set.pt` — a small PyTorch tensor of synthetic anomaly
examples generated from Phase D archetypes (see §4, cold start).

**Output contract:** `vae_v2_scores.csv` — `application_id`,
`vae_anomaly_score` (higher = more anomalous, i.e. negative of reconstruction
probability), plus `recon_error_vector` — per-feature MSE as a JSON list for the
XAI layer.

**Architecture:**
- Same encoder → μ, σ → reparameterize → decoder structure as v1's VAE.
- Output activation: Sigmoid (same as v1).
- Loss: ELBO = reconstruction MSE + KL divergence (same as v1).

**Training curriculum (per `OutlierExposure` + `LOE`, §6.1):**

```
Stage 1 — synthetic anomaly exposure pretraining
  L = L_reconstruction(normal_data)
      + λ(t) · L_exposure(synthetic_anomaly_set)
  L_exposure: auxiliary head on latent μ, trained to push synthetic
  anomaly embeddings away from the normal cluster centroid.
  The synthetic set comes from Phase D archetypes — these are not rules;
  they are configuration examples. λ(t) decays to 0 over Stage 1.

Stage 2 — free reconstruction discovery
  λ(t) = 0. Train purely on reconstruction across all 15,000 records.
  Anomaly score = exp(-MSE), inverted so higher = more anomalous.
  The Stage 1 auxiliary head is frozen and not used for scoring.
```

**Anomaly score direction convention:**
`vae_anomaly_score` is defined such that **higher = more anomalous** throughout
v2. This is the inverse of v1's `vae_reconstruction_prob` (where higher = more
normal). All consuming modules must respect this convention. This is documented
here and in the output contract of every downstream module.

**Do not:**
- Do not let the Stage 1 auxiliary head become the scoring mechanism. Scores
  come from reconstruction error only, post Stage 2.
- Do not introduce any feature that encodes a rule threshold (see §5.1 contract).

---

### 5.4 `graph_autoencoder_v2.py`

**Responsibility:** learn what "normal" looks like in the identity graph and
produce a per-node anomaly score decomposed into attribute and structural
components.

**Input contract:** `identity_graph.pt` from `graph_builder_v2.py`, plus the same
`synthetic_exposure_set.pt` as §5.3 (node-level synthetic anomaly examples
mapped into the graph).

**Output contract:** `graph_v2_scores.csv` — `application_id`,
`graph_anomaly_score` (higher = more anomalous), `attr_recon_error`,
`struct_recon_error`.

**Architecture (per `DOMINANT` + `DeepSVDD`, §6.1):**

DOMINANT dual-decoder baseline:
- GCN encoder (or `RGCNConv` for typed edges) produces node embeddings.
- Attribute decoder: reconstructs node feature vectors from embeddings.
- Structure decoder: dot-product inner product reconstructs adjacency.
- Anomaly score: weighted sum of attribute and structure reconstruction errors.

DeepSVDD hypersphere complement (applied to graph embeddings):
- After the DOMINANT encoder produces node embeddings, a secondary
  DeepSVDD objective minimizes the volume of the hypersphere enclosing
  normal node embeddings — <a neural network learns a transformation that
  maps most of the data representations into a hypersphere of minimum
  volume; mappings of normal examples fall within, whereas mappings of
  anomalies fall outside> (per `DeepSVDD`, §6.1).
- Distance from hypersphere center becomes an additional anomaly signal,
  composited with DOMINANT's reconstruction error.
- This catches anomalies that reconstruct well but embed far from the normal
  cluster — a failure mode DOMINANT alone can miss.

**Stage 1 / Stage 2 curriculum:** same pattern as §5.3. Synthetic anomaly
node examples push the Stage 1 encoder to form a tighter normal cluster before
Stage 2 free reconstruction begins.

**Do not:**
- Do not export raw node embeddings. Only `graph_anomaly_score`,
  `attr_recon_error`, and `struct_recon_error` leave this module.
  This is the single most important constraint — see §7.2.

---

### 5.5 `graph_builder_v2.py`

**Responsibility:** construct the typed identity graph from application records.

**Input contract:** `engineered_features_v2.csv` (from §5.1).

**Output contract:** `identity_graph.pt` — PyG `HeteroData` object with:
- One node per application, node features = engineered tabular features from §5.1.
- Typed edges: `shares_mobile`, `shares_ip`, `shares_father_name`,
  `shares_mother_name`, `shares_pincode`.

**Implementation notes:**
- Build with `networkx` first for inspection (sanity-check clusters visually),
  convert to PyG last.
- 15,000 nodes — no mini-batching infrastructure needed.
- Do not collapse edge types. Typed edges are what `RGCNConv` exists to use.

---

### 5.6 `self_training_loop_v2.py`

**Responsibility:** grow the positive label set beyond the initial EVT-tail cold
start, round by round, without any rule involvement.

**Input contract:** `vae_v2_scores.csv`, `graph_v2_scores.csv`,
`evt_thresholds_v2.json`, and the current fusion classifier's predictions.

**Output contract:** `pseudo_labels_v2.json`:
```json
{
  "positive_set": [
    {
      "application_id": "...",
      "round": 1,
      "trigger": "EVT_MUTUAL_TAIL",
      "vae_anomaly_score": 0.91,
      "graph_anomaly_score": 0.87,
      "classifier_confidence": 0.82
    }
  ],
  "negative_set": [...],
  "round": 2,
  "timestamp": "..."
}
```

**Promotion rule (per `ASTRA-SelfTrain` + `LOE`, §6.1):**
A record is promoted to pseudo-positive only if ALL THREE hold:
1. `vae_anomaly_score` exceeds its EVT threshold (§5.2).
2. `graph_anomaly_score` exceeds its EVT threshold (§5.2).
3. The current fusion classifier assigns it probability ≥ the EVT-derived
   threshold on the classifier's own score distribution (not a hand-set cutoff).

**Round 0 specifics:** the fusion classifier does not exist yet. Condition 3
is waived at round 0 — promotion is based on EVT mutual tail agreement alone.
From round 1 onward, all three conditions are required. This must be enforced
in code, not just documented.

**Do not:**
- Do not use any rule code, rule score, or rule-derived label as a promotion
  signal at any round. Rule involvement here re-introduces the ceiling v2 exists
  to remove.
- Do not run rounds automatically. Each round requires a Phase D PR-AUC check
  before its label set is used for the next training cycle.

---

### 5.7 `fusion_classifier_v2.py`

**Responsibility:** combine all scalar anomaly signals into a final risk score.

**Input contract:**
- Tabular features from `v2_feature_schema.json`
- `vae_anomaly_score` and `recon_error_vector` (as individual columns)
- `graph_anomaly_score`, `attr_recon_error`, `struct_recon_error`
- `pseudo_labels_v2.json` (positive = EVT-confirmed + self-training promoted;
  negative = rest)

**Output contract:** `risk_scores_v2.csv`:
```csv
application_id, vae_anomaly_score, graph_anomaly_score,
lgbm_risk_score_v2, label_source, top_shap_features
```

`label_source` values: `evt_cold_start`, `self_training_round_N`, `negative`.

**Do not:**
- Do not include raw GNN embeddings as features. See §7.2.
- Do not include any v1 model output as a feature. See §2.1.
- Do not treat `label_source` as a feature — it is metadata for audit only.

---

### 5.8 `xai_layer_v2.py`

**Responsibility:** produce a human-readable explanation card for every
flagged application, covering all three signal channels.

**Input contract:** trained fusion classifier (§5.7), trained graph AE (§5.4),
`recon_error_vector` from §5.3.

**Output contract:** `explanation_cards_v2.json`, one entry per application:
```json
{
  "application_id": "...",
  "risk_score": 0.84,
  "label_source": "self_training_round_2",
  "anomaly_channel": "graph",
  "top_shap_features": ["graph_anomaly_score", "fee_income_ratio", "..."],
  "top_graph_neighbors": [
    {"application_id": "...", "relation": "shares_ip", "weight": 0.81}
  ],
  "top_reconstruction_dims": ["ip_application_count", "name_similarity_score"],
  "narrative": "This application shares an IP address with 12 others and
    reconstructs poorly on name-similarity features."
}
```

**Note on the `narrative` field:** this is a human-readable string generated
from the top signals, not a model output. It is templated from
`anomaly_channel`, `top_graph_neighbors`, and `top_reconstruction_dims`.
It replaces the "rule_codes_fired" column v1 used to explain flags. The
absence of rule codes does not mean the absence of an explanation — structural
anomalies (neighborhood concentration, reconstruction divergence, hypersphere
distance) are the explanations.

**Mechanism per signal type:**
- **Tabular/classifier-level:** `shap.TreeExplainer` (unchanged from v1/v2).
- **Graph-level:** `GNNExplainer` for per-case explanations. Switch to
  `PGExplainer` when explanation volume warrants the one-time training cost.
- **Reconstruction-level:** per-feature MSE from the `recon_error_vector`.
- **Hypersphere-level:** distance-to-centroid decomposition from the DeepSVDD
  component of §5.4, attributable to which node-feature dimensions contribute
  most to the displacement.

**Do not:**
- Do not treat explainer outputs as ground truth without validating against
  Phase D synthetic cases where the true fraud mechanism is known.

---

## 6. Research References

### 6.1 Primary (peer-reviewed / arXiv, directly grounds a component)

| Short Ref | Full Citation | Grounds |
|---|---|---|
| `OutlierExposure` | Hendrycks, Mazeika, Dietterich, "Deep Anomaly Detection with Outlier Exposure," ICLR 2019, arXiv:1812.04606 | Synthetic anomaly exposure curriculum in §5.3, §5.4 |
| `LOE` | Qiu et al., "Latent Outlier Exposure for Anomaly Detection with Contaminated Data," ICML 2022, arXiv:2202.08088 | Cold-start joint label inference (§4, §5.6); allows anomaly detection without any pre-labeled anomalies |
| `DOMINANT` | Ding, Li, Bhanushali, Liu, "Deep Anomaly Detection on Attributed Networks," SDM 2019 | Dual-decoder graph AE architecture in §5.4 |
| `DeepSVDD` | Ruff et al., "Deep One-Class Classification," ICML 2018 | Hypersphere anomaly boundary in §5.4; anomaly score = distance from normal centroid |
| `EVT-SPOT` | Siffer, Fouque, Termier, Largouet, "Anomaly Detection in Streams with Extreme Value Theory," KDD 2017 | Statistically-derived thresholds in §5.2; replaces all hand-set numeric cutoffs |
| `ASTRA-SelfTrain` | Karamanolakis et al., "Self-Training with Weak Supervision," NAACL 2021, arXiv:2104.05514 | Self-training promotion mechanism in §5.6 |
| `R-GCN` | Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks," ESWC 2018 | Typed-edge encoder in §5.4, §5.5 |
| `GNNExplainer` | Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks," NeurIPS 2019, arXiv:1903.03894 | §5.8 |
| `PGExplainer` | Luo et al., "Parameterized Explainer for Graph Neural Network," NeurIPS 2020, arXiv:2011.04573 | §5.8 (volume-scale upgrade from GNNExplainer) |

### 6.2 Inherited from v1 (load-bearing baselines only, no architecture carried)

| Short Ref | Full Citation |
|---|---|
| `VAE-AnomalyProb` | "Variational Autoencoder based Anomaly Detection using Reconstruction Probability," Semantic Scholar |
| `AED-LGB 2024` | "An AutoEncoder enhanced LightGBM method for credit card fraud detection," PMC/NCBI, 2024 |
| `FraudHandbook` | "Reproducible ML for Credit Card Fraud Detection," fraud-detection-handbook.github.io |

### 6.3 Important Caveat: Synthetic Tabular Generation for Fraud

A 2026 benchmark found that <standard tabular generators (CTGAN, TVAE,
GaussianCopula, TabularARGN) fail severely at preserving behavioral fraud patterns
— including temporal, velocity, and multi-account signals — with composite
degradation ratios of 24x or more> (arXiv:2604.13125). This directly applies to
the synthetic anomaly exposure set in §5.3/§5.4:

**Do not** generate the synthetic exposure set using a general-purpose tabular
GAN or VAE. The Phase D harness archetypes (IP clustering, name collisions, fee
inflation) should be constructed programmatically from the actual feature
distributions — e.g., sample a real application, duplicate its IP field across
N rows, perturb the name fields — rather than sampling from a generative model.
The goal is structurally valid fraud-shaped examples, not statistically faithful
synthetic data.

### 6.4 Tech Stack (v2 Additions Over v1)

| Purpose | Library | Version Constraint | New in v2? |
|---|---|---|---|
| Data loading & manipulation | `pandas` | >= 1.5 | No |
| Numerical ops | `numpy` | >= 1.23 | No |
| VAE implementation | `torch` (PyTorch) | >= 2.0 | No |
| Gradient boosting (fusion) | `lightgbm` | >= 4.0 | No |
| SHAP explainability | `shap` | >= 0.44 | No |
| Evaluation metrics | `scikit-learn` | >= 1.2 | No |
| Graph neural networks | `torch_geometric` (PyG) | >= 2.4 | **Yes** |
| Graph construction / inspection | `networkx` | >= 3.0 | **Yes** |
| EVT / GPD fitting | `scipy.stats.genpareto` | >= 1.11 | **Yes** |
| Graph explainability | PyG `GNNExplainer`, `PGExplainer` | via torch_geometric | **Yes** |
| Deep SVDD | Custom (built on PyTorch) | — | **Yes** |

> **Do not introduce:** `tensorflow`, `keras`, `xgboost` (unless discussed),
> SMOTE/oversampling, any autoML library, any tabular GAN for synthetic
> exposure (see §6.3). These prohibitions carry forward from v1.

---

## 7. Hard Constraints

1. **Inherited dataset constraints from v1:** never use `sanity`,
   `application_id`, or `jwt` as features; do not impute 100% null columns
   (§2.2) or missing bank fields (§2.5); all file paths must be relative.
2. **No raw GNN node embeddings leave `graph_autoencoder_v2.py`.** Only scalar
   scores and decomposed reconstruction errors are valid exports. Violating this
   silently breaks the explainability guarantee.
3. **No rule codes, rule scores, or hand-set domain thresholds anywhere in this
   pipeline.** The only numeric thresholds are EVT-derived (§5.2) or learned
   (Stage 1 / Stage 2 from synthetic exposure).
4. **`vae_anomaly_score` direction is higher = more anomalous throughout.** Any
   module that inverts this convention must document the inversion explicitly at
   the point of inversion.
5. **Self-training rounds are not automatic.** Each round's label set must clear
   a Phase D PR-AUC check before promotion into the next cycle.
6. **Round 0 uses EVT mutual tail only for promotion.** The classifier-agreement
   condition (condition 3 in §5.6) is code-enforced to waive at round 0 and
   require from round 1. This is not a documentation note — it must be in the
   implementation.
7. **The synthetic exposure set is programmatically constructed, not GAN-generated.**
   See §6.3. A GAN-generated exposure set is a known failure mode on fraud data.
8. **GNNExplainer/PGExplainer outputs are not trusted by default.** Validate
   against Phase D synthetic cases before treating explanations as accurate.

---

## 8. Evaluation Standards

### 8.1 Primary Gate — Phase D Synthetic Harness

The Phase D synthetic harness (inherited from v1 §11.5) injects 150 synthetic
anomalies per category into the real dataset. Each injection category exercises
a specific fraud archetype:

| Category | What Is Injected | Exercises |
|---|---|---|
| AGE_VIOLATION | Pre-matric age > 20 or post-matric age > 35 | Tabular anomaly in `age_at_registration` |
| INCOME_VIOLATION | Income set to extreme-low values (< 1000) | Tabular anomaly in `annual_family_income` |
| IP_CONCENTRATION | Same IP address across 15+ rows, distinct mobiles | Graph cluster density in `shares_ip` edges |
| MOTHER_NAME_COLLISION | `father_name == mother_name`, distinct from `applicant_name` | Graph attribute anomaly in identity fields |
| FEE_INFLATION | `fee_income_ratio > 1.0` while `annual_family_income > 20,000` | Tabular anomaly in fee/income relationship |

**v1.4 VAE-alone baselines (the floor to beat — same data as §2.8):**

| Category | v1 VAE-Alone ROC-AUC | v1 VAE-Alone PR-AUC |
|---|---|---|
| INCOME_VIOLATION | 0.9465 | 0.1162 |
| AGE_VIOLATION | 0.8737 | 0.0506 |
| MOTHER_NAME_COLLISION | 0.8012 | 0.0258 |
| FEE_INFLATION | 0.7961 | 0.0264 |
| IP_CONCENTRATION | 0.7672 | 0.0239 |

> v2's combined tabular VAE + graph AE must beat these PR-AUC figures on every
> relational category (IP_CONCENTRATION, MOTHER_NAME_COLLISION, FEE_INFLATION)
> before any production discussion. The tabular-only categories
> (AGE_VIOLATION, INCOME_VIOLATION) are expected to match or improve with the
> new Stage 1 synthetic exposure curriculum.

**Mandatory ablations before claiming any component helps:**
- Stage 1 vs no Stage 1 (`λ(t) ≡ 0`): does synthetic exposure pretraining
  actually improve Stage 2 PR-AUC? Report both.
- DeepSVDD component vs DOMINANT-only: does the hypersphere loss add
  independent signal over reconstruction alone? Report both.
- Self-training round N vs round 0: track Phase D PR-AUC across rounds.
  A declining trend after round N is a stop signal, not a reason to continue.

**What replaces v1's "PR-AUC vs rule labels" metric:**
v2 has no rule labels to evaluate against. The primary internal metric during
development is PR-AUC vs Phase D synthetic ground truth (where the injected
fraud mechanism is known). The 4 originally-flagged records from v1 §2.7 are
confirmed valid and are not used as an evaluation signal.

**Explainer faithfulness check:** for Phase D cases with a known injection
mechanism, confirm GNNExplainer/PGExplainer's identified subgraph contains the
planted relationship. Report as a hit rate, not just "explainer ran successfully."

---

## 9. Multi-Agent Ownership Map

| Module | File | Reads from | Writes to | Must not touch |
|---|---|---|---|---|
| Feature engineering | `tabular_feature_engine_v2.py` | raw CSV | `engineered_features_v2.csv`, `v2_feature_schema.json` | any model training code |
| Graph construction | `graph_builder_v2.py` | `engineered_features_v2.csv` | `identity_graph.pt` | model training code |
| EVT thresholds | `evt_scorer.py` | score CSVs from §5.3/§5.4 | `evt_thresholds_v2.json` | which features are scored (reads scores only) |
| Tabular detector | `tabular_vae_v2.py` | `v2_feature_schema.json`, `synthetic_exposure_set.pt` | `vae_v2_scores.csv` | `identity_graph.pt` |
| Graph detector | `graph_autoencoder_v2.py` | `identity_graph.pt`, `synthetic_exposure_set.pt` | `graph_v2_scores.csv` | tabular feature files |
| Self-training | `self_training_loop_v2.py` | both score CSVs, `evt_thresholds_v2.json`, current classifier | `pseudo_labels_v2.json` | model architectures |
| Fusion classifier | `fusion_classifier_v2.py` | all scalar scores, `pseudo_labels_v2.json` | `risk_scores_v2.csv` | raw GNN embeddings (§7.2), any rule signals (§7.3) |
| Explainability | `xai_layer_v2.py` | trained classifier, trained graph AE, error vectors | `explanation_cards_v2.json` | training code in any module |

**Working rule:** if your task requires reading or writing outside your module's
row, stop and confirm scope with whoever owns the other module.

---

## 10. Open Questions (v2)

- [ ] **Synthetic exposure set construction:** how many synthetic examples per
  archetype? What perturbation strategy best preserves structural validity
  without becoming an implicit rule? Needs a small ablation on Phase D.
- [ ] **λ(t) annealing schedule:** linear, step, or cosine decay in Stage 1?
  Needs ablation before locking in.
- [ ] **DOMINANT attribute/structure weight:** the composite anomaly score weights
  attribute vs structure reconstruction error. Equal weighting is the starting
  point; tuning needs Phase D validation.
- [ ] **DeepSVDD hypersphere centroid initialization:** should it be initialized
  as the mean embedding of normal samples (standard) or as the mean of Stage 1
  synthetic-normal embeddings? Needs a decision before §5.4 is implemented.
- [ ] **Self-training stability:** at what round does PR-AUC typically stabilize
  on this dataset? Unknown until first full run.
- [ ] **Appeals framing for EVT/self-training flags:** a flag with no rule-code
  anchor has different standing in a government appeals process than a
  rule-matched flag. This is a policy decision, not a code decision.
- [ ] **`state_match_flag` dormant guard:** still absent (no AISHE/DISE data).
  Carry the placeholder note forward from v1 §10.

---

## 11. Approach Comparison: v1.5 (Rule-Informed Distillation) vs v2 (Rule-Free)

During the design phase two distinct next-stage architectures were considered
before v2 was chosen. This section documents both approaches honestly so the
project lead can revisit the choice if v2's Phase D results warrant it.

---

### 11.1 What v1.5 Was

v1.5 was a hybrid extension of v1 that kept the rule system but upgraded how
it interacted with the learning components. Its key decisions:

- v1's trained `lgbm_risk_score` (pinned to the v1.4 canonical run) used as a
  **distillation teacher** for the tabular VAE's Stage 1 alignment loss —
  feeding a continuous soft signal rather than the binary `rule_violation_score > 0`.
- EVT-derived thresholds replaced hand-set numeric cutoffs in the rule layer
  (`ip_application_count >= 15` etc.), but the rule *concepts* remained.
- A DOMINANT + RGCNConv graph autoencoder was added as a new relational channel
  alongside the tabular VAE, not replacing it.
- Self-training loop seeded by rule labels at round 0, with v1's model standing
  in as the round-0 classifier until v2's own fusion classifier existed.
- Explanations: SHAP + GNNExplainer + reconstruction decomposition, with
  `rule_codes_fired` still present in the output alongside graph neighbors.

The architecture remained seven modules but with v1's model as a
read-only frozen artifact feeding into `tabular_vae_v2.py` and the round-0
bootstrap of `self_training_loop.py`.

---

### 11.2 What v2 Is (This File)

v2 drops v1's model, all rule codes, all rule-derived labels, and all
hand-set numeric thresholds entirely. The detectors learn what "normal"
looks like from scratch. Cold start is solved via LOE joint inference and
EVT mutual tail agreement. Synthetic anomaly examples (programmatically
constructed from Phase D archetypes) shape Stage 1 latent geometry without
encoding any specific numeric rule. The graph autoencoder adds a DeepSVDD
hypersphere component alongside DOMINANT to catch nodes that reconstruct
well but embed anomalously.

---

### 11.3 Honest Comparison (Unbiased)

| Dimension | v1.5 | v2 |
|---|---|---|
| **Rule dependency** | Reduced but present — EVT replaces thresholds, but rule *concepts* (age cutoffs, income violations) still shape Stage 1 | Eliminated — no rule concepts, thresholds, or codes anywhere |
| **Cold start confidence** | High — v1's model provides a warm, validated round-0 signal; first self-training round benefits from 1,986 rule-confirmed positives | Lower — round 0 relies on EVT mutual tail only; first positive set is statistically defined but empirically unvalidated on real fraud |
| **Novel pattern detection ceiling** | Lower — Stage 1 is shaped by v1's learned rule-bounded geometry; self-training starts from a rule-anchored seed and must drift away from it | Higher — no rule ceiling; Stage 2 discovery and self-training start from a genuinely neutral geometry |
| **Explainability at launch** | Stronger — rule codes provide an auditable, named justification for every flag from day one | Weaker at launch — explanations are reconstruction dims + graph neighbors + EVT tail membership; no named pattern anchor |
| **Auditability in a government context** | More straightforward — "this application violated documented rule X7" is a clear statement | More complex — "this application reconstructed poorly and sits in the EVT tail of both detectors" requires investigator training to interpret |
| **Risk of v1's blind spots persisting** | Real — distilling v1's score into Stage 1 bakes v1's learned limitations into v2's starting geometry; the self-training loop must actively overcome this | None — no inherited geometry from v1 |
| **Implementation risk** | Lower — v1's codebase is validated and the rule layer is well-tested; v1.5 adds components incrementally | Higher — entirely new pipeline with no validated baseline to fall back on at round 0 |
| **Self-training stability** | More stable — rule-confirmed positives provide a high-quality, low-noise anchor for round 0; drift risk is lower | Less stable — EVT-tail-only round 0 is noisier; the LOE joint inference mechanism has not been validated on this specific dataset |
| **Adaptation to new fraud types** | Partial — EVT thresholds adapt automatically; new *relational* fraud is caught by the graph AE regardless of rules; but entirely new *tabular* fraud patterns still need a self-training round to propagate | Full — any deviation from learned normality, tabular or relational, is a candidate from day one |
| **Phase D baseline to beat** | Same (v1 standalone-VAE: IP PR-AUC 0.0239) | Same |

---

### 11.4 When v1.5 Would Be the Better Choice

- The project operates in a high-stakes government context where every flag
  must be backed by a named, documented rule at launch — where "the model
  found it statistically unusual" is not sufficient justification for an
  investigation or an appeals decision.
- The team wants to validate the graph component incrementally against a known
  rule-anchored baseline before trusting EVT-only cold starts.
- Timeline is constrained — v1.5 can reuse v1's validated rule infrastructure
  and only adds the graph AE and distillation teacher components on top.
- The 1,986 rule-confirmed positives from v1.4 are considered a reliable enough
  signal to anchor the first self-training round.

### 11.5 When v2 Is the Better Choice

- The explicit goal is to detect fraud patterns that no existing rule describes —
  where a rule-anchored cold start is not just unnecessary but actively harmful
  because it pre-biases the system toward known patterns.
- The team is willing to invest in investigator training so that EVT-tail and
  graph-neighbor explanations are understood and trusted without rule-code anchors.
- Long-term maintainability matters more than launch-day explainability — v2
  requires no rule-book maintenance as fraud patterns evolve.
- The appeals/audit framing for rule-free flags has been resolved at the policy
  level (§10 open question) before deployment begins.

### 11.6 What Cannot Be Said Without Running Both

Neither approach has been validated on this dataset's real fraud distribution.
The comparisons above are structural and research-grounded, not empirical.
Before treating v1.5 as "safer" or v2 as "better," the only honest next step
is running both against the Phase D harness and comparing PR-AUC on the
relational categories. v1.5 may have a higher floor; v2 may have a higher
ceiling. Whether the ceiling matters more than the floor is a domain decision,
not a technical one.

<!-- END OF v2 RULE-FREE ARCHITECTURE CONTEXT FILE -->
<!-- Total sections: 11 | Human-curated: YES | Status: draft -->
<!-- Last updated: 2026-06 -->
<!-- Changes from previous draft: removed 4-known-fraud-record evaluation -->
<!--   target (confirmed valid); added §11 v1.5 vs v2 unbiased comparison; -->
<!--   renamed from AGENTS_v3_rulefree to AGENTS_v2_rulefree. -->
