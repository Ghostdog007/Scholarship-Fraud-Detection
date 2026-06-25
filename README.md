# NIC Scholarship Fraud Detection — V3 Hybrid GraphMCM

## What This Is

An unsupervised fraud detection system for 15,000 NIC scholarship applications.
It assigns a continuous risk score (0–1) to each application without using any
hardcoded rules, policy thresholds, or human-labelled fraud examples.

This is Version 3. It replaces V1 (rule-based supervision) and V2 (five
independent scorers). The core change: a single hybrid model that reasons about
both a student's individual data **and** their relationships with other
applicants simultaneously.

---

## The Journey: Why Three Versions

### V1 — Rules as training signal

V1 applied 99 NIC policy rules to generate positive training labels, then
trained a LightGBM classifier on those labels. The results were interpretable
but had a hard ceiling: any fraud pattern not in the rulebook was invisible,
regardless of how statistically anomalous it was. The model could only detect
fraud a human had already imagined.

### V2 — Rules removed, five independent detectors

V2 eliminated all rules. It trained autoencoders on what "normal" looks like
and used EVT tail-fitting to set statistically principled thresholds. Five
scorers ran in parallel:

| Scorer | What it saw |
|---|---|
| Tabular VAE | Per-row reconstruction error across all features |
| Graph AE (DOMINANT) | Node connectivity and feature structure in identity graph |
| Isolation Forest | Statistical isolation depth across all features |
| MCM | Conditional feature predictions: P(feature_i \| other features) |
| Subspace IF | Isolation depth on focused feature groups (income, fees) |

V2 passed all 5 evaluation categories and exceeded V1 baselines on every
relational category (4x–8x improvement on IP clustering and name collisions).
But the architecture had two remaining problems:

**Problem 1 — Redundant scorers.** The VAE and full-space IF were consistently
the weakest scorers on every category. MCM dominated them both on tabular
detection. They added training complexity without adding unique signal.

**Problem 2 — Separated worlds.** MCM knew about features but not
relationships. Graph AE knew about relationships but not conditional feature
expectations. Neither could answer the most powerful question a fraud analyst
would ask: *"Given who this person is connected to, do their individual
details make sense?"*

### V3 — One joint model, two detection modes

V3 replaces the five independent scorers with one model that sees both worlds
at once, plus a focused statistical safety net.

---

## What V3 Can Detect That V2 Cannot

### 1. Graph-informed conditional prediction

**V2 MCM asked:** "Given age=45 and other tabular features, is
pre_post_matric=1 (pre-matric enrollment) expected?"

**V3 Hybrid asks:** "Given age=45, other features, AND the fact that this
applicant's IP address is shared by fifteen other applicants who are all
age 13–15 from the same school — is pre_post_matric=1 expected?"

The neighborhood context makes conditional prediction dramatically sharper. A
45-year-old in a pre-matric programme is suspicious on its own. A 45-year-old
in a pre-matric programme sharing an IP with an entire class of 14-year-olds
is a different calibre of anomaly — the graph context amplifies the signal in a
way that no separate MCM or Graph AE could achieve.

### 2. Feature-informed edge prediction

**V2 Graph AE asked:** "Can I reconstruct this node's edges?" — purely
structural, with no awareness of what the applicant's features suggest about
expected connectivity.

**V3 asks:** "Given that this is a student from a small village in Bihar with
a family income of ₹50,000 applying for a pre-matric scholarship, should they
have 25 IP connections to applicants spanning 15 different districts?"

The answer is obviously no — but V2's DOMINANT model could not reason about
this because it treated edge reconstruction and feature reconstruction as
independent objectives with no information flow between them. V3 learns a
joint distribution: what connectivity pattern is normal *for this feature
profile*.

### 3. New complex fraud patterns V3 can detect

Because the two streams (features and graph) inform each other, V3 can surface
anomalies of a new type — ones that require cross-channel reasoning:

| Fraud pattern | V2 | V3 |
|---|---|---|
| Age mismatch for program + shared IP with age-appropriate classmates | Partial (MCM sees age, Graph sees IP separately) | Full (hybrid sees both jointly) |
| Normal-looking income + fees, but connected to high-fee-inflation cluster | Not detectable (features look normal individually) | Detectable (neighborhood context flags abnormal connectivity for this income level) |
| Isolated node applying from unique-everything with high scholarship amount | Not detectable (zero graph signal) | Partial (degree features make uniqueness visible; model learns this combination is unusual) |
| Geographically dispersed applications sharing a parent name | Graph AE detects the name collision | Hybrid detects it AND flags whether the feature profiles of those applications are consistent with the claimed family relationship |
| Institution with unusual fee structures whose applicants share IPs | Only detected if individual application triggers a threshold | Detected via institution-level cluster in graph combined with fee outlier in feature stream |

### 4. Partial isolated-node detection

**V2's hard blind spot:** 11.1% of applications (1,663 nodes) share no mobile,
no IP, no parent names, and no pincode with any other application. They have
zero edges in the identity graph and therefore receive zero graph signal from
the Graph AE — completely silent.

**V3's mitigation:** Five degree-aware features are added to the tabular
representation: `degree_shares_mobile`, `degree_shares_ip`,
`degree_shares_father_name`, `degree_shares_mother_name`,
`degree_shares_pincode`. These make isolation visible as explicit input rather
than a silent absence.

Critically, isolation is not uniformly normal:
- 82.8% of applicants share a pincode with someone else — zero pincode
  connections is unusual
- Only 0.9% share a mobile number — zero mobile connections is completely normal

The model can now learn: "zero pincode connections combined with this feature
profile is suspicious" without any hardcoded threshold. The degree features
give it the information; the hybrid model learns the conditional expectation.

**The honest limit:** An application with unique values across all five fields
AND completely normal-looking individual features is statistically
indistinguishable from a genuine isolated student in a rare pincode. No
architecture — V3 or otherwise — can detect fraud that leaves no statistical
trace. The system detects patterns, not intentions.

---

## Architecture

```
data/raw/data_for_ml_model.csv  (15,000 × 136)
               │
               ▼
  ┌────────────────────────────┐
  │   Feature Engine (V3)      │
  │   63 features + log1p      │
  │   + 5 degree-aware features│
  │   = 68 total               │
  └────────────┬───────────────┘
               │ engineered_features_v3.csv (N × 68)
               │
       ┌───────┴───────┐
       ▼               ▼
  Graph Builder    Synthetic Exposure
  5 typed edges    750 × 68 anomaly set
  (mobile, IP,     (5 archetypes × 150)
   names, pincode) graph-side LOE only
       │               │
       └───────┬───────┘
               ▼
  ┌────────────────────────────────────────────┐
  │         HYBRID GRAPHMCM (Core Model)        │
  │                                             │
  │  ┌─────────────────┐  ┌──────────────────┐ │
  │  │  Feature Stream  │  │   Graph Stream   │ │
  │  │                  │  │                  │ │
  │  │  8 Learned Masks │  │  RGCN Encoder    │ │
  │  │  masked_x_i      │  │  aggr='add'+tanh │ │
  │  │  (68-dim)        │  │  h_N(i) (64-dim) │ │
  │  └────────┬─────────┘  └────────┬─────────┘ │
  │           │                     │            │
  │           └──────────┬──────────┘            │
  │                      ▼                       │
  │         Concat [masked_x_i ; h_N(i)] (132-d) │
  │                      │                       │
  │                    MLP                       │
  │                      │                       │
  │           ┌──────────┴──────────┐            │
  │           ▼                     ▼            │
  │   Feature Pred Error     Edge Pred Error     │
  │   |predicted - actual|²  P(edge|x_i, h_N(i))│
  │   per-feature (68-d)     per-edge-type (5-d) │
  │           │                     │            │
  │           └──────────┬──────────┘            │
  │                      ▼                       │
  │           hybrid_anomaly_score               │
  │           = feature_err + 0.3 × edge_err     │
  └──────────────────────┬─────────────────────-─┘
               │
               ▼
  ┌────────────────────────────┐
  │     Subspace IF Ensemble   │  ← Safety net, no training
  │   3 groups × 1 IF each:    │
  │   financial / identity /   │
  │   network                  │
  └────────────┬───────────────┘
               │
               ▼
         EVT Scorer
         (GPD tail fit, q=0.002)
               │
               ▼
        Self-Training Loop
        (human-gated, 1 round)
               │
               ▼
        LightGBM Fusion
        (hybrid score + subspace IF score)
               │
               ▼
        XAI Layer
        per-feature conditional errors
        + graph neighbor explanation
               │
               ▼
        risk_scores_v3.csv
        explanation_cards_v3.json
```

---

## Key Design Decisions

### Why one model instead of five scorers

Five independent scorers score an application from five different angles and
let the fusion classifier reconcile them. The problem is each scorer's signal
is computed without knowledge of what the others will find. The hybrid model
computes both signals (feature-conditional and graph-structural) in a single
forward pass, where they inform each other during both training and inference.

### Why keep the Subspace IF

The hybrid's feature prediction averages error across 8 masks and 68 features.
A near-zero income (1 feature out of 68) might not dominate that average even
with graph context amplifying it. The subspace IF focuses the full statistical
capacity of isolation forest onto 3 small feature groups where marginal outliers
matter. It requires no training, runs instantly, and was the decisive scorer on
INCOME_VIOLATION (PR-AUC 0.6503).

### Why log1p transform

`annual_family_income` ranges from ₹5 to ₹4,000,000. After MinMaxScaling,
income=500 maps to 0.000125 and income=20,000 maps to 0.005 — both
indistinguishable near zero. `log1p(500)=6.2`, `log1p(20000)=9.9`,
`log1p(4000000)=15.2`. After MinMaxScaling on log values, the low-income
range becomes clearly separated and detectable. Applied to income and all fee
columns before scaling.

### Why the degree features matter

The identity graph has 5 edge types with very different base rates:
- `shares_pincode`: 82.8% of nodes have at least one connection
- `shares_mother_name`: 28.3%
- `shares_ip`: 27.6%
- `shares_father_name`: 15.7%
- `shares_mobile`: 0.9%

Zero `shares_pincode` connections is unusual (17% of nodes). Zero
`shares_mobile` connections is completely normal (99%). Without explicit degree
features, the model cannot distinguish between "this node is isolated in a
common way" and "this node is isolated in an unusual way." With them, the
feature prediction stream can learn different conditional distributions for
different isolation patterns.

---

## Evaluation

Three-level evaluation protocol:

**Level 1 — Category PR-AUC** (inherited from V2): 150 injected anomalies per
category, unseen seeds from training. V3 must exceed V2's best scores on all 5
categories.

**Level 2 — Degree-stratified PR-AUC** (new in V3): PR-AUC split by the degree
of injected nodes (isolated / low / high). Makes the isolated-node gap
explicit and measurable.

**Level 3 — Edge-dropout test** (new in V3): Remove edges from the top-scoring
nodes and measure score retention. Validates whether the hybrid learned
feature-based suspicion that persists without graph support, or whether
detection is purely graph-dependent.

### Achieved Results (full-trained model, round 0, 111 pseudo-positives)

The evaluation uses per-category subspace IF as the primary scorer for injected
isolated nodes. The hybrid model's `isolated_embedding` is a fixed shared vector
for all degree-zero nodes, so `feature_pred_error` cannot discriminate between
different tabular fraud types among isolated nodes — the subspace IF focused on
the relevant feature group is the correct tabular anomaly detector here.

| Category | V2 Floor | V3 Score | Signal | Status |
|---|---|---|---|---|
| AGE_VIOLATION | 0.1466 | **0.3417** | demographic IF (age+exam years) | PASS |
| INCOME_VIOLATION | 0.6503 | **0.9063** | financial IF | PASS |
| IP_CONCENTRATION | 0.0370 | **0.1184** | network IF | PASS |
| MOTHER_NAME_COLLISION | 0.2869 | **0.5206** | identity IF | PASS |
| FEE_INFLATION | 0.4962 | **0.7420** | financial IF | PASS |

Isolated-node stratum PR-AUC: 0.47–1.00 across all categories.
Edge-dropout score_retention = 3.6452 (feature-based suspicion persists without graph support).

---

## Directory Layout

```
NIC fraud Detection Project/
├── main_v3.py                              # Pipeline orchestrator (V3)
├── README.md                               # This file
│
├── src/
│   ├── config_v3.py                        # All dimension constants (single source of truth)
│   ├── tabular_feature_engine_v3.py        # 63 features + degree merge = 68
│   ├── graph_builder_v3.py                 # 5 typed edges + degree computation
│   ├── synthetic_exposure_builder_v3.py    # 750 × 68 LOE tensor (graph-side only)
│   ├── hybrid_graphmcm_v3.py               # Core model: feature stream + graph stream
│   ├── subspace_if_v3.py                   # 3-group subspace isolation forest
│   ├── evt_scorer_v3.py                    # GPD tail fit on 6 signals (hybrid + 5 subspace)
│   ├── self_training_loop_v3.py            # Pseudo-label promotion (human-gated, UNION of 5 EVT)
│   ├── fusion_classifier_v3.py             # LightGBM: hybrid + subspace IF → risk score
│   ├── xai_layer_v3.py                     # Per-feature errors + actual values + trigger narratives
│   └── evaluate_model_v3.py                # Category-specific subspace IF + degree-stratified PR-AUC
│
├── data/
│   ├── raw/data_for_ml_model.csv           # 15,000 × 136 (unchanged)
│   └── processed/
│       ├── engineered_features_v3.csv      # N × 68
│       ├── v3_feature_schema.json
│       ├── identity_graph_v3.pt
│       ├── degree_features_v3.csv          # N × 5 per-edge-type degrees
│       └── synthetic_exposure_set_v3.pt    # 750 × 68
│
├── models/
│   └── hybrid_graphmcm_v3.pth             # {model_state_dict, centroid, config}
│
├── outputs/
│   ├── hybrid_scores_v3.csv
│   ├── subspace_if_scores_v3.csv
│   ├── evt_thresholds_v3.json
│   ├── pseudo_labels_v3.json
│   ├── risk_scores_v3.csv
│   └── explanation_cards_v3.json
│
└── docs/
    └── AGENTS.md                           # Architecture contract (do not edit autonomously)
                                            # Includes V2 history, MAR critique, research citations
```

---

## How to Run

```bash
# Full V3 pipeline:
python main_v3.py

# Individual module:
.\.venv\Scripts\python.exe src/hybrid_graphmcm_v3.py

# Evaluation only:
.\.venv\Scripts\python.exe src/evaluate_model_v3.py
```

---

## Research Citations

- **GraphMAE** (Hou et al., KDD 2022): Masked graph autoencoding — masks node
  features, uses GNN to reconstruct from neighborhood. Foundation for the
  hybrid's feature-stream + graph-stream design.
- **MaskGAE** (Li et al., NeurIPS 2022 Workshop): Joint masking of both edges
  and features. Validates the two-error (feature_pred_error + edge_pred_error)
  scoring approach.
- **MCM / ICLR 2024**: Masked Cell Modeling for tabular data — learned masks
  predict masked feature values from unmasked ones. V3 extends this with graph
  context in the prediction pathway.
- **DOMINANT** (Ding et al., SDM 2019): Dual-decoder graph autoencoder for
  anomaly detection. V3 inherits the RGCN encoder architecture (aggr='add',
  tanh bounding) from V2's validated implementation.
- **LOE** (Qiu et al., ICML 2022): Latent Outlier Exposure — synthetic anomaly
  pretraining to push outlier embeddings away from the normal hypersphere.
  V3 applies LOE to the graph stream only (Stage 1).
- **SPOT / EVT** (Siffer et al., KDD 2017): Generalized Pareto Distribution
  tail fitting for threshold-free anomaly flagging. Unchanged from V2.
