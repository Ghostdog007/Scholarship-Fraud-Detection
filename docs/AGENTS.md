# NIC Fraud Detection — V3 Hybrid GraphMCM: Agent Context File
<!-- VERSION: 3.0-draft | OWNER: Project Lead | LAST REVIEWED: 2026-06 -->
<!-- DO NOT MODIFY AUTONOMOUSLY. Flag outdated content; only project lead edits this file. -->

---

## AGENT QUICK-START — Read This First

**What this system is:** Unsupervised fraud detection for 15,000 NIC scholarship
applications. Outputs a 0–1 risk score per application. No hardcoded rules allowed.

**Current state:** V3 fully implemented and evaluated. MLOps Phase 1 complete
(MLflow SQLite backend, DVC, pre-commit). XAI explanation cards improved
(human-readable feature labels, resolved neighbor application IDs, actionable
narratives, `review_status` replacing raw `label_source`). MLflow UI working
via `mlflow_ui.bat` (port 5000, SQLite backend). README has full MLflow Audit
Guide. AGENTS.md has LLM quick-start block and closing instruction.

**Next task:** Phase 2 MLOps — build ADR-005 (FastAPI inference + supervisor
server) and a Dockerfile together so the API can be tested via Docker Desktop
on Windows (no WSL). Start with the FastAPI server in `src/api/`, then
containerise immediately. Order: ADR-005 → Dockerfile → ADR-008
(checkpoint manager) → ADR-007 (structured logging) → ADR-006 (Celery + Redis).

**Three rules you must never break:**
1. No domain-threshold rules anywhere in code (no `age > 35`, no rule codes). §10 stop #1.
2. No raw GNN embeddings outside `hybrid_graphmcm_v3.py`. §10 stop #2.
3. Never advance self-training rounds without project lead approval. §10 stop #5.

**Jump to what you need:**
- Your module boundary → §9 (Module Ownership table)
- All hard stops → §10
- File input/output contracts → §7
- Dimension constants → §6 (import from `src/config_v3.py`, never hardcode)
- Current evaluation results → §5.1
- MLOps decisions → Appendix F

---

## AGENT CLOSING INSTRUCTION — Do This Before Every Session End

Before ending any session, you MUST update the **AGENT QUICK-START** block
above with the current state of the project. Specifically update these two lines:

```
**Current state:** <what is fully implemented and verified>
**Next task:**     <exactly what the next session should start with>
```

Also update AGENTS.md and README.md to reflect any code changes made during
the session (module boundaries, file contracts, hard stops, evaluation results).

This is mandatory — not optional. A future agent reading this file cold must
be able to pick up exactly where this session left off without asking the user
to re-explain context.

---

## 0. How to Use This File

**Read this entire file before writing a single line of code.**

This is the authoritative contract for V3. Every module boundary, every tensor
dimension, every file path, and every inter-module message format is specified
here. Agents working in parallel MUST respect these contracts — a dimension
mismatch discovered at fusion time costs a full retrain.

**Parallel agent rules:**
- Each agent owns exactly one module (§9).
- Agents communicate only through the file-based contracts in §8.
- If your output shape deviates from §7, stop and flag it — do not silently
  adjust to make downstream code compile.
- If two modules need to change simultaneously, stop and confirm scope with
  the project lead before proceeding.

---

## 1. Why V3 Exists

### V1 → V2: Remove the rules ceiling

V1 used 99 NIC rules + 8 engineered bridges to generate positive training
labels. Any fraud pattern not in the rulebook was invisible regardless of how
anomalous the detectors found it. V2 removed all rules, replacing them with
unsupervised autoencoders seeded by synthetic outlier exposure and EVT tail
thresholds.

### V2 → V3: Close the conditional gap and unify the detection pathway

V2 ran five independent scorers (VAE, Graph AE, Isolation Forest, MCM,
Subspace IF) and fused their outputs. The XAI demo and PR-AUC ablations
revealed two problems:

1. **The VAE and full-space IF were redundant.** MCM strictly dominated both
   on every tabular category. The VAE's two-stage LOE training added complexity
   without adding detection capability.

2. **MCM and Graph AE operated in separate worlds.** MCM asked "given this
   row's features, is feature X expected?" but had no graph context. Graph AE
   asked "can I reconstruct this node's edges?" but had no conditional masking.
   Neither could ask the more powerful question: "given this node's features
   AND its neighborhood, does the full picture make sense?"

V3 replaces the five independent scorers with:

- **One Hybrid GraphMCM model** — masked feature prediction informed by graph
  neighborhood context. Single training loop, joint feature+edge scoring,
  richer XAI.
- **One Subspace IF ensemble** — focused marginal detection on 3 feature
  groups. Cheap, no training, catches extreme values that conditional models
  miss.
- **Five degree-aware features** added to the feature engine — makes isolation
  visible as an explicit input signal rather than a silent absence.

**Scorer count: 5 → 2. Training loops: 4 → 1 (plus instant IF fitting).**

---

## 2. What V3 Can Detect That V2 Cannot

### Graph-informed conditional prediction

V2 MCM predicted: *"Given age=45 and other tabular features, is
pre_post_matric=1 expected?"*

V3 Hybrid predicts: *"Given age=45, other features, AND the fact that this
node's IP neighbors are all age 13–15 from the same school — is
pre_post_matric=1 expected?"*

The neighborhood context sharpens the conditional. A 45-year-old in pre-matric
is suspicious alone. A 45-year-old in pre-matric sharing an IP with fifteen
14-year-olds is far more suspicious — but V2 could only see one or the other,
never both simultaneously.

### Feature-informed edge prediction

V2 Graph AE asked: *"Can I reconstruct this node's edges?"* — purely
structural, blind to what the application's features imply about expected
connectivity.

V3 asks: *"Given that this is a rural student from Bihar with income 50k,
should they have 25 IP connections spanning 15 districts?"*

DOMINANT could not reason about this because it treated edge reconstruction
and feature reconstruction as independent objectives. The hybrid model learns a
joint distribution: what connectivity is normal FOR this feature profile.

### Partial isolated-node detection

V2: isolated nodes (11.1% of dataset — 1,663 nodes with degree=0 across all
5 edge types) received zero graph signal. Silent.

V3: degree-aware features (`degree_shares_pincode`, `degree_shares_ip`, etc.)
make isolation visible as tabular input. The model can learn "zero pincode
connections is unusual — 82.8% of nodes share a pincode" and flag the
combination of unusual isolation with an anomalous feature profile.

**Hard limit that no architecture can overcome:** an application with unique
values across all 5 edge fields AND normal-looking tabular features is
statistically indistinguishable from a genuine isolated student. The system
detects fraud that leaves statistical traces — forensically clean applications
are outside the detection boundary.

---

## 3. Architecture Overview

```
data/raw/data_for_ml_model.csv
        │
        ▼
src/tabular_feature_engine_v3.py
        │  68 features (63 original + 5 degree-aware)
        │  data/processed/engineered_features_v3.csv
        │  data/processed/v3_feature_schema.json
        ▼
src/graph_builder_v3.py
        │  data/processed/identity_graph_v3.pt
        │  data/processed/degree_features_v3.csv   ← 5 per-edge-type degrees
        ▼
src/synthetic_exposure_builder_v3.py
        │  data/processed/synthetic_exposure_set_v3.pt   (750 × 68)
        ▼
src/hybrid_graphmcm_v3.py          ← THE CORE MODEL
        │  models/hybrid_graphmcm_v3.pth
        │  outputs/hybrid_scores_v3.csv
        │    columns: application_id, hybrid_anomaly_score,
        │             feature_pred_error, edge_pred_error,
        │             per_feature_error_json
        ▼
src/subspace_if_v3.py
        │  outputs/subspace_if_scores_v3.csv
        │    columns: application_id, subspace_if_score,
        │             group_scores_json (financial/identity/network)
        ▼
src/evt_scorer_v3.py
        │  outputs/evt_thresholds_v3.json
        ▼
src/self_training_loop_v3.py
        │  outputs/pseudo_labels_v3.json
        ▼
src/fusion_classifier_v3.py
        │  outputs/risk_scores_v3.csv
        ▼
src/xai_layer_v3.py
        │  outputs/explanation_cards_v3.json
        ▼
src/evaluate_model_v3.py
        │  Console: degree-stratified PR-AUC + edge-dropout test
        ▼
      [END]
```

---

## 4. The Hybrid GraphMCM: Internal Design

### 4.1 Two streams, one model

```
Node i: features x_i (68-dim) + neighbors N(i)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
 FEATURE STREAM       GRAPH STREAM
 Learned Masks        RGCN Encoder
 (K=8 masks)         (aggr='add', tanh)
 masked_x_i          h_N(i) = neighborhood
    │                    │
    └─────────┬──────────┘
              ▼
    Concat([masked_x_i ; h_N(i)])   → 68 + GRAPH_EMB_DIM
              │
             MLP
              │
         predicted x_i
              │
    ┌─────────┴──────────┐
    ▼                    ▼
feature_pred_error    edge_pred_error
|predicted - actual|  P(edge | x_i, h_N(i))
per-feature           per-edge-type
    │                    │
    └─────────┬──────────┘
              ▼
   hybrid_anomaly_score = feature_err + λ × edge_err
```

### 4.2 Training

**Stage 1 — Synthetic Exposure (LOE):** Graph-side warm-start only. Inject
synthetic nodes, push their embeddings away from the normal centroid using the
exposure loss from V2. The masked prediction objective is self-supervised and
does NOT need synthetic data — it learns from the normal data distribution.

**Stage 2 — Free joint reconstruction:** Masked feature prediction + edge
prediction simultaneously on real data. The feature stream and graph stream
learn to inform each other.

No Stage 1 is needed for the feature prediction pathway (unlike V2 VAE). LOE
is graph-side only.

### 4.3 Isolated node handling

For degree=0 nodes, `h_N(i)` is a learned zero-vector embedding (not literal
zeros — a trainable `isolated_embedding` parameter). The model sees:

```
Concat([masked_x_i ; isolated_embedding])
```

This means: isolated nodes still get a prediction, and the model can learn
"features that look suspicious given that there are NO neighbors" as a distinct
conditional distribution from "features that look suspicious given rich
connectivity."

---

## 5. Evaluation Protocol

### 5.1 Standard PR-AUC (inherited from V2)

Same 5 categories × 150 injected anomalies × different seeds from training.
V3 must beat V2's best scores.

**Achieved results (full-trained model, 111 pseudo-positives, round 0):**

| Category | V2 Floor | V3 Score | Primary scorer | Status |
|---|---|---|---|---|
| AGE_VIOLATION | 0.1466 | **0.3417** | demographic IF | PASS |
| INCOME_VIOLATION | 0.6503 | **0.9063** | financial IF | PASS |
| IP_CONCENTRATION | 0.0370 | **0.1184** | network IF | PASS |
| MOTHER_NAME_COLLISION | 0.2869 | **0.5206** | identity IF | PASS |
| FEE_INFLATION | 0.4962 | **0.7420** | financial IF | PASS |

Isolated-node degree stratum: 0.47–1.00 across all categories.
Edge-dropout score_retention = 3.6452 (feature-based suspicion persists without graph edges).

### 5.2 Degree-Stratified PR-AUC (new in V3)

For each category, report PR-AUC split by the degree of the injected nodes:
- `isolated` (degree=0)
- `low` (degree 1–5)
- `high` (degree 6+)

This makes the isolated-node gap explicit and measurable rather than hidden in
the aggregate number.

**Evaluation methodology note — injected nodes are always isolated:**
The hybrid model's `isolated_embedding` is a single trained vector shared by
ALL degree-zero nodes. This means `feature_pred_error` is nearly uniform across
different injected-anomaly types — the hybrid cannot discriminate between, say,
age fraud and income fraud among isolated nodes. The correct primary scorer for
isolated-node evaluation is therefore the category-specific subspace IF group,
which focuses its full statistical capacity on the feature dimensions that
define each fraud type.

The evaluation harness uses an eval-only "demographic" group for AGE_VIOLATION
(`age_at_registration`, `competitive_exam_year`, `admission_year`,
`c_course_year`) that is not part of the production pipeline config. This group
is fitted fresh from real data during evaluation and is not exported to any
pipeline output file.

For scoring, both real-node and inject-node subspace IF scores are normalized
on the same real-data range, so inject scores > 1.0 indicate a node more
extreme than any real node in that group.

### 5.3 Edge-Dropout Test (new in V3)

After training, pick the top 100 highest-scoring nodes from the hybrid. Remove
all their edges. Re-score. Report:

```
score_retention = median(score_after_dropout) / median(score_before_dropout)
```

If `score_retention > 0.5`, the model learned feature-based suspicion that
persists without graph support. If `score_retention < 0.2`, detection is
graph-dependent and isolated-node performance will be poor.

---

## 6. Fixed Dimension Constants

**These are the single source of truth. Every module reads these values.
Never hardcode these numbers in module code — import from `src/config_v3.py`.**

```python
# src/config_v3.py  (agent must create this file)

N_FEATURES      = 68    # 63 original + 5 degree-aware features
N_EDGE_TYPES    = 5     # shares_mobile, shares_ip, shares_father_name,
                        # shares_mother_name, shares_pincode
MASK_NUM        = 8     # number of learned masks in GraphMCM
GRAPH_HIDDEN    = 128   # RGCN hidden channels
GRAPH_EMB_DIM   = 64    # RGCN output embedding dimension (h_N(i))
MLP_HIDDEN      = 256   # MLP hidden dim after concat
Z_DIM           = 64    # latent dimension after concat+MLP
LOE_MARGIN      = 2.0   # exposure loss margin (graph side only)
LAMBDA_EDGE     = 0.3   # weight of edge_pred_error in hybrid score
LAMBDA_EXPOSURE = 1.0   # initial LOE weight (decays linearly to 0)
EPOCHS_STAGE1   = 80    # graph LOE warm-start epochs
EPOCHS_STAGE2   = 120   # free joint reconstruction epochs
LR              = 1e-3  # Adam learning rate
BATCH_SIZE      = 256
RANDOM_SEED     = 42

# Subspace IF feature groups
SUBSPACE_GROUPS = {
    "financial":  ["annual_family_income", "fee_income_ratio",
                   "income_rank_in_district", "income_deviation_from_state_median",
                   "admission_fee", "tution_fee", "misc_fee"],
    "identity":   ["name_similarity_score", "is_father_name_eq_mother",
                   "is_applicant_name_eq_father", "is_applicant_name_eq_mother",
                   "mobile_unique_names", "mobile_unique_fathers"],
    "network":    ["ip_application_count", "ip_to_mobile_ratio",
                   "mobile_application_count", "institute_application_count",
                   "degree_shares_ip", "degree_shares_mobile",
                   "degree_shares_pincode"],
}

# Log1p-transform these columns before MinMaxScaling
LOG1P_COLS = ["annual_family_income", "admission_fee", "tution_fee", "misc_fee"]

# ── Risk-mitigation constants (added 2026-06-30) ──────────────────────────────

# Self-training: minimum number of EVT signals that must fire simultaneously
# for a node to be promoted to pseudo-positive in Round 0.
# 1 = original OR logic (any single signal promotes). 2 = multi-signal agreement.
# Raised to 2 to reduce confirmation bias from single-signal data-entry noise
# (e.g. income=5 INR fires EVT_FINANCIAL alone but passes no other signal).
MIN_SIGNALS_FOR_PROMOTION = 2

# EVT GPD shape validity range. Fits outside this range indicate a distribution
# that violates GPD regularity assumptions (discrete cluster spikes, heavy-tailed
# or bounded distributions). Bad fits fall back to empirical quantile.
EVT_SHAPE_MIN = -0.5
EVT_SHAPE_MAX = 1.0

# DeepSVDD centroid: fraction of nodes KEPT when computing the initial centroid.
# The top (100 - CENTROID_CLEAN_PERCENTILE)% of nodes by embedding norm are
# excluded before averaging. These are the highest-anomaly embeddings and are
# likely to include fraud — including them shifts the centroid toward fraud,
# causing the DeepSVDD hypersphere to silently expand to accept fraud as normal.
CENTROID_CLEAN_PERCENTILE = 95
```

### EVT per-signal threshold overrides

`evt_scorer_v3.py` fits 6 signals: `hybrid_anomaly_score`, `subspace_if_score`,
`subspace_if_financial`, `subspace_if_identity`, `subspace_if_network`,
`edge_pred_error`. The default `U_PERCENTILE = 95` (POT baseline) applies to
all signals except where overridden:

```python
U_PERCENTILE_OVERRIDES = {
    "subspace_if_identity": 97,   # identity IF over-flagged at 95th; 97th reduces noise
}
```

The only non-EVT numeric in this module is `Q = 0.002` (false-positive rate).
All other thresholds are GPD-derived.

### Self-training pseudo-label signals (Round 0)

Round 0 requires **at least `MIN_SIGNALS_FOR_PROMOTION` (= 2) EVT signals to
fire simultaneously** (no classifier agreement required). A node is
pseudo-positive if it clears the threshold on ≥ 2 of:
- `EVT_HYBRID` — `hybrid_anomaly_score >= hybrid_threshold`
- `EVT_FINANCIAL` — `subspace_if_financial >= financial_threshold`
- `EVT_IDENTITY` — `subspace_if_identity >= identity_threshold`
- `EVT_NETWORK` — `subspace_if_network >= network_threshold`
- `EVT_EDGE_RING` — `edge_pred_error >= edge_pred_error_threshold`

**Rationale for change (2026-06-30):** The original OR logic (any 1 of 5)
promoted 111 nodes in the full-trained run. A meaningful fraction of these
were single-signal hits driven by data-entry noise (e.g. `annual_family_income
= 5 INR` fires `EVT_FINANCIAL` alone but looks normal on every other signal).
Requiring 2 signals ensures each promoted node has independent corroboration
from two different fraud-detection lenses before entering LightGBM training.

Each positive record in `pseudo_labels_v3.json` carries a `trigger` field
listing which signals fired for that application.

### Concat dimension check (all agents must verify)

```
masked_x_i shape:  (B, N_FEATURES)         = (B, 68)
h_N(i) shape:      (B, GRAPH_EMB_DIM)      = (B, 64)
concat shape:      (B, N_FEATURES + GRAPH_EMB_DIM) = (B, 132)
MLP input:         132
MLP output (z):    Z_DIM = 64
decoder output:    N_FEATURES = 68
```

If any module produces a tensor that violates these shapes, it must raise a
`DimensionError` with the actual vs expected shape — never silently broadcast.

---

## 7. File-Based Contracts (Inter-Module Messages)

Every module reads from and writes to these exact files. No module reads
another module's source files directly. No embeddings cross module boundaries.

| File | Producer | Consumer | Schema |
|---|---|---|---|
| `data/processed/engineered_features_v3.csv` | feature_engine | graph_builder, hybrid, subspace_if, evaluate | N×68 numeric + application_id |
| `data/processed/v3_feature_schema.json` | feature_engine | all modules | `{features, aggregation_features, degree_features, excluded, n_features, log1p_cols}` |
| `data/processed/identity_graph_v3.pt` | graph_builder | hybrid, evaluate | PyG HeteroData, 5 edge types |
| `data/processed/degree_features_v3.csv` | graph_builder | feature_engine (written back) | N×5, cols=`degree_shares_*` |
| `data/processed/synthetic_exposure_set_v3.pt` | synthetic_builder | hybrid | (750, 68) float32 tensor |
| `models/hybrid_graphmcm_v3.pth` | hybrid | xai, evaluate | `{model_state_dict, centroid, config}` |
| `outputs/hybrid_scores_v3.csv` | hybrid | evt, self_training, fusion, xai | `application_id, hybrid_anomaly_score, feature_pred_error, edge_pred_error, per_feature_error_json` |
| `outputs/subspace_if_scores_v3.csv` | subspace_if | fusion, xai | `application_id, subspace_if_score, group_scores_json` |
| `outputs/evt_thresholds_v3.json` | evt | self_training | 6 signals: `{hybrid, subspace_if, subspace_if_financial, subspace_if_identity, subspace_if_network, edge_pred_error}` each with `{u, scale, shape, threshold, n_flagged}` |
| `outputs/pseudo_labels_v3.json` | self_training | fusion, xai | `positive_set` array, each record: `{application_id, round, trigger: [list of EVT signal names], hybrid_anomaly_score, subspace_if_*, edge_pred_error}` |
| `outputs/risk_scores_v3.csv` | fusion | xai, evaluate | `application_id, risk_score_v3, label_source` |
| `outputs/explanation_cards_v3.json` | xai | [end user] | per-application JSON. `top_feature_errors`: `{feature, feature_label, error, value, magnitude}` — `feature_label` is human-readable display name. `top_graph_neighbors`: `{edge_type, application_id}` — neighbor index resolved to actual application ID (no raw array indices). `review_status`: human-readable string replacing raw `label_source` (e.g. "Pending Review — no confirmed fraud label assigned yet"). `narrative`: full actionable prose including recommended action and explanation of why features are suspicious. |
| `data/processed/confirmed_fraud.json` | API / supervisor endpoint | retraining_orchestrator, self_training, fusion | `{confirmed: [{application_id, fraud_type, confirmed_by, cycle, feature_vec, confirmed_at, notes}], false_positives: [{application_id, confirmed_by, confirmed_at, notes}]}` |
| `models/checkpoints/hybrid_v3_<cycle>_<run_id>.pth` | checkpoint_manager | rollback command, MLflow | same schema as `hybrid_graphmcm_v3.pth` (`model_state_dict`, `centroid`, `config`); keep last 5; filename encodes cycle and mlflow_run_id |
| `outputs/prev_cycle_scores_ks.json` | retraining_orchestrator (end of cycle) | next-cycle drift check | `{scores: [float, ...]}` — score distribution baseline for KS test; overwritten after every completed inference cycle |
| `outputs/feature_drift_v3.json` | retraining_orchestrator | API `/monitoring/drift` endpoint | `{feature_name: {ks_stat, p_value, mean_prev, mean_curr}}` for all 68 engineered features |

**Hard rule:** `per_feature_error_json` is a JSON string of
`{feature_name: float}` for all 68 features. Downstream XAI reads this column
— its key set must exactly match the feature names in `v3_feature_schema.json`.
If they diverge, the XAI layer must raise, not silently skip unknown keys.

---

## 8. Degree-Aware Features: Build Order

The 5 degree features create a **build dependency**: graph_builder needs the
raw CSV to build edges, but the feature engine needs the degrees to write the
final feature CSV. Resolution:

1. `tabular_feature_engine_v3.py` runs **first** with 63 features, writes a
   temp CSV `data/processed/engineered_features_v3_nodeg.csv`.
2. `graph_builder_v3.py` reads the temp CSV, builds the graph, computes per-
   node degrees per edge type, writes `degree_features_v3.csv`.
3. `tabular_feature_engine_v3.py` has a second entry point `add_degree_features()`
   that reads `degree_features_v3.csv`, merges into the temp CSV, writes the
   final `engineered_features_v3.csv` with all 68 features, and updates
   `v3_feature_schema.json`.

`main_v3.py` must call them in this order:
```
feature_engine.build_base()       # step 1
graph_builder.build_graph()       # step 2
feature_engine.add_degree_features()  # step 3
```

---

## 9. Module Ownership

One agent, one module, one session. If your task requires editing outside your
row, stop and confirm scope.

| Module | File | Reads from | Writes to | Hard boundary |
|---|---|---|---|---|
| Feature engine | `src/tabular_feature_engine_v3.py` | raw CSV | `engineered_features_v3.csv`, `v3_feature_schema.json` | no model code |
| Graph builder | `src/graph_builder_v3.py` | `engineered_features_v3_nodeg.csv` | `identity_graph_v3.pt`, `degree_features_v3.csv` | no training code |
| Config | `src/config_v3.py` | — | — | no logic, constants only |
| Synthetic exposure | `src/synthetic_exposure_builder_v3.py` | `engineered_features_v3.csv` | `synthetic_exposure_set_v3.pt` (750×68) | no training code |
| Hybrid model | `src/hybrid_graphmcm_v3.py` | `identity_graph_v3.pt`, `synthetic_exposure_set_v3.pt`, `engineered_features_v3.csv` | `hybrid_scores_v3.csv`, `hybrid_graphmcm_v3.pth` | no rule thresholds |
| Subspace IF | `src/subspace_if_v3.py` | `engineered_features_v3.csv` | `subspace_if_scores_v3.csv` | no neural network code |
| EVT scorer | `src/evt_scorer_v3.py` | `hybrid_scores_v3.csv`, `subspace_if_scores_v3.csv` | `evt_thresholds_v3.json` | no model training |
| Self-training | `src/self_training_loop_v3.py` | score CSVs, `evt_thresholds_v3.json` | `pseudo_labels_v3.json` | no architecture changes |
| Fusion | `src/fusion_classifier_v3.py` | score CSVs, `pseudo_labels_v3.json` | `risk_scores_v3.csv` | no raw embeddings |
| XAI | `src/xai_layer_v3.py` | `hybrid_scores_v3.csv`, `risk_scores_v3.csv`, `pseudo_labels_v3.json`, `engineered_features_v3.csv` | `explanation_cards_v3.json` | no training code |
| Evaluate | `src/evaluate_model_v3.py` | `engineered_features_v3.csv`, `models/*.pth` | console stdout | no training code |
| Orchestrator | `main_v3.py` | — | calls all modules | no business logic |
| Checkpoint manager | `src/checkpoint_manager.py` | incoming `.pth` (temp path), `models/hybrid_graphmcm_v3.pth` | `models/hybrid_graphmcm_v3.pth` (live), `models/hybrid_graphmcm_v3.pth.bak`, `models/checkpoints/` | no training code; no model forward pass; validation and file operations only |

---

## 10. Hard Stops (Inherited + New)

**All V2 hard stops apply unchanged:**

1. **No rules. No exceptions.** No numeric threshold against a domain concept,
   no named rule code, no `apply_rules()` call, no feature whose definition
   encodes a policy boundary. The only allowed thresholds are EVT-derived or
   learned from synthetic exposure.
2. **No raw GNN embeddings leave `hybrid_graphmcm_v3.py`.** Only
   `hybrid_anomaly_score`, `feature_pred_error`, `edge_pred_error`, and
   `per_feature_error_json` are valid exports.
3. **Score direction: higher = more anomalous.** Any module that inverts this
   must document the inversion explicitly at the point of inversion.
4. **`sanity` column is never used.** Drop at load time in every pipeline file.
5. **Self-training rounds are not automatic.** Each round requires a Phase D
   PR-AUC check. Round 0 classifier-agreement condition must be code-enforced
   off, not just noted in a comment.
6. **No v1 or v2 model outputs in v3.** No v1/v2 checkpoints, no
   `lgbm_risk_score`, no `vae_anomaly_score`, no `graph_anomaly_score` from
   prior versions.
7. **Synthetic exposure set is programmatically constructed.** Never use CTGAN,
   TVAE, GaussianCopula, or any tabular GAN — composite degradation is 24x or
   more on fraud behavioral signals (arXiv:2604.13125).
8. **Never modify this file autonomously.** Flag outdated content explicitly.

**Additional V3 stops:**

9. **Dimension constants are in `config_v3.py` only.** Never hardcode `68`,
   `64`, `132`, `8`, or `5` inside module code.
10. **`h_N(i)` never leaves `hybrid_graphmcm_v3.py`.** Raw graph embeddings
    are not outputs.
11. **Isolated nodes use the learned `isolated_embedding`, not zero vectors.**
    `torch.zeros(GRAPH_EMB_DIM)` for isolated nodes is forbidden. The model
    must have a trainable `nn.Parameter` named `isolated_embedding` of shape
    `(GRAPH_EMB_DIM,)` initialized to `torch.randn`.
12. **Degree features are always in the 63:68 slice.** Columns 0–62 are the
    original 63 features in V2 order. Columns 63–67 are
    `degree_shares_mobile`, `degree_shares_ip`, `degree_shares_father_name`,
    `degree_shares_mother_name`, `degree_shares_pincode`. Use named indexing
    via `v3_feature_schema.json` — never positional hardcoding.
13. **`score_retention` must be printed by `evaluate_model_v3.py`.** If
    `score_retention < 0.2`, print: "Graph-dependent detection: isolated node
    performance will be degraded."
14. **`hybrid_graphmcm_v3.pth` is never written directly.** The live checkpoint
    path is a read-only destination at runtime. All writes go to a temp path
    (`models/incoming_<timestamp>_<uuid>.pth`) first. `checkpoint_manager.py`
    performs the atomic rename after validation passes. Any code that calls
    `torch.save(..., "models/hybrid_graphmcm_v3.pth")` directly is wrong —
    route through `checkpoint_manager.validate_and_hotswap()` instead.
15. **Checkpoint schema must embed a `config` dict.** `hybrid_graphmcm_v3.py`
    must save checkpoints with exactly these top-level keys:
    `{model_state_dict, centroid, config}`. The `config` dict must contain at
    minimum `N_FEATURES`, `GRAPH_EMB_DIM`, and `N_EDGE_TYPES` sourced from
    `config_v3.py`. This is the contract `checkpoint_manager.py` validates
    against before any swap. A checkpoint missing these keys is rejected with
    no change to the live model.
16. **`nic-worker` replica count is fixed at 1.** Training jobs write to fixed
    output paths (`outputs/*.csv`, `models/*.pth`). Scaling `nic-worker` to 2
    causes concurrent runs to overwrite each other's intermediates. Enforce
    in the k8s Deployment manifest with an explicit comment — `concurrency=1`
    at the Celery level alone is not sufficient.

---

## 11. Open Architecture Questions — Do Not Resolve Autonomously

- Optimal `LAMBDA_EDGE` (currently 0.3) — needs ablation.
- Whether `isolated_embedding` should be shared or per-node (currently shared).
- `MASK_NUM=8` — no ablation done with graph context.
- `EPOCHS_STAGE1=80` — V2 used 100; reduced because feature stream doesn't
  need LOE. May need tuning.
- Whether Subspace IF `network` group is final — degree features are untested
  as IF inputs.
- Optimal `MIN_SIGNALS_FOR_PROMOTION` — currently 2; no ablation on whether
  3-signal agreement would improve pseudo-label precision at the cost of recall.
- Archetype expansion — `_add_context_noise()` widens existing archetype
  geometry but the 5 archetype types are unchanged. 3–5 additional archetypes
  (cross-cycle IP reuse, institute-cluster, income-rounding) should be evaluated
  before the next full retrain. See Appendix B.

---

## 12. Quantitative Claims Protocol (inherited from V2, unchanged)

1. Raw stdout only. No number enters a doc without a traceable print line.
2. Name the baseline explicitly. V3 must beat V2 best scores from §5.1.
3. Seed everything before comparing runs.
4. Row-level counting only.
5. No same-turn resolution.
6. Conflicting numbers halt.

---

---

# Appendix A — Dataset Ground Truth (Inherited, Do Not Re-Derive)

> Sourced from executed analysis on `data_for_ml_model.csv`, confirmed across
> V1 and V2 runs. These facts apply to V3 unchanged — the raw CSV is the same.

## A.1 Primary Dataset

| Property | Value |
|---|---|
| Rows | 15,000 |
| Columns | 136 |
| All applicants | Fresh applicants only (`fresh_renewal = 'F'`) |
| Pre-Matric (`pre_post_matric = 1`) | 5,073 |
| Post-Matric (`pre_post_matric = 2`) | 9,908 |
| Fraud-labeled records (`sanity` not null) | 4 (0.027%) — confirmed valid in this slice; duplicate counterparts exist outside the 15,000-record boundary. **Never use as evaluation target.** |

## A.2 Confirmed 100% Null Columns — Drop at Load Time

```
updated_by, delete_record, deleted_by, delete_on, delete_ip_address,
deleted_by_level, c_university_id, p_institution_id, x_institution_id,
xii_institution_id, competitive_exam_score, xii_course_id,
new_entitled_fee_amount_centre_share, sub_category_id,
updated_by-2, updated_on-2
```

## A.3 Confirmed Duplicate Columns — Keep Only One per Group

```
# State ID group (all identical): domicile_state_id == state_id == state_id-2 == pfms_state_code
# State name group (identical):   state_name == state_name-2
# District ID group (identical):  permanent_district_id == district_id
# District name group (identical): district_name == district_name-2
```

Keep: `domicile_state_id`, `state_name`, `permanent_district_id`, `district_name`.

## A.4 Key High-Nullity Fields

| Column | Null % | Handling |
|---|---|---|
| `disability_percentage`, `disablity_type` | 99.49% | Fill 0 — disability is rare, expected |
| `orphan_flag` | 99.75% | Fill 0 |
| `gaurdian_name` | 99.77% | Fill 0 |
| `enroll_udid_no` | 99.49% | Fill 0 |
| `ration_card_no`, `ration_card_member_no` | 96.49% | Fill 0 |
| `district_short_name` | 99.97% | Drop |

## A.5 Confirmed Missing Fields (Do Not Engineer Proxies)

```
bank_account_no, bank_name, ifsc_code
```
These appear in NIC revalidation rules but are entirely absent from the CSV.

## A.6 Confirmed Data Anomalies (Sanity-Check Reference)

- **1 IP address submitted 39 applications.** Top 10 IPs: 15–39 applications each.
- **1 mobile number shared by 6 applicants.** 59 mobiles shared by 2–3.
- **Family income as low as 5 INR** — likely data entry errors or fraud.
- **3 Post-Matric applicants exceed the 35-year age limit** but are NOT flagged
  in `sanity`. Confirms enforcement gaps in the source system.
- **Institute `c_institution_id=10791`** has 151 applications — highest concentration.

## A.7 Fields That Must Never Appear as Features

```
sanity          — never a feature, never a label, never for evaluation
application_id  — row identifier only
jwt             — row identifier only
```

Also never use: `rule_violation_score`, `rule_codes_fired`, `apply_rules()`,
any NIC rule code (X1, X7, YF, UW, YK…), any engineered bridge
(IP_CONC_ENG, FEE_ENG, FM_ENG…), or any V1/V2 model checkpoint output.

---

# Appendix B — Synthetic Exposure: GAN Prohibition (Hard Reference)

> Source: arXiv:2604.13125 (2026 benchmark). This prohibition applies to
> `synthetic_exposure_builder_v3.py` and any future exposure set construction.

Standard tabular generators (CTGAN, TVAE, GaussianCopula, TabularARGN) fail
severely at preserving behavioral fraud patterns — including temporal, velocity,
and multi-account signals — with composite degradation ratios of **24x or
more**.

**Required approach:** construct the synthetic exposure set programmatically
from the actual feature distributions. Sample a real application, duplicate its
IP field across N rows, perturb name fields, etc. The goal is structurally
valid fraud-shaped examples, not statistically faithful synthetic data.

The five V3 archetypes (same as V2):

| Archetype | Construction method |
|---|---|
| IP_CONCENTRATION | Sample real application; duplicate `ip_address` across 15 rows; vary `mobile_no` |
| MOTHER_NAME_COLLISION | Pair applications; set `father_name == mother_name`; differ from `applicant_name` |
| FEE_INFLATION | Sample high-income application; set `fee_income_ratio > 1.0` |
| AGE_VIOLATION | Pre-matric: set `age_at_registration > 20`; post-matric: set > 35 |
| INCOME_VIOLATION | Set `annual_family_income < 1000` |

150 synthetic examples per archetype = 750 total. Use seeds distinct from
evaluation harness seeds.

---

# Appendix C — V1 / V2 Architecture Evolution (Historical Reference Only)

> This section documents what was tried before V3 and why it was superseded.
> None of the V1 or V2 code, model outputs, or rule logic carries into V3.

## C.1 V1 — Rule-Based Supervision

V1 applied 99 NIC policy rules + 8 engineered bridges to generate positive
training labels, then trained a LightGBM classifier on those labels. Key
outputs: `rule_violation_score`, `rule_codes_fired`, `lgbm_risk_score`.

**Ceiling:** any fraud pattern not in the rulebook was invisible regardless of
how statistically anomalous the detectors found it.

**V1 standalone-VAE baseline PR-AUC** (the original evaluation floor):

| Category | V1 VAE-Alone ROC-AUC | V1 VAE-Alone PR-AUC |
|---|---|---|
| INCOME_VIOLATION | 0.9465 | 0.1162 |
| AGE_VIOLATION | 0.8737 | 0.0506 |
| MOTHER_NAME_COLLISION | 0.8012 | 0.0258 |
| FEE_INFLATION | 0.7961 | 0.0264 |
| IP_CONCENTRATION | 0.7672 | 0.0239 |

## C.2 V1.5 (Considered, Not Built)

A hybrid extension that would have kept V1's rule system but upgraded how it
interacted with learning components. V1's trained `lgbm_risk_score` would have
been used as a distillation teacher for the tabular VAE's Stage 1 alignment
loss; rule *concepts* would have remained while EVT replaced hand-set thresholds.

**Why V2 was chosen over V1.5:** V1.5 would have baked V1's rule-bounded
geometry into the new model's starting point, preserving the detection ceiling
it was meant to raise. The self-training loop seeded by rule labels would have
been biased toward known patterns from day one. See the full comparison below.

| Dimension | V1.5 | V2 |
|---|---|---|
| Rule dependency | Reduced but present | Eliminated |
| Cold start confidence | High — V1 model provides warm round-0 signal | Lower — EVT mutual tail only |
| Novel pattern ceiling | Lower — Stage 1 shaped by rule-bounded geometry | Higher — no rule ceiling |
| Explainability at launch | Stronger — rule codes provide auditable named justification | Weaker — reconstruction dims + graph neighbors |
| Risk of V1 blind spots persisting | Real — distilling V1 score bakes in its limitations | None |
| Self-training stability | More stable — rule-confirmed positives anchor round 0 | Less stable — EVT-only round 0 is noisier |

## C.3 V2 — Five Independent Scorers

V2 ran: Tabular VAE, Graph AE (DOMINANT + DeepSVDD), Isolation Forest, MCM,
Subspace IF. EVT tail + self-training pseudo-labels replaced all rule labels.

**V2 best PR-AUC results** (the floor V3 must beat — see §5.1):

| Category | V2 Best | Scorer |
|---|---|---|
| AGE_VIOLATION | 0.1466 | MCM |
| INCOME_VIOLATION | 0.6503 | Subspace IF |
| IP_CONCENTRATION | 0.0370 | Graph AE |
| MOTHER_NAME_COLLISION | 0.2869 | Graph AE |
| FEE_INFLATION | 0.4962 | Subspace IF |

**Why V3 supersedes V2:** VAE and full-space IF were redundant (MCM dominated
both). MCM and Graph AE operated in separate worlds with no information flow
between them. V3 unifies them into a single joint model.

---

# Appendix D — V2 Architecture Critique (MAR Reference)

> Source: MAR_v2.md (internal Model and Architecture Review, June 2026).
> Some failure modes are partially mitigated in V3; others remain.
> Read before making changes to the hybrid model, EVT scorer, or self-training loop.

## D.1 Structural Weaknesses That Persist in V3

| Component | Core Assumption | Failure Condition | Failure Mode | V3 Status |
|---|---|---|---|---|
| DeepSVDD / Hybrid centroid | Normal data density is clean | Fraud dominates the dataset | Hypersphere inflates to accept fraud as normal — silent | **Partially mitigated** — contamination-aware init excludes top 5% embedding-norm nodes before computing centroid (`CENTROID_CLEAN_PERCENTILE=95`). Does not eliminate risk if fraud fraction > 5%. |
| EVT Scorer | Tail fits GPD smoothly | Score distribution has discrete cluster spikes or violates regularity | Threshold explodes or drops to 0 | **Partially mitigated** — score jitter (σ=0.001) smooths discrete spikes; shape validation rejects GPD fits outside `[EVT_SHAPE_MIN, EVT_SHAPE_MAX]` = `[-0.5, 1.0]` and falls back to empirical quantile. Verified live: caught `subspace_if_score` (shape=-0.513) and `subspace_if_network` (shape=-0.533) on first run. |
| Self-Training Loop | EVT tail contains true positives | EVT tail is mostly data entry typos (e.g., income = 5 INR) | Classifier anchors on typos, misses sophisticated fraud | **Partially mitigated** — `MIN_SIGNALS_FOR_PROMOTION=2` requires independent corroboration from ≥2 EVT signals before promotion. Single-signal noise hits (e.g. income=5 INR firing EVT_FINANCIAL alone) no longer promoted. Human EVT-tail review before Round 1 remains mandatory. |
| Isolated nodes | Degree features make isolation visible | Applicant has unique values on all 5 edge fields AND normal-looking features | Statistically indistinguishable from a genuine isolated student | **Partially mitigated** (degree features help; forensically clean isolated fraud remains undetectable). Diagnostic added: scoring pass prints isolated-node fraction in top-100 risk scores. |
| Stage 1 Synthetic Exposure | Archetypes represent real fraud geometry | Archetypes are too narrow or obvious | Stage 2 biased toward obvious fraud; subtle patterns treated as normal | **Partially mitigated** — each archetype now produces `N_CLEAN=50` peak-signal examples + `N_PERTURB=100` graduated variants spanning 85th–97th percentile signal strength, plus `_add_context_noise()` perturbing 25% of non-target features per row. Wider geometry. Novel fraud archetypes remain unrepresented — still needs archetype expansion. |

## D.2 What Would Break First in Production

**The self-training label promotion.** Without robust ground truth, relying on
EVT mutual agreement to seed the LightGBM is the highest-risk step. A slight
misalignment in EVT score distribution tails seeds the classifier with false
positives, triggering semantic drift in Round 1.

**Recommended mitigation (not yet implemented):** replace fully automated Round 0
with an active learning interface where an investigator explicitly reviews the
EVT tail before the LightGBM is ever allowed to train.

## D.3 What the Metrics Don't Measure

Phase D synthetic harness PR-AUC proves the system detects injected archetypes.
It does NOT prove the system can detect zero-day real-world fraud patterns
outside those specific synthetic topologies.

## D.4 Outstanding External Dependencies

- **AISHE/DISE data:** institutional geo-location data needed to structurally
  ground `institute_application_count` into a true physical feature. Until
  available, `state_match_flag` remains dormant.
- **Human review of Round 0 pseudo-labels:** before Round 1 self-training,
  NIC investigators should perform a blind review of the top EVT-flagged
  applications to confirm cold-start precision.

---

# Appendix E — Research Citations

## E.1 Primary (directly grounds a V3 component)

| Short Ref | Full Citation | Grounds |
|---|---|---|
| `GraphMAE` | Hou et al., "Masked Autoencoders Are Scalable Vision Learners," KDD 2022 | Feature-stream + graph-stream design in `hybrid_graphmcm_v3.py` |
| `MaskGAE` | Li et al., NeurIPS 2022 Workshop | Joint masking of edges and features; two-error scoring |
| `MCM` | Masked Cell Modeling, ICLR 2024 | Learned masks for tabular conditional prediction |
| `LOE` | Qiu et al., "Latent Outlier Exposure for Anomaly Detection with Contaminated Data," ICML 2022, arXiv:2202.08088 | Graph-side Stage 1 warm-start in hybrid model |
| `DOMINANT` | Ding et al., "Deep Anomaly Detection on Attributed Networks," SDM 2019 | RGCN encoder architecture (aggr='add', tanh bounding) |
| `DeepSVDD` | Ruff et al., "Deep One-Class Classification," ICML 2018 | Hypersphere centroid; anomaly = distance from normal centroid |
| `EVT-SPOT` | Siffer et al., "Anomaly Detection in Streams with Extreme Value Theory," KDD 2017 | GPD tail fitting in `evt_scorer_v3.py` |
| `R-GCN` | Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks," ESWC 2018 | Typed-edge RGCN encoder |
| `ASTRA-SelfTrain` | Karamanolakis et al., "Self-Training with Weak Supervision," NAACL 2021, arXiv:2104.05514 | Self-training promotion logic |
| `OutlierExposure` | Hendrycks et al., "Deep Anomaly Detection with Outlier Exposure," ICLR 2019, arXiv:1812.04606 | Synthetic anomaly exposure curriculum |
| `GNNExplainer` | Ying et al., NeurIPS 2019, arXiv:1903.03894 | XAI layer per-case explanations |
| `PGExplainer` | Luo et al., NeurIPS 2020, arXiv:2011.04573 | XAI layer volume-scale upgrade |
| Tabular GAN benchmark | arXiv:2604.13125 (2026) | Hard prohibition on GAN-generated exposure sets (Appendix B) |

## E.2 Tech Stack

| Purpose | Library | Version constraint |
|---|---|---|
| Data loading | `pandas` | >= 1.5 |
| Numerical ops | `numpy` | >= 1.23 |
| Neural networks | `torch` (PyTorch) | >= 2.0 |
| Graph neural networks | `torch_geometric` (PyG) | >= 2.4 |
| Graph construction | `networkx` | >= 3.0 |
| Gradient boosting | `lightgbm` | >= 4.0 |
| SHAP explainability | `shap` | >= 0.44 |
| EVT / GPD fitting | `scipy.stats.genpareto` | >= 1.11 |
| Evaluation metrics | `scikit-learn` | >= 1.2 |

**Do not introduce:** `tensorflow`, `keras`, `xgboost` (unless discussed),
SMOTE/oversampling, any autoML library, any tabular GAN (Appendix B).

---

# Appendix F — MLOps Architecture Decisions (ADR Format)
<!-- STATUS: Proposed | OWNER: Project Lead | DATE: 2026-06-29 -->
<!-- These ADRs define the operational infrastructure for V3. -->
<!-- No ADR is implemented until explicitly authorized by the project lead. -->

## F.0 Context

V3 has a well-designed ML architecture with zero MLOps infrastructure.
There is no experiment tracking, model registry, artifact versioning, CI/CD,
containerization, or observability. The file-based inter-module contracts
(§8) are an MLOps asset — every improvement here wraps the existing `src/`
modules without touching them.

**Three-phase migration:**
- **Phase 1 (zero-risk):** Fix the broken reproducibility baseline.
- **Phase 2 (low-risk):** Operational layer — API, async jobs, registry, logging.
- **Phase 3 (infrastructure):** Containerization, Kubernetes, PostgreSQL, CI/CD.

**Invariant:** no `src/*.py` module is modified in any phase. Wrappers only.

---

## ADR-001 — Fix `requirements.txt` to declare all dependencies

**Status:** Implemented (2026-06-29)
**Context:** `requirements.txt` did not declare `torch`, `torch_geometric`, or
any operational dependencies. A fresh `pip install -r requirements.txt` on the
production server produced a broken environment.
**Decision:** Add pinned minimum versions for PyTorch, PyG, FastAPI, Celery,
Redis, psycopg3, MLflow, and structlog to `requirements.txt`. Add a separate
`requirements-dev.txt` for DVC, ruff, mypy, pytest, and pre-commit.
**Note on PyTorch install:** the wheel is platform-specific. CPU server:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.
GPU laptop: `pip install torch` (default index picks the CUDA wheel).
**Consequences:** Reproducible environments. Docker builds will succeed.
**Files changed:** `requirements.txt`, `requirements-dev.txt` (new).
**Phase:** 1

---

## ADR-002 — MLflow experiment tracking

**Status:** Proposed
**Context:** Every training run overwrites `models/hybrid_graphmcm_v3.pth`
in place. No hyperparameters, metrics, or run metadata are recorded.
**Decision:** Wrap `main_v3.py` and `retraining_orchestrator.py` entry points
in `mlflow.start_run()` context managers. Log all `config_v3.py` constants as
params, all PR-AUC values from `evaluate_model_v3.py` as metrics, and model
checkpoints as artifacts. Use local file-based MLflow tracking for Phase 1–2;
migrate to a tracking server in Phase 3 if needed.
**Technology chosen:** MLflow over W&B (W&B sends metadata to cloud — NIC
data cannot leave premises) and Aim (smaller community, less mature registry).
**Consequences:** Every run is reproducible by checkpoint. Year-over-year
PR-AUC comparison is automatic. No `src/` module is modified.
**Files to change:** `main_v3.py` (wrap), `retraining_orchestrator.py` (wrap),
`src/evaluate_model_v3.py` (log metrics).
**Phase:** 1

---

## ADR-003 — DVC for data and artifact versioning

**Status:** Proposed
**Context:** `data/processed/` artifacts change every cycle and are not
version-controlled. `models/*.pth` are overwritten in place. No data lineage
exists between a model checkpoint and the dataset that produced it.
**Decision:** `dvc init` in the repository. Track `data/processed/`,
`models/`, and `outputs/` under DVC. Use local remote storage initially;
upgrade to Azure Blob or S3-compatible in Phase 3.
**Technology chosen:** DVC over LakeFS (requires S3-compatible backend,
adds infrastructure) and Delta Lake (requires Spark, Parquet-only).
**Consequences:** Every artifact is content-addressed. `dvc checkout` reproduces
any prior cycle's exact state. `dvc push` provides off-site backup.
**Files to change:** `.dvc/` (new), `.dvcignore` (new), `.gitignore` (update).
**Phase:** 1

---

## ADR-004 — Pre-commit hooks for code quality

**Status:** Proposed
**Context:** No automated code quality enforcement. A type error or import
error in any `src/` module is discovered only at pipeline runtime — which may
be months after the change was merged.
**Decision:** Add `.pre-commit-config.yaml` with `ruff` (lint + format),
`mypy` (type check), and `pytest --smoke` (2-epoch pipeline smoke test).
**Consequences:** Breaking changes are caught before they reach `main`. The
smoke test adds ~3 minutes to commit time but catches import errors immediately.
**Files to change:** `.pre-commit-config.yaml` (new), `pyproject.toml` (new).
**Phase:** 1

---

## ADR-005 — FastAPI inference and supervisor server

**Status:** Proposed
**Context:** The supervisor workflow (confirm fraud, mark false positives,
trigger retraining) currently requires a developer to run Python scripts
directly. This is a deployment blocker for any non-technical user.
**Decision:** Implement a 20-endpoint REST API in `src/api/`. Six endpoint
groups: inference, supervisor feedback, training, monitoring, configuration,
health. All training commands return a `job_id` immediately (async via
Celery — see ADR-006). No `src/` module is modified — handlers call the
existing `confirmed_fraud_store`, `retraining_orchestrator`, and `evt_scorer`
functions directly.
**Key endpoints:**
```
POST /v3/supervisor/confirm-fraud        → confirmed_fraud_store.add_confirmed()
POST /v3/supervisor/mark-false-positive  → confirmed_fraud_store.add_false_positive()
POST /v3/training/incremental            → retraining_orchestrator [async]
POST /v3/training/full                   → main_v3.run_pipeline() [async]
GET  /v3/training/jobs/{job_id}          → Celery task status
GET  /v3/monitoring/drift                → _check_drift() result
GET  /v3/monitoring/fraud-store-summary  → confirmed_fraud_store.summary()
GET  /health                             → 200 if model loaded
GET  /ready                              → 200 if scores CSV exists
POST /v3/training/upload-checkpoint      → checkpoint_manager.validate_and_hotswap() [multipart .pth]
POST /v3/training/pull-checkpoint        → dvc pull + checkpoint_manager.validate_and_hotswap() [async]
GET  /v3/model/checkpoint-info           → {version, loaded_at, mlflow_run_id, n_features, graph_emb_dim}
POST /v3/model/rollback                  → checkpoint_manager.validate_and_hotswap(versioned_path) [async]
```
**Open question:** public-facing vs internal-only (VPN). If public, TLS
termination and authentication (API keys or OAuth2) must be added. Resolve
before Phase 2 implementation.
**Files to change:** `src/api/main.py` (new), `src/api/handlers/` (new),
`src/api/schemas.py` (new).
**Phase:** 2

---

## ADR-006 — Celery + Redis for async job management

**Status:** Proposed
**Context:** `train_incremental()` runs ~15 minutes. `run_pipeline()` runs
2–4 hours. HTTP endpoints cannot block for this duration.
**Decision:** Celery with Redis as broker. Each training command returns a
`job_id` immediately. Callers poll `GET /v3/training/jobs/{job_id}` for
status. On the CPU server, a single Celery worker with `concurrency=1`
serializes training jobs (training must not run concurrently).
**Technology chosen:** Celery + Redis over Prefect (adds a UI server; overkill
for twice-yearly jobs) and Kubeflow Pipelines (requires full K8s cluster,
YAML pipelines, weeks of setup — too heavy for batch-once-per-year).
**Files to change:** `src/api/tasks.py` (new), `celeryconfig.py` (new),
`docker-compose.yml` (Redis service).
**Phase:** 2

---

## ADR-007 — Structured logging with structlog

**Status:** Proposed
**Context:** All modules use `print()`. On a server with cron-scheduled runs,
logs from different runs interleave without timestamps or module tags. There
is no machine-parseable format for alerting or aggregation.
**Decision:** Replace `print()` calls at orchestrator and API level with
`structlog` JSON logging. Module internals may keep `print()` — only
`main_v3.py`, `retraining_orchestrator.py`, and `src/api/main.py` are changed.
`structlog` is drop-in compatible with Python's `logging` module.
**Consequences:** Logs are shippable to ELK, Loki, or CloudWatch. Each log
line includes timestamp, module name, cycle label, and log level.
**Files to change:** `main_v3.py`, `retraining_orchestrator.py`,
`src/api/main.py`.
**Phase:** 2

---

## ADR-008 — Versioned model registry and rollback

**Status:** Proposed
**Context:** `train_incremental()` writes a single `.pth.bak` before
overwriting. After two incremental runs the first checkpoint is gone. There
is no way to roll back to a known-good model more than one step.
**Decision:** After every training run, copy the checkpoint to a versioned
path: `models/checkpoints/hybrid_graphmcm_v3_<cycle>_<mlflow_run_id>.pth`.
Keep the last 5 versioned checkpoints. Register each in MLflow as an artifact
with its associated PR-AUC and cycle label. Add
`rollback_to_checkpoint(run_id)` to `retraining_orchestrator.py` that copies
the versioned file back to `models/hybrid_graphmcm_v3.pth` and restores its
DVC hash.
**Consequences:** Any prior cycle's model can be restored in one command.
PR-AUC regression is recoverable.
**Files to change:** `retraining_orchestrator.py` (add rollback function),
`src/hybrid_graphmcm_v3.py` (update `_backup_checkpoint()`).
**Phase:** 2

---

## ADR-009 — Feature-level drift monitoring

**Status:** Proposed
**Context:** The existing KS test (in `retraining_orchestrator.py`) monitors
`hybrid_anomaly_score` (output). Distribution shift in individual input
features (e.g., a policy change that inflates all reported incomes by 2×) is
not caught until it shows up as anomaly score drift — a lagging indicator.
**Decision:** After each cycle, compute and store mean and standard deviation
of each of the 68 engineered features. At the next cycle, compute KS
statistics per feature and flag any feature where p < 0.01. Log to the MLflow
run as a metric artifact `feature_drift_v3.json`. Alert but do not block
inference — the supervisor decides whether to proceed or wait for full retrain.
**Consequences:** Earlier warning of dataset shift. Preserves human-gated
philosophy of the self-training loop.
**Files to change:** `retraining_orchestrator.py` (add `_check_feature_drift()`),
`outputs/feature_drift_v3.json` (new output artifact).
**Phase:** 2

---

## ADR-010 — Docker containerization (two-image strategy)

**Status:** Proposed — do not implement until Phase 1 is complete
**Context:** The application runs only in the developer's Python environment.
Deployment to the CPU server requires manual environment setup.
**Decision:** Two Docker images:
- `nic-fraud-trainer`: GPU-capable, includes PyTorch + PyG + full pipeline.
  Used on the GPU laptop for initial training and full retrains.
- `nic-fraud-server`: CPU-only, includes FastAPI + Celery worker + **full
  pipeline** (`main_v3.py`). This image deploys to the production server and
  supports both incremental fine-tune (~15 min) and full CPU retrain (8–16 hr
  via `POST /v3/training/full`). GPU hardware is not required on the server.
`WORKDIR /app` in both images ensures all `Path("outputs/...")` calls resolve
correctly. The `.pth` checkpoint is transferred from trainer to server via
`dvc push` (trainer) + `dvc pull` (server) after each full retrain.
**Files to change:** `Dockerfile.trainer` (new), `Dockerfile.server` (new),
`docker-compose.yml` (new), `.dockerignore` (new).
**Phase:** 3

---

## ADR-011 — Kubernetes deployment on the CPU server

**Status:** Proposed — do not implement until Docker images are stable
**Context:** Docker Compose has no self-healing or health-check-based restart.
The CPU server must run unattended for the year between cycles.
**Decision:** Single-node Kubernetes (k3s) with three deployments:
- `nic-api`: FastAPI, 2 replicas, request 2 vCPU / 8 GB, limit 4 vCPU / 16 GB
- `nic-worker`: Celery worker, 1 replica, request 8 vCPU / 32 GB, limit 16 vCPU / 56 GB
  (concurrency=1 — training must not run concurrently)
- `mlflow`: 1 replica, limit 1 vCPU / 2 GB (local file-based tracking)
- `redis`: single pod, 512 MB
Liveness and readiness probes hit `/health` and `/ready`. Rolling deploy for
`nic-api` on image update; `nic-worker` is restarted manually after training
jobs (to avoid splitting a long-running job across deploys).
**Files to change:** `k8s/` directory (new — Deployment, Service, ConfigMap
manifests for each component).
**Phase:** 3

---

## ADR-012 — PostgreSQL for ego-graph inference queries

**Status:** Proposed
**Context:** At inference time, a new application needs its relational
neighbors (across 5 edge types) to build a mini-subgraph for the RGCN
encoder. Querying a `.pt` file for this at inference time is not practical.
**Decision:** Load `data/raw/data_for_ml_model.csv` into a PostgreSQL table
(`applications`) with indexed columns for all 5 edge-type fields (`mobile_no`,
`ip_address`, `father_name`, `mother_name`, `pincode`). At inference time,
query neighbors per edge type, build a mini-subgraph, run RGCN, return
`hybrid_anomaly_score` for the new node only.
**Open question:** cross-cycle schema design — does the `applications` table
accumulate across years (enabling cross-cycle IP cluster detection) or rebuild
each year? No decision made. Resolve before Phase 3 implementation.
**Files to change:** `src/inference_server_v3.py` (new), `db/schema.sql`
(new), `docker-compose.yml` (postgres service).
**Phase:** 3

---

## ADR-013 — CI/CD pipeline with GitHub Actions

**Status:** Proposed
**Context:** No automated validation of code changes. A change to any `src/`
module goes directly to the main branch with no gate.
**Decision:** Three GitHub Actions jobs:
1. `lint` — ruff + mypy on every push to any branch.
2. `smoke` — `python main_v3.py --smoke` (2-epoch run, CPU) on pull requests
   to `main`.
3. `docker-build` — build both Docker images on merge to `main`.
No GPU in CI — smoke test uses CPU with synthetic data.
**Consequences:** Breaking changes caught within minutes. Docker images always
current on `main`.
**Files to change:** `.github/workflows/ci.yml` (new).
**Phase:** 3

---

## F.1 Open MLOps Questions — Do Not Resolve Autonomously

- **OQ-1:** Where does the MLflow tracking server live in production? Options:
  co-locate on CPU server (risk: single point of failure), separate VM,
  or Azure ML as tracking backend. Depends on NIC IT constraints.
- **OQ-2:** Is the FastAPI server public-facing or internal-only (VPN)?
  If public: requires TLS termination, authentication, rate limiting.
  The current API design does not include auth. Resolve before Phase 2.
- **OQ-3:** Data retention policy for `confirmed_fraud.json` — should
  confirmed fraud from prior years continue to influence the LOE exposure
  set? No policy decision has been made.
- **OQ-4:** Is `data/raw/data_for_ml_model.csv` manually exported or is there
  an automated data ingestion step? If manual, the yearly pipeline has an
  undocumented human dependency.
- **OQ-5:** Should the PostgreSQL `applications` table accumulate cross-cycle
  (enabling cross-cycle relational pattern detection) or rebuild each year?
  Resolve before ADR-012 implementation.

---

## F.2 Migration Roadmap Summary

| Phase | Duration | Key deliverables | Risk |
|---|---|---|---|
| 1 — Zero-risk | 1–2 weeks | Fix requirements.txt (done), DVC init, MLflow wrappers, pre-commit | Zero — additive only |
| 2 — Operational | 2–4 weeks | FastAPI server, Celery worker, model registry, structured logging, feature drift | Low — new files only |
| 3 — Infrastructure | 4–8 weeks | Docker images, Kubernetes, PostgreSQL, CI/CD | Medium — infrastructure |

**Invariant across all phases:** no `src/*.py` module is modified.

---

# Appendix G — Production Server Environment
<!-- These are fixed hardware and memory constraints for the target deployment. -->
<!-- An agent must not make architectural choices that violate these bounds. -->

> **Agent instruction:** read this appendix before choosing batch sizes,
> deciding whether to load the full graph in memory, or designing any module
> that runs on the production server. Violating the memory budget causes an
> OOM kill that terminates the entire training run with no checkpoint saved.

## G.1 Hardware

| Parameter | Value |
|---|---|
| CPU | 16 vCPU |
| RAM | 64 GB |
| OS | Ubuntu 22.04 LTS |
| GPU | None — CPU-only PyTorch build required on server |
| PyTorch install | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Storage | k3s local-path PersistentVolume mounted at `/app` |

## G.2 Memory Budget

| Operational mode | Peak RSS |
|---|---|
| Inference-only (`nic-api`, per replica) | ~1 GB |
| Full pipeline training (`nic-worker`) | ~8–12 GB |
| Both running simultaneously | ~14 GB — 4× safety margin within 64 GB |

No module may hold more than one full copy of the feature matrix
(15,000 × 68 float32 ≈ 4 MB) AND the full graph (`identity_graph_v3.pt`,
≈ 400–800 MB) AND the model weights (≈ 300–600 MB) simultaneously in the
same process outside of the training window.

**At inference time the graph `.pt` file is not loaded.** Only the model
weights and the feature schema are needed. The graph is rebuilt from the
PostgreSQL ego-graph query (ADR-012) for individual-application scoring, or
loaded once per batch for full-cycle scoring.

## G.3 CPU Full Retrain Time Estimate

| Stage | GPU laptop (CUDA) | Server 16 vCPU (CPU only) |
|---|---|---|
| Stage 1: LOE warm-start (80 epochs) | 30–60 min | 2–4 hr |
| Stage 2: joint reconstruction (120 epochs) | 60–90 min | 4–8 hr |
| All other pipeline steps combined | ~50 min | ~50 min |
| **Total** | **~2–4 hr** | **~8–16 hr** |

A full retrain on the server is a blocking operation for the `nic-worker` pod.
`nic-api` (inference serving) continues uninterrupted in its own pod.
The smoke test (`--smoke`, 2 epochs) completes in ~5 minutes on CPU and is
the CI gate for every pull request (ADR-013).

## G.4 PyTorch CPU Thread Configuration

Set these at the top of any module that runs heavy tensor operations on the
server. Omitting them lets PyTorch default to all 16 vCPU, which causes
contention with the `nic-api` replicas.

```python
import torch
torch.set_num_threads(8)          # intra-op parallelism (BLAS, matmul)
torch.set_num_interop_threads(4)  # inter-op parallelism (async ops)
```

Do not call these inside `src/` modules — set them in the Celery task wrapper
(`src/api/tasks.py`) so the values are applied once per worker process and
do not affect local development or the GPU laptop.
