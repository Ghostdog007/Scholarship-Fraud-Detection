# NIC Fraud Detection — V3 Hybrid GraphMCM: Agent Context File
<!-- VERSION: 3.0-draft | OWNER: Project Lead | LAST REVIEWED: 2026-06 -->
<!-- DO NOT MODIFY AUTONOMOUSLY. Flag outdated content; only project lead edits this file. -->

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
V3 must beat V2's best scores:

| Category | V2 Best | Scorer | Floor to beat |
|---|---|---|---|
| AGE_VIOLATION | 0.1466 | MCM | 0.1466 |
| INCOME_VIOLATION | 0.6503 | Subspace IF | 0.6503 |
| IP_CONCENTRATION | 0.0370 | Graph AE | 0.0370 |
| MOTHER_NAME_COLLISION | 0.2869 | Graph AE | 0.2869 |
| FEE_INFLATION | 0.4962 | Subspace IF | 0.4962 |

### 5.2 Degree-Stratified PR-AUC (new in V3)

For each category, report PR-AUC split by the degree of the injected nodes:
- `isolated` (degree=0)
- `low` (degree 1–5)
- `high` (degree 6+)

This makes the isolated-node gap explicit and measurable rather than hidden in
the aggregate number.

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
```

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
| `outputs/evt_thresholds_v3.json` | evt | self_training | `{hybrid: {u, scale, shape, threshold}, subspace_if: {...}}` |
| `outputs/pseudo_labels_v3.json` | self_training | fusion | `{application_id: label_source}` |
| `outputs/risk_scores_v3.csv` | fusion | xai, evaluate | `application_id, risk_score_v3, label_source` |
| `outputs/explanation_cards_v3.json` | xai | [end user] | per-application JSON cards |

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
| XAI | `src/xai_layer_v3.py` | `hybrid_scores_v3.csv`, `risk_scores_v3.csv` | `explanation_cards_v3.json` | no training code |
| Evaluate | `src/evaluate_model_v3.py` | `engineered_features_v3.csv`, `models/*.pth` | console stdout | no training code |
| Orchestrator | `main_v3.py` | — | calls all modules | no business logic |

---

## 10. Hard Stops (Inherited + New)

**All V2 hard stops apply unchanged.** Additional V3 stops:

**9. Dimension constants are in `config_v3.py` only.**
Never hardcode `68`, `64`, `132`, `8`, or `5` inside module code. Import from
`config_v3`. If you find a hardcoded dimension, fix it before continuing.

**10. `h_N(i)` never leaves `hybrid_graphmcm_v3.py`.**
The graph embedding is internal to the hybrid model. Downstream modules receive
only `hybrid_anomaly_score`, `feature_pred_error`, `edge_pred_error`, and
`per_feature_error_json`. Raw embeddings are not outputs.

**11. Isolated nodes use the learned `isolated_embedding`, not zero vectors.**
`torch.zeros(GRAPH_EMB_DIM)` for isolated nodes is forbidden. The model must
have a trainable `nn.Parameter` named `isolated_embedding` of shape
`(GRAPH_EMB_DIM,)` initialized to `torch.randn`.

**12. Degree features are always in the 63:68 slice.**
`engineered_features_v3.csv` columns 0–62 are the original 63 features in the
same order as V2. Columns 63–67 are the 5 degree features in this order:
`degree_shares_mobile`, `degree_shares_ip`, `degree_shares_father_name`,
`degree_shares_mother_name`, `degree_shares_pincode`. Any module that slices
the feature matrix must use named indexing via `v3_feature_schema.json`, never
positional hardcoding.

**13. `score_retention` must be printed by evaluate_model_v3.py.**
The edge-dropout test (§5.3) must run as part of every evaluation. If
`score_retention < 0.2`, print a warning: "Graph-dependent detection: isolated
node performance will be degraded."

---

## 11. Open Architecture Questions — Do Not Resolve Autonomously

- Optimal `LAMBDA_EDGE` (currently 0.3) — balance between feature and edge
  prediction in the hybrid score. Needs ablation.
- Whether `isolated_embedding` should be shared across all isolated nodes or
  per-node (currently shared — cheaper, may under-fit large isolated sets).
- `MASK_NUM=8` inherited from V2 MCM — no ablation done with graph context.
- `EPOCHS_STAGE1=80` for graph LOE — V2 used 100, reduced here because feature
  stream doesn't need LOE. May need tuning.
- Whether Subspace IF groups are final — `network` group includes degree
  features which are new and untested as IF inputs.

---

## 12. Quantitative Claims Protocol (inherited from V2, unchanged)

1. Raw stdout only. No number enters a doc without a traceable print line.
2. Name the baseline explicitly. V3 must beat V2 best scores from §5.1.
3. Seed everything before comparing runs.
4. Row-level counting only.
5. No same-turn resolution.
6. Conflicting numbers halt.
