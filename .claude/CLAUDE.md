# CLAUDE.md — NIC Scholarship Fraud Detection

## What This Project Is

ML-based fraud detection for the NIC scholarship portal. The system produces
a per-application anomaly risk score (0–1) for 15,000 fresh applications.
There are two architectures: **v1** (rule + bridge supervised, canonical run
done, do not retrain without explicit instruction) and **v2** (fully
rule-free, currently under development). You are almost always working on v2.

---

## Read These Files First — Every Session

1. **`docs/AGENTS.md`** — primary reference for all v2 work.
   Read in full before writing any code.
2. **`docs/MAR_v2.md`** — Model and Architecture Review. Contains the
   unbiased critique, known failure modes, and the critique summary table.
   Read Layer 3 (critique) before proposing architectural changes.

If these files are not in the working directory, stop and say so before
writing any code.

---

## Project Directory Layout

```
NIC fraud Detection Project/
├── main_v2.py                          # Pipeline orchestrator (entry point)
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                                # All Python source modules
│   ├── tabular_feature_engine_v2.py    # Phase A: feature engineering
│   ├── graph_builder_v2.py             # Phase B: identity graph
│   ├── synthetic_exposure_builder_v2.py # Phase B: LOE exposure set
│   ├── tabular_vae_v2.py              # Phase C: tabular VAE
│   ├── graph_autoencoder_v2.py        # Phase C: DOMINANT + DeepSVDD
│   ├── evt_scorer.py                  # Phase D: EVT thresholds
│   ├── self_training_loop_v2.py       # Phase D: pseudo-label promotion
│   ├── fusion_classifier_v2.py        # Phase E: LightGBM fusion
│   ├── xai_layer_v2.py               # Phase E: SHAP + GNNExplainer
│   ├── evaluate_model_v2.py           # Phase F: synthetic harness
│   └── get_metrics.py                 # Standalone metric helper
│
├── data/
│   ├── raw/                            # Original untouched datasets
│   │   └── data_for_ml_model.csv       # 15,000 × 136 primary dataset
│   └── processed/                      # Generated intermediate artifacts
│       ├── engineered_features_v2.csv  # 15,000 × 63 numeric features
│       ├── v2_feature_schema.json      # Column names and exclusions
│       ├── identity_graph.pt           # PyG HeteroData graph
│       └── synthetic_exposure_set.pt   # 750 × 63 LOE tensor
│
├── models/                             # Saved PyTorch checkpoints
│   ├── tabular_vae_v2.pth             # VAE state_dict
│   └── graph_autoencoder_v2.pth       # DOMINANT state_dict + centroid
│
├── outputs/                            # Final pipeline outputs
│   ├── vae_v2_scores.csv
│   ├── graph_v2_scores.csv
│   ├── evt_thresholds_v2.json
│   ├── pseudo_labels_v2.json
│   ├── risk_scores_v2.csv
│   └── explanation_cards_v2.json
│
└── docs/
    ├── AGENTS.md           # Architecture contract (do not edit)
    └── MAR_v2.md                       # Model and Architecture Review
```

**Convention:** all code paths are relative to the project root. Source
modules live in `src/`. Data reads from `data/raw/` or `data/processed/`.
Model checkpoints save to `models/`. Pipeline outputs write to `outputs/`.
Never write outputs to the project root.

---

## Architecture in Brief

```
data/raw/data_for_ml_model.csv [CSV]
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
src/tabular_feature_engine_v2.py          src/graph_builder_v2.py
        │                                              │
        │ data/processed/                              │ data/processed/
        │   engineered_features_v2.csv [CSV]           │   identity_graph.pt [Tensor]
        │   v2_feature_schema.json [JSON]              │
        ▼                                              ▼
src/tabular_vae_v2.py                     src/graph_autoencoder_v2.py
        │                                              │
        │ outputs/vae_v2_scores.csv [CSV]              │ outputs/graph_v2_scores.csv [CSV]
        │ models/tabular_vae_v2.pth                    │ models/graph_autoencoder_v2.pth
        └───────────────────┬──────────────────────────┘
                            ▼
                   src/evt_scorer.py
                            │ outputs/evt_thresholds_v2.json [JSON]
                            ▼
               src/self_training_loop_v2.py
                            │ outputs/pseudo_labels_v2.json [JSON]
                            ▼
               src/fusion_classifier_v2.py
                            │ outputs/risk_scores_v2.csv [CSV]
                            ▼
                   src/xai_layer_v2.py
                            │ outputs/explanation_cards_v2.json [JSON]
                            ▼
               src/evaluate_model_v2.py
                            │ Console PR-AUC output
                            ▼
                          [END]
```

Seven source modules, strict file-based contracts. Find your module in
`docs/AGENTS.md §9` and stay inside it. If a task spans two
modules, stop and confirm scope before writing anything.

---

## Current Hyperparameters (Locked Unless Ablation Justifies Change)

| Parameter | Value | File | Line |
|---|---|---|---|
| VAE latent dim | 8 | `src/tabular_vae_v2.py` | 12 |
| VAE encoder layers | 63 → 32 → 16 → μ/σ(8) | `src/tabular_vae_v2.py` | 15–22 |
| VAE decoder layers | 8 → 16 → 32 → 63 (Sigmoid) | `src/tabular_vae_v2.py` | 24–31 |
| VAE learning rate | 1e-3 (Adam) | `src/tabular_vae_v2.py` | 94 |
| VAE batch size | 256 | `src/tabular_vae_v2.py` | 84 |
| VAE Stage 1 epochs | 20 | `src/tabular_vae_v2.py` | 96 |
| VAE Stage 2 epochs | 30 | `src/tabular_vae_v2.py` | 97 |
| VAE LOE margin | 5.0 | `src/tabular_vae_v2.py` | 98 |
| VAE λ(t) schedule | linear decay: `1.0 - (epoch / epochs_stage1)` | `src/tabular_vae_v2.py` | 104 |
| Graph AE hidden channels | 64 | `src/graph_autoencoder_v2.py` | 67 |
| Graph AE output channels | 32 | `src/graph_autoencoder_v2.py` | 68 |
| Graph AE learning rate | 1e-3 (Adam) | `src/graph_autoencoder_v2.py` | 71 |
| Graph AE Stage 1 epochs | 100 | `src/graph_autoencoder_v2.py` | 69 |
| Graph AE Stage 2 epochs | 100 | `src/graph_autoencoder_v2.py` | 70 |
| Graph AE λ(t) schedule | linear decay: `1.0 - (epoch / epochs_stage1)` | `src/graph_autoencoder_v2.py` | 98 |
| Graph AE exposure loss | `exp(-sqrt(dist))` (no hard margin) | `src/graph_autoencoder_v2.py` | 123 |
| EVT q (false-positive rate) | 0.002 | `src/evt_scorer.py` | 41 |
| EVT u_percentile (POT) | 95 | `src/evt_scorer.py` | 49 |
| Synthetic anomalies per archetype | 150 | `src/synthetic_exposure_builder_v2.py` |  |
| Total synthetic exposure set | 750 (5 × 150) | `data/processed/synthetic_exposure_set.pt` |  |
| Input feature dimensions | 63 numeric columns | `data/processed/v2_feature_schema.json` |  |
| Graph edge types | 5 (`shares_mobile`, `shares_ip`, `shares_father_name`, `shares_mother_name`, `shares_pincode`) | `src/graph_builder_v2.py` |  |
| Random seed (Graph AE) | 42 | `src/graph_autoencoder_v2.py` | 11–12 |

---

## Current Evaluation Results (Last Verified Run)

```
--- PR-AUC Comparison: V2 Models vs V1 Baseline ---
AGE_VIOLATION             | V1: 0.0506 | V2 Tabular VAE: 0.0120 | FAILED
INCOME_VIOLATION          | V1: 0.1162 | V2 Tabular VAE: 0.0109 | FAILED
IP_CONCENTRATION          | V1: 0.0239 | V2 Graph AE: 0.0953 | PASSED
MOTHER_NAME_COLLISION     | V1: 0.0258 | V2 Graph AE: 0.1428 | PASSED
FEE_INFLATION             | V1: 0.0264 | V2 Graph AE: 0.2112 | PASSED
```

**Interpretation:** The Graph AE dominates V1 on relational categories (4x–8x
improvement). The Tabular VAE underperforms V1 on univariate tabular
categories (AGE, INCOME). This is expected — V1's rule engine had hardcoded
boundaries for these exact patterns. The VAE must discover them from
reconstruction error alone, which is harder without domain hints.

**Known gap:** Phase D PR-AUC proves archetype detection. It does not prove
zero-day detection of novel fraud patterns outside the 5 synthetic topologies.

---

## How to Run

```bash
# Full pipeline (from project root):
python main_v2.py

# Individual module (from project root):
.\.venv\Scripts\python.exe src/tabular_vae_v2.py
```

`main_v2.py` calls each `src/` module sequentially via subprocess. All
modules assume the working directory is the project root, not `src/`.

---

## Hard Stops — Never Proceed Past These

**1. No rules. No exceptions.**
If you find yourself writing any of the following, stop immediately:
- A numeric threshold against a domain concept (`ip_count >= 15`, `age > 35`)
- A named rule code (`X1`, `YF`, `IP_CONC_ENG`, etc.)
- A call to `apply_rules()` or any equivalent
- A feature whose definition encodes a policy boundary

The only numeric thresholds allowed are EVT-derived (`src/evt_scorer.py`) or
learned from synthetic exposure (Stage 1 training).

**2. No raw GNN embeddings leave `src/graph_autoencoder_v2.py`.**
Only `graph_anomaly_score`, `attr_recon_error`, and `struct_recon_error` are
valid exports. If downstream code requires embeddings, that is a design error.
Stop and flag it.

**3. Score direction in v2 is higher = more anomalous.**
`vae_anomaly_score` and `graph_anomaly_score` are both inverted relative to
v1's `vae_reconstruction_prob`. Any module that inverts this convention must
document the inversion explicitly at the point of inversion. Do not silently
flip the sign.

**4. `sanity` column is never used.**
Never as a feature. Never as a label. Never for evaluation. Drop it at load
time in every pipeline file. See `docs/AGENTS.md §2.7` for why.

**5. Self-training rounds are not automatic.**
Each round requires a Phase D PR-AUC check before its label set is used for
the next training cycle. Never write a loop that advances rounds without a
human check. The Round 0 classifier-agreement condition must be code-enforced
off, not just noted in a comment.

**6. No v1 model outputs in v2.**
`lgbm_risk_score` (v1), the v1 LightGBM checkpoint, and the v1 VAE checkpoint
do not exist in the v2 pipeline. Not as teachers, not as warm-starts, not as
round-0 stand-ins.

**7. Synthetic exposure set is programmatically constructed.**
Never use CTGAN, TVAE, GaussianCopula, or any tabular GAN to generate the
exposure set. See `docs/AGENTS.md §6.3` — composite degradation
is 24x or more on fraud behavioral signals.

**8. Never modify `docs/AGENTS.md` autonomously.**
Flag outdated content explicitly. Only the project lead updates this file.

---

## Known Structural Weaknesses (from MAR Layer 3)

Read `docs/MAR_v2.md` for full critique. Summary of the five load-bearing
failure conditions:

| Component | Core Assumption | What Breaks |
|---|---|---|
| DeepSVDD Graph AE | Normal data density is clean | If fraud dominates, hypersphere silently inflates to include fraud |
| EVT Scorer | Tail fits GPD smoothly | Discontinuous distributions cause threshold to explode or collapse |
| Self-Training Loop | EVT tail is true fraud | If tail is data-entry errors, classifier anchors on typos |
| Graph AE (isolated nodes) | Every node has ≥1 typed edge | Unique-mobile + unique-IP nodes get zero structural signal — silent |
| Stage 1 Synthetic Exposure | Archetypes represent real fraud geometry | Too-narrow archetypes bias Stage 2 toward obvious fraud only |

**What would break first in production:** the self-training label promotion.
A slight misalignment in EVT score distribution tails seeds the LightGBM with
false positives, triggering semantic drift in Round 1.

---

## Quantitative Claims Protocol

Before any number enters documentation, a summary, or these files:

1. **Raw stdout only.** Paste the literal unedited printed output. No number
   enters a doc without a traceable print line.
2. **Name the baseline explicitly.** "Before vs after" requires a specific
   prior run (timestamp or conversation line).
3. **Seed everything** before comparing runs — VAE, row sampling, train/test
   split, random module.
4. **Row-level counting only.** `rule_codes_fired.explode().value_counts()`
   counts code occurrences, not unique flagged rows. Use isolated masking.
5. **No same-turn resolution.** An open question does not get resolved in the
   same turn its supporting number was generated. Log as "proposed, pending."
6. **Conflicting numbers halt.** Surface the conflict. Do not reconcile by
   narrative. Re-derive from scratch.

---

## Module Ownership — One Module Per Response

| Module | File | Reads from | Writes to | Do not touch |
|---|---|---|---|---|
| Feature engineering | `src/tabular_feature_engine_v2.py` | `data/raw/data_for_ml_model.csv` | `data/processed/engineered_features_v2.csv`, `data/processed/v2_feature_schema.json` | any model training code |
| Graph construction | `src/graph_builder_v2.py` | `data/processed/engineered_features_v2.csv` | `data/processed/identity_graph.pt` | model training code |
| Synthetic exposure | `src/synthetic_exposure_builder_v2.py` | `data/processed/engineered_features_v2.csv` | `data/processed/synthetic_exposure_set.pt` | model training code |
| Tabular detector | `src/tabular_vae_v2.py` | `data/processed/*.csv`, `data/processed/*.pt` | `outputs/vae_v2_scores.csv`, `models/tabular_vae_v2.pth` | `identity_graph.pt` |
| Graph detector | `src/graph_autoencoder_v2.py` | `data/processed/identity_graph.pt`, `data/processed/synthetic_exposure_set.pt` | `outputs/graph_v2_scores.csv`, `models/graph_autoencoder_v2.pth` | tabular feature files |
| EVT thresholds | `src/evt_scorer.py` | `outputs/vae_v2_scores.csv`, `outputs/graph_v2_scores.csv` | `outputs/evt_thresholds_v2.json` | which features are scored |
| Self-training | `src/self_training_loop_v2.py` | score CSVs, `outputs/evt_thresholds_v2.json` | `outputs/pseudo_labels_v2.json` | model architectures |
| Fusion classifier | `src/fusion_classifier_v2.py` | all scalar scores, `outputs/pseudo_labels_v2.json` | `outputs/risk_scores_v2.csv` | raw GNN embeddings, rule signals |
| Explainability | `src/xai_layer_v2.py` | trained classifier, trained graph AE, error vectors | `outputs/explanation_cards_v2.json` | training code in any module |
| Evaluation | `src/evaluate_model_v2.py` | `data/processed/*.csv`, `models/*.pth` | Console stdout | training code |

**Working rule:** if your task requires reading or writing outside your
module's row, stop and confirm scope before writing code.

---

## v1 vs v2 — Do Not Mix These

| Concept | v1 | v2 |
|---|---|---|
| Anomaly score | `vae_reconstruction_prob` (higher = normal) | `vae_anomaly_score` (higher = anomalous) |
| Labels | `rule_violation_score > 0` | EVT tail + self-training pseudo-labels |
| Feature list file | `selected_features.json` | `data/processed/v2_feature_schema.json` |
| Risk score output | `risk_scores.csv` | `outputs/risk_scores_v2.csv` |
| Rule involvement | 99 NIC rules + 8 bridges | none |
| Architecture doc | `AGENTS.md` (v1 only) | `docs/AGENTS.md` |

If you see v1 column names in v2 code, that is a regression. Stop and flag it.

---

## Evaluation Standard

The primary gate is the **Phase D synthetic harness** (150 injected anomalies
per category, entirely unseen from the training synthetic set — different
random seeds). v2 must beat the v1 standalone-VAE PR-AUC floor on every
relational category before any production discussion:

| Category | Floor PR-AUC | Evaluated by |
|---|---|---|
| IP_CONCENTRATION | 0.0239 | Graph AE |
| MOTHER_NAME_COLLISION | 0.0258 | Graph AE |
| FEE_INFLATION | 0.0264 | Graph AE |
| AGE_VIOLATION | 0.0506 | Tabular VAE |
| INCOME_VIOLATION | 0.1162 | Tabular VAE |

Mandatory ablations before claiming any component helps:
- Stage 1 vs no Stage 1 (`λ(t) ≡ 0`): report both PR-AUC figures
- DeepSVDD vs DOMINANT-only: report both
- Round N vs round 0: track PR-AUC across all rounds, report all

---

## When to Stop and Ask

Stop and ask the project lead before proceeding if:

- A task requires modifying two modules simultaneously
- You are about to introduce a new dependency not in
  `docs/AGENTS.md §6.4`
- A quantitative result conflicts with a number stated earlier in the session
- The synthetic exposure set does not exist and you need to create it
- A self-training round is ready to advance (do not advance without
  confirmation)
- You are unsure whether a threshold is EVT-derived or domain-set
- Any file you need to read is not present in the working directory
- A proposed change would affect the Phase D evaluation harness definition

---

## Open Architecture Questions — Do Not Resolve Autonomously

These are unresolved by design. Surface the relevant one when working in the
related area; do not make a unilateral decision:

- λ(t) annealing schedule (linear / step / cosine) — currently linear in both
  models, no ablation evidence yet
- DeepSVDD centroid initialisation strategy — currently mean of normal
  embeddings at init time (pre-training)
- DOMINANT attribute vs structure reconstruction weight — currently equal
  (both contribute to the z-score sum)
- Synthetic exposure set size per archetype — currently 150, no sensitivity
  analysis done
- Isolated-node handling strategy — currently no mitigation; isolated nodes
  get zero structural signal
- VAE tabular underperformance on AGE/INCOME — whether to accept the tradeoff
  or introduce targeted Stage 1 curriculum adjustments
- Appeals framing for EVT-derived flags (policy decision, not code)
