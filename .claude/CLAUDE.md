# CLAUDE.md — NIC Scholarship Fraud Detection

## What This Project Is

ML-based fraud detection for the NIC scholarship portal. The system produces
a per-application anomaly risk score (0–1) for 15,000 fresh applications.

Three architectural generations exist:
- **v1** — rule + bridge supervised (99 NIC rules + 8 bridges). Canonical run
  done; do not retrain without explicit instruction.
- **v2** — rule-free: Tabular VAE + Graph AE (DOMINANT + DeepSVDD) + Isolation
  Forest + MCM. **Superseded** — no `_v2.py` source remains in `src/`.
- **v3** — rule-free **Hybrid GraphMCM**: one two-stream detector (masked
  feature prediction + RGCN graph stream) + subspace Isolation Forest + EVT +
  human-gated self-training + LightGBM fusion + XAI. **This is the current
  system. You are almost always working on v3.**

An experimental **Phase 2 (V4 capability layer)** sits on top of v3 on branch
`v4-han-graphmcm`: an attention read-out (Tier 1) and a subgraph ring-classifier,
compared head-to-head against baseline. These are **opt-in, default OFF**. The
settled architecture is in `docs/IMPLEMENTATION.md`; all comparison results are in
`docs/AGENTS.md` Appendix H.

> **Naming:** "V4" is a capability label. The source files stay `_v3`. Do NOT
> rename `_v3`→`_v4` or `/v3`→`/v4`.

---

## Read These Files First — Every Session

1. **`docs/AGENTS.md`** — primary architecture reference for all v3 work.
   Read before writing any code. (Editable only under explicit project-lead
   direction — see hard stop #8.)
2. **`docs/IMPLEMENTATION.md`** — the settled V4 layered architecture and the
   remaining validation gate. Comparison results/tables are in `docs/AGENTS.md`
   Appendix H.
3. The MAR (Model & Architecture Review) critique is **folded inline** — see
   "Known Structural Weaknesses" below and `docs/AGENTS.md`. There is no
   separate `MAR_v2.md`/`MAR_v3.md` file.

If a referenced file is not in the working directory, stop and say so before
writing any code.

---

## Project Directory Layout

```
NIC fraud Detection Project/
├── main_v3.py                          # Pipeline orchestrator (entry point)
├── README.md
├── requirements.txt
│
├── src/                                # All Python source modules (_v3)
│   ├── tabular_feature_engine_v3.py    # Phase A: feature engineering
│   ├── graph_builder_v3.py             # Phase B: identity graph + degree features
│   ├── synthetic_exposure_builder_v3.py# Phase B: LOE exposure (tabular + topology)
│   ├── hybrid_graphmcm_v3.py           # Phase C: Hybrid GraphMCM detector (RGCN|HAN)
│   ├── subspace_if_v3.py               # Phase C: per-group Isolation Forest
│   ├── evt_scorer_v3.py                # Phase D: EVT thresholds
│   ├── self_training_loop_v3.py        # Phase D: pseudo-label promotion (human-gated)
│   ├── fusion_classifier_v3.py         # Phase E: LightGBM fusion
│   ├── xai_layer_v3.py                 # Phase E: evidence-first explanation cards (JSON)
│   ├── xai_card_html_v3.py             # Phase E: interactive reviewer cards (HTML) + lazy Plotly rings
│   ├── evaluate_model_v3.py            # Phase F: synthetic harness (isolated + connected)
│   │
│   ├── ring_candidate_v3.py            # V4 Phase 2 (experimental): ring candidate gen
│   ├── ring_fingerprint_v3.py          # V4 Phase 2: structural fingerprint
│   ├── ring_classifier_v3.py           # V4 Phase 2: ring classifier + open-set novelty
│   ├── compare_architectures_v3.py     # V4 Phase 2: baseline/tier1/ring/max_fusion harness
│   ├── graph_viz_v3.py                 # V4 Phase 2: interactive Plotly ring viz
│   └── confirmed_fraud_graph_store.py  # confirmed-fraud lifecycle store
│
├── data/
│   ├── raw/data_for_ml_model.csv       # 15,000 primary dataset
│   └── processed/
│       ├── engineered_features_v3.csv  # 15,000 × 68 numeric features (+ application_id)
│       ├── v3_feature_schema.json      # 68 feature names + exclusions
│       ├── degree_features_v3.csv
│       ├── identity_graph_v3.pt        # PyG HeteroData graph (5 edge types)
│       ├── synthetic_exposure_set_v3.pt# tabular LOE tensor
│       └── synthetic_exposure_graph_v3.pt # topology LOE (clusters + edges)
│
├── models/
│   └── hybrid_graphmcm_v3.pth          # detector state_dict + centroid (+ seed variants)
│
├── outputs/
│   ├── hybrid_scores_v3.csv            # detector scores + per-feature error/predicted JSON
│   ├── subspace_if_scores_v3.csv
│   ├── evt_thresholds_v3.json
│   ├── pseudo_labels_v3.json
│   ├── risk_scores_v3.csv              # final fused risk
│   ├── explanation_cards_v3.json
│   ├── ablation/                       # comparison + ablation JSON
│   └── viz/                            # interactive ring HTML
│
└── docs/
    ├── AGENTS.md                       # Architecture contract (v3) + Appendix H results
    ├── IMPLEMENTATION.md               # Settled V4 layered architecture
    ├── OPERATIONS_RUNBOOK.md
    └── API_TESTING_GUIDE.md
```

**Convention:** all paths are relative to the project root. Source in `src/`,
data in `data/`, checkpoints in `models/`, outputs in `outputs/`. Never write
outputs to the project root.

---

## Architecture in Brief

```
data/raw/data_for_ml_model.csv
        │
        ▼
src/tabular_feature_engine_v3.py ──► engineered_features_v3.csv, v3_feature_schema.json
        │                                         │
        ▼                                         ▼
src/graph_builder_v3.py ──► identity_graph_v3.pt  src/synthetic_exposure_builder_v3.py
        │                                         │   ──► synthetic_exposure_set_v3.pt
        │                                         │       synthetic_exposure_graph_v3.pt
        └──────────────┬──────────────────────────┘
                       ▼
     src/hybrid_graphmcm_v3.py  (feature stream: K=8 masks over 68-dim
                       │         graph stream: RGCN, 5 edge types → h_N(64)
                       │         concat → MLP → predicted x + edge probs)
                       │  ──► hybrid_scores_v3.csv, models/hybrid_graphmcm_v3.pth
                       │        hybrid_anomaly_score = feature_pred_error + 0.3·edge_pred_error
        ┌──────────────┤
        ▼              ▼
src/subspace_if_v3.py  src/evt_scorer_v3.py ──► evt_thresholds_v3.json
   │ subspace_if_scores_v3.csv          │
   └───────────────┬────────────────────┘
                   ▼
     src/self_training_loop_v3.py ──► pseudo_labels_v3.json  (human-gated)
                   ▼
     src/fusion_classifier_v3.py ──► risk_scores_v3.csv
                   ▼
     src/xai_layer_v3.py ──► explanation_cards_v3.json
                   ▼
     src/evaluate_model_v3.py ──► console PR-AUC / ablation JSON
```

Strict file-based contracts. Find your module in `docs/AGENTS.md` and stay
inside it. If a task spans two modules, stop and confirm scope before writing.

**Key v3 change vs v2:** the separate Tabular VAE and Graph AE (DOMINANT +
DeepSVDD) are replaced by a single **Hybrid GraphMCM** detector. Full-space IF
is replaced by a **subspace IF** (per feature group). Score direction is still
higher = more anomalous.

---

## Current Hyperparameters (Locked Unless Ablation Justifies Change)

Source of truth: `src/config_v3.py`.

| Parameter | Value |
|---|---|
| Input feature dimensions | 68 numeric columns (`N_FEATURES`) |
| Graph edge types | 5 (`shares_mobile`, `shares_ip`, `shares_father_name`, `shares_mother_name`, `shares_pincode`) |
| Feature-stream masks | 8 (`MASK_NUM`) |
| Graph hidden / embedding dim | 128 / 64 (`GRAPH_HIDDEN` / `GRAPH_EMB_DIM`) |
| MLP hidden / Z dim | 256 / 64 (`MLP_HIDDEN` / `Z_DIM`) |
| LOE margin | 2.0 (`LOE_MARGIN`) |
| Edge-loss weight λ_edge | 0.3 (`LAMBDA_EDGE`) |
| Exposure-loss weight λ_exposure | 1.0 (`LAMBDA_EXPOSURE`) |
| Stage 1 / Stage 2 epochs | 80 / 120 (`EPOCHS_STAGE1` / `EPOCHS_STAGE2`) |
| Learning rate / batch size | 1e-3 / 256 |
| Random seed | 42 (`RANDOM_SEED`, override via `V4_SEED`) |
| Encoder | `rgcn` default (`han` available; HAN drop-in regresses −0.091, 3-seed) |
| Incremental fine-tune | 10 epochs @ 1e-4, RGCN frozen |
| Confirmed-fraud LightGBM weight | 3.0 (`CONFIRMED_WEIGHT`) |
| Self-training min signals for promotion | 2 (`MIN_SIGNALS_FOR_PROMOTION`) |
| Centroid clean percentile | 95 (`CENTROID_CLEAN_PERCENTILE`) |
| EVT GPD shape valid range | [-0.5, 1.0] |
| Subspace IF groups | financial / identity / network |

---

## Current Evaluation State (source-traceable)

The **isolated-node harness** (`src/evaluate_model_v3.py::evaluate()`) uses the
V2 PR-AUC floors as the pass bar (in `evaluate_model_v3.py`):

```
AGE_VIOLATION 0.1466 | INCOME_VIOLATION 0.6503 | IP_CONCENTRATION 0.0370
MOTHER_NAME_COLLISION 0.2869 | FEE_INFLATION 0.4962
```

The **connected-cluster harness** (`evaluate_connected()`) is the relational
test. The six-mode head-to-head numbers (baseline / tier1 / ring / max_fusion /
dense_block_fusion / dense_block_only) live in `docs/AGENTS.md` Appendix H and
`outputs/ablation/tier_comparison.json` — **the single source of
truth for V4 results.** Do not restate PR-AUC numbers from memory; cite the JSON.

> Reproducibility caveat: `RGCNConv(aggr="add")` uses CUDA scatter-add atomics
> that are not seed-controlled, giving a ±0.03–0.04 run-to-run noise floor on
> detector-derived scores. Use `torch.use_deterministic_algorithms(True)` or CPU
> scoring before comparing effects smaller than that.

---

## How to Run

```bash
# Full pipeline (from project root):
python main_v3.py

# Individual module (from project root):
.\.venv\Scripts\python.exe -m src.hybrid_graphmcm_v3

# Head-to-head comparison (V4 Phase 2):
.\.venv\Scripts\python.exe -m src.compare_architectures_v3
```

Modules assume the working directory is the project root, not `src/`, and are
run as modules (`-m src.<name>`) so package imports resolve.

---

## Hard Stops — Never Proceed Past These

**1. No rules. No exceptions.**
If you find yourself writing any of the following, stop immediately:
- A numeric threshold against a domain concept (`ip_count >= 15`, `age > 35`)
- A named rule code (`X1`, `YF`, `IP_CONC_ENG`, etc.)
- A call to `apply_rules()` or any equivalent
- A feature whose definition encodes a policy boundary

The only numeric thresholds allowed are EVT-derived (`src/evt_scorer_v3.py`) or
learned from synthetic exposure (Stage 1 training).

**2. No raw GNN embeddings leave `src/hybrid_graphmcm_v3.py`.**
Only scalar scores (`hybrid_anomaly_score`, `feature_pred_error`,
`edge_pred_error`) and attention *weights* (per-relation β_r, α entropy/top-1)
are valid exports. The 64-dim `h_N` never leaves the module. If downstream code
requires embeddings, that is a design error. Stop and flag it.

**3. Score direction in v3 is higher = more anomalous.**
`hybrid_anomaly_score` and `subspace_if_score` are both higher = more anomalous.
Any module that inverts this convention must document the inversion explicitly at
the point of inversion. Do not silently flip the sign.

**4. `sanity` column is never used.**
Never as a feature. Never as a label. Never for evaluation. It is in
`EXCLUDED_FROM_FEATURES` (`config_v3.py`). See `docs/AGENTS.md` for why.

**5. Self-training rounds are not automatic.**
Each round requires a Phase D PR-AUC check before its label set is used for the
next training cycle. Never write a loop that advances rounds without a human
check. The Round 0 classifier-agreement condition must be code-enforced off, not
just noted in a comment.

**6. No v1 or v2 model outputs in v3.**
`lgbm_risk_score` (v1), `vae_anomaly_score` / `graph_anomaly_score` (v2), and any
v1/v2 checkpoint do not exist in the v3 pipeline. Not as teachers, not as
warm-starts, not as round-0 stand-ins.

**7. Synthetic exposure set is programmatically constructed.**
Never use CTGAN, TVAE, GaussianCopula, or any tabular GAN to generate the
exposure set. See `docs/AGENTS.md` — composite degradation is 24x or more on
fraud behavioral signals.

**8. `docs/AGENTS.md` is project-lead-owned.**
Do not modify it autonomously. Flag outdated content explicitly and propose a
redline; apply only under explicit project-lead direction for the current branch.

---

## Known Structural Weaknesses (MAR critique, folded inline)

Summary of the load-bearing failure conditions:

| Component | Core Assumption | What Breaks |
|---|---|---|
| DeepSVDD centroid (Hybrid GraphMCM) | Normal data density is clean | If fraud dominates, hypersphere silently inflates to include fraud |
| EVT Scorer | Tail fits GPD smoothly | Discontinuous distributions cause threshold to explode or collapse |
| Self-Training Loop | EVT tail is true fraud | If tail is data-entry errors, classifier anchors on typos |
| Isolated nodes | Every node has ≥1 typed edge | Unique-mobile + unique-IP nodes get zero structural signal — rely on subspace IF |
| Stage 1 Synthetic Exposure | Archetypes represent real fraud geometry | Too-narrow archetypes bias Stage 2 toward obvious fraud only |
| Reconstruction on dense rings | Anomalies are hard to reconstruct | Dense fraud cliques reconstruct *easily* (smoothing) → weak relational signal; see V4 comparison |

**What would break first in production:** self-training label promotion. A
slight misalignment in EVT score-distribution tails seeds the LightGBM with false
positives, triggering semantic drift in Round 1.

---

## Quantitative Claims Protocol

Before any number enters documentation, a summary, or these files:

1. **Raw stdout only.** Paste the literal unedited printed output. No number
   enters a doc without a traceable print line.
2. **Name the baseline explicitly.** "Before vs after" requires a specific
   prior run (timestamp or conversation line).
3. **Seed everything** before comparing runs — model init, row sampling,
   train/test split, random module. (Note the GPU scatter-add caveat above.)
4. **Row-level counting only.** Count unique flagged rows, not code occurrences.
   Use isolated masking.
5. **No same-turn resolution.** An open question does not get resolved in the
   same turn its supporting number was generated. Log as "proposed, pending."
6. **Conflicting numbers halt.** Surface the conflict. Do not reconcile by
   narrative. Re-derive from scratch.

---

## Module Ownership — One Module Per Response

| Module | File | Reads from | Writes to |
|---|---|---|---|
| Feature engineering | `tabular_feature_engine_v3.py` | `data/raw/data_for_ml_model.csv` | `engineered_features_v3.csv`, `v3_feature_schema.json` |
| Graph construction | `graph_builder_v3.py` | `engineered_features_v3.csv` | `identity_graph_v3.pt`, `degree_features_v3.csv` |
| Synthetic exposure | `synthetic_exposure_builder_v3.py` | `engineered_features_v3.csv` | `synthetic_exposure_set_v3.pt`, `synthetic_exposure_graph_v3.pt` |
| Hybrid detector | `hybrid_graphmcm_v3.py` | `engineered_features_v3.csv`, `identity_graph_v3.pt`, exposure `.pt` | `hybrid_scores_v3.csv`, `models/hybrid_graphmcm_v3.pth` |
| Subspace IF | `subspace_if_v3.py` | `engineered_features_v3.csv` | `subspace_if_scores_v3.csv` |
| EVT thresholds | `evt_scorer_v3.py` | score CSVs | `evt_thresholds_v3.json` |
| Self-training | `self_training_loop_v3.py` | score CSVs, `evt_thresholds_v3.json` | `pseudo_labels_v3.json` |
| Fusion classifier | `fusion_classifier_v3.py` | scalar scores, `pseudo_labels_v3.json` | `risk_scores_v3.csv` |
| Explainability | `xai_layer_v3.py` | trained detector, error/predicted vectors, score CSVs | `explanation_cards_v3.json` (+ closed-form fusion split) |
| Reviewer cards (HTML) | `xai_card_html_v3.py` | `explanation_cards_v3.json`, `risk_scores_v3.csv`, graph (lazy ring) | `outputs/cards/*.html`; served via API `/card` + `/ring` |
| Evaluation | `evaluate_model_v3.py` | `data/processed/*`, `models/*.pth` | console / `outputs/ablation/*.json` |
| **(V4)** ring pipeline | `ring_candidate_v3.py`, `ring_fingerprint_v3.py`, `ring_classifier_v3.py` | public structure + scores only (never `h_N`) | `models/ring_classifier_v3.pkl` |
| **(V4)** comparison | `compare_architectures_v3.py` | cached detectors + scores | `outputs/ablation/tier_comparison.json` |
| **(V4)** ring viz | `graph_viz_v3.py` | structure + `hybrid_scores_v3.csv` | `outputs/viz/*.html` |

**Working rule:** if your task requires reading or writing outside your module's
row, stop and confirm scope before writing code.

---

## v3 Conventions — Do Not Regress to v1/v2

| Concept | v1 | v2 (superseded) | v3 (current) |
|---|---|---|---|
| Anomaly score | `vae_reconstruction_prob` (higher=normal) | `vae_anomaly_score`, `graph_anomaly_score` | `hybrid_anomaly_score` (higher=anomalous) |
| Labels | `rule_violation_score > 0` | EVT tail + self-training | EVT tail + human-gated self-training |
| Feature schema | `selected_features.json` | `v2_feature_schema.json` | `data/processed/v3_feature_schema.json` |
| Risk output | `risk_scores.csv` | `risk_scores_v2.csv` | `outputs/risk_scores_v3.csv` |
| Rules | 99 rules + 8 bridges | none | none |

If you see v1/v2 column names or `_v2.py` imports in v3 code, that is a
regression. Stop and flag it.

---

## When to Stop and Ask

Stop and ask the project lead before proceeding if:

- A task requires modifying two modules simultaneously
- You are about to introduce a new dependency not already in `requirements.txt`
- A quantitative result conflicts with a number stated earlier in the session
- A synthetic exposure set does not exist and you need to create it
- A self-training round is ready to advance (do not advance without confirmation)
- You are unsure whether a threshold is EVT-derived or domain-set
- Any file you need to read is not present in the working directory
- A proposed change would affect the Phase D / connected-cluster evaluation harness

---

## Open Architecture Questions — Do Not Resolve Autonomously

- λ(t) annealing schedule (linear / step / cosine) — currently linear
- DeepSVDD centroid initialisation strategy — currently mean of clean-percentile
  embeddings at init
- Feature vs edge reconstruction weight — `LAMBDA_EDGE=0.3`, no ablation yet
- Synthetic exposure set size per archetype — no sensitivity analysis
- Isolated-node handling — currently rely on subspace IF; no structural signal
- Reconstruction weakness on dense rings — **settled:** adopt dense-block gated to
  `shares_ip` only (`docs/IMPLEMENTATION.md`), pending the IP-gated validation run.
  Baseline is the backbone; RGCN retirement disproven (AGENTS.md Appendix H).
- Encoder choice (RGCN vs HAN) — RGCN default; HAN drop-in regresses
- Appeals framing for EVT-derived flags (policy decision, not code)
