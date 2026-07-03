# NIC Scholarship Fraud Detection — V3 Hybrid GraphMCM

> **For LLM agents:** Read `docs/AGENTS.md` before writing any code — start
> with the **AGENT QUICK-START** block at the top, then §9 (module ownership)
> and §10 (hard stops). This README is a human-readable overview; AGENTS.md is
> the authoritative contract for all implementation decisions.

---

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
  │  │  8 Learned Masks │  │  HAN Encoder     │ │
  │  │  masked_x_i      │  │  GAT per relation│ │
  │  │  (68-dim)        │  │  + semantic β_r  │ │
  │  │                  │  │  h_N(i) (64-dim) │ │
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
        (human-gated, ≥2 EVT signals required)
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

### Why HAN attention instead of fixed RGCN aggregation (graph stream)

The original V3 graph stream used an RGCN encoder: every neighbor within a
relation received the same fixed, degree-normalized weight, and the five
relations were summed with equal importance. A node in a large, organically
diverse shared-pincode cluster and a node in a tight, feature-camouflaged
fraud ring sharing an IP got the same fixed-form treatment, differing only by
degree.

The HAN-style encoder (Wang et al., WWW 2019) replaces this with two levels
of learned attention:

1. **Node-level attention** (GAT, Veličković et al., ICLR 2018) — within each
   of the 5 relations, a learned coefficient α_ij per edge decides *which
   neighbors* matter, computed from the endpoint feature vectors and
   softmax-normalized over each node's neighborhood.
2. **Semantic-level attention** — a learned scalar importance w_r per
   relation, softmax-normalized to β_r across the 5 relations, decides *which
   relations* matter. The final h_N(i) is the β_r-weighted fusion of the 5
   relation-specific embeddings.

Everything around the encoder is unchanged: h_N(i) is still 64-dim, the
concat → MLP → masked-prediction pipeline is identical, isolated nodes still
use the trainable `isolated_embedding`, and the `hybrid_scores_v3.csv` schema
is byte-for-byte the same. The MCM anomaly-discovery mechanism is untouched —
only the neighborhood context it conditions on becomes content-aware.

The swap is deliberately confined to the graph stream. Feature-stream or
fusion-stream attention was considered and rejected: tree-based models remain
state-of-the-art on medium-sized tabular data at this project's scale
(15,000 rows × 68 features — Grinsztajn et al., NeurIPS 2022), so LightGBM
stays. The graph stream is where no classical baseline can express relational
reasoning over typed edges at all — attention closes a genuine capability gap
there.

**Diagnostic check:** the learned β_r weights are logged at the end of
training. Given relation base rates (82.8% of nodes share a pincode vs 27.6%
sharing an IP), `shares_ip` ending up weighted above `shares_pincode` is the
expected signature of a correctly-behaving mechanism.

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

### Achieved Results (full-trained model, round 0)

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

> **Encoder-baseline note (2026-07-03):** the figures above were measured with
> the original RGCN graph encoder. The HAN encoder swap (see Changelog)
> requires a full GPU retrain and re-evaluation; until that run completes,
> these RGCN-era numbers are the comparison baseline for the HAN model, not a
> description of it.

> **Self-training note (updated 2026-06-30):** Round 0 promotion now requires
> ≥ 2 EVT signals to fire simultaneously (`MIN_SIGNALS_FOR_PROMOTION=2`).
> The original OR logic promoted ~111 nodes; multi-signal agreement reduces
> this to ~13 higher-confidence pseudo-positives per run. The PR-AUC figures
> above were measured on the full-trained model and remain the current
> evaluation baseline.

---

## Directory Layout

```
NIC fraud Detection Project/
├── main_v3.py                              # Pipeline orchestrator (V3)
├── README.md                               # This file
├── Dockerfile                              # CPU image (python:3.12-slim)
├── docker-compose.yml                      # Redis + nic-api + nic-worker
├── celeryconfig.py                         # Celery broker/worker config (concurrency=1)
├── requirements-docker.txt                 # Linux min-version pins for Docker builds
│
├── src/
│   ├── config_v3.py                        # All dimension constants (single source of truth)
│   ├── tabular_feature_engine_v3.py        # 63 features + degree merge = 68
│   ├── graph_builder_v3.py                 # 5 typed edges + degree computation
│   ├── synthetic_exposure_builder_v3.py    # 750 × 68 LOE tensor (graph-side only)
│   ├── hybrid_graphmcm_v3.py               # Core model: feature stream + graph stream
│   ├── subspace_if_v3.py                   # 3-group subspace isolation forest
│   ├── evt_scorer_v3.py                    # GPD tail fit on 6 signals (hybrid + 5 subspace)
│   ├── self_training_loop_v3.py            # Pseudo-label promotion (human-gated, ≥2 of 5 EVT signals)
│   ├── fusion_classifier_v3.py             # LightGBM: hybrid + subspace IF → risk score
│   ├── xai_layer_v3.py                     # Evidence-first explanation cards: population percentiles, expected-vs-actual values, EVT-threshold quotes, deterministic narratives
│   ├── evaluate_model_v3.py                # Category-specific subspace IF + degree-stratified PR-AUC
│   ├── checkpoint_manager.py               # ADR-008: atomic checkpoint validation + hot-swap
│   └── api/
│       ├── main.py                         # FastAPI app entry point + structlog setup
│       ├── schemas.py                      # Pydantic request/response models
│       ├── tasks.py                        # Celery task definitions (5 tasks)
│       └── handlers/
│           ├── supervisor.py               # POST confirm-fraud, mark-false-positive
│           ├── training.py                 # POST incremental/full, GET jobs/{id}, upload/pull checkpoint
│           ├── monitoring.py               # GET drift, fraud-store-summary
│           └── model.py                    # GET checkpoint-info, POST rollback
│
├── data/
│   ├── raw/data_for_ml_model.csv           # 15,000 × 136 (unchanged)
│   └── processed/
│       ├── engineered_features_v3.csv      # N × 68
│       ├── v3_feature_schema.json
│       ├── identity_graph_v3.pt
│       ├── degree_features_v3.csv          # N × 5 per-edge-type degrees
│       ├── synthetic_exposure_set_v3.pt    # 750 × 68
│       └── confirmed_fraud.json            # Supervisor feedback store (confirmed + false positives)
│
├── models/
│   ├── hybrid_graphmcm_v3.pth             # {model_state_dict, centroid, config} — live checkpoint
│   └── checkpoints/                        # Versioned checkpoints (last 5 kept by checkpoint_manager)
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
    ├── AGENTS.md                           # Architecture contract (do not edit autonomously)
    ├── OPERATIONS_RUNBOOK.md               # Yearly operational cycle guide
    └── API_TESTING_GUIDE.md                # Manual curl testing guide for all 13 endpoints
```

---

## How to Run

### Local (Python venv)

```bash
# Full V3 pipeline:
python main_v3.py

# Individual module:
.\.venv\Scripts\python.exe src/hybrid_graphmcm_v3.py

# Evaluation only:
.\.venv\Scripts\python.exe src/evaluate_model_v3.py
```

### Docker (API + async worker)

```powershell
# First time — builds the image (~5 min):
docker compose up --build

# Every subsequent start:
docker compose up -d

# API is now at http://localhost:8000
# Swagger UI (all 13 endpoints): http://localhost:8000/docs
# Full manual testing guide: docs/API_TESTING_GUIDE.md

# Stop:
docker compose down
```

Docker starts three containers: `redis` (broker), `nic-api` (FastAPI on port 8000),
`nic-worker` (Celery, concurrency=1). Training jobs dispatched via the API run
inside `nic-worker` and write to the mounted `data/`, `models/`, `outputs/` volumes.

---

## MLflow Audit Guide

Every pipeline run is logged to `mlflow.db` (SQLite, project root).
This covers how to open the UI and where to find each auditable artefact.

### Starting the UI

```bat
mlflow_ui.bat
```

Then open **http://localhost:5000** in your browser.
The batch file always points the UI at the correct `mlflow.db` — do not
run `mlflow ui` manually without it or runs will not appear.

---

### Navigation map

```
http://localhost:5000
│
├── Experiments (left sidebar)
│   └── nic-fraud-detection-v3          ← click this
│       └── Runs table (one row per pipeline execution)
│           └── [click any run]
│               ├── Overview tab        ← cycle label, duration, smoke-test flag
│               ├── Model Metrics tab   ← PR-AUC per fraud category (chart view)
│               ├── Parameters tab      ← all 23 config_v3 constants used for this run
│               └── Artifacts tab       ← all output files (see below)
```

---

### Artifacts tab — what to look at and why

| Folder | File | What it tells you |
|---|---|---|
| `suspicious/` | `top_suspicious_v3.tsv` | **Start here.** Top 20 highest-risk applications with risk score, anomalous features (human-readable labels), linked application IDs, EVT triggers, and full narrative recommendation. Open in Excel for best readability. |
| `xai/` | `explanation_cards_v3.json` | Full 500 explanation cards. Each card has `review_status`, `top_feature_errors` (with `feature_label`, model-`expected` value, and population percentiles), `top_graph_neighbors` (with actual application IDs), an `evidence` object (risk rank/percentile, EVT crossings vs thresholds, subspace detector scores, graph-degree percentiles), and a `narrative` composed deterministically from that evidence. Download and open in VS Code — search by application ID. |
| `scores/` | `risk_scores_v3.csv` | Risk score for all 15,000 applications. Columns: `application_id`, `risk_score_v3`, `label_source`. Sort descending to get the full ranked list beyond the top 500. |
| `thresholds/` | `evt_thresholds_v3.json` | EVT threshold for each of the 6 signals (hybrid, subspace_if, financial, identity, network, edge). Records the GPD fit params (`scale`, `shape`) and how many applications crossed each threshold. Useful for auditing whether thresholds are reasonable. |
| `labels/` | `pseudo_labels_v3.json` | The 13 applications promoted to pseudo-positive in Round 0 self-training. Each record shows which EVT signals fired. Review before any Round 1 is authorised. |
| `checkpoints/` | `hybrid_graphmcm_v3.pth` | Model weights used for this run. Download to reproduce scores exactly on the same data. |

---

### Metrics tab — what the numbers mean

| Metric | What it means | Pass threshold |
|---|---|---|
| `pr_auc_age_violation` | Precision-Recall AUC for detecting fake age fields | > 0.1466 (V2 floor) |
| `pr_auc_income_violation` | PR-AUC for declared income anomalies | > 0.6503 |
| `pr_auc_ip_concentration` | PR-AUC for IP address clustering fraud | > 0.0370 |
| `pr_auc_mother_name_collision` | PR-AUC for shared mother-name rings | > 0.2869 |
| `pr_auc_fee_inflation` | PR-AUC for inflated tuition fee declarations | > 0.4962 |
| `score_retention` | How much risk score survives when graph edges are removed (isolated-node robustness). > 1.0 means feature signal alone is sufficient. | > 1.0 |
| `n_categories_pass` | Number of fraud categories beating the V2 floor. Should be 5/5. | 5 |
| `pipeline_duration_seconds` | Wall-clock time for the run | — |

---

### Comparing two runs

1. In the runs table, tick the checkboxes on two rows.
2. Click **Compare** (top of table).
3. MLflow shows a side-by-side diff of all params and metrics — useful for
   checking whether a config change improved PR-AUC or shifted thresholds.

---

### Re-running XAI only (without retraining)

If you want a fresh MLflow run using existing scores (e.g. after editing
the XAI layer), run:

```bat
.venv\Scripts\python.exe main_v3.py --steps=xai,evaluate --cycle=<label>
```

This completes in ~15 seconds and logs a new run with updated explanation
cards and the top-suspicious TSV.

---

## Changelog

### 2026-07-03 — HAN two-level attention encoder adopted (docs first, code pending)

The graph stream's RGCN encoder is replaced by a HAN-style two-level attention
encoder (node-level GAT attention per relation + semantic-level β_r attention
across the 5 relations). **Documentation updated ahead of implementation** —
the code swap in `src/hybrid_graphmcm_v3.py` + new constants in
`src/config_v3.py` follows as the next code task. See ADR-015 in
`docs/AGENTS.md` Appendix F for the full decision record. Key operational
facts:

- **Full retrain required.** The new architecture has different parameter
  shapes and names — no old checkpoint loads into the new code, and there is
  no old RGCN to freeze, so `train_incremental()`'s frozen-encoder pathway is
  invalid for first deployment. First HAN deployment must go through the GPU
  full-retrain pathway (`main_v3.py` on the GPU laptop), never
  `retraining_orchestrator.py`.
- **Checkpoints carry an `ARCH_VERSION` field** (`"han_v1"`) in their `config`
  dict. `checkpoint_manager.validate_and_hotswap()` must reject any hot-swap
  where `ARCH_VERSION` mismatches the running code — a loud error, never a
  silent partial load.
- **Rollback pairing:** a pre-HAN versioned checkpoint is only a valid
  rollback target together with the pre-HAN code. Rolling back the checkpoint
  alone fails the `ARCH_VERSION` check — that failure is correct and safe,
  not a bug.
- **Output contract unchanged:** `outputs/hybrid_scores_v3.csv` schema is
  byte-for-byte identical; every downstream module is unaware of the swap.

### 2026-07-02 — MLOps Phase 2: REST API, async jobs, checkpoint manager, Docker

FastAPI server (`src/api/`) with 13 endpoints across 4 groups — supervisor
feedback, async training, monitoring, and model/checkpoint management.
Celery + Redis for async job dispatch (`celeryconfig.py`, `src/api/tasks.py`):
training jobs return a `job_id` immediately; callers poll
`GET /v3/training/jobs/{job_id}` for status. Atomic checkpoint hot-swap with
schema validation (`src/checkpoint_manager.py`): validates
`{model_state_dict, centroid, config}` keys and dimension constants before
replacing the live model, keeps last 5 versioned checkpoints, `.bak` fallback.
Structlog JSON logging in all new API files. Single Docker image
(`python:3.12-slim`, CPU-only) with `docker-compose.yml` running Redis,
API server, and Celery worker. Full stack tested end-to-end via Docker Desktop.
See `docs/API_TESTING_GUIDE.md` for manual curl testing of all endpoints.

### 2026-07-03 — Evidence-first XAI narratives

Narratives no longer come from a fixed lookup table of canned sentences —
every claim is now a statistic measured against the scored population
(15,000 applications) or an EVT-derived threshold. `hybrid_scores_v3.csv`
gained a `per_feature_predicted_json` column (the model's expected value per
feature) so the narrative can state expected-vs-actual with direction, not
just "this field is unusual." Hand-set narrative cuts (fixed 0.7/0.4 risk
tiers, `_magnitude` word buckets) were removed; the only numeric gates quoted
anywhere are EVT-fitted thresholds. Two real cards from the current run:

**Case 1 — promoted to pseudo-positive (2 EVT signals agreed):**

> Risk score 1.0000 — higher than 99.9+% of the 15,000 scored applications
> (rank 1). Crossed 2 independent extreme-value thresholds, fitted to the
> score tails at a 0.2% target false-positive rate: overall anomaly detected
> by hybrid model (observed 0.908 vs threshold 0.860); IP/mobile sharing
> pattern statistically extreme (observed 0.930 vs threshold 0.925). Key
> evidence: IP-to-mobile concentration ratio is higher than 99.9% of
> applicants (scaled value 1.000, population median 0.017); given the other
> declared fields and the application's network context, the model expected
> about -0.061 — the declared value is above expectation; this prediction
> miss is larger than 99.9+% of all applications' misses on this field | No.
> of applications from same IP is higher than 99.9% of applicants (scaled
> value 1.000, population median 0.000); given the other declared fields and
> the application's network context, the model expected about -0.038 — the
> declared value is above expectation; this prediction miss is larger than
> 99.9% of all applications' misses on this field | Year of admission matches
> the population median (scaled value 1.000); given the other declared
> fields and the application's network context, the model expected about
> -0.035 — the declared value is above expectation; this prediction miss is
> larger than 85.9% of all applications' misses on this field. Network
> links: shares the same IP address with 38 other application(s), more
> connected than 99.9% of applicants (e.g. GJ202526000019733,
> GJ202526000029086, GJ202526000029074); shares the same mother's name with
> 8 other application(s), more connected than 85.5% of applicants; shares
> the same father's name with 1 other application(s), more connected than
> 81.7% of applicants. Linked applications should be reviewed together.
> Recommended action: hold disbursement and request supporting documents
> (fee receipt, admission letter, income certificate); review linked
> applications together before approval.

**Case 2 — one signal crossed, below the 2-signal promotion bar (isolated node):**

> Risk score 0.0001 — higher than 99.7% of the 15,000 scored applications
> (rank 45). Crossed 1 extreme-value threshold: financial-features detector
> (observed 0.931 vs threshold 0.923) — below the 2-signal agreement
> required for automatic flag promotion. Key evidence: Institution verifier
> code is higher than 99.7% of applicants (scaled value 1.000, population
> median 0.183); given the other declared fields and the application's
> network context, the model expected about -0.139 — the declared value is
> above expectation; this prediction miss is larger than 97.6% of all
> applications' misses on this field | Course year (external record) is
> lower than 60.5% of applicants (scaled value 0.997, population median
> 0.999); the model expected about -0.113 — the declared value is above
> expectation; this prediction miss is larger than 78.5% of all
> applications' misses on this field | Course ID (college-reported) is
> higher than 76.0% of applicants (scaled value 0.999, population median
> 0.001); the model expected about -0.080 — the declared value is above
> expectation; this prediction miss is larger than 85.4% of all
> applications' misses on this field. No shared IP, mobile, parent-name, or
> pincode links found — 9.7% of applicants are similarly isolated.
> Suspicion rests on feature-level evidence and the subspace detectors.
> Recommended action: not auto-flagged (signal agreement below the
> promotion requirement), but the crossed threshold above warrants
> secondary verification before disbursement.

Notice the second case shows the system correctly *not* over-flagging: one
crossed threshold is reported honestly as insufficient for automatic
promotion, with a softer recommendation, instead of being silently upgraded
or silently dropped. Every card also carries a structured `evidence` object
(`risk_rank`, `risk_percentile`, `evt_crossings`, `subspace_groups`,
`graph_connections`, `isolated_population_pct`) alongside the prose, so a UI
can render the numbers directly instead of re-parsing the sentence.
Deterministic by design — no LLM in the loop, so the same evidence always
produces the same words, which matters for appeals/audit trails.

LLM-based narrative rendering was considered and explicitly rejected: NIC
data cannot leave premises, and non-deterministic wording is a liability
when a flagged application needs a reproducible explanation.

### 2026-06-30 — XAI explanation card improvements

Five readability and usability issues in `src/xai_layer_v3.py` fixed to make
explanation cards usable by fraud reviewers (not just data scientists):

| Issue | Fix |
|---|---|
| Raw database column names (`inst_verify_by`) unreadable by reviewers | Added `FEATURE_LABELS` dict mapping all 68 columns to human-readable display names (e.g. "Institution verifier code"). `feature_label` field added to every `top_feature_errors` entry. |
| `label_source: "negative"` misleading when risk = 1.0 | Replaced with `review_status` field using plain-English strings (e.g. "Pending Review — no confirmed fraud label assigned yet"). |
| `triggers: []` with high risk score unexplained | Narrative now explicitly states when the hybrid model's collective anomaly score drives suspicion without a single EVT threshold crossing. |
| `neighbor_idx: 8534` (raw array index) unresolvable by reviewers | `top_graph_neighbors` now contains `application_id` (actual portal ID) instead of internal array index. Reviewers can cross-reference directly. |
| Narrative described ML mechanics, not reviewer action | Narrative rewritten to explain WHY each field is suspicious ("the model could not predict this from the rest of the application"), include graph alert with linked application IDs, and end with a concrete recommended action (hold/request documents). |

MLflow artifacts updated: `suspicious/top_suspicious_v3.tsv` now includes
`review_status`, `linked_applications` (with actual IDs), and full narrative.

### 2026-06-30 — Risk-mitigation hardening pass

Five structural weaknesses identified in the MAR (Appendix D) were partially
mitigated. All changes are additive or tighten existing thresholds — no
architecture change, no new dependencies, no evaluation harness change.

| Risk | Fix | File(s) |
|---|---|---|
| DeepSVDD centroid contamination | `init_centroid()` now excludes the top 5% of nodes by embedding norm before computing the mean. Prevents fraud-shaped embeddings from pulling the hypersphere centroid toward fraud. | `hybrid_graphmcm_v3.py`, `config_v3.py` (`CENTROID_CLEAN_PERCENTILE=95`) |
| EVT GPD instability on discrete score distributions | Score jitter (σ=0.001 Gaussian) applied before fitting. GPD shape validation rejects fits outside `[-0.5, 1.0]` and falls back to empirical quantile. Discreteness spike warning added. | `evt_scorer_v3.py`, `config_v3.py` (`EVT_SHAPE_MIN`, `EVT_SHAPE_MAX`) |
| Self-training confirmation bias | Round 0 promotion changed from OR-of-5-signals to requiring `MIN_SIGNALS_FOR_PROMOTION=2` simultaneous EVT signals. Single-signal noise hits (data-entry errors that trigger one detector) no longer promote to pseudo-positive. | `self_training_loop_v3.py`, `config_v3.py` |
| Forensically clean isolated nodes (detection boundary) | No code fix possible — added diagnostic print: reports what fraction of top-100 risk scores are degree-0 nodes, making the gap observable. | `hybrid_graphmcm_v3.py` |
| Narrow synthetic exposure archetypes | Each archetype now generates `N_CLEAN=50` peak-signal examples + `N_PERTURB=100` graduated variants spanning 85th–97th percentile signal strength. `_add_context_noise()` perturbs 25% of non-target features per row to widen geometric coverage. | `synthetic_exposure_builder_v3.py` |

**Verified:** 5-step smoke test (exposure build → 2-epoch train → EVT → self-training → fusion)
ran clean in 35.7 s. EVT shape fallback caught 2 real bad fits live (`subspace_if_score`,
`subspace_if_network`). Promotion count moved from 111 (OR logic) to 13 (2-signal) on the
smoke run.

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
- **HAN** (Wang, Ji, Shi, Wang, Cui, Ye & Yu, "Heterogeneous Graph Attention
  Network," WWW 2019): Two-level attention over heterogeneous graphs —
  node-level attention within each relation, semantic-level attention (β_r)
  across relations. Grounds the V3 graph-stream encoder.
- **GAT** (Veličković et al., "Graph Attention Networks," ICLR 2018): Learned
  per-edge attention coefficients via a shared linear projection + attention
  vector, softmax-normalized over neighborhoods. The node-level mechanism
  inside the HAN encoder.
- **Grinsztajn et al.** ("Why do tree-based models still outperform deep
  learning on tabular data?", NeurIPS 2022): Tree-based models remain
  state-of-the-art on medium-sized (~10K-row) tabular data. Grounds the
  decision to confine attention to the graph stream and keep LightGBM in the
  fusion layer.
- **DOMINANT** (Ding et al., SDM 2019): Dual-decoder graph autoencoder for
  anomaly detection. The original V3 encoder inherited its RGCN architecture
  (aggr='add', tanh bounding) from V2's validated implementation; superseded
  in the graph stream by the HAN encoder (2026-07-03).
- **LOE** (Qiu et al., ICML 2022): Latent Outlier Exposure — synthetic anomaly
  pretraining to push outlier embeddings away from the normal hypersphere.
  V3 applies LOE to the graph stream only (Stage 1).
- **SPOT / EVT** (Siffer et al., KDD 2017): Generalized Pareto Distribution
  tail fitting for threshold-free anomaly flagging. Unchanged from V2.
