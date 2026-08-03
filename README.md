# NIC Scholarship Fraud Detection — V3 Hybrid GraphMCM

> **For LLM agents:** Read `docs/AGENTS.md` before writing any code — start
> with the **AGENT QUICK-START** block at the top, then §9 (module ownership)
> and §10 (hard stops). This README is a human-readable overview; AGENTS.md is
> the authoritative contract for all implementation decisions.

> **New to this codebase?** Start with `docs/PROJECT_OVERVIEW.md` for a
> conceptual, no-jargon walkthrough of what the system does and why it's
> built this way. For "how do I add a feature / detector / schema change"
> task recipes, see `docs/MAINTAINER_PLAYBOOK.md`.

---

## V4 — Final Detection Architecture (this branch, LOCKED)

V4 keeps the V3 detectors but **changes how they are combined**. The detectors
were never the problem — the old LightGBM fusion, trained on only ~14
auto-generated pseudo-labels, was *burying* their signal. V4 replaces it with a
**label-free weighted score-level fusion** of three specialised raw detectors.
It is a *capability* label, not a rename — all `_v3` file names, `/v3/...` API
routes, MLOps tooling, and curl commands are unchanged. Full record:
`docs/IMPLEMENTATION.md` and `docs/AGENTS.md` Appendix H.

**The three detectors (each strong where the others are blind):**

1. **Subspace Isolation Forest** — the **tabular backbone**, and the single
   strongest component (raw mean PR-AUC **0.727**; INCOME 0.966, FEE 0.916). Wins
   4 of 5 categories on its own. No training.
2. **Dense-block detector (IP-gated)** — the **IP-concentration specialist**
   (FRAUDAR-style camouflage-resistant greedy peeling on `shares_ip`; raw IP
   **0.713**). Fills subspace's one blind spot. Deterministic, no training.
3. **RGCN + topology exposure** (Hybrid GraphMCM) — the **relational signal**
   (raw IP **0.51**, MOTHER **0.45**), with the best generalisation to *unseen*
   fraud shapes. Its outlier-exposure layer is validated to learn new topologies
   (before/after test: +0.148 on a never-seen star/bipartite ring).

**The fusion (locked):**

```
risk = minmax( 1.0·subspace_if_score + 0.5·dense_block_ip + 0.3·hybrid_anomaly_score )
       (each component min-max normalised first)
```

Label-independent — there is no learned gate that can suppress a strong signal.
Subspace dominates; dense-block-IP boosts the IP blind spot; RGCN adds the
relational/topology signal.

**Measured (frozen detector, one run):** connected mean **0.639** (IP **0.538**),
held-out **0.640** — versus the old LightGBM fusion's **~0.22**.

**Why the LightGBM fusion was dropped:** with only ~14 EVT pseudo-labels the tree
learned "fraud = these 14 particular points" and discounted the very detectors
that work — it dragged subspace INCOME **0.966 → 0.315** and RGCN IP **0.51 →
0.169**. It is parked until confirmed labels accumulate (then revisited with
monotonic constraints).

**Dropped (measured out):** HAN encoder (−0.091 drop-in — attention over-smooths
dense fraud cliques), Tier-1 attention read-out, equal-weight/max fusion, and
"retire the RGCN" (disproven). **Standing by:** the deviation layer is wired but
dormant (cold-start, activates as confirmations grow), and the ring classifier is
kept as an independent audit signal, not a fusion input.

> **Honest caveat:** the numbers above are on a single frozen detector; a
> multi-seed confirmation on a representative detector is the remaining validation
> before production sign-off. See `docs/AGENTS.md` Appendix H for the full
> component-by-component evidence.

---

## V4-Scale — PostgreSQL + Kubernetes Remodel (`V4-Scale` branch, implemented)

The detection architecture above is **fixed** — V4-Scale doesn't touch model
math, it rebuilds the I/O around it so the system can process **30–40 lakh
(3–4 million) applications** on the production Kubernetes server (16 vCPU,
64 GB RAM, no GPU) instead of the 15k it was built against. Full detail,
including the real Postgres schema, the external-GPU-checkpoint mechanism,
and measured scale numbers: **`docs/TECHNICAL_REFERENCE_AND_SCALING.md`**.

**What changed:**
- **PostgreSQL is the system of record** — every application, feature
  vector, score, label, LOE pattern, and training run lives in Postgres
  (10 tables, `deploy/postgres/schema.sql`), not flat files. A `db-init`
  container auto-applies the schema and ingests/replays data on every
  `docker compose up`, so Postgres is populated the moment the API starts.
- **SQL-pushdown feature engineering** replaces whole-frame pandas — proven
  **bit-exact** against the old pipeline on all 44 model features.
- **Hub-capped identity graph** replaces all-pairs edge construction (a
  shared value with thousands of members would otherwise produce millions of
  edges) — capped to a clique below a threshold, a star above it.
- **Exact-neighborhood mini-batch training** replaces full-graph RGCN
  passes — proven **bit-exact** against full-graph scoring on 15k, and
  measured at **3.53 GB peak memory on a synthetic 1M-application test**
  (memory is not the constraint at 3.5M; training *time*, projected at
  ~101 h for a full retrain, is — hence the next point).
- **External GPU-checkpoint ingestion** — a model trained entirely outside
  the cluster (e.g. a GPU laptop, same `config_v3.py`) can be uploaded
  through the admin console, schema-validated, and atomically hot-swapped
  in, with zero in-cluster training time.
- **Console CSV intake is unchanged** — upload/Evaluate/Decide behave
  identically; underneath, rows now land in a Postgres staging batch instead
  of a file.

All five migration steps are implemented and gate-tested (bit-for-bit parity
checks, live-stack verification via Playwright) — see `docs/IMPLEMENTATION.md`
for the step-by-step evidence and `docs/HISTORY.md` for how the detection
architecture above was locked in the first place.

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
  │   = 44 total               │
  └────────────┬───────────────┘
               │ engineered_features_v3.csv (N × 44)
               │
       ┌───────┴───────┐
       ▼               ▼
  Graph Builder    Synthetic Exposure
  5 typed edges    750 × 44 anomaly set
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
  │  │  (44-dim)        │  │  h_N(i) (64-dim) │ │
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
  │   per-feature (44-d)     per-edge-type (5-d) │
  │           │                     │            │
  │           └──────────┬──────────┘            │
  │                      ▼                       │
  │           hybrid_anomaly_score               │
  │           = feature_err + 0.3 × edge_err     │
  └──────────────────────┬─────────────────────-─┘
               │
               ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  THREE RAW DETECTORS (each strong where the others are blind)   │
  │                                                                 │
  │  Subspace IF          Dense-Block (IP)      RGCN + topology     │
  │  tabular backbone     FRAUDAR peeling on    (from Hybrid above) │
  │  financial/identity/  shares_ip only        relational signal   │
  │  network (no train)   raw IP 0.713          raw IP 0.51         │
  │  raw mean 0.727       (no train)            best generalisation │
  └───────┬───────────────────┬─────────────────────┬──────────────┘
          │ subspace_if_score  │ dense_block_ip      │ hybrid_anomaly_score
          └───────────┬────────┴──────────┬──────────┘
                      ▼                    │
              EVT Scorer (GPD tail)        │   (EVT + self-training run in
                      ▼                    │    parallel; feed thresholds +
              Self-Training Loop           │    label_source metadata)
              (human-gated, ≥2 signals)    │
                      └─────────┬──────────┘
                                ▼
              WEIGHTED SCORE-LEVEL FUSION  (LOCKED — replaces LightGBM)
              risk = minmax(1.0·subspace + 0.5·dense_ip + 0.3·hybrid)
                                │  each component min-max normalised
                                ▼
              EVT/SPOT threshold on the fused score
                                ▼
                          XAI Layer
              per-feature conditional errors + graph neighbor explanation
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

The hybrid's feature prediction averages error across 8 masks and 44 features.
A near-zero income (1 feature out of 44) might not dominate that average even
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
├── docker-compose.yml                      # postgres + db-init + redis + nic-api + nic-worker + nginx
├── docker-compose.override.yml             # Dev-only frontend hot-mount (auto-merged)
├── celeryconfig.py                         # Celery broker/worker config (concurrency=1)
├── requirements-docker.txt                 # Linux min-version pins for Docker builds
│
├── src/
│   ├── config_v3.py                        # All dimension constants (single source of truth)
│   ├── model_registry.py                   # Run/metric/checkpoint history (replaces MLflow)
│   ├── tabular_feature_engine_v3.py        # engineer, then drop 24 identifiers → 44 (+ SQL-pushdown path)
│   ├── graph_builder_v3.py                 # 5 typed edges + degree computation (+ hub-capped PG path)
│   ├── synthetic_exposure_builder_v3.py    # 750 × 44 LOE tensor (graph-side only)
│   ├── hybrid_graphmcm_v3.py               # Core model: feature stream + graph stream (+ NeighborLoader path)
│   ├── subspace_if_v3.py                   # 3-group subspace isolation forest (tabular backbone)
│   ├── dense_block_detector_v3.py          # FRAUDAR greedy peeling on shares_ip (IP specialist)
│   ├── evt_scorer_v3.py                    # GPD tail fit on 6 signals (hybrid + 5 subspace)
│   ├── self_training_loop_v3.py            # Pseudo-label promotion (human-gated, ≥2 of 5 EVT signals)
│   ├── fusion_classifier_v3.py             # Weighted SCORE-LEVEL fusion: subspace + dense-block-IP + hybrid → risk
│   ├── xai_layer_v3.py                     # Evidence-first explanation cards: population percentiles, expected-vs-actual values, EVT-threshold quotes, deterministic narratives
│   ├── xai_card_html_v3.py                 # Interactive reviewer cards (base + cohort-preview, PG-backed rings)
│   ├── topology_view.py                    # Ego-graph extraction (PG-backed + staged-cohort fallback)
│   ├── retraining_orchestrator.py          # Drift check + retrain dispatch (PG-backed baselines)
│   ├── evaluate_model_v3.py                # Category-specific subspace IF + degree-stratified PR-AUC
│   ├── pattern_ingest_v3.py                # Supervisor CSV fraud-ring intake: test (read-only) + ingest as topology-exposure pattern
│   ├── checkpoint_manager.py               # ADR-008: atomic checkpoint validation + hot-swap (external GPU checkpoints too)
│   ├── confirmed_fraud_store.py / confirmed_fraud_graph_store.py  # JSON stores, dual-written to Postgres
│   ├── db/                                 # ALL SQL lives here (hard stop 14) — see docs/TECHNICAL_REFERENCE_AND_SCALING.md §11
│   │   ├── connection.py · migrate.py · bootstrap.py   # pool/config, schema apply, one-shot startup init
│   │   ├── ingest.py                       # primary-batch ingest + staged-batch lifecycle (stage/evaluate/merge/delete)
│   │   ├── reads.py                        # payload-exact PG read mirrors (queue, ego-graph, fraud summary)
│   │   ├── features.py                     # SQL-pushdown aggregates, persisted-scaler save/load, hub-capped edges
│   │   ├── stores.py / drift.py            # dual-write mirrors for the JSON stores + drift baselines
│   └── api/
│       ├── main.py                         # FastAPI app entry point + structlog setup
│       ├── schemas.py                      # Pydantic request/response models
│       ├── tasks.py                        # Celery task definitions
│       └── handlers/
│           ├── supervisor.py               # POST confirm-fraud, mark-false-positive, confirm-batch, pattern/test, pattern/ingest, patterns/*
│           ├── training.py                 # POST incremental/full/decision, GET jobs/{id}, upload/pull checkpoint
│           ├── monitoring.py               # GET drift, drift-explain, fraud-store-summary, top-suspicious, {id}/card|ring|topology|export, export/bulk, export/selected; POST evaluate-dataset, upload-dataset
│           └── model.py                    # GET checkpoint-info, stats, registry; POST rollback
│
├── scripts/
│   └── profile_group_sizes.py              # K_CAP profiling query (rerun against real-scale data)
│
├── data/
│   ├── raw/data_for_ml_model.csv           # 15,000 × 136 (unchanged)
│   └── processed/
│       ├── engineered_features_v3.csv      # N × 44
│       ├── v3_feature_schema.json
│       ├── identity_graph_v3.pt
│       ├── degree_features_v3.csv          # N × 5 per-edge-type degrees
│       ├── synthetic_exposure_set_v3.pt    # 750 × 44
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
│   ├── explanation_cards_v3.json
│   └── model_registry.json                 # Run/metric/checkpoint history (replaces MLflow)
│
├── frontend/                               # Review console (vanilla HTML/CSS/JS, nginx-served)
│   ├── index.html · app.js · style.css · config.js
│
├── deploy/
│   ├── README.md                           # Local Docker + Kubernetes deploy guide
│   ├── postgres/schema.sql                 # The system-of-record schema (10 tables)
│   ├── nginx/                              # front-door image (serves frontend + proxies API)
│   └── k8s/nic-fraud.yaml                  # Kubernetes manifests (incl. postgres pod)
│
└── docs/
    ├── AGENTS.md                           # Architecture + scale contract (do not edit autonomously)
    ├── TECHNICAL_REFERENCE_AND_SCALING.md  # Full model + Postgres + scaling reference (start here for depth)
    ├── IMPLEMENTATION.md                   # 5-step Postgres migration plan + gate evidence
    ├── HISTORY.md                          # How the detection architecture was locked, with metrics
    ├── OPERATIONS_RUNBOOK.md               # Start Docker → open the console → what each screen does
    └── API_TESTING_GUIDE.md                # Manual curl testing guide for the API
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

### Docker (console + API + async worker)

```powershell
# First time — builds the images (~5 min):
docker compose up --build

# Every subsequent start:
docker compose up -d

# Review console (the UI):        http://localhost:8080/
# Swagger UI (proxied):           http://localhost:8080/docs
# API direct (debugging):         http://localhost:8000/health
# Stop:
docker compose down
```

Docker starts six containers: `postgres` (system of record, host port **5433**
— not 5432, to avoid colliding with a locally-installed PostgreSQL), `db-init`
(one-shot — applies the schema and ingests/replays data into Postgres, then
exits; `nic-api`/`nic-worker` wait for it to finish before starting), `redis`
(broker), `nic-api` (FastAPI), `nic-worker` (Celery, concurrency=1), and
`nginx` (the **front door** — serves the `frontend/` console and
reverse-proxies the API, so the browser only needs `:8080`). Training jobs
dispatched from the console run inside `nic-worker` and write to the mounted
`data/`, `models/`, `outputs/` volumes and to Postgres.

**Gotcha:** if you rebuild only `nic-api`/`nic-worker` (not the whole stack),
`nginx` keeps the old container's cached IP and every request 502s — run
`docker compose restart nginx` after a partial rebuild.

For operating the console screen-by-screen see `docs/OPERATIONS_RUNBOOK.md`;
for the full model architecture, the Postgres schema, and the scaling story
see `docs/TECHNICAL_REFERENCE_AND_SCALING.md`; for the full local +
Kubernetes deployment (storage, worker-singleton, no-GPU notes) see
`deploy/README.md`; for raw endpoint testing see `docs/API_TESTING_GUIDE.md`.

---

## Model Audit (in the console — MLflow retired)

MLflow has been removed. Model state, run history, and the deployment loop now
live in the **console's "Model audit & deploy" tab** (`http://localhost:8080/`),
backed by a local JSON registry (`outputs/model_registry.json`) that the
pipeline writes on every training run and checkpoint swap. No MLflow server, no
`mlflow.db`, no `mlruns/`.

What the tab surfaces:

- **Running-model status strip** — live checkpoint (size, feature/edge counts),
  scored-population size, confirmed/false-positive counts, drift recommendation,
  last-eval PR-AUC, and the last run. `GET /v3/model/stats`.
- **Deployment loop** — upload a cohort CSV (schema-validated) → evaluate
  read-only (drift + preview) → human-gated decide (incremental / full /
  no-action, or retrain on current data) → watch the job.
- **Run history** — every training run and checkpoint swap, newest first, with
  metrics and checkpoint size. `GET /v3/model/registry`.
- **Rollback** — restore any versioned checkpoint.

The PR-AUC pass floors still apply (age > 0.1466, income > 0.6503, ip > 0.0370,
mother-name > 0.2869, fee > 0.4962; 5/5 categories) — they're computed by
`evaluate_model_v3.py` and recorded as run metrics in the registry. For the
screen-by-screen operator walkthrough see `docs/OPERATIONS_RUNBOOK.md`.

---

## Changelog

### 2026-08-03 — NeighborLoader sampled path closes the ENCODER_RELATION_IDS gap; stale fanout default fixed

Follow-up to the entry below: the mini-batch/NeighborLoader path
(`train_sampled`, `score_sampled`, `_make_neighbor_loader`, `_batch_inputs`
in `src/hybrid_graphmcm_v3.py` — step-5 scale-migration code, **not yet
called anywhere in production**, confirmed by repo-wide grep) was the one
place the father_name/mother_name exclusion below hadn't reached, since it
samples the graph independently of the full-batch path's edge builder.

- **Why this matters even though nothing calls it today**: this is the code
  path that *will* train the model once the system moves to 30–40 lakh
  (3–4M) applications (`docs/IMPLEMENTATION.md` step 5) — a full-batch
  forward/backward pass over millions of nodes doesn't fit in memory or a
  reasonable step time, so training switches to sampling a bounded
  neighborhood per node via PyG's `NeighborLoader` instead of loading the
  whole graph every step. Left unfixed, the scale-up retrain would have
  silently reintroduced `shares_father_name`/`shares_mother_name` into the
  encoder — undoing today's finding at the exact point (larger population,
  more shared surnames) where those relations would be *most* diffuse and
  least informative.
- `_make_neighbor_loader()` and `_batch_inputs()` both take a new
  `relation_ids` param (default `ENCODER_RELATION_IDS`); the loader sets
  fan-out to `[0, 0]` for excluded relations so `NeighborLoader` never
  samples/materializes their neighbors, and `_batch_inputs` skips them again
  as a second check. `train_sampled()`/`score_sampled()` pass
  `relation_ids=ENCODER_RELATION_IDS` at all 3 call-site pairs.
  `_global_edge_target()` is unchanged by design — it builds the
  edge-*prediction* target (all 5 relations, matching the unchanged
  5-relation edge-predictor head), not the encoder's input.
- Verified end-to-end on the real 15k graph (smoke run, both fanout
  settings): sampled-batch `edge_type` values only ever take `{0, 1, 4}`
  across a full pass — `2`/`3` (father/mother-name) never leak through.
- **Separate, pre-existing bug found and fixed in the same pass**: both
  functions' `fanout` parameter defaulted to `(25, 10)` (truncated,
  per-hop-capped sampling), but `docs/AGENTS.md` §7 records this as a
  **CLOSED lead decision since 2026-07-21** — `docs/IMPLEMENTATION.md`
  step 5's 1M-scale ablation found truncated fan-outs deviate from the
  full-graph score by up to 0.44, while exact-neighborhood `(-1, -1)`
  (sample every neighbor, no cap; memory-bounded by the hub cap instead) is
  bit-equal to the full graph for this 2-layer RGCN. Defaults corrected to
  `(-1, -1)`. `.claude/CLAUDE.md`'s Open Decisions list still named this as
  open — corrected to match `AGENTS.md`.

### 2026-08-03 — Hybrid GraphMCM RGCN encoder drops father_name/mother_name edges (redlined, lead-approved)

`shares_father_name`/`shares_mother_name` **excluded from what the Hybrid
GraphMCM's RGCN encoder trains/scores on**. `graph_builder_v3.py` is
unchanged — the full 5-relation identity graph is still built and stored,
and both relations remain available to `graph_viz_v3.py` (XAI 3D rings) and
`dense_block_detector_v3.py` (unaffected either way — it only ever reads
mobile/ip).

Evidence (session log, full record in
`outputs/ablation/relation_ablation_result.json` and
`src/ablation_relation_v3.py`):
- **Structural** (real 15k population, no retrain): both relations form
  large diffuse hubs, the same shape as the already-rejected dense-block
  `pincode` relation — `shares_mother_name` mean degree 5.09, max 109,
  touching 35.7% of nodes; `shares_father_name` mean degree 1.22, max 35,
  touching 23.2% — vs. `shares_mobile`'s 0.9% of nodes / max degree 5.
- **Behavioral** (locked checkpoint, post-hoc `compute_relation_ablation()`):
  neither relation separated the top-5%-risk population from the rest
  (~1.0x ratio), unlike `shares_mobile` (19x) and `shares_ip` (2.2x).
- **Retrain-based** (6 fresh RGCN trainings, 2 injected-cluster-density
  settings, seeds 42–47): dropping the two relations never showed harm on
  the connected-cluster harness, and consistently improved
  `MOTHER_NAME_COLLISION` detection in all 3 higher-precision-setting seeds
  (+0.042, +0.050, +0.018 PR-AUC).

Implementation:
- `src/config_v3.py`: new `ENCODER_RELATION_IDS = [0, 1, 4]` (mobile/ip/
  pincode) — `N_EDGE_TYPES` stays **5**, architecture unchanged; relations
  2/3 are simply never populated for the encoder.
- `src/hybrid_graphmcm_v3.py`: `_build_edge_index_and_types()` takes an
  optional `relation_ids` filter (default `None` = unrestricted, so
  `graph_viz_v3`/`dense_block_detector_v3` callers are unaffected); the
  Hybrid GraphMCM's own consumption points (`train()`, `load_model_and_inputs()`
  used by `score()`/`compute_relation_ablation()`/`attention_summary`,
  `train_incremental()`, `_get_synth_h_topology()`'s per-relation exposure
  grouping, `score_sampled()`/`train_sampled()`) now all pass
  `relation_ids=ENCODER_RELATION_IDS`. New `build_fixed_slot_edge_index_list()`
  helper (always returns a length-5 list, empty tensor for excluded/empty
  relations) fixes a **latent positional-indexing bug** this change would
  otherwise have exposed: `evaluate_model_v3.evaluate_connected()` and
  `compare_architectures_v3.py`'s synthetic-cluster injection index the base
  edge list by `RELATION_MAP[category]` (a fixed relation id, e.g.
  `MOTHER_NAME_COLLISION: 3`) — `_build_edge_index_and_types()`'s compaction
  (skips empty relations from the returned list) silently misaligned that
  positional assumption once a relation had zero real edges; both files now
  use the fixed-slot builder for their base edge list.
- `src/evaluate_model_v3.py`, `src/compare_architectures_v3.py`,
  `src/deploy_gate.py` updated so evaluation/deploy-gate scoring stays
  consistent with what the retrained model actually consumes.
- **Known gap**: the NeighborLoader mini-batch path (`train_sampled`,
  `_make_neighbor_loader`, `_batch_inputs` — step-5 scale work, not yet used
  in production) still samples all 5 relations per-batch and is not yet
  wired to `ENCODER_RELATION_IDS`; flag before that path goes live.
- Production checkpoint (`models/hybrid_graphmcm_v3.pth`) retrained end-to-end
  (80+120 epochs, topology-exposure ON, production seed) with the new
  encoder restriction; `compute_relation_ablation()` re-run on the retrained
  checkpoint confirms `shares_father_name`/`shares_mother_name` deltas are
  now ~0.000000 (never in the encoder's input), while `shares_mobile`/
  `shares_ip`/`shares_pincode` retain real deltas.
- `docs/AGENTS.md` §1 redlined to match (version 3.1).

### 2026-07-23 — CSV-free portal/ETL push endpoint (auto-preprocess, no auto-merge/retrain)

`POST /v3/monitoring/push-dataset` (new, `src/api/handlers/monitoring.py`):
a portal/ETL job names a raw-schema CSV it already wrote to the shared
`data/` volume (mounted into both `nic-api` and `nic-worker`) — no inline
JSON row payload, deliberately, since 3-4M rows in a request body doesn't
hold up at scale where a file path does. Schema-checked the same way as the
console's `upload-dataset`; on pass, stages the batch in Postgres
(`stage_raw_csv()`, unchanged) and dispatches Evaluate as a background
Celery task, returning a `job_id` immediately instead of blocking the
request for however long feature/graph rebuild + scoring takes.

Refactored `evaluate_dataset()`'s body into `_run_evaluate()` so the
console's synchronous route and the new async task run the *identical*
tested code path — no duplicated logic to drift out of sync. Concurrency
safety needed no new locking: `worker_concurrency=1` (hard stop 16,
`celeryconfig.py`) already serializes every Celery task, so an
auto-triggered Evaluate can't collide with another training/evaluate job's
canonical-file backup/restore.

**Deliberately scoped to auto-preprocess only** — Merge (into the base data)
and retrain dispatch stay separate, human-gated calls
(`POST /v3/training/decision`), exactly as for a console-uploaded cohort.
See `docs/OPERATIONS_RUNBOOK.md` §5c for the operator/integrator steps.

### 2026-07-23 — Opt-in Postgres-sourced full retrain via API (no CSV required)

`POST /v3/training/full` gained a `data_source` query param (`"file"`
default, unchanged; `"postgres"` new) that lets a full retrain read every
**merged** batch straight from Postgres (`src/db/features.py
fetch_raw_frame()`/`edge_groups()`) instead of `data/raw/data_for_ml_model.csv`
— no CSV round-trip needed once data is already staged + merged. Wired via
`config_v3.DATA_SOURCE` (`NIC_DATA_SOURCE` env), which `main_v3.py`'s
`build_base`/`build_graph` steps branch on to call `build_base_pg()`/
`build_graph_pg()` (already built and gate-4-verified bit-exact, step 4 of
the V4-Scale migration) writing to the same canonical file paths the rest
of the pipeline expects — every downstream step is unaffected either way.
See `docs/OPERATIONS_RUNBOOK.md` §5a for the operator steps.

Fixed en route: `fetch_raw_frame()` previously derived its row set/order
from the raw CSV's own `application_id` list (a known gap called out in its
own docstring), so any batch merged into Postgres *after* the primary 15k
would have been silently dropped from a Postgres-sourced read. It now reads
row set/order entirely from Postgres (every `status='merged'` batch); only
the static 136-column header still comes from the file (schema names, not
row data). Re-verified bit-exact against the canonical
`engineered_features_v3_nodeg.csv` after the fix, aligned by
`application_id` (max abs diff 0.0).

**Still file-based, not yet closed:** this is an opt-in read-path switch,
not a default flip — nothing about hard stop 13 ("authoritative only after
demonstrated parity") changes by default. Also, landing raw rows in
Postgres in the first place still requires a CSV via `stage_raw_csv()` — a
CSV-free *ingestion* entry point (portal/DB push -> auto-preprocess) is
separate, not-yet-scoped work (see `docs/IMPLEMENTATION.md`, "After step 5").

### 2026-07-23 — hybrid_anomaly_score drops edge_pred_error's contribution (redlined, lead direction)

`LAMBDA_EDGE_SCORE` (new, `config_v3.py`) decouples the inference-time score
weight on `edge_pred_error` from `LAMBDA_EDGE` (unchanged at 0.3, still the
training-loss weight for the edge-prediction head). `hybrid_anomaly_score =
feature_pred_error + LAMBDA_EDGE_SCORE * edge_pred_error`, with
`LAMBDA_EDGE_SCORE = 0.0` — was effectively `LAMBDA_EDGE` (0.3) prior to this
change.

Evidence: `edge_pred_error` showed no usable signal on any of its 3 designed
relational categories on a held-out stress cohort (`stress_testing_1`) —
IP_CLUSTER/MOBILE_CLUSTER/PINCODE_CLUSTER PR-AUC 0.011-0.023, at or below
what a random ranking would score — and on MOBILE_CLUSTER specifically it
actively diluted `feature_pred_error`'s real standalone signal (0.268 alone
vs 0.017 combined). Mechanistically consistent with the documented MAR
critique (`.claude/CLAUDE.md` "Known Structural Weaknesses": dense cliques
reconstruct/predict too easily) — ring members' edges are *easier*, not
harder, for the RGCN to predict, so `edge_pred_error` points the wrong way
for exactly the fraud shape it was meant to specialise in.

Dropping the term from the score improved 6/7 held-out categories (mean
PR-AUC 0.017 -> 0.056), with the only regression (IP_CLUSTER, -0.0007)
smaller than the documented GPU noise floor. **Replicated on a second,
independently-seeded stress population** (`stress_testing_2`, different
seed, different sampled rows/rings, same archetype proportions): same sign
on every one of the 7 categories, mean PR-AUC 0.017 -> 0.048. Raw stdout:
`outputs/ablation_lambda_edge_v3_44.json`,
`outputs/ablation_lambda_edge_v3_44_stress2.json`.

Training is unaffected — `LAMBDA_EDGE` (the loss weight) is untouched, so
the edge-prediction head still trains and `edge_pred_error` is still
computed and available for XAI narration; it is excluded only from the
ranking score. Untested: whether removing the edge-prediction objective
from training entirely (not just from the score) would change embedding
quality — a separate question, not addressed by this change.

**Real-population caveat, stated plainly because it matters:** on the real
15k population (no ground truth available), the new formula reorders
rankings substantially — old vs. new `hybrid_anomaly_score` correlation
0.56, top-100-most-suspicious overlap only 6/100. The synthetic evidence
above supports that the new formula is better at catching the fraud shapes
it was tested against; it does NOT by itself confirm the reordering is an
improvement for the real population, since no ground truth exists there to
check against. Flagged, not resolved, by this change.

### 2026-07-22 — Explainability build-out: RGCN relation ablation, EVT empirical rates, ring fixes, cohort-preview signal drivers

Same-day follow-up: three requested explainability enhancements, plus two
real bugs found and fixed while verifying them live via Playwright against
the rebuilt Docker API.

- **RGCN per-relation ablation (new, XAI-only)** —
  `hybrid_graphmcm_v3.compute_relation_ablation()`: the production RGCN
  encoder has no learned attention (unlike the rejected HAN path, whose
  `beta_r`/`top_alpha` stay dormant), so "expected this based on neighbours"
  couldn't say WHICH relation drove that expectation. Re-scores the locked
  checkpoint 5 extra times (once per edge type, masked via
  `edge_type_tensor`, no retraining) and reports, per node, which relation's
  removal improved feature-reconstruction fit the most. New pipeline step
  `relation_ablation` (`main_v3.py`, between `train_hybrid` and
  `subspace_if`); writes `outputs/relation_ablation_v3.csv`; narrated as
  "Neighbourhood-expectation driver" on cards and shown as a bar on the
  Signal drivers tab. Never feeds fusion or any threshold.
- **EVT empirical-rate framing** — every `evt_scorer_v3._fit_evt()` branch
  (POT-GPD and both fallbacks) now returns `n_flagged`; trigger sentences
  cite the actual measured flagged rate for that signal ("this pattern was
  this extreme in 31 of 15,000 applications, ~0.21%") instead of only the
  aspirational target `Q` the tail was fit towards.
- **Dense-block core highlighting on the 3D ring** — a gold diamond outline
  now marks nodes that are part of the actual Charikar-peeled dense core
  (`dense_block_score_relational > 0`), distinguishing real ring members
  from incidental shares-X neighbours. `xai_card_html_v3._dense_core_app_ids()`.
- **Per-relation edge toggle on the ring** — each relation's legend entry now
  explicitly supports click-to-hide / double-click-to-isolate
  (`itemclick="toggle"`, `itemdoubleclick="toggleothers"`), for decluttering
  dense identity cliques.
- **Bug fix: `shares_mother_name` (and any relation) could be silently
  dropped from the ring.** `graph_viz_v3._figure_for_ring` used to keep only
  the FIRST relation seen for a node pair (`edge_rel.setdefault`), so a pair
  sharing both e.g. `shares_ip` and `shares_mother_name` only ever drew the
  IP edge. Fixed to track every relation connecting a pair and draw one line
  per relation — this also means the new edge-toggle now works correctly
  for such pairs (previously, toggling the shadowing relation off did not
  reveal the hidden one, because it was never rendered in the first place).
- **Cohort-preview Signal drivers were empty; now populated.**
  `POST /evaluate-dataset` now also computes subspace IF
  (`subspace_if_v3.compute_subspace_if_scores`, refactored out as a reusable
  pure function shared with `run_subspace_if()`) and dense-block scores over
  the batch's merged population, plus a preview fusion score via the SAME
  `fusion_classifier_v3.score_level_fusion()` /
  `xai_layer_v3.build_fusion_contributions()` the committed pipeline uses.
  `build_staged_card_html()` now reads these into `subspace_groups` /
  `dense_block_relational` / `fusion_contributions` instead of hardcoded
  empty dicts. Still clearly labeled PREVIEW, still not `risk_score_v3`
  (renormalised over the cohort's own population, not the canonical one's
  fixed scale), still without EVT triggers or model-traceability margins.
  Verified live via Playwright against the rebuilt `nic-api` image on the
  `stress_testing_1` cohort — bars, fusion-composition footer, and the
  corrected preview-note text all render correctly.

### 2026-07-22 — Dense-block pincode gate reverted (redlined, sole-author lead direction)

Same-day follow-up to the dense-block relational extension below: pincode
**removed** from the dense-block gate per lead direction. Shared pincode
reflects legitimate geographic clustering, not collusion — it is not a
valid fraud signal **on its own** for this detector (the RGCN/hybrid
detector still consumes `shares_pincode` edges as part of the unchanged
5-relation identity graph; this only affects the dense-block specialist's
own gate).

- **`DENSE_BLOCK_RELATIONS` reverted from `[0, 1, 4]` (mobile+ip+pincode) to
  `[0, 1]` (mobile+ip only)**; `DENSE_BLOCK_RELATION_WEIGHTS` reverted from
  `{0: 0.3, 1: 1.0, 4: 0.2}` to `{0: 0.3, 1: 1.0}` (mobile: 0.3, ip: 1.0).
  `src/config_v3.py`.
- `src/dense_block_detector_v3.py` no longer emits `dense_block_score_pincode`
  — output columns are `dense_block_score_mobile`/`dense_block_score_ip` +
  `dense_block_score_relational`.
- `src/xai_layer_v3.py`, `src/xai_card_html_v3.py`, `src/fusion_classifier_v3.py`,
  `src/compare_architectures_v3.py` updated to match — the dense-block
  evidence section on XAI cards now names only mobile/IP as the shared
  identity value that can drive a dense-block flag.
- History: dense-block was `shares_ip`-only → extended to
  mobile + ip + pincode earlier 2026-07-22 (see the entry below) → pincode
  dropped same day, reverted to mobile + ip.

### 2026-07-22 — RGCN root_weight fix + Deep SAD supplementary signal (XAI cards)

Two real architecture changes, both prototyped on `stress_testing_1` before
touching production, both now live.

**`root_weight=False` on Hybrid GraphMCM's RGCN encoder.** `RGCNConv`
defaults to `root_weight=True`, so `h_n` (the "neighborhood context" fed to
the MCM feature predictor) contained a direct learned self-transform of a
node's own UNMASKED features, independent of the intentional masking in
`_apply_masks()` — leaking self-signal around the mask for every connected
node (isolated nodes were already unaffected, overridden by
`isolated_embedding`). Disabling it makes `h_n` pure multi-relation neighbor
aggregation — the actual MCM contract. stress_testing_1: overall PR-AUC
0.153→0.201, mobile-ring 0.029→0.078, IP-ring 0.032→0.055, no low-degree
regression (3-5 degree bucket 0.193→0.385, 6+ bucket 0.153→0.201). Real
15k-dataset retrain: 5/5 V2 floors still pass, edge-dropout retention 2.34.
Two other prototyped alternatives were tested and rejected first: a scoped
Graph Matching Network (cross-graph attention against exposure clusters,
0.147 overall — flat vs baseline) and a scoped UniGAD-inspired learned-
subgraph + spectral-energy scorer (0.123 overall — worse, particularly on
IP/mobile). See `outputs/stress_testing_1_{gmn,unigad,rootweightoff}_stats.json`.

**Deep SAD center-distance (`src/deepsad_detector_v3.py`, new module).**
Separate encoder, separate objective (Ruff et al., ICLR 2020): pulls real
nodes toward a learned normal center, pushes topology exposure's synthetic
archetypes away via an inverted-distance term — no reconstruction loss, so
it doesn't inherit the MAR reconstruct-too-easily failure mode. stress_testing_1:
0.201 overall / 0.093 mobile-ring / 0.050 IP-ring — the strongest single
relational signal found this session, beating both hybrid_reconstruction and
the root_weight fix on mobile-ring specifically. A companion per-cluster
"nearest known archetype" prototype-match mechanism was tested alongside it
and rejected (0.116, near-random on every category — no inter-prototype
separation term). A further CARE-GNN-style self-adjusting neighbor filter
(dormant until a relation sees ≥20 exposure examples, then loosens/tightens
via reward feedback) was layered on top of Deep SAD and also rejected net
(0.189 vs 0.201 — helped mother-name/pincode, cost IP/mobile through the
shared encoder). Deliberately kept OUT of `FUSION_COMPONENTS` /
`final_risk_score` — promoting a 4th signal into the locked fusion is a
bigger structural decision than this session resolved. Instead surfaced on
XAI cards (`xai_layer_v3.py`, `xai_card_html_v3.py`) as a supplementary
"Deep SAD center-distance" panel + narrative sentence when >75th percentile,
clearly marked as not driving the fused score, plus a `deepsad_percentile`
column in scorecard exports. Pipeline: new `deepsad` step in
`main_v3.py` between `dense_block` and `evt`.

Both changes redlined under explicit lead direction (sole author); see
`docs/AGENTS.md` staleness note below.

**Follow-up, same day — Deep SAD fusion inclusion tested and rejected.**
With the card wiring live, tested whether `center_dist_score` actually
improves the fused score if added as a 4th max-fusion input (candidate:
`max(subspace, dense_relational, hybrid, center_dist_score)`), re-scoring
`stress_testing_1`'s already-computed columns (no retraining). Result:
overall PR-AUC 0.4182 (locked 3-way) vs 0.4181 (4-way) — noise-level, not
an improvement. Deep SAD won the argmax driver role in only 483 of 50,000
nodes (<1%): the existing three detectors already cover their specialty
categories so completely (e.g. mobile-ring alone reaches 0.53 PR-AUC under
the current fusion, far above Deep SAD's standalone 0.09-0.10 there) that a
4th max-input rarely gets to matter, even though Deep SAD carries real
information on its own. Confirms the XAI-card-only placement was the right
call rather than an open question — not promoting it into
`FUSION_COMPONENTS`. See `outputs/stress_testing_1_fusion4_stats.json`.

**Also tested and rejected this session, for the record:** Latent Outlier
Exposure (Qiu et al., ICML 2022) — jointly re-estimating which real nodes
are likely-contaminated during Deep SAD training, instead of the static
`CENTROID_CLEAN_PERCENTILE=95` heuristic. At alpha=0.05 (matching production's
existing percentile convention): net regression, 0.201→0.193 overall. At
alpha=0.15 (matched to stress_testing_1's actual ~15% injected fraud rate):
overall recovered to near-parity (0.199) with the best-ever mobile-ring
result this session (0.105), at a small cost to IP/pincode/mother-name.
Not adopted — alpha=0.15 matches this stress-test's deliberately inflated
contamination rate, not production's real (unmeasured, presumably much
lower) rate; using it as-is on real data would be an unjustified guess.
See `outputs/stress_testing_1_loe_qiu*_stats.json`. Also note the acronym
collision: this is unrelated to the project's own "Learning-from-Only-
Exposure" (LOE).

### 2026-07-22 — On-demand explanation cards + 3D rings (scale prep for 3-4M)

Found while reviewing what happens to card/ring serving at 30-40L scale: the
**3D ring was already fully on-demand** (`build_ring_html` is PG-indexed —
`ego_neighbors`/`induced_subgraph_edges`, no full-graph load, computed live
per `/ring` request) — nothing to fix there. The **explanation card was not**:
`build_card_html` only worked for an application inside the pre-computed
top-500 batch (`explanation_cards_v3.json`), and the batch generator
(`run_xai()`) did a full-population rescan (percentile distributions, the
whole graph's neighbor index) on every call regardless of how many cards it
wrote — a cost that scales with population size and would dominate at
3-4M rows.

- **`src/xai_layer_v3.py` refactored**: population-wide setup (percentile
  stats, neighbor index, degree counts, closed-form fusion contributions) is
  now a **cached context** (`get_xai_context`, mtime-keyed on its source
  files — same pattern as `confirmed_fraud_graph_store.py`'s `_ip_cache`),
  built once per scoring cycle instead of once per call. The per-application
  assembly logic that used to live inline in `run_xai()`'s loop is now a
  standalone `_assemble_card(app_id, ctx)`, and both `run_xai()` (the
  top-500 batch, unchanged externally) and the new **`build_card_for_app(app_id)`**
  (any application, on demand) call the same function — so batch and
  on-demand cards can never drift apart. Also skips loading the hybrid model
  entirely for attention extraction when `ENCODER_ARCH != "han"` (RGCN, the
  production default, never uses `beta_r` — this was previously loaded and
  forward-passed unconditionally).
- **`src/xai_card_html_v3.build_card_html()`** now falls back to
  `build_card_for_app()` when the requested application isn't in the
  pre-computed batch, so the `/card` API route serves a real card for
  *any* scored application, not just the top 500.
- **Verified**: on a 15k population, the first on-demand call pays the
  context-build cost (~6.3s at this scale); every subsequent on-demand card
  for a different arbitrary application — including the single lowest-ranked
  one, rank 15,000 — costs ~1ms. A card for application #2,847,193 will cost
  the same as one for #1.
- **`src/xai_card_html_v3._graph_ctx`** (the ring's `.pt`-graph fallback, used
  only when Postgres is unavailable) is now also mtime-keyed cached instead
  of rebuilding the whole graph's neighbor index on every fallback request.
- Static per-application HTML file generation (`render_cards()`, gated to
  `suspicious_only=True` already) is unchanged — it was already bounded by
  flag rate, not population size, so it wasn't the actual problem.

### 2026-07-22 — Dense-block relational extension, max fusion, LOE margin fix (redlined, sole-author lead direction)

Three locked-architecture changes, all validated on the stress_testing_1
ablation before being adopted into production. `docs/AGENTS.md`'s hard-stop
table needs a corresponding redline (dense-block relation gate, fusion
formula, LOE margin are all named there) — flagged, not yet applied to that
file.

- **Dense-block extended from `shares_ip`-only to mobile + IP + pincode**
  (`DENSE_BLOCK_RELATIONS = [0, 1, 4]`), each relation scored independently
  then combined via an **IP-priority-weighted max**
  (`DENSE_BLOCK_RELATION_WEIGHTS = {mobile: 0.3, ip: 1.0, pincode: 0.2}` —
  `dense_block_score_relational`). Equal-weighting was tried first and
  rejected: it gained more overall (0.268 PR-AUC) but let ordinary, non-fraud
  density in mobile/pincode outrank true IP-ring members (IP PR-AUC collapsed
  0.220→0.067) — unacceptable given IP is the dominant real fraud vector.
  IP-priority-strong keeps IP detection ~unchanged (0.220→0.220) while mobile
  goes from near-zero (0.030) to real signal (0.149-0.349 depending on
  weighting). `src/dense_block_detector_v3.py`, `src/config_v3.py`.
- **Fusion changed from a weighted sum to an unweighted max**:
  `risk = minmax(max(minmax(subspace), minmax(dense_relational), minmax(hybrid)))`.
  The weighted-sum was found to dilute whichever detector actually found the
  fraud with near-random noise from the other two on every category tested
  (e.g. mobile-ring: subspace alone 0.674 PR-AUC vs the old fused 0.349).
  Overall PR-AUC on the ablation: 0.403 (sum) → 0.447 (max). A rank-based
  (Borda) alternative was also tried and rejected — it scored worse (0.295),
  likely because most detector outputs are exact zero for the vast majority
  of rows, so rank-averaging gets dominated by tie blocks.
  `src/fusion_classifier_v3.py`.
- **XAI cards refactored for max-fusion attribution**: "share of a blend" no
  longer means anything under max fusion, so `build_fusion_contributions`
  now reports each detector's own normalised value + an `is_driver` flag
  (the argmax) + `margin_over_next` (how clearly it won). Cards show a
  DRIVER badge on the winning detector instead of a percentage-share bar;
  the dense-block evidence section now names WHICH shared identity value
  (mobile/IP — pincode was briefly included here, dropped later the same
  day, see the entry above) actually drove the flag. `src/xai_layer_v3.py`,
  `src/xai_card_html_v3.py`, `src/export_v3.py` (scorecard columns renamed
  `subspace_normalized`/`dense_relational_normalized`/`hybrid_normalized` +
  `driving_margin`, replacing the old `*_share` columns).
- **LOE topology-exposure margin fixed** — found via direct measurement,
  not assumption, while testing why hybrid's ring detection stayed weak: the
  locked `LOE_MARGIN=2.0` constant is ~3x SMALLER than even the REAL
  population's own median embedding-to-centroid distance at this embedding
  dimensionality (measured: real median ≈5.9, exposure mean ≈6.9, margin
  2.0) — meaning exposure embeddings were already past the "margin" before
  training even started, so the LOE warm-start term contributed exactly
  `0.0000` throughout training regardless of formula (`exp(-sqrt(dist))`,
  the old formula, and a fixed hinge were both tested and both stayed at
  zero). Fixed in two parts: (1) `_loe_loss` is now a hinge
  (`clamp(margin - dist, min=0)`) instead of the old exponential, which
  saturates to ~0 once distances exceed a handful of units; (2) the margin
  is now **derived from the current epoch's real embedding distribution**
  (`_derive_loe_margin`, a percentile of real dist-to-centroid — same
  principle as `CENTROID_CLEAN_PERCENTILE`, not a hand-picked constant) so
  it self-calibrates to whatever scale the network's embedding space
  actually lives at. Stage 2 also gained a small **persistent** LOE term
  (`LOE_STAGE2_WEIGHT=0.15`, not decayed to zero) — previously Stage 2 had
  no exposure term at all, so any separation Stage 1 bought could be freely
  re-absorbed by 120 epochs of unconstrained reconstruction (dense synthetic
  cliques reconstruct too easily — the MAR critique). Verified on a 15-epoch
  test run: LOE now goes 0.85→0.25 in early epochs (genuine gradient-driven
  separation, not saturation-from-init), decaying toward zero as embeddings
  clear the margin — convergence, not failure to engage. `src/config_v3.py`,
  `src/hybrid_graphmcm_v3.py` (`train`, `train_incremental`, and the
  NeighborLoader mini-batch path all updated to keep the three training
  loops in sync).

Two prototype alternatives to Hybrid GraphMCM itself were tried and
**rejected** on the same stress-test data before landing on the margin fix
above — kept for the record, not adopted:
- A GRACE/DGI-style contrastive encoder + embedding-redundancy anomaly
  score: 0.097 PR-AUC at 120 epochs on GPU (vs hybrid's 0.276), no better
  than a 25-epoch run — properly tested, genuinely underperforms for this
  fraud shape.
- A "predict from real vs. randomly-swapped neighbors" margin score on a
  simplified MCM/RGCN: underperformed even its own model's raw error (0.148
  vs 0.213) at 150 epochs on GPU — a clean, controlled negative result.

Full numbers: `outputs/stress_testing_1_v2_stats.json` (dense-block +
contrastive), `outputs/stress_testing_1_v2b_stats.json` (dense-block weight
sweep), `outputs/stress_testing_1_v3_stats.json` (MCM-margin prototype).
Prototype scripts (not part of the production module ownership):
`scripts/prototype_v2_components.py`, `scripts/prototype_dense_weighted_max.py`,
`scripts/prototype_mcm_margin.py`.

### 2026-07-21 — 50k-application stress test ("stress_testing_1")

Ad hoc scale/quality exercise, not a formal migration gate (that's the 3.5M
K_CAP profiling in `docs/IMPLEMENTATION.md`). Generated a 50,000-row synthetic
cohort (`scripts/generate_stress_test_dataset.py`) — 85% valid, 15% fraud
across 4 tabular archetypes (fee inflation, income violation, age violation,
mother-name collision) + 3 relational ring types (IP/mobile/pincode-sharing,
147 rings, sizes 6-40), sampled/perturbed from the real 15k population per
AGENTS.md Appendix B (no GAN/CTGAN/TVAE). Ground truth in
`data/uploads/stress_testing_1_ground_truth.csv`.

Ingested via the existing intake path (`upload-dataset` → Postgres batch
`stress_testing_1`, 50k rows / 0 conflicts, 31s; `evaluate-dataset` → merge
with the base 15k, rebuild features + identity graph, score with the current
checkpoint, restore — 146s for ~65k merged nodes). Since the cohort endpoint
only stages the pre-fusion hybrid score, `scripts/stress_test_1_analysis.py`
separately computed subspace IF + dense-block-IP on the staged artifacts
(same code paths, read-only) and applied the unmodified locked fusion formula
to get the real three-detector picture: fused `risk_score_v3` PR-AUC 0.403 /
ROC-AUC 0.800 at a 15% base rate; strongest on mobile-sharing rings (PR-AUC
0.349, 98.6% of ring members in a top-5,000 queue) and identity-collision
(0.187); weakest on pincode-sharing rings and the fee/age tabular archetypes
(PR-AUC <0.05 across all three detectors — flagged as "observed, not yet
explained," not diagnosed). Full numbers: `outputs/stress_testing_1_stats.json`,
per-row scores in `outputs/stress_testing_1_full_scores.csv`.

### 2026-07-21 — V4-Scale: PostgreSQL system of record + Kubernetes-scale remodel

Branch `V4-Scale`. Detection architecture unchanged (locked, see above);
every I/O boundary rebuilt for 30–40 lakh (3–4M) applications. Full detail:
`docs/TECHNICAL_REFERENCE_AND_SCALING.md`; step-by-step gate evidence:
`docs/IMPLEMENTATION.md`.

- **PostgreSQL system of record** — 10-table schema (`deploy/postgres/
  schema.sql`): `applications`, `identity_keys`, `features`,
  `feature_scaling`, `scores`, `confirmed_fraud`, `loe_patterns`,
  `evt_thresholds`, `training_runs`, `drift_baselines`. All access through
  the new `src/db/` package (hard stop: no inline SQL elsewhere).
- **`db-init` bootstrap** — a one-shot container service applies the schema
  and ingests/replays data into Postgres on every `docker compose up`;
  `nic-api`/`nic-worker` wait for it, so Postgres is populated the moment
  the API starts, not just reachable.
- **Reads flip to Postgres by default** (`NIC_READS_FROM_PG=1`), falling
  back to files automatically on any query failure — the review queue,
  status tiles, 3D rings, and ego-graphs are now served from indexed SQL
  instead of whole-file parses / an in-memory graph. Verified identical to
  the old file/graph paths across 150 sampled ego-graphs and 60 rings.
- **CSV intake unchanged, staging added underneath** — uploads land in a
  Postgres staging batch (raw rows only; nothing derived until Evaluate),
  Evaluate populates preview scores, Decide → Merge makes it permanent. The
  cohort-preview reviewer card was rebuilt to share the same rendering
  components as the base card (real identity-network view, ranked reason
  codes, expandable fields) instead of a separate flat-table template.
- **SQL-pushdown feature engineering + persisted scaler** — proven
  bit-exact against the file pipeline on all 44 features; the scaler now
  persists its fitted parameters instead of refitting per batch (closes a
  batch-statistics leak).
- **Hub-capped identity graph + exact-neighborhood training** — replaces
  all-pairs edge construction and full-graph RGCN passes. Every truncating
  NeighborLoader fan-out tested deviated from full-graph scores; exact
  2-hop batching reproduces them bit-for-bit while keeping memory bounded.
  Measured on a synthetic 1M-application population: **3.53 GB peak RSS**;
  full-retrain wall-clock projects to ~101 h at 3.5M on CPU (training time,
  not memory, is the real scaling constraint).
- **External GPU-checkpoint ingestion** — a checkpoint trained entirely
  outside the cluster can be uploaded via the admin console, schema-
  validated (rejected outright on any mismatch, live model untouched), and
  atomically hot-swapped in.
- **K_CAP profiling query built** (`scripts/profile_group_sizes.py`) and
  dry-run tested on the 15k primary population; the production threshold
  still needs a real-scale ingest to derive (open decision).
- Two real deployment bugs found and fixed via a live `docker compose up`
  test: `postgres:18`'s changed volume-mount convention, and the Dockerfile
  not copying `deploy/postgres/schema.sql` into the image.

### 2026-07-21 — Flagged-pattern export + explicit incremental/full retrain choice

Found and fixed a real gap while checking that "flag a pattern → the model
learns its topology" actually holds end-to-end: `append_ring_to_topology_exposure()`
correctly writes a promoted ring's **real** identity-graph edges to
`synthetic_exposure_graph_v3.pt`, but every retrain the console/API dispatched
(`patterns/promote`, `confirm-batch`, `pattern/ingest`, `training/decision`)
called `train_incremental()`, which never reads that file — it only fine-tunes
on isolated (edge-free) feature vectors, so a promoted ring's structure sat
unused until someone manually ran a full `main_v3.py`. The topology-consuming
path (`hybrid_graphmcm_v3.train()` Stage 1 LOE against `synthetic_exposure_graph_v3.pt`)
already existed via the full pipeline — it just was never wired to the pattern
flows.

- **Explicit retrain-mode choice, everywhere a pattern retrain is dispatched.**
  `POST /v3/supervisor/patterns/promote` now takes `mode: "incremental" |
  "full_retrain"` (default unchanged: `incremental`). `"full_retrain"` dispatches
  the existing `run_full_pipeline_task` (same job the admin **Full pipeline**
  button uses) instead of the incremental fine-tune, so the RGCN actually
  trains against the ring's topology when you want that.
- **New `POST /v3/supervisor/patterns/retrain`** — retrain directly from the
  flagged-history store, independent of the pending-queue promote flow. Covers
  patterns already `PROMOTED` (topology already appended — this just (re)runs
  training) as well as any still `CONFIRMED`/`SELECTED` in the selection
  (promoted first). Empty `pattern_ids` = every non-rejected pattern in the
  store. Console: **Flagged history** panel gets a retrain-mode dropdown +
  smoke-test checkbox + **▶ Retrain selected** / **▶ Retrain all** buttons,
  reusing the existing multi-select.
- **Flagged-pattern export**, matching the application queue's export
  affordance: **⤓ Export selected** / **⤓ Export all** buttons on the flagged-
  history panel. New endpoints `GET /v3/supervisor/patterns/export/bulk` and
  `.../export/selected?ids=…`, `build_pattern_bulk_export()` /
  `build_pattern_selected_export()` in `src/export_v3.py` — zip of
  `manifest.csv` + `patterns/<pattern_id>.json` (the full stored record),
  same shape as the existing application-export bundles.

### 2026-07-15 — Supervisor CSV fraud-pattern intake (relational LOE)

Supervisors can now bring in a **brand-new, relationally-complex fraud ring the
model has never seen** as a CSV of full raw-schema rows — not only patterns the
model surfaced. New module `src/pattern_ingest_v3.py` + a purpose toggle on the
admin **Intake** step ("New cohort to score" vs "New fraud pattern"):

- **Test (read-only)** — `POST /v3/supervisor/pattern/test`: merges the ring,
  rebuilds features **and the identity graph** (so the members' shared
  IP/mobile/name/pincode edges are real), scores just the ring with the current
  checkpoint, then restores every canonical file. Answers "does the model already
  catch this?"
- **Ingest as relational pattern** — `POST /v3/supervisor/pattern/ingest`:
  permanently merges the ring, extracts its **real intra-ring subgraph across all
  5 relations**, appends it as a new **topology-exposure cluster**
  (`synthetic_exposure_graph_v3.pt`), records it in both confirmed stores (graph
  pattern + tabular rows), and optionally dispatches the human-gated incremental
  fine-tune. A **re-test** then shows detection after the model has learned it.
- **Implemented the previously-stubbed topology-LOE injection.**
  `confirmed_fraud_graph_store.promote()` now actually appends promoted patterns
  to the topology-exposure set (extracting real edges, falling back to a clique on
  the reviewer-asserted relation) — so the **Pattern queue → Promote** path also
  feeds the RGCN exposure stream, not just the state machine.

### 2026-07-17 — Cohort preview queue (review ingested data + 3D rings, read-only)

The Review queue gained a **Dataset switcher** (`#cohort-select`): **Primary
dataset · 15k scored applications** vs any **evaluated cohort**. (The primary
dataset is the population the unsupervised detector is fit on *and* scores — not a
held-out test set; genuinely unseen data is scored via an evaluated cohort, or the
synthetic harness in `evaluate_model_v3.py`.) Selecting a cohort re-points the
whole triage surface — pagination, filters, multi-select, open-card — at that
cohort's **staged** scores, so you can see how the model behaves on data you
ingested from the front end *before* committing it.

- **Scores are pre-fusion** (`hybrid_anomaly_score`, higher = more anomalous),
  bucketed by **within-cohort percentile** (80th=high / 50th=med) so badges + the
  risk filter stay meaningful on an arbitrary-scale score. A cyan banner states
  this plainly. This is **not** the committed fused `risk_score_v3`.
- **3D identity rings work for cohort apps.** `evaluate-dataset` now persists a
  cohort bundle — `outputs/staged_graph_<name>.pt` + `staged_nodeorder_<name>.csv`
  (the merged base+cohort graph + node order it would otherwise discard on
  restore). The ring builder (`build_ring_html`) takes an optional graph source,
  so a cohort app's ring shows its real edges into the base 15k **and** other
  cohort apps. `build_staged_card_html` renders a lightweight pre-fusion preview
  card.
- **Full read-only parity**: card, **3D ring**, **ego-graph**, and **export**
  (single / bulk / selected) all work on cohort apps — rendered/bundled from the
  staged graph + scores, same as the base run. Ring inclusion mirrors the
  committed exporters (only *selected* embeds the heavy Plotly rings; bulk/single
  stay light). Only the **training-feeding** actions (Flag-for-LOE, label/retrain)
  stay gated with an "ingest to enable" hint — those write to the live graph/stores
  and only make sense once the cohort is committed.
- New read-only endpoints: `GET /v3/monitoring/cohorts`,
  `/cohort/{name}/top-suspicious`, `/cohort/{name}/{app_id}/card|ring|topology|export`,
  `/cohort/{name}/export-bulk`, `/cohort/{name}/export-selected`. `extract_ego`
  and `build_ring_html` gained optional staged-graph-source params;
  `build_cohort_{single,bulk,selected}_export` in `export_v3.py`.
- **Remove cohort** (demo reset): a `✕ Remove cohort` button by the Dataset
  dropdown (cohort mode only) drops that cohort from the console via
  `POST /v3/monitoring/cohort/{name}/delete` — deletes only its staged files
  (+ its uploaded CSV) by exact filename; base data and the downloadable sample
  CSV are untouched. Re-evaluate the CSV to bring it back.
- **Intake help + sample CSV.** The admin Intake step now spells out what a CSV
  needs (raw schema incl. the identity fields that build the ring) and reassures
  that **the system does its own feature engineering** — you supply raw columns
  only. A **Download sample CSV** button serves `frontend/sample_cohort.csv`
  (full raw schema, fresh application_ids, two planted shared-IP rings).

### 2026-07-17 — Paginated queue + bulk Flag-for-LOE

Frontend-only (no endpoint changes; reuses `top-suspicious` and
`patterns/confirm`):

- **Paginated review queue** — the queue now pulls the full flagged set (top
  `QUEUE_FETCH_N=500`, all of which have reviewer cards) and pages through it
  **50 rows per page** with ← Prev / Next → controls, instead of hard-capping at
  the top 50. The ID/risk filters reset to page 1; the pager shows
  "Showing a–b of N flagged · Page x / y". To widen to the whole 15k scored
  population, raise `QUEUE_FETCH_N` in `frontend/app.js` — but rows past the ~500
  carded apps have no card, so their "Open" 404s.
- **Bulk Flag-for-LOE** — a new **◈ Flag for LOE (selected)** toolbar button
  takes every checkbox-selected application (selection **persists across pages**)
  and opens the same Flag-for-LOE modal pre-filled with all of them as one
  candidate ring; the per-card single-app "⚑ Flag for LOE" button is unchanged.
  On record, the console clears the selection and jumps to the **Pattern queue**,
  where the new candidate is visibly "pending" — closing the "flagged patterns
  don't appear" gap (the modal always posted correctly; the UI just never
  surfaced the result). The LOE modal is now driven by `loeCenterId` rather than
  `selectedAppId`, so it serves both entry points.
- **Soft IP-cluster coverage guard** — when a reviewer opens a card or the
  Flag-for-LOE modal, the console cross-checks the application against every
  previously-flagged pattern (all sessions) on the **`shares_ip`** edge and, if a
  match is found, shows a **soft, non-blocking** banner: *"Looks like this cluster
  may already be flagged … verify via the 3D identity ring before re-adding."* It
  distinguishes patterns **already in LOE exposure** from merely pending/promoted.
  Deliberately advisory (not a hard block) and IP-only — the reviewer confirms the
  ring. New read-only endpoint `GET /v3/supervisor/patterns/coverage/{app_id}`
  backed by `confirmed_fraud_graph_store.ip_coverage_for_app()` (mtime-cached
  `shares_ip` adjacency, so only the first check per graph pays the load).
- **Flagged history directory** — a new panel on the Pattern queue lists **every**
  pattern flagged across all sessions (all states, newest first) with state badges,
  the "in LOE exposure" tag + exposure cluster id, members, and who/when. Backed by
  `GET /v3/supervisor/patterns/all` → `confirmed_fraud_graph_store.list_all()`.
  This is the persistent store the coverage guard matches against. Rows are
  **selectable with a hard-delete** (`✕ Delete selected` → `POST
  /v3/supervisor/patterns/delete` → `remove_patterns()`) for cleaning up mistaken
  or test flags; delete removes the **record only** — a promoted pattern's ring
  may already be in the exposure set / checkpoint, so the confirm dialog warns
  that deleting it does not un-train the model (needs a rebuild). `removed_promoted`
  in the response names any promoted ids that were deleted.

### 2026-07-15 — Triage-first console: multi-select batch label/retrain, export selected, drift explanation

Reworked the review queue into a triage surface and added three supervisor
workflows on top of the existing console (every prior control + endpoint
preserved):

- **Multi-select queue** — per-row checkboxes + select-all, an application-ID
  filter and risk-level filter, colored **risk badges** (high/med/low), and a
  session-local **remove/restore** so triaged rows stay out of the way without
  mutating server data.
- **Label / retrain selected** — a batch modal tags each selected application as
  confirmed-fraud (with type) or false-positive, writes them to the confirmed-
  fraud store, and optionally dispatches the **human-gated** incremental
  fine-tune (confirmed at 3× weight, RGCN frozen). New endpoint
  `POST /v3/supervisor/confirm-batch`. Recording labels never auto-advances
  training — the "Record + retrain" click is the gate (hard stop #5).
- **Export selected** — bundles a chosen subset as one zip (per-app scorecard
  CSV + reviewer-card HTML + interactive identity-ring HTML + evidence JSON +
  combined manifest). New endpoint `GET /v3/monitoring/export/selected?ids=…`
  and `build_selected_export()` in `src/export_v3.py`.
- **Drift explanation panel** (admin) — plain-English full-retrain rationale from
  existing stats only: overall score-KS p-value vs `DRIFT_KS_THRESHOLD` plus the
  per-feature KS table. New endpoint `GET /v3/monitoring/drift-explain`. Counts
  are scoped to the **44 model features** (the 24 dropped nominal identifiers are
  excluded via `v3_feature_schema.json`), so it reflects what a retrain relearns.

### 2026-07-10 — Review console + nginx front door, MLflow retired

Added a browser **review console** (`frontend/`, vanilla HTML/CSS/JS) served by
an **nginx front door** (`deploy/nginx/`) that also reverse-proxies the API — one
origin at `http://localhost:8080/`, portable unchanged to Kubernetes
(`deploy/k8s/nic-fraud.yaml`, see `deploy/README.md`). Three tabs: review queue
(embeds the XAI cards + a topology detail modal), pattern queue, and a **model
audit & deploy** console (status strip + CSV-upload → evaluate → decide → watch
loop + run history + rollback). **MLflow fully removed** and replaced by a local
registry (`src/model_registry.py` → `outputs/model_registry.json`); new endpoints
`GET /v3/model/stats`, `GET /v3/model/registry`, `POST /v3/monitoring/upload-dataset`.
Fixes: `promote_patterns` job_id/task bug; global NaN/inf→null JSON encoder so no
endpoint 500s on non-finite floats. `docs/OPERATIONS_RUNBOOK.md` rewritten as a
console operator guide; `docs/DEPLOYMENT_PLAN.md` (draft) removed, superseded by
`deploy/README.md`.

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
| Raw database column names (`inst_verify_by`) unreadable by reviewers | Added `FEATURE_LABELS` dict mapping all 44 model columns to human-readable display names (e.g. "Institution verifier code"). `feature_label` field added to every `top_feature_errors` entry. |
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
- **DOMINANT** (Ding et al., SDM 2019): Dual-decoder graph autoencoder for
  anomaly detection. V3 inherits the RGCN encoder architecture (aggr='add',
  tanh bounding) from V2's validated implementation.
- **LOE** (Qiu et al., ICML 2022): Latent Outlier Exposure — synthetic anomaly
  pretraining to push outlier embeddings away from the normal hypersphere.
  V3 applies LOE to the graph stream only (Stage 1).
- **SPOT / EVT** (Siffer et al., KDD 2017): Generalized Pareto Distribution
  tail fitting for threshold-free anomaly flagging. Unchanged from V2.
