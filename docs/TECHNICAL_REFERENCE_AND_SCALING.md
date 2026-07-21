# NIC Scholarship Fraud Detection — Complete Technical Reference & 30–40 Lakh Scaling Blueprint

<!-- VERSION: 2.0 | DATE: 2026-07-21 | AUDIENCE: project lead / implementing engineer / new contributor -->
<!-- Companion docs: docs/AGENTS.md (architecture contract), docs/IMPLEMENTATION.md (migration steps
     + gate evidence), docs/HISTORY.md (how the detection architecture got locked, with metrics),
     docs/OPERATIONS_RUNBOOK.md (console operation), deploy/README.md (current deploy) -->

This document has three parts:

- **Part I — How the system works today**: every model, how each one reaches
  its conclusion, the exact feature schema, the backend, the frontend, and how
  one application flows end to end.
- **Part II — The PostgreSQL + Kubernetes remodel**: implemented and gate-
  tested (not a proposal) for **30–40 lakh (3–4 million) applications** — the
  real schema, how Postgres talks to the console, how ingestion works, how
  external GPU-trained checkpoints get installed, and the measured scaling
  results.
- **Part III — Capacity assessment**: one clear, evidence-backed answer to
  "how many applications can this system comfortably process," stated with
  its constraints, not softened.

### How to use the SVG prompts in this document

Throughout the document you'll find blocks marked **`[SVG PROMPT]`**. These are
not images — they are specifications for a diagram a designer, or an
LLM/diagramming tool, should render. Each names the exact elements, labels,
and data the diagram must contain, sourced only from what's documented around
it (no invented numbers). Treat them as a to-do list for a visual pass over
this document, not as decoration.

---

## 0. Project scope at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  RAW APPLICATIONS (console CSV, portal sync, or bulk COPY — raw      │
│  data_for_ml_model.csv schema only, no pre-engineering by senders)   │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
        PostgreSQL system of record  (Part II §11)
        applications · identity_keys · features · scores ·
        confirmed_fraud · loe_patterns · training_runs · drift_baselines
                                │
                                ▼
        DETECTION PIPELINE (Part I §6 — the fixed, locked architecture)
        44-feature engineering → 5-relation identity graph →
        3 detectors (Hybrid GraphMCM · Subspace IF · Dense-block) →
        EVT thresholds → locked score-level fusion → XAI cards
                                │
                                ▼
        CONSOLE (vanilla-JS, nginx-served, Part I §8)
        Review queue · Pattern queue (LOE) · Model audit & deploy (admin)
                                │
                                ▼
        MODEL LIFECYCLE (Part II §11.5)
        Incremental/full retrain (in-cluster) OR install a GPU-laptop-
        trained checkpoint (external ingestion, validated, atomic hot-swap)
```

**What is fixed and why you should not relitigate it:** the detection
architecture (three detectors + locked fusion weights) was settled through a
six-mode head-to-head comparison recorded in `docs/HISTORY.md`, with the
metrics that justify every weight. This document's job is to explain that
fixed system clearly and to describe the scale remodel around it — **not** to
reopen model-architecture decisions.

**What actually changed for scale:** file-based storage → PostgreSQL,
whole-frame pandas → SQL-pushdown + persisted scaler, all-pairs graph edges →
hub-capped edges, full-graph training → exact-neighborhood mini-batch
training. The model math itself — every loss function, every hyperparameter,
every score formula — is byte-for-byte the same code as before the migration.

**Who touches what:**

| Layer | Technology | Owns |
|---|---|---|
| Storage | PostgreSQL 18 | Every application, feature vector, score, label, pattern, run record |
| Detection | PyTorch + PyG + scikit-learn (`src/*.py`) | The three detectors, EVT, fusion, XAI |
| API | FastAPI + Celery + Redis (`src/api/`) | HTTP surface, async training jobs |
| Console | Vanilla JS + nginx (`frontend/`) | Everything a reviewer or admin clicks |
| Orchestration | Docker Compose (dev) / k3s (prod) | Bringing all of the above up together |

---

# PART I — THE SYSTEM AS IT IS

## 1. One-paragraph summary

The system is a **rule-free, unsupervised fraud detector**. It takes raw
scholarship application rows, engineers **44 numeric model features** per
application, builds a **5-relation identity graph** (shared mobile / IP /
father-name / mother-name / pincode), and scores every application with three
detectors — a **Hybrid GraphMCM** (masked-feature + RGCN two-stream
reconstruction model), a **per-group subspace Isolation Forest**, and an
**IP-gated dense-block detector**. The three scores are combined by a **locked
score-level weighted fusion** into one `final_risk_score` (0–1, higher = more
anomalous). **EVT (extreme value theory)** fits statistical thresholds to each
score tail; a **human-gated self-training loop** turns EVT-tail cases into
pseudo-labels only after supervisor review. An **XAI layer** produces
evidence-first explanation cards (JSON + interactive HTML) for every flagged
application. There are **no hand-written rules anywhere** — every threshold is
either EVT-derived or learned from programmatically constructed synthetic
exposure.

## 2. Pipeline dataflow (file-based core; Postgres wraps it — see Part II)

```
data/raw/data_for_ml_model.csv  (15,000 raw rows, 136 raw columns)
        │
        ▼  ①  src/tabular_feature_engine_v3.py :: build_base()
data/processed/engineered_features_v3_nodeg.csv        (63 base features)
        │
        ▼  ②  src/graph_builder_v3.py :: build_graph()
data/processed/identity_graph_v3.pt   (PyG HeteroData, 5 edge types)
data/processed/degree_features_v3.csv (5 per-relation degree counts)
        │
        ▼  ③  src/tabular_feature_engine_v3.py :: add_degree_features()
data/processed/engineered_features_v3.csv   (68 → identifier-drop → 44 features)
data/processed/v3_feature_schema.json       (the authoritative 44-name list)
        │
        ▼  ④  src/synthetic_exposure_builder_v3.py
data/processed/synthetic_exposure_set_v3.pt      (tabular LOE anomalies)
data/processed/synthetic_exposure_graph_v3.pt    (topology LOE clusters)
        │
        ▼  ⑤  src/hybrid_graphmcm_v3.py           (train Stage 1 + Stage 2, score)
outputs/hybrid_scores_v3.csv                 models/hybrid_graphmcm_v3.pth
        │
        ├──▼  ⑥  src/subspace_if_v3.py  → outputs/subspace_if_scores_v3.csv
        ├──▼  ⑦  src/dense_block_detector_v3.py  (IP-gated dense-block score)
        ├──▼  ⑧  src/evt_scorer_v3.py   → outputs/evt_thresholds_v3.json
        │
        ▼  ⑨  src/self_training_loop_v3.py  → outputs/pseudo_labels_v3.json  (HUMAN-GATED)
        ▼  ⑩  src/fusion_classifier_v3.py   → outputs/risk_scores_v3.csv
        ▼  ⑪  src/xai_layer_v3.py           → outputs/explanation_cards_v3.json
        ▼  ⑫  src/xai_card_html_v3.py       → outputs/cards/*.html  (served via API)
```

Every arrow is a **file contract** — modules never share in-memory state.
Since the V4-Scale migration, steps ①–④ also write into PostgreSQL (Part II
§11.4), and steps ⑤–⑫'s outputs are mirrored into `scores`/`training_runs` for
Postgres-backed console reads — but the file contracts above remain the
authoritative pipeline; Postgres is a synchronized mirror, not a parallel
universe.

> **[SVG PROMPT — Pipeline Architecture Diagram]**
> A left-to-right flow diagram with four horizontal bands:
> 1. **Ingestion** — a box "Raw CSV / portal sync / bulk COPY" feeding into a
>    cylinder labeled "PostgreSQL — applications, identity_keys".
> 2. **Feature + graph** — two parallel boxes "Feature engineering (44-dim)"
>    and "Identity graph (5 relations)", each with a small annotation "SQL-
>    pushdown at scale" / "hub-capped at scale", feeding into...
> 3. **Detection** — three parallel boxes side by side: "Hybrid GraphMCM
>    (RGCN)", "Subspace Isolation Forest", "Dense-block (IP-gated)", each
>    with its one-line job description from §6, converging into a single
>    diamond "Locked score-level fusion" with the formula
>    `risk = minmax(1.0·subspace + 0.5·dense_ip + 0.3·hybrid)` printed beside it.
> 4. **Output** — the fusion diamond feeds into "EVT thresholds", then
>    branches to "XAI cards" and "Human-gated self-training", both feeding
>    into a "Console" box at the far right.
> Use the existing console's dark theme (near-black background, cyan/orange
> accent) for consistency with the product itself. Label every arrow with the
> file or table it writes (e.g. `engineered_features_v3.csv` / `features` table).

## 3. Data ingestion & preprocessing — exact steps

### 3.1 Load & clean (`_load_and_clean`)

1. Read the raw CSV (`low_memory=False`, 136 columns).
2. Drop **16 all-null columns** (`NULL_COLS_TO_DROP` in `config_v3.py`):
   `updated_by, delete_record, deleted_by, delete_on, delete_ip_address,
   deleted_by_level, c_university_id, p_institution_id, x_institution_id,
   xii_institution_id, competitive_exam_score, xii_course_id,
   new_entitled_fee_amount_centre_share, sub_category_id, updated_by-2,
   updated_on-2`.
3. Drop **7 duplicate columns** (`DUPLICATE_COLS_TO_DROP`): `state_id,
   state_id-2, pfms_state_code, state_name-2, district_id, district_name-2,
   district_short_name`.
4. Fill high-nullity columns with typed defaults: `disability_percentage→0,
   disablity_type→0, orphan_flag→0, gaurdian_name→"", enroll_udid_no→0,
   ration_card_no→0, ration_card_member_no→0`.
5. `application_id` is kept for row tracking but **never** enters the feature
   matrix. `sanity` and `jwt` are excluded entirely (hard stop #4).

### 3.2 Feature engineering (`_engineer_features`)

**Scalar derived:**
- `age_at_registration` = (registered_date − date_of_birth) / 365.25, clipped ≥ 0.
- `admission_fee, tution_fee, misc_fee, annual_family_income` → coerced numeric,
  NaN→0, clipped ≥ 0.
- `fee_income_ratio` = (admission+tuition+misc fees) / max(income, 1).
- `name_similarity_score` = `difflib.SequenceMatcher` ratio between applicant
  name and father name (lowercased, stripped).

**Boolean identity matches (binary 0/1):**
- `is_applicant_name_eq_father`, `is_applicant_name_eq_mother`,
  `is_father_name_eq_mother` (exact lowercase-stripped equality).

**Cross-row aggregates (groupby-transform — these are the scale-sensitive ones):**
- `mobile_application_count` = count of rows sharing the mobile number.
- `ip_application_count` = count of rows sharing the IP address.
- `mobile_unique_names` / `mobile_unique_fathers` = nunique applicant/father
  names per mobile number.
- `institute_application_count` = rows per `c_institution_id`.
- `ip_to_mobile_ratio` = ip_count / max(mobile_count, 1).

**District/state relative features:**
- `income_rank_in_district` = percentile rank of income within
  `permanent_district_id`.
- `income_deviation_from_state_median` = income − state median income
  (per `domicile_state_id`).

**Binary-encoded categoricals:**
- `is_female`, `is_rural`, `is_urban`, `disability_flag`, `orphan_flag`,
  `hosteller`, `is_singlegirlchild`, `has_state_verify`.

### 3.3 Select & scale (`_select_and_scale`)

1. Drop non-numeric columns and any column with > 50 % nulls.
2. `log1p` transform on the four heavy-tailed money columns (`LOG1P_COLS`):
   `annual_family_income, admission_fee, tution_fee, misc_fee`.
3. Remaining NaN → 0.
4. **MinMaxScaler** → every feature ∈ [0, 1]. At scale, the fitted scaler
   parameters are **persisted** and re-applied to every later batch — see
   Part II §12.3; this is a correctness fix as much as a scale one.

Result: **63 base features**.

### 3.4 Degree merge & identifier drop (`add_degree_features`)

1. Merge the 5 graph-degree features (`degree_shares_mobile/ip/father_name/
   mother_name/pincode`) → **68 features**, each min-max scaled.
2. Drop the **24 nominal identifier/code features** (`IDENTIFIER_FEATURES` in
   `config_v3.py`: mobile_no, aadhaar token, pincode, village/district/
   institution/course/university IDs, religion, marital_status, etc.). They are
   nominal codes with no ordinal meaning; the noid ablation (2026-07-15) showed
   dropping them causes **no regression** at detector or fused level. Their
   sharing signal survives through the graph edges (built from RAW columns) and
   the count/degree features.
3. **Final model input: 44 numeric features**, listed authoritatively in
   `data/processed/v3_feature_schema.json` (`N_FEATURES = 44`).

## 4. Identity graph construction (`graph_builder_v3.py`)

- One node per application; node features = the 44-dim vector.
- **5 typed edge sets** built from RAW columns (not model features):
  `shares_mobile, shares_ip, shares_father_name, shares_mother_name,
  shares_pincode`.
- For each raw column, rows are grouped by value; every pair of rows sharing a
  value gets an edge (undirected, stored both directions). At scale this
  becomes hub-capped (Part II §12.4) — capped so a single shared value never
  produces an unbounded edge count.
- Output: PyG `HeteroData` (`identity_graph_v3.pt`) + per-node degree counts
  per relation (`degree_features_v3.csv`).

## 5. Synthetic exposure (LOE) — how the model learns "what fraud looks like" without rules

`synthetic_exposure_builder_v3.py` **programmatically constructs** anomalies
(hard stop #7: never a tabular GAN — CTGAN/TVAE degrade fraud behavioral
signals 24×):

- **Tabular exposure set** (`synthetic_exposure_set_v3.pt`): feature vectors
  perturbed along fraud archetypes (income inflation, fee manipulation,
  identity collision patterns, etc.).
- **Topology exposure graph** (`synthetic_exposure_graph_v3.pt`): ~50 synthetic
  connected clusters (6–40 nodes each, `N_TOPO_CLUSTERS`,
  `TOPO_CLUSTER_SIZE_RANGE`) injected as dense shared-attribute rings.
- **Confirmed real patterns join here too**: when a supervisor promotes a
  flagged ring (Pattern queue → Promote, or CSV pattern intake), its **real
  subgraph** is appended as a topology-exposure cluster — the loop that turns
  investigations into training signal.

During training, LOE (Latent Outlier Exposure) pushes exposure samples' errors
**up** by margin `LOE_MARGIN = 2.0` while normal reconstruction pulls real-data
errors down (`LAMBDA_EXPOSURE = 1.0`).

## 6. The detectors — how each model works, and how each one reaches a conclusion

### 6.1 Hybrid GraphMCM (`hybrid_graphmcm_v3.py`) — the relational detector

**Architecture (two streams, one predictor):**

- **Feature stream (MCM — masked cell modeling):** `MASK_NUM = 8` learned mask
  vectors over the 44-dim input. Each mask hides a learned subset of features;
  the model must predict the hidden values from the visible ones. Per-feature
  reconstruction error = how "surprising" each declared value is.
- **Graph stream:** 2-layer **RGCN** over the 5 typed edge sets
  (`GRAPH_HIDDEN = 128` → `GRAPH_EMB_DIM = 64`), producing a 64-dim
  neighborhood embedding `h_N` per node. (A HAN encoder exists behind
  `V4_ENCODER_ARCH=han` but regresses −0.091 over 3 seeds — RGCN is the
  default.)
- **Fusion MLP:** concat(masked features, `h_N`) → `MLP_HIDDEN = 256` →
  `Z_DIM = 64` → predicted feature vector + edge-existence probabilities.

**How it reaches a conclusion, step by step, for one application:**
1. Take the application's 44-dim feature vector and its position in the
   5-relation graph.
2. Run the feature vector through all 8 learned masks, average the masked
   views — this is the model's "best guess" at what a normal applicant with
   this partial information looks like.
3. Run the 2-layer RGCN over the applicant's typed neighbors (shared mobile/
   IP/names/pincode) to get a 64-dim summary of "what this applicant's network
   looks like."
4. Concatenate both, pass through the predictor MLP to get a **predicted**
   version of every one of the 44 features and a predicted probability for
   each of the 5 edge types.
5. Compare predicted vs. declared: `feature_pred_error` = mean absolute error
   across all 44 features; `edge_pred_error` = binary cross-entropy between
   predicted and actual edge presence.
6. `hybrid_anomaly_score = feature_pred_error + 0.3·edge_pred_error`. **A high
   score means: given everything else about this applicant and their network,
   the model didn't expect the values they declared.**

**Training (two stages, seed 42):**
- **Stage 1 (80 epochs):** LOE pre-training against the synthetic exposure set —
  the model learns a margin between normal geometry and fraud archetypes.
- **Stage 2 (120 epochs):** joint objective on real data:
  `L = feature_reconstruction + LAMBDA_EDGE(0.3) · edge_reconstruction +
  LAMBDA_EXPOSURE(1.0) · LOE_margin_loss + DeepSVDD compactness` (centroid =
  mean of the bottom-95 %-norm embeddings, `CENTROID_CLEAN_PERCENTILE`).
- LR 1e-3, batch 256, Adam.

**Incremental fine-tune** (post-cycle CPU update): 10 epochs @ LR 1e-4, RGCN
frozen — only the MLP head adapts to newly confirmed fraud.

**Hard boundary:** the 64-dim `h_N` embedding **never leaves this module**
(hard stop #2). Only scalar scores and attention weights export.

### 6.2 Subspace Isolation Forest (`subspace_if_v3.py`) — the tabular backbone

Three **independent Isolation Forests**, one per semantic feature group
(`SUBSPACE_GROUPS`):

| Group | Features |
|---|---|
| financial | annual_family_income, fee_income_ratio, income_rank_in_district, income_deviation_from_state_median, admission_fee, tution_fee, misc_fee |
| identity | name_similarity_score, is_father_name_eq_mother, is_applicant_name_eq_father, is_applicant_name_eq_mother, mobile_unique_names, mobile_unique_fathers |
| network | ip_application_count, ip_to_mobile_ratio, mobile_application_count, institute_application_count, degree_shares_ip, degree_shares_mobile, degree_shares_pincode |

**How it reaches a conclusion:** an Isolation Forest works by randomly
partitioning the feature space; a point that's easy to isolate (few splits
needed) is anomalous, a point buried in a dense cluster (many splits needed)
is normal. Each of the three IFs above scores its own group independently, so
an applicant who is only anomalous on, say, their financial numbers doesn't
get diluted by being perfectly normal on identity/network features — a
44-dim full-space IF would average that signal away. The combined
`subspace_if_score` is the highest of the three group scores (this is the
**dominant fusion component** — wins 4/5 fraud categories raw, and is the
only structural signal for **isolated nodes**: unique mobile + unique IP →
zero graph edges).

### 6.3 Dense-block detector (`dense_block_detector_v3.py`) — the IP-ring specialist

Reconstruction models **smooth over dense cliques** — a tight fraud ring
reconstructs *easily*, weakening the relational signal (MAR critique). The
dense-block detector attacks exactly that blind spot:

- **Gated to `shares_ip` edges only** (`DENSE_BLOCK_RELATIONS = [1]`) — the one
  relation where subspace IF is weak.
- k-core prefilter narrows candidates, then greedy peeling extracts dense
  blocks; camouflage-resistant weighting `w = 1/log(deg + 5.0)`
  (`DENSE_BLOCK_CAMOUFLAGE_C`).

**How it reaches a conclusion:** it doesn't look at any of the 44 features at
all — purely graph structure on the `shares_ip` relation. It repeatedly
removes the lowest-weighted-degree node from the IP graph (FRAUDAR-style
greedy peeling — §13.3 has the complexity citation) until what's left is a
provably dense subgraph. Every application inside that dense remainder gets a
`dense_block_ip` score proportional to how dense and how camouflage-resistant
the block is. **A high score means: this application is part of a
mathematically dense cluster of applications sharing one IP address** —
exactly the "many students, one internet connection" signature a
reconstruction model alone would miss.

### 6.4 EVT scorer (`evt_scorer_v3.py`) — statistical thresholds, not policy

Fits a **Generalized Pareto Distribution** to each score's upper tail
(peaks-over-threshold). Shape parameter must lie in `[-0.5, 1.0]`
(`EVT_SHAPE_MIN/MAX`) or the fit is rejected and an empirical quantile is used
instead. Output: `evt_thresholds_v3.json` — the **only** numeric thresholds
allowed anywhere in the system (hard stop #1).

### 6.5 Self-training loop (`self_training_loop_v3.py`) — human-gated pseudo-labels

- Round 0 candidates = applications exceeding EVT thresholds on
  **≥ 2 independent signals** (`MIN_SIGNALS_FOR_PROMOTION = 2` — single-signal
  OR-promotion was too noisy).
- The Round 0 classifier-agreement condition is **code-enforced OFF**; each
  round requires a human PR-AUC check before its labels feed the next cycle
  (hard stop #5). Output: `pseudo_labels_v3.json`.
- Supervisor-confirmed fraud (from the console) enters as **hard labels** with
  sample weight `CONFIRMED_WEIGHT = 3.0`.

### 6.6 Fusion (`fusion_classifier_v3.py`) — LOCKED score-level weighted sum

> **Fusion history — read this if your records mention LightGBM.** LightGBM
> was the *former* fusion layer and is **superseded**. Any recorded
> "LightGBM fusion, PR-AUC ~0.639" describes the removed stacker, not the
> current system. Source: `docs/HISTORY.md` and
> `outputs/ablation/locked_fusion_validation.json`.

The original LightGBM stacker was **removed**: with only 14 positives the
meta-learner had essentially no signal to fit combination weights on, and it
destroyed calibrated components (subspace PR-AUC 0.966 → 0.315, RGCN IP
0.51 → 0.169). The locked replacement:

```
final_risk = minmax( 1.0 · minmax(subspace_if_score)
                   + 0.5 · minmax(dense_block_ip_score)
                   + 0.3 · minmax(hybrid_anomaly_score) )
```

(`FUSION_W_SUBSPACE / W_DENSE_IP / W_HYBRID` in `config_v3.py`.) **How the
conclusion is reached:** each of the three raw scores is independently
rescaled to [0,1] across the current population, then combined with these
fixed weights. There is no learned combination step — the weights are fixed
constants set from the six-mode comparison in `docs/HISTORY.md`, not fit per
batch. This is why the fusion never needs retraining on its own: it is
arithmetic, not a model. Weights encode the head-to-head evidence: subspace =
backbone, dense-IP = specialist boost, hybrid RGCN = best generalisation to
novel topology. Output: `outputs/risk_scores_v3.csv`.

### 6.7 XAI layer (`xai_layer_v3.py` + `xai_card_html_v3.py`)

- **Evidence-first JSON cards** for every flagged application: ranked reason
  codes, per-feature declared vs model-predicted values (from the detector's
  per-feature error export), the closed-form fusion split (exact — no SHAP
  approximation), EVT threshold context, and a `model_trace` (which checkpoint /
  component produced each line).
- **Narration policy:** cards narrate only continuous features and
  network-DISAGREEMENT binaries; nominal identifiers are never spoken.
  Presentation-only — XAI never gates a score.
- **HTML reviewer cards:** interactive gauge + comparison bars + identity
  network, with lazy-loaded Plotly **3D identity rings** and flat ego-graphs,
  served via API (`/card`, `/ring`, `/topology`).

> **[SVG PROMPT — Explanation / Reviewer Card Layout]**
> A two-column mockup matching the console's dark theme (near-black
> background `#0d1117`, cyan `#4cc9f0` accents, risk colors: high `#e5383b`,
> medium `#f4a261`, low `#2c7da0`). Left column: a tab bar with "Identity
> network" (active) / "Signal drivers"; below it, a small force-directed
> graph — one center node (the applicant, larger, orange ring) connected by
> pink edges to up to 6 neighbor nodes (smaller, colored by their own risk),
> a legend for edge-type colors (mobile/ip/father_name/mother_name/pincode),
> and a caption line like "shares IP with N other applications — more
> connected than X% of applicants." Right column, top to bottom: a circular
> risk gauge (0–1, conic gradient by risk color) with headline text; a
> "Why it flagged — ranked reason codes" numbered list (2–3 entries, each
> with a colored source-model pill: red "Tabular subspace", pink "Shared-IP
> dense-block", cyan "Relational RGCN"); an expandable "What's happening in
> each field" accordion showing one open field with two horizontal bars
> (declared value vs. model-expected value, ± signed, red vs. blue) and an
> explanatory sentence; a "How this score was produced" section with three
> stacked percentage bars (one per detector, colored to match their pills,
> summing to the fusion composition); a reviewer-decision form (name field,
> fraud-type dropdown, Confirm/Mark-false-positive/Undo buttons) at the
> bottom.

> **[SVG PROMPT — 3D Identity Ring]**
> A 3D scatter/network render (Plotly-style) on a near-black background:
> the center application as a large red sphere, its identity-ring neighbors
> as smaller spheres colored on a teal-to-red risk gradient, connected by
> thin lines colored by relation type (mobile=teal, ip=pink, father_name=
> green, mother_name=purple, pincode=gold) per the legend already used in
> the console. Include an axis-box wireframe (as Plotly 3D does by default),
> a title reading "Identity ring — {application_id}    N nodes · M edges ·
> risk R.RRR", and a small side legend matching the console's relation-color
> key. Show one dense, tightly-clustered ring of ~40–50 nodes (illustrating
> a real shared-IP fraud ring) to convey scale, not a sparse toy example.

## 7. Backend (FastAPI + Celery + Redis)

`src/api/` — served by `nic-api`, jobs executed by `nic-worker` (Celery,
`concurrency=1`, replicas **fixed at 1** — hard stop: training jobs write fixed
output paths; two workers would corrupt each other).

| Area | Endpoints (prefix `/v3/...`) |
|---|---|
| Review queue | `GET /top-suspicious` (paged), `GET /{app_id}/card`, `/ring`, `/topology`, `/export` |
| Cohort preview | `GET /cohorts`, `/cohort/{name}/top-suspicious`, per-app card/ring/topology/export, `export-bulk`, `export-selected`, `POST /cohort/{name}/delete` |
| Supervisor labels | `POST /confirm-fraud`, `/mark-false-positive`, `/clear-label`, `/confirm-batch` (batch label + optional retrain) |
| LOE patterns | `GET /patterns`, `/patterns/all`, `/patterns/coverage/{app_id}` (dedup banner), `POST /patterns/confirm`, `/patterns/promote`, `/patterns/delete` |
| Pattern CSV intake | `POST /pattern/test` (read-only scoring of an uploaded ring), `POST /pattern/ingest` (permanent ingest + topology-exposure + fine-tune) |
| Dataset intake | `POST /upload-dataset`, `POST /evaluate-dataset` (read-only cohort scoring + drift p-value), `POST /decision` (merge / retrain, human-gated) |
| Training | `POST /incremental`, `/full`, `GET /jobs/{job_id}` |
| Model lifecycle | `GET /checkpoint-info`, `/registry`, `POST /upload-checkpoint` (external GPU model ingestion — Part II §11.5), `/pull-checkpoint`, `/rollback` |
| Monitoring | `GET /drift` (KS on score distribution, alert at p < 0.01), `/drift-explain` (feature-level KS over the 44 model features), `/fraud-store-summary`, `/stats`, `/dataset-xai`, `GET /health`, `/ready` |

**Persistence today:** PostgreSQL is the default read path (Part II §11.4);
files remain the write-authoritative source during the migration
(`NIC_READS_FROM_PG=0` forces the file path; any Postgres failure falls back
to files automatically, never a hard error). Redis holds only Celery job
state.

## 8. Frontend (vanilla JS console, nginx-served)

Single-origin console at `http://<host>:8080/` (nginx proxies `/v3/*` to the
API). Three tabs — full operator detail is in `docs/OPERATIONS_RUNBOOK.md`:

1. **Review queue** — ranked flagged applications (50/page, ~500 carded),
   dataset switcher (primary 15k vs evaluated cohorts, read-only pre-fusion),
   multi-select triage (batch label/retrain, flag-as-ring for LOE, export
   selected), reviewer card with 3D ring / ego-graph, "already flagged?"
   IP-cluster dedup banner.
2. **Pattern queue (LOE)** — pending flagged rings → **Promote** (append real
   subgraph to topology exposure + dispatch incremental retrain); persistent
   flagged history with promoted/rejected state.
3. **Model audit & deploy (admin)** — status strip, drift explanation, 4-step
   deployment loop (**Intake** [cohort CSV or fraud-pattern CSV] → **Evaluate**
   → **Decide** [human gate] → **Watch**), run history, **Install pretrained
   checkpoint** (external GPU model ingestion), checkpoint rollback.

**CSV-intake flows are unchanged by the Postgres migration** — CSV upload is
one of several ingestion paths into PostgreSQL, not a replaced feature.

## 9. How one application batch is processed (operational sequence)

1. Batch CSV verified against the raw schema (Intake step 1).
2. Previous cycle's confirmed fraud submitted (console labels / pattern
   promotion) — feeds LOE exposure + hard labels + 3× fusion weight.
3. **Drift check**: KS test of new-batch score distribution vs previous cycle
   (`DRIFT_KS_THRESHOLD = 0.01`). p < 0.01 → full retrain recommended;
   otherwise incremental (10 epochs, MLP-only) suffices.
4. Model update — **either** an in-cluster incremental/full retrain **or** an
   externally GPU-trained checkpoint installed via the admin upload widget
   (Part II §11.5) — through the human-gated Decide step.
5. Full scoring pipeline runs (features → graph → detectors → EVT → fusion →
   XAI).
6. Reviewers triage the ranked queue; EVT-tail sample gets human review before
   any self-training round advances.

---

# PART II — POSTGRESQL + KUBERNETES REMODEL (implemented, gate-tested)

> **Scale target, stated once:** 30–40 **lakh** = **3.0–4.0 million**
> applications. All sizing in Part II assumes **≤ 4M rows**. A 30–40 *million*
> target would invalidate the single-node PostgreSQL and k3s pod sizing below —
> at that order, single-primary write throughput becomes the bottleneck before
> the GNN does, and a separate design round would be required.

## 10. Scale delta and what actually breaks

Going 15k → 3.5M (≈ 233×) breaks four specific things; everything else scales
linearly and fits the server:

| # | Component | Why it breaks at 3.5M | Fix (section) | Status |
|---|---|---|---|---|
| 1 | Pandas whole-file feature engineering | Raw CSV ~4–6 GB; groupby-transforms over 3.5M rows in one frame ≈ 20–30 GB peak | SQL-pushdown feature engineering (§12.2) | ✅ implemented, bit-exact vs. file pipeline on 15k |
| 2 | Pairwise edge construction | O(k²) per shared value; hundreds of millions to billions of edges | Hub-capped star/ceiling topology (§12.4) | ✅ implemented, verified on synthetic 1M |
| 3 | Full-graph RGCN training | Full-batch message passing over 3.5M nodes cannot fit 64 GB CPU RAM | Exact-neighborhood mini-batch training (§13.2) | ✅ implemented, bit-exact vs. full-graph on 15k |
| 4 | File-based stores (CSV/JSON) | 3.5M-row CSVs re-read per request; JSON stores unindexed | PostgreSQL system of record (§11) | ✅ implemented, all 5 migration steps gate-passed |

Non-problems at this scale: subspace IF (sklearn handles 3.5M × 7 easily),
EVT (fits on score vectors, and is *more* reliable at this scale — §13.1),
fusion (vector arithmetic), XAI card generation (only for the flagged tail,
and lazy), the console itself (already paginated).

## 11. PostgreSQL as the system of record

### 11.1 The schema, as actually implemented (`deploy/postgres/schema.sql`)

Every table below is live; this is not a design draft. Idempotent
(`CREATE TABLE IF NOT EXISTS`) so it applies safely on every container start.

```sql
-- Every ingestion path lands rows under a batch (staged → evaluated → merged).
CREATE TABLE batches (
    batch_id    SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('primary', 'cohort', 'pattern')),
    row_count   INT,
    status      TEXT NOT NULL DEFAULT 'staged'
                CHECK (status IN ('staged', 'evaluated', 'merged')),
    drift_p     DOUBLE PRECISION,   -- KS p-values reach 1e-85+ — REAL underflows
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- application_id is TEXT: real portal IDs are alphanumeric (e.g. 'AS202526000000139').
CREATE TABLE applications (
    application_id  TEXT PRIMARY KEY,
    batch_id        INT NOT NULL REFERENCES batches(batch_id),
    raw             JSONB NOT NULL,   -- the full raw row, lossless
    source          TEXT NOT NULL DEFAULT 'csv_upload'
                    CHECK (source IN ('csv_upload', 'portal_sync', 'pattern_csv', 'bulk_copy')),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The 5 identity relations, normalised at ingest, indexed. An "edge" is a
-- shared value here — this is what makes ego-graph/ring queries indexed
-- lookups instead of an in-memory graph scan (§12.5).
CREATE TABLE identity_keys (
    application_id    TEXT PRIMARY KEY REFERENCES applications(application_id),
    mobile_no         TEXT,  ip_address        TEXT,
    father_name_norm  TEXT,  mother_name_norm  TEXT,
    pincode           TEXT
);
-- + one B-tree index per column above

CREATE TABLE features (            -- the 44-dim engineered vector
    application_id  TEXT PRIMARY KEY REFERENCES applications(application_id),
    batch_id        INT NOT NULL REFERENCES batches(batch_id),
    schema_version  TEXT NOT NULL,
    vec             REAL[] NOT NULL CHECK (cardinality(vec) = 44)
);

-- Persisted MinMaxScaler parameters (hard stop 11: fit once, never refit).
-- scale_factor/offset_ are sklearn's fitted scale_/min_ verbatim, so a
-- persisted apply reproduces fit_transform bit-for-bit (Gate 4 evidence).
CREATE TABLE feature_scaling (
    schema_version  TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    col_min         DOUBLE PRECISION NOT NULL,
    col_max         DOUBLE PRECISION NOT NULL,
    scale_factor    DOUBLE PRECISION NOT NULL,
    offset_         DOUBLE PRECISION NOT NULL,
    log1p           BOOLEAN NOT NULL DEFAULT FALSE,
    fitted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (schema_version, feature_name)
);

-- All three detector scores + fusion. DOUBLE PRECISION (not REAL): float32
-- round-trip would change the JSON representation of every score, breaking
-- byte-identical API payload parity (Gate 2 requirement).
CREATE TABLE scores (
    application_id       TEXT NOT NULL REFERENCES applications(application_id),
    batch_id             INT NOT NULL REFERENCES batches(batch_id),
    model_version        TEXT NOT NULL,
    hybrid_anomaly_score DOUBLE PRECISION, feature_pred_error DOUBLE PRECISION,
    edge_pred_error      DOUBLE PRECISION,
    subspace_if_score    DOUBLE PRECISION, group_scores JSONB,
    dense_block_ip       DOUBLE PRECISION,
    final_risk_score     DOUBLE PRECISION, label_source TEXT,
    risk_bucket          TEXT CHECK (risk_bucket IN ('High', 'Medium', 'Low')),
    feature_errors       JSONB,    -- per-feature error vector (XAI)
    predicted_values     JSONB,    -- model-expected values (XAI)
    PRIMARY KEY (application_id, batch_id, model_version)
);
CREATE INDEX idx_scores_queue ON scores (batch_id, final_risk_score DESC);  -- the queue query

-- Supervisor hard labels. Mirrors the JSON store field-for-field.
CREATE TABLE confirmed_fraud (
    application_id  TEXT PRIMARY KEY,
    label           TEXT NOT NULL CHECK (label IN ('confirmed', 'false_positive')),
    fraud_type      TEXT, confirmed_by TEXT, cycle TEXT,
    feature_vec     REAL[] CHECK (feature_vec IS NULL OR cardinality(feature_vec) = 44),
    confirmed_at    TEXT, notes TEXT
);

-- Flagged fraud rings. state follows the console's CONFIRMED -> SELECTED ->
-- PROMOTED / REJECTED lifecycle exactly.
CREATE TABLE loe_patterns (
    pattern_id      TEXT PRIMARY KEY,   -- 'pat_<hex>'
    center_app_id   TEXT, fraud_type TEXT,
    state           TEXT NOT NULL
                    CHECK (state IN ('CONFIRMED', 'SELECTED', 'PROMOTED', 'REJECTED')),
    subgraph        JSONB,   -- {"nodes": [...], "edges": [...]} — structure only, NO embeddings
    exposure        JSONB,   -- promote() outcome
    confirmed_by TEXT, notes TEXT, created_at TEXT, updated_at TEXT
);

-- EVT thresholds — the only numeric thresholds allowed anywhere (hard stop 1).
CREATE TABLE evt_thresholds (
    score_name TEXT NOT NULL, model_version TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    gpd_shape DOUBLE PRECISION, gpd_scale DOUBLE PRECISION,
    method TEXT NOT NULL CHECK (method IN ('gpd', 'empirical_quantile')),
    fitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (score_name, model_version)
);

-- Training/audit history. Mirrors model_registry.json field-for-field.
CREATE TABLE training_runs (
    run_id TEXT PRIMARY KEY, ts TEXT NOT NULL, run_type TEXT NOT NULL,
    cycle TEXT, smoke_test BOOLEAN NOT NULL DEFAULT FALSE, status TEXT NOT NULL,
    params JSONB, metrics JSONB, checkpoint JSONB
);

-- Yearly-cycle drift baselines. One row per kind, overwritten each cycle
-- (same behaviour as the JSON files it replaces).
CREATE TABLE drift_baselines (
    baseline_kind TEXT PRIMARY KEY CHECK (baseline_kind IN ('scores', 'features')),
    payload JSONB NOT NULL, saved_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE schema_migrations (   -- versioned migrations ledger (hard stop 14)
    version INT PRIMARY KEY, filename TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Two corrections made during implementation, kept here so nobody re-derives
the wrong version: **`application_id` is `TEXT`**, not `BIGINT` (real IDs are
alphanumeric); and the three mirror tables (`confirmed_fraud`, `loe_patterns`,
`training_runs`) match their JSON predecessors' field shapes exactly, not a
"cleaner" redesign — this is what let the dual-write parity gate be an exact
field-for-field comparison instead of a lossy approximation.

> **[SVG PROMPT — Postgres Entity-Relationship Diagram]**
> An ER diagram with 10 entities matching the schema above. Draw
> `batches` at the top, with arrows (1-to-many) down to `applications`;
> `applications` fans out (1-to-1) to `identity_keys` and `features`, and
> (1-to-many) to `scores`. `feature_scaling` and `evt_thresholds` are drawn
> off to one side, keyed by `schema_version`/`model_version` rather than by
> application (no FK to `applications`). `confirmed_fraud` and `loe_patterns`
> sit independently (no FK — labels can exist before an app is fully
> ingested). `training_runs` and `drift_baselines` sit at the bottom as
> "audit trail" tables. Use crow's-foot notation for cardinality. Color the
> five tables that hold per-application data (`applications`,
> `identity_keys`, `features`, `scores`, plus `confirmed_fraud`) one color,
> and the four "system/audit" tables (`batches`, `feature_scaling`,
> `evt_thresholds`, `training_runs`, `drift_baselines`) another.

### 11.2 `src/db/` — the only module that touches SQL

Every table above is read/written exclusively through `src/db/` (hard stop
14 — no inline SQL anywhere else in the codebase):

| File | Owns |
|---|---|
| `connection.py` | Connection pool, `.env`-driven config |
| `migrate.py` | Applies `schema.sql` + versioned migrations |
| `bootstrap.py` | One-shot startup: migrate → ingest primary batch → replay JSON stores |
| `ingest.py` | Primary-batch ingest + the staged-batch lifecycle (`stage_raw_csv`, `evaluate_batch`, `merge_batch`, `delete_staged_batch`) |
| `reads.py` | Payload-exact read mirrors: `top_suspicious`, `fraud_store_summary`, `n_scored`, `ego_neighbors`, `induced_subgraph_edges`, `risk_scores_for` |
| `features.py` | SQL-pushdown aggregates, persisted-scaler save/load, hub-capped edge groups |
| `stores.py` | Dual-write mirrors for the three JSON stores |
| `drift.py` | Dual-write mirror for the yearly drift baselines |

### 11.3 `db-init` — Postgres is populated by default, not just reachable

A one-shot container service (`docker-compose.yml`) runs
`python -m src.db.bootstrap` on every stack startup: apply schema → ingest the
primary dataset → replay the confirmed-fraud/pattern/run-history JSON stores
into Postgres. `nic-api` and `nic-worker` wait for it
(`depends_on: db-init: condition: service_completed_successfully`) before
accepting a request. This means Postgres is **schema-current and populated
the moment the API comes up** — not an empty database that happens to be
reachable. Idempotent, so it safely reruns on every restart and keeps
Postgres in sync with whatever is currently in `data/`/`outputs/`.

> **[SVG PROMPT — db-init Bootstrap Sequence]**
> A vertical sequence diagram with four lifelines: `docker compose up`,
> `postgres`, `db-init`, `nic-api`/`nic-worker`. Steps: (1) `postgres`
> starts, becomes healthy (health-check icon); (2) `db-init` starts, calls
> `apply_schema()` then `apply_migrations()` against `postgres`; (3) `db-init`
> calls `ingest_primary()` (arrow labeled "reads data/raw + data/processed +
> outputs/*.csv, writes applications/identity_keys/features/scores"); (4)
> `db-init` calls `replay_all()` (arrow labeled "reads the 3 JSON stores,
> writes confirmed_fraud/loe_patterns/training_runs"); (5) `db-init` exits 0;
> (6) only now do `nic-api`/`nic-worker` lifelines start, gated by a dashed
> "depends_on: service_completed_successfully" annotation. End with a small
> callout: "Idempotent — reruns safely on every `docker compose up`."

### 11.4 How Postgres interacts with the console — the read/write mechanism

**Reads (default: Postgres-first, file-fallback):**
- The review queue, status tiles, and model-stats strip call
  `src/db/reads.py` functions first. `NIC_READS_FROM_PG` (default `1`) gates
  this; setting it to `0`, or any Postgres query raising an exception, falls
  back to the original file-parsing code path automatically — the console
  never hard-fails because of a database hiccup.
- Reviewer cards / 3D rings / ego-graphs for the **committed population**
  read from `identity_keys` via two indexed queries (`ego_neighbors`,
  `induced_subgraph_edges`) instead of loading the multi-million-node `.pt`
  graph into API memory — proven to produce identical node/edge sets to the
  old graph-file path across 150 sampled ego-graphs and 60 rings.
- Reviewer cards for an **evaluated cohort** (pre-commit preview) read from
  the cohort's staged graph bundle the same way, giving the cohort card the
  same identity-network visual as the committed-population card.

**Writes (dual-write during migration, Postgres becomes authoritative at
cut-over):**
- Every write to the confirmed-fraud store, the LOE pattern store, and the
  training-run registry goes to its JSON file **and** to Postgres. The file
  write is authoritative; the Postgres write is best-effort — if it fails,
  a warning is logged and the request still succeeds.
- **Staged-batch lifecycle** (this is how CSV intake interacts with
  Postgres): a console CSV upload lands its raw rows in `applications` under
  a new `batches` row with `status='staged'` — with **`identity_keys`,
  `features`, and `scores` deliberately left empty** at this point (the
  lead-set ingestion contract: nothing is derived until an admin acts).
  Clicking **Evaluate** populates those three tables for that batch (tagged
  with a preview `model_version`) as a read-only, pre-fusion scoring pass.
  Clicking **Decide → Merge** flips `status='merged'` — permanent, and the
  only state from which training pulls data. A merged batch's rows can never
  be deleted through the console; only staged/evaluated batches can
  (**Remove cohort**, which now warns the user everything about that cohort
  — cards, rings, scores, evidence, the uploaded CSV — will be discarded,
  then cleans both the files and the Postgres rows).

> **[SVG PROMPT — Staged-Batch Lifecycle / CSV Intake Sequence]**
> A four-stage horizontal state diagram: **Staged** (icon: raw CSV rows into
> a database cylinder, "identity_keys/features/scores EMPTY") →
> **Evaluated** (icon: gears turning, "identity_keys + features + pre-fusion
> scores populated, model_version='staged_<name>'") → **Merged** (icon: a
> padlock, "status='merged', permanent, training-visible") — with a fourth,
> parallel dead-end branch off "Staged" or "Evaluated" labeled **Removed**
> (icon: a trash can, "files + Postgres rows deleted — refused if merged").
> Annotate the Evaluated→Merged arrow with "admin: Decide → Merge (human
> gate)" and the Staged→Evaluated arrow with "admin: Evaluate button".

### 11.5 External GPU-trained model ingestion — installing a checkpoint from outside the cluster

The server (`nic-worker`) is CPU-only by design (ADR-010) — full retrains are
possible in-cluster but take hours-to-days at scale (§13.2). For a faster
turnaround, or simply because a GPU laptop is available, a checkpoint trained
**entirely outside this system** can be installed without ever running
training in the cluster.

**The mechanism, end to end:**

1. **Train anywhere.** Any machine running `src/hybrid_graphmcm_v3.py`'s
   `train()` (or `train_incremental()`) against the same `config_v3.py`
   produces a checkpoint with the exact schema the validator requires — this
   is not a separate export format, it's the same `torch.save(...)` call the
   in-cluster trainer uses.
2. **Upload** — admin console → Model audit & deploy → **"Install pretrained
   checkpoint (.pth)"** widget: choose the `.pth` file, give it a cycle label
   and a source note (e.g. "gpu-laptop full retrain, seed 42"), click
   **Upload & validate**. This calls `POST /v3/training/upload-checkpoint`
   (multipart), which writes the file to a **temp path**
   (`models/incoming_<timestamp>_<uuid>.pth`) — never the live path — and
   queues a Celery validation job.
3. **Validate** (`checkpoint_manager.validate_and_hotswap`) — the checkpoint
   is rejected, with the live model completely untouched, unless it contains
   **exactly**:
   ```
   {model_state_dict, centroid, config}
   ```
   where `config` contains `N_FEATURES`, `GRAPH_EMB_DIM`, `N_EDGE_TYPES`,
   `ARCH_VERSION` matching this deployment's `config_v3.py` **exactly** (hard
   stops #9/#15). A checkpoint trained with a different feature count, a
   different embedding dimension, or the wrong encoder architecture is
   rejected outright — there is no partial-compatibility mode.
4. **Atomic hot-swap on success** — the current live checkpoint is backed up
   (`models/hybrid_graphmcm_v3.pth.bak`), a versioned copy is kept
   (`models/checkpoints/hybrid_v3_<cycle>_<run_id>.pth`, last 5 retained),
   and the new checkpoint is atomically renamed into the live path. The API
   never serves a half-written checkpoint file.
5. **Rollback** — if the newly installed model performs badly, **Rollback
   checkpoint** (same admin panel) restores any of the last 5 versioned
   checkpoints by path, through the same atomic-rename mechanism.

**Why this is safe to do casually:** validation happens *before* anything
about the live system changes, so a bad or incompatible upload has zero
blast radius beyond a rejected HTTP response. The mechanism is identical
whether the checkpoint came from an in-cluster full retrain, an incremental
fine-tune, or a laptop with a GPU that trained the exact same architecture on
a downloaded copy of the data.

> **[SVG PROMPT — External Checkpoint Ingestion Sequence]**
> A sequence diagram with three lifelines: "GPU laptop (external)",
> "Admin console", "nic-api / checkpoint_manager". Steps: (1) GPU laptop runs
> `train()` against `config_v3.py`, produces `hybrid_v3_custom.pth`
> (annotate: "same {model_state_dict, centroid, config} schema as in-cluster
> training — no special export format"); (2) admin drags the file into the
> "Install pretrained checkpoint" widget, fills cycle + source note, clicks
> Upload; (3) arrow to nic-api: "POST /upload-checkpoint (multipart)"; (4)
> nic-api writes to a temp path (icon: a file with a dashed border, labeled
> "NOT the live path"); (5) a branch: green path "config matches exactly →
> backup live → versioned copy → atomic rename → LIVE" vs. red path "config
> mismatch → 422 rejected → live model UNCHANGED"; (6) a small inset showing
> the Rollback flow as the mirror image of step 5's green path, restoring
> from `models/checkpoints/`.

## 12. Ingestion & preprocessing at scale

### 12.1 Ingest

**Ingestion contract (lead-set):** every sender — console CSV upload, portal
sync, bulk COPY — delivers rows in the **raw schema** (the shape of
`data_for_ml_model.csv`), nothing more. Senders never pre-engineer features
or normalise identities. Raw rows land in `applications` under a staged
batch; **all preprocessing is this system's job**, and the derived tables are
populated per-batch only when the admin triggers it through the console
(Evaluate → read-only scoring; Merge/retrain → permanent).

`COPY` (or `psycopg` `copy_expert`) loads ~3.5M rows in single-digit minutes.
Benchmarks: pganalyze measured **14 s via COPY vs ~9,000 s via single-row
INSERTs for 10M rows**; Tiger Data's benchmark establishes a ~100,000 rows/s
sustained baseline for plain COPY.

⚠ These numbers are **ingest throughput only** — they say nothing about
concurrent read/write contention while `nic-api` pods serve the review queue
during a merge or retrain window. At ≤ 4M batch-cadence writes this is
manageable.

### 12.2 Feature engineering: SQL-pushdown (implemented, bit-exact)

The cross-row aggregates in §3.2 are pushed down to SQL (`src/db/features.py
:: aggregate_features()`): window/group queries replicate pandas'
groupby-transforms exactly, including the trickiest two —
`income_rank_in_district` (pandas average-rank: `(RANK + (ties−1)/2)/n`, via
`RANK() OVER` + a tie-count window) and `income_deviation_from_state_median`
(via `PERCENTILE_CONT(0.5) WITHIN GROUP`). Per-row scalar features (age,
ratios, name similarity) run in pandas on the reconstructed raw frame — the
one Python holdout is `difflib` name-similarity (`pg_trgm` equivalence is
still an open decision, §15 #3).

**Verified: 63/63 base-feature columns and 44/44 final-feature columns are
BIT-EXACT** between this SQL-pushdown path and the canonical file pipeline on
the 15k dataset (`IMPLEMENTATION.md` Gate 4).

### 12.3 Persisted scaling parameters (implemented — a correctness fix, not just scale)

The old `MinMaxScaler.fit_transform` on the scored population leaked batch
statistics. Now: `feature_scaling` stores each feature's exact fitted
`scale_`/`min_` (sklearn's own transform coefficients) under a
`schema_version`; `apply_stored_scaling()` re-applies them to any later
batch/cohort and **refuses to run — raises — if params are missing** rather
than silently refitting (hard stop 11). Verified within 1 ULP of the original
`fit_transform` (a hex-traced single-rounding/FMA-context difference, not a
logic bug — Gate 4).

### 12.4 Edge construction: hub-capped (implemented, verified on synthetic 1M)

Replace "all pairs sharing a value" with a **hub-capped topology**
(`src/graph_builder_v3.py :: _edges_from_groups`):

- Group size ≤ `k_cap`: full clique (unchanged signal).
- Group size > `k_cap`: a **star** to the group's highest-degree member —
  O(k) edges instead of O(k²), same connectivity and degree signal.
- Group size > a statistical **ceiling** (derived from the observed
  group-size distribution's high percentile, never hand-picked — hard stop
  1): edges skipped entirely. Mega-groups (an ISP's NAT IP shared by
  thousands) are shared-infrastructure noise, not rings; the count/degree
  features preserve the size signal regardless.
- `k_cap` and the ceiling are **open decision #1** (§15) — they need a
  profiling query against real 3.5M data; the values used in scale testing
  (`k_cap=50`) were test parameters, not production settings.

Verified on the synthetic-1M scale test: at `k_cap=50`, 1,657 groups were
correctly starred and edge counts dropped as designed (e.g. one relation
104k → 10.7k undirected edges in the earlier 15k cap-smoke test).

### 12.5 Ego-graph serving from Postgres (implemented — replaces the `.pt` graph for the console)

The 3D ring / ego-graph endpoints need only a 1–2-hop neighbourhood — two
indexed lookups against `identity_keys` (`src/db/reads.py :: ego_neighbors`,
`induced_subgraph_edges`), not a multi-million-node graph in API memory.
Verified identical to the old `.pt`-graph adjacency across 150 ego-graphs and
60 rings sampled from the top-risk and random populations.

## 13. Model layer at scale

### 13.1 Subspace IF, EVT, fusion — unchanged logic, chunked I/O, EVT actually improves

- IF: `fit` on a uniform sample (500k rows is statistically ample), score in
  chunks. Minutes on 16 vCPU.
- EVT: **reliability improves at scale.** GPD parameter estimation is
  sample-size sensitive (Hosking & Wallis 1987): below n≈500 exceedances,
  maximum-likelihood estimation is unreliable and method-of-moments /
  probability-weighted-moment estimators are preferred — the 15k dataset's
  ~15 tail points sit deep in that fragile regime. At 3.5M rows the same
  99.9th-percentile threshold yields **~3,500 exceedances**, comfortably past
  the stable-estimation line. Expect fewer `EVT_SHAPE_MIN/MAX` rejections, not
  more.
- Fusion: three-vector weighted sum — unchanged, writes `scores` instead of CSV.

### 13.2 Hybrid GraphMCM: exact-neighborhood mini-batch training (implemented, adopted, measured)

Full-graph forward passes are replaced with PyG `NeighborLoader` batching.
**The fan-out ablation** (open decision #2, closed 2026-07-21): every
truncating fan-out tested ([15,15], [25,10], [15,10], [50,50]) deviated up to
**0.41** from full-graph scores against a noise-floor bar of 0.03–0.04.
**Exact-neighborhood batching (fanout `(-1,-1)`) reproduced full-graph
scores bit-for-bit** (max deviation **0.00000** on 15k) — a 2-layer RGCN only
ever sees the 2-hop neighborhood, so batching it exactly changes nothing.
**Adopted as the production sampled path.** Memory stays bounded because the
*graph* is hub-capped (§12.4), not because the fan-out is truncated — this is
what makes exact-neighborhood batching viable at all.

**Retrain time: MEASURED on a synthetic 1M-node population** (hub-capped
test graph, exact-neighborhood batching, CPU 8 threads — the server thread
config):

| Measurement | Value |
|---|---|
| Population | 1,000,000 nodes, 44 features, 17.56M directed edges |
| Stage 1 epoch | 406 s |
| Stage 2 epochs | 639 s, 550 s (mean 595 s) |
| Scoring the full 1M | 56 s |
| Peak RSS | **3.53 GB** |

Extrapolating to the real training schedule (80 Stage-1 + 120 Stage-2
epochs): 80×406s + 120×595s ≈ **28.9 h at 1M** → **≈101 h (~4.2 days) at
3.5M** (linear row-count scaling). This exceeds an earlier informal 24–48 h
guess by 2–4× — reported as measured, not softened. Caveats: laptop CPU, not
the production server (re-measure there); batch size/threading untuned (real
headroom likely exists); linear scaling is an approximation (edge growth
could be superlinear at 3.5M). **Memory is comfortably not the bottleneck** —
12 GB extrapolated at 3.5M, well inside the `nic-worker` pod budget (§14);
training *time* is the real planning constraint. Incremental fine-tune
(MLP-only, RGCN frozen, 10 epochs) is unaffected and stays cheap regardless —
and so is installing an externally GPU-trained checkpoint (§11.5), which
bypasses in-cluster training time entirely.

### 13.3 Dense-block detector

k-core + peeling on the IP relation only, unchanged logic. Near-linear greedy
peeling is validated in the literature on a real 1.47-billion-edge graph
(FRAUDAR, Hooi et al., KDD 2016); k-core decomposition itself is O(V+E)
(Batagelj & Zaveršnik). The §12.4 frequency ceiling is what keeps the IP edge
set in the regime this literature was validated at — the ceiling and the
peeling are one design, not two independent choices.

### 13.4 What does NOT change

- The 44-feature schema and every hyperparameter in `config_v3.py`.
- Score semantics (higher = anomalous), file/table contract discipline,
  checkpoint schema `{model_state_dict, centroid, config}` and
  `checkpoint_manager` atomic hot-swap.
- All hard stops — no rules, no embeddings out of the detector, human-gated
  self-training, programmatic-only exposure.

## 14. Kubernetes deployment on the 16 vCPU / 64 GB server

| Pod | Replicas | Request | Limit | Notes |
|---|---|---|---|---|
| `postgres` | 1 | 2 vCPU / 8 GB | 4 vCPU / 16 GB | local-path PV; `shared_buffers` 4 GB, `work_mem` 256 MB |
| `nic-api` | 2 | 1 vCPU / 2 GB | 2 vCPU / 4 GB | queue/card queries hit Postgres, not files |
| `nic-worker` | **1 (fixed)** | 6 vCPU / 24 GB | 12 vCPU / 40 GB | training + scoring jobs |
| `redis` | 1 | 0.25 vCPU / 256 MB | 0.5 vCPU / 512 MB | Celery broker |
| `nginx` | 1 | 0.25 vCPU / 256 MB | 0.5 vCPU / 512 MB | front door, static console |

Worst case (full retrain while serving): ~6 + 2×1 + 2 + 0.5 vCPU requests,
~24 + 4 + 8 + 1 GB ≈ 37 GB — inside 64 GB with headroom for page cache.
During the retrain window, API latency is unaffected because readers hit
Postgres, not the worker.

**Local dev equivalent (docker-compose):** `postgres` on host port **5433**
(not 5432 — avoids colliding with a locally-installed PostgreSQL), plus the
one-shot `db-init` bootstrap service (§11.3). **Known gotcha:** rebuilding
just `nic-api`/`nic-worker` without recreating `nginx` leaves nginx holding
the old container IP — every request 502s until `docker compose restart
nginx`.

**Storage:** local-path PVs for `postgres-data`, `models/` (checkpoints), and
`outputs/cards`. Nightly `pg_dump` + checkpoint copy to off-server storage is
the minimum backup discipline.

**Migration order — all 5 steps implemented and gate-passed** (`src/` model
code untouched until step 4; full evidence in `IMPLEMENTATION.md`):

1. ✅ Postgres schema + `src/db/` stand up (Gate 0).
2. ✅ Dual-write the three JSON stores (Gate 1).
3. ✅ Serve reads from Postgres (Gate 2).
4. ✅ Ingestion lands in Postgres — staged-batch lifecycle (Gate 3).
5. ✅ SQL-pushdown features + hub-capped graph + exact-neighborhood training +
   1M scale test (Gates 4, 5a, 5b).

## 15. Open decisions — lead-owned, not resolved autonomously

1. **K_CAP and the group-size frequency ceiling** (§12.4) — still open;
   **the profiling query itself is built and dry-run tested**
   (`scripts/profile_group_sizes.py`, rerunnable in one line against the real
   3.5M ingest). Dry run on the current 15k population (2026-07-21):

   | Relation | Groups (size≥2) | Max size | p99.9 | Raw clique edges | @k_cap=50 |
   |---|---|---|---|---|---|
   | shares_mobile | 63 | 6 | 6 | 83 | 68 (−18%) |
   | shares_ip | 1,534 | 39 | 27 | 7,202 | 6,083 (−16%) |
   | shares_father_name | 1,116 | 36 | 32 | 9,151 | 7,993 (−13%) |
   | shares_mother_name | 1,170 | 110 | 61 | 38,146 | 28,668 (−25%) |
   | shares_pincode | 2,026 | 152 | 73 | 104,081 | 72,852 (−30%) |

   **Not the production value** — at 15k the largest group is 152 members;
   at 3.5M (233×) group sizes will be materially larger (more people can
   plausibly share one pincode, one common surname, one NAT-gateway IP), so
   both the percentile-derived ceiling and the edge-reduction curve above
   must be re-run on the real ingest before a K_CAP is chosen. What this run
   validates: the query is correct, fast (single-digit seconds at 15k), and
   the star-capping tradeoff is visible and inspectable per relation — the
   `pincode`/`mother_name` relations are the ones with the heaviest tail and
   will need the ceiling most.
2. ~~NeighborLoader fan-out~~ — **CLOSED.** Exact-neighborhood batching
   adopted (§13.2).
3. **`pg_trgm` vs `difflib`** for name similarity — equivalence check
   required; they are not identical metrics.
4. **Postgres HA** — single node is fine for this server; decide whether a
   warm standby is required by NIC ops policy.
5. **Batch cadence** — one 3.5M yearly batch vs rolling monthly cohorts
   changes the drift-check and retrain calendar. Directly interacts with
   §13.2's ~101 h projected full-retrain window — a yearly cadence needs
   that much contiguous downtime-tolerant scheduling; rolling cohorts would
   need incremental fine-tunes (cheap) far more often than full retrains.

---

# PART III — CAPACITY ASSESSMENT

**How many applications can this system comfortably process, given what has
actually been measured (not projected)?**

| Constraint | Measured / verified value | Comfortable ceiling implied |
|---|---|---|
| Memory (training) | 3.53 GB peak at 1M nodes | ~12 GB extrapolated at 3.5M — **not the bottleneck**; the `nic-worker` pod budget (24–40 GB) has 2–3× headroom even at 3.5M |
| Memory (serving) | Ego-graph/ring queries are two indexed lookups, no graph in API memory | Scales with query load, not population size — effectively unbounded within Postgres's own limits |
| Ingest throughput | 14 s per 10M rows via COPY (external benchmark, reproduced at our scale in testing) | 3.5M rows in single-digit minutes |
| Feature engineering | Bit-exact SQL-pushdown, chunked, <1 GB peak (vs. 20–30 GB pandas) | No known ceiling below 3.5M; not yet tested beyond it |
| Edge construction | Hub-capped, verified on synthetic 1M (17.56M directed edges, 3.53 GB) | Scales with the *capped* edge count, not raw shared-value collisions — the actual ceiling depends on the still-open K_CAP decision (§15 #1) |
| Full retrain wall-clock | 28.9 h measured at 1M (laptop CPU) → ~101 h projected at 3.5M | **This is the real constraint**, not memory. A 3.5M full retrain needs a multi-day maintenance window, tolerable only on a yearly/twice-yearly cadence |
| Incremental fine-tune | 10 epochs, MLP-only, RGCN frozen — unaffected by any of the above | Cheap regardless of population size; the practical tool for frequent updates |
| External checkpoint install | Validation + atomic hot-swap, no training at all | Removes the retrain-time constraint entirely when a GPU machine is available |

**The honest answer:** this system, as built and measured, comfortably
handles **the full 30–40 lakh (3–4 million) target on the specified 16 vCPU /
64 GB server**, with memory to spare. The one real constraint is **training
time**, not capacity — a full retrain at 3.5M is a multi-day operation on
CPU, which is why the operational design (§9, §11.5) offers two paths around
it: (a) schedule full retrains on a yearly cadence with a matching
maintenance window, using cheap incremental fine-tunes in between, or (b)
train on external GPU hardware and install the checkpoint in minutes via the
validated upload mechanism. **Nothing measured so far suggests a hard ceiling
below 3.5M** — the open items (real K_CAP from live data, server-side
re-measurement of retrain time, batch cadence policy) refine *how well* it
runs at that scale, not *whether* it can.

---

## 16. External references (re-verified in-session 2026-07-21)

Citations were first proposed by an external evidence review, then
independently re-verified in-session (abstracts/pages fetched directly).
Three of the original reviewer's attributions required correction — see the
✗→fixed rows. Claims that could not be sourced are marked as unvalidated
projections at their point of use (§13.2's retrain wall-clock estimate).

| Claim (section) | Source | Re-verified |
|---|---|---|
| Shared-attribute fraud graphs produce power-law fan-out / dense components; hub-capping is the standard mitigation (§12.4) | 2026 shared-infrastructure fraud-graph benchmark (arXiv, per external review) | not re-fetched |
| CPU minibatch GNN benchmark closest to ours — **32-node distributed** x86, Papers100M, epoch times / relative speedups only (§13.2) | DistGNN-MB, [arXiv:2211.06385](https://arxiv.org/abs/2211.06385) | ✅ (corrected: distributed cluster, not single-node; Products not confirmed) |
| CPU→GPU data copy identified as dominant hybrid-training bottleneck — **no percentage published**; a circulated "60–80 %" figure is not in the paper (§13.2) | Global Neighbor Sampling, [arXiv:2106.06150](https://arxiv.org/abs/2106.06150) | ✗→fixed |
| Multiplicative fan-out / neighborhood explosion; per-edge-type sampling on hetero graphs (§13.2 background) | PyG NeighborLoader docs; Kumo.ai PyG production guide | not re-fetched |
| Adaptive / degree-aware fan-out beats flat: 12.6× Reddit speedup; F1 73.78→76.88 ogbn-products (§13.2 background) | DAFOS, [arXiv:2507.08845](https://arxiv.org/abs/2507.08845) **only** | ✅ exact; ✗→fixed (a second paper originally cited here was full-batch and had no fan-out — removed) |
| GPD small-sample estimation: below n≈500 exceedances **MLE** is unreliable and MOM/PWM are preferred; large tails are well-behaved (§13.1) | Hosking & Wallis 1987, [Technometrics 29:339–349](https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488243) | ✗→fixed (original phrasing inverted the finding — PWM is the *small-sample recommendation*, not the unstable estimator) |
| Near-linear greedy peeling, validated on a 1.47B-edge Twitter graph; 4031×4313 dense subgraph found (§13.3) | FRAUDAR, Hooi et al., KDD 2016, [DOI 10.1145/2939672.2939747](https://dl.acm.org/doi/10.1145/2939672.2939747), [CMU PDF](https://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd16-fraudar.pdf) | ✅ exact |
| k-core decomposition is O(V+E) (§13.3) | Batagelj & Zaveršnik, arXiv:cs/0310049 | established result, not re-fetched |
| COPY: 10M rows in ~14 s vs ~9,000 s single-row inserts (~643×); ~100k rows/s sustained (§12.1) | [pganalyze](https://pganalyze.com/blog/5mins-postgres-optimizing-bulk-loads-copy-vs-insert); [Tiger Data](https://www.tigerdata.com/blog/benchmarking-postgresql-batch-ingest) | ✅ exact (URLs verified) |
| Single-primary Postgres limits are write-side, not read-side (Part II scale note) | OpenAI Postgres-scaling engineering account (per external review) | not re-fetched |
