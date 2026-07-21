# NIC Scholarship Fraud Detection — Complete Technical Reference & 30–40 Lakh Scaling Blueprint

<!-- VERSION: 1.0 | DATE: 2026-07-21 | AUDIENCE: project lead / implementing engineer -->
<!-- Companion docs: docs/AGENTS.md (architecture contract), docs/IMPLEMENTATION.md (V4 layers),
     docs/OPERATIONS_RUNBOOK.md (console operation), deploy/README.md (current deploy) -->

This document has two halves:

- **Part I — How the system works today** (15k applications, file-based pipeline):
  every model, every preprocessing step, the exact feature schema, the backend,
  the frontend, and how one application flows end to end.
- **Part II — The remodel for 30–40 lakh (3–4 million) applications** on the
  production Kubernetes server (16 vCPU, 64 GB RAM, Ubuntu 22.04, no GPU) with
  **PostgreSQL as the system of record** for training data, LOE patterns, scores,
  and all ingestion — while the console's CSV-intake features stay exactly as
  they are.

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

## 2. Pipeline dataflow (current, file-based)

```
data/raw/data_for_ml_model.csv  (15,000 raw rows, ~90 raw columns)
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

Every arrow is a **file contract** — modules never share in-memory state. This
is the property that makes the PostgreSQL remodel in Part II low-risk: any file
boundary can be swapped for a table without touching model code.

## 3. Data ingestion & preprocessing — exact steps

### 3.1 Load & clean (`_load_and_clean`)

1. Read the raw CSV (`low_memory=False`, ~90 columns).
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
4. **MinMaxScaler over the whole population** → every feature ∈ [0, 1].
   ⚠ The scaler is fit on the scored population itself (fit_transform). At
   scale, scaler parameters must be **persisted from the training population**
   and re-applied to new batches (Part II §12.3).

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
  value gets an edge (undirected, stored both directions). ⚠ This is
  **O(k²) per shared value** — a group of 1,000 rows sharing one pincode
  produces ~500k edges. Fine at 15k rows; the single biggest scale hazard at
  4M rows (Part II §12.4).
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

## 6. The detectors — how each model works

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

**Training (two stages, seed 42):**
- **Stage 1 (80 epochs):** LOE pre-training against the synthetic exposure set —
  the model learns a margin between normal geometry and fraud archetypes.
- **Stage 2 (120 epochs):** joint objective on real data:
  `L = feature_reconstruction + LAMBDA_EDGE(0.3) · edge_reconstruction +
  LAMBDA_EXPOSURE(1.0) · LOE_margin_loss + DeepSVDD compactness` (centroid =
  mean of the bottom-95 %-norm embeddings, `CENTROID_CLEAN_PERCENTILE`).
- LR 1e-3, batch 256, Adam.

**Score:** `hybrid_anomaly_score = feature_pred_error + 0.3 · edge_pred_error`
(higher = more anomalous). Per-feature error and predicted-value vectors are
exported as JSON columns in `hybrid_scores_v3.csv` — these power the XAI
"declared vs model-expected" bars.

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

Per-group anomaly scores + a combined `subspace_if_score`. Subspacing prevents
a 44-dim full-space IF from diluting a strong 3-feature signal. **This is the
dominant fusion component** — it wins 4/5 fraud categories raw. It is also the
only structural signal for **isolated nodes** (unique mobile + unique IP →
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
- Emits a per-application `dense_block_ip` score. Default ON — it is part of
  the locked architecture (AGENTS.md Appendix H.5).

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
> current system. Source: `docs/AGENTS.md` §H.8 and
> `outputs/ablation/tier_comparison.json`.

The original LightGBM stacker was **removed**: with only 14 positives the
meta-learner had essentially no signal to fit combination weights on, and it
destroyed calibrated components (subspace PR-AUC 0.966 → 0.315, RGCN IP
0.51 → 0.169). The locked replacement (AGENTS.md H.8):

```
final_risk = minmax( 1.0 · minmax(subspace_if_score)
                   + 0.5 · minmax(dense_block_ip_score)
                   + 0.3 · minmax(hybrid_anomaly_score) )
```

(`FUSION_W_SUBSPACE / W_DENSE_IP / W_HYBRID` in `config_v3.py`.) Weights encode
the head-to-head evidence: subspace = backbone, dense-IP = specialist boost,
hybrid RGCN = best generalisation to novel topology. Output:
`outputs/risk_scores_v3.csv`.

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
| Model lifecycle | `GET /checkpoint-info`, `/registry`, `POST /upload-checkpoint`, `/pull-checkpoint`, `/rollback` (all through `checkpoint_manager.py`: temp-path write → validate `{model_state_dict, centroid, config}` → atomic rename; versioned copies in `models/checkpoints/`, keep last 5) |
| Monitoring | `GET /drift` (KS on score distribution, alert at p < 0.01), `/drift-explain` (feature-level KS over the 44 model features), `/fraud-store-summary`, `/stats`, `/dataset-xai`, `GET /health`, `/ready` |

**Persistence today is entirely file-based:** CSVs/JSON under `outputs/` and
`data/`, checkpoints under `models/`, confirmed-fraud + flagged-pattern stores
as JSON (`confirmed_fraud_store.py`, `confirmed_fraud_graph_store.py`),
`model_registry.json` for run history. Redis holds only Celery job state.

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
   → **Decide** [human gate] → **Watch**), run history, checkpoint rollback.

**These CSV-intake flows are kept as-is in Part II** — CSV upload becomes one
of several ingestion paths into PostgreSQL, not a replaced feature.

## 9. How one application batch is processed (operational sequence)

1. Batch CSV verified against the raw schema (Intake step 1).
2. Previous cycle's confirmed fraud submitted (console labels / pattern
   promotion) — feeds LOE exposure + hard labels + 3× fusion weight.
3. **Drift check**: KS test of new-batch score distribution vs previous cycle
   (`DRIFT_KS_THRESHOLD = 0.01`). p < 0.01 → full retrain recommended;
   otherwise incremental (10 epochs, MLP-only) suffices.
4. Model update (incremental or full) through the human-gated Decide step.
5. Full scoring pipeline runs (features → graph → detectors → EVT → fusion →
   XAI).
6. Reviewers triage the ranked queue; EVT-tail sample gets human review before
   any self-training round advances.

---

# PART II — REMODEL FOR 30–40 LAKH APPLICATIONS (PostgreSQL + Kubernetes)

> **Scale target, stated once:** 30–40 **lakh** = **3.0–4.0 million**
> applications. All sizing in Part II assumes **≤ 4M rows**. A 30–40 *million*
> target would invalidate the single-node PostgreSQL and k3s pod sizing below —
> at that order, single-primary write throughput (one-shot ingest + feature +
> score table rewrites) becomes the bottleneck before the GNN does, and a
> separate design round would be required. Do not reuse these numbers for it.

## 10. Scale delta and what actually breaks

Going 15k → 3.5M (≈ 233×) breaks four specific things; everything else scales
linearly and fits the server:

| # | Component | Why it breaks at 3.5M | Fix (section) |
|---|---|---|---|
| 1 | Pandas whole-file feature engineering | Raw CSV ~4–6 GB; groupby-transforms over 3.5M rows in one frame ≈ 20–30 GB peak — collides with the 64 GB budget alongside everything else | SQL-pushdown feature engineering (§12) |
| 2 | Pairwise edge construction | O(k²) per shared value. One pincode shared by 5,000 apps → 12.5M edges. Realistic totals: **hundreds of millions to billions of edges** | Capped star/hub edge topology (§12.4) |
| 3 | Full-graph RGCN training | Full-batch message passing over 3.5M nodes + huge edge set cannot fit 64 GB CPU RAM | Neighbor-sampled mini-batch training (§13.2) |
| 4 | File-based stores (CSV/JSON) | 3.5M-row CSVs re-read per request; JSON stores unindexed; no concurrent access control | **PostgreSQL system of record** (§11) |

Non-problems at this scale: subspace IF (sklearn handles 3.5M × 7 easily),
EVT (fits on score vectors), fusion (vector arithmetic), XAI card generation
(only for the flagged tail, and lazy), the console itself (already paginated).

## 11. PostgreSQL as the system of record

### 11.1 Schema (proposed)

```sql
-- Raw ingested applications: every ingestion path lands here.
CREATE TABLE applications (
    application_id   BIGINT PRIMARY KEY,
    batch_id         INT NOT NULL REFERENCES batches(batch_id),
    -- all raw columns as typed cols (money NUMERIC, dates DATE, ids TEXT) --
    raw              JSONB,          -- lossless original row (audit)
    ingested_at      TIMESTAMPTZ DEFAULT now(),
    source           TEXT            -- 'csv_upload' | 'portal_sync' | 'pattern_csv'
);

-- The 5 identity keys, extracted + normalised at ingest, INDEXED.
-- This replaces pairwise edge building: an "edge" is a shared value here.
CREATE TABLE identity_keys (
    application_id   BIGINT PRIMARY KEY REFERENCES applications,
    mobile_no        TEXT,  ip_address  TEXT,
    father_name_norm TEXT,  mother_name_norm TEXT,
    pincode          TEXT
);
CREATE INDEX ON identity_keys (mobile_no);
CREATE INDEX ON identity_keys (ip_address);
CREATE INDEX ON identity_keys (father_name_norm);
CREATE INDEX ON identity_keys (mother_name_norm);
CREATE INDEX ON identity_keys (pincode);

CREATE TABLE features (            -- the 44-dim engineered vector
    application_id BIGINT PRIMARY KEY REFERENCES applications,
    batch_id       INT NOT NULL,
    schema_version TEXT NOT NULL,   -- ties to v3_feature_schema.json
    vec            REAL[44] NOT NULL
);

CREATE TABLE scores (
    application_id BIGINT NOT NULL REFERENCES applications,
    batch_id       INT NOT NULL,
    model_version  TEXT NOT NULL,           -- checkpoint tag (hard stop #15 config)
    hybrid_anomaly_score  REAL, feature_pred_error REAL, edge_pred_error REAL,
    subspace_if_score REAL, subspace_financial REAL,
    subspace_identity REAL, subspace_network REAL,
    dense_block_ip REAL,
    final_risk_score REAL,
    risk_bucket TEXT,                       -- High/Medium/Low (EVT-derived)
    feature_errors JSONB,                   -- per-feature error (XAI)
    predicted_values JSONB,                 -- model-expected values (XAI)
    PRIMARY KEY (application_id, batch_id, model_version)
);
CREATE INDEX ON scores (batch_id, final_risk_score DESC);  -- the queue query

CREATE TABLE confirmed_fraud (              -- replaces confirmed_fraud_store.py JSON
    application_id BIGINT PRIMARY KEY,
    label          TEXT NOT NULL CHECK (label IN ('confirmed','false_positive')),
    fraud_type     TEXT, confirmed_by TEXT, cycle INT,
    feature_vec    REAL[44], confirmed_at TIMESTAMPTZ DEFAULT now(), notes TEXT
);

CREATE TABLE loe_patterns (                 -- replaces confirmed_fraud_graph_store JSON
    pattern_id   SERIAL PRIMARY KEY,
    state        TEXT NOT NULL CHECK (state IN ('pending','promoted','rejected')),
    fraud_type   TEXT, relation_asserted TEXT,
    member_ids   BIGINT[] NOT NULL,
    edges        JSONB,                     -- extracted real subgraph
    in_loe_exposure BOOLEAN DEFAULT FALSE, exposure_cluster_id INT,
    flagged_by TEXT, flagged_at TIMESTAMPTZ DEFAULT now(),
    promoted_at TIMESTAMPTZ
);

CREATE TABLE batches (
    batch_id SERIAL PRIMARY KEY, name TEXT, kind TEXT,   -- 'primary'|'cohort'|'pattern'
    row_count INT, status TEXT,      -- 'staged'|'evaluated'|'merged'
    drift_p REAL, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE evt_thresholds (score_name TEXT, model_version TEXT,
    threshold REAL, gpd_shape REAL, gpd_scale REAL, method TEXT,
    PRIMARY KEY (score_name, model_version));

CREATE TABLE training_runs (                -- replaces model_registry.json
    run_id UUID PRIMARY KEY, kind TEXT, cycle INT, started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ, status TEXT, metrics JSONB,
    checkpoint_path TEXT, config JSONB
);
```

### 11.2 What each consumer pulls from Postgres

- **Training** pulls `features.vec` (chunked, `ORDER BY application_id`) + edge
  lists derived from `identity_keys` — no CSV reads.
- **LOE / exposure builder** pulls promoted `loe_patterns` (real subgraphs) and
  `confirmed_fraud.feature_vec` alongside its programmatic archetypes.
- **Self-training / fusion** pulls `confirmed_fraud` (hard labels, 3× weight)
  and `scores` from the previous model version.
- **API** serves the review queue straight from
  `scores ORDER BY final_risk_score DESC LIMIT 50 OFFSET n` — no CSV parse per
  request.
- **Drift monitoring** compares `scores` distributions across `batch_id` /
  `model_version` with SQL sampling.
- **Other ingestion means**: anything that can write rows into
  `applications` (portal DB sync job, bulk COPY, API POST) is automatically a
  first-class data source — CSV upload is just one writer among several.

### 11.3 CSV intake stays exactly as-is

The console flow is unchanged: drop a CSV → schema check → Evaluate → Decide.
Under the hood the upload handler does `COPY ... FROM STDIN` into a staging
batch (`batches.kind='cohort'`, `status='staged'`) instead of writing a file.
Evaluate scores the staged batch read-only (writes `scores` rows tagged with a
preview `model_version`); Decide's "Merge" flips `status='merged'`, making the
rows visible to training pulls. Pattern-CSV intake likewise lands in
`applications` + `loe_patterns`. **No frontend change is required** — only the
handler internals behind the same endpoints.

## 12. Ingestion & preprocessing at scale

### 12.1 Ingest

`COPY` (or `psycopg` `copy_expert`) loads ~3.5M rows in single-digit minutes.
Benchmarks: pganalyze measured **14 s via COPY vs ~9,000 s via single-row
INSERTs for 10M rows**; Tiger Data's benchmark establishes a ~100,000 rows/s
sustained baseline for plain COPY; CYBERTEC's PostgreSQL 16 10M-row benchmark
confirms the same COPY-vs-INSERT gap. At ingest time, per-row normalisation
happens once: identity keys lowercased/stripped into `identity_keys`, dates
parsed, money coerced. The 23 null/duplicate column drops become simply "not
selected".

⚠ These numbers are **ingest throughput only** — they say nothing about
concurrent read/write contention while `nic-api` pods serve the review queue
during a merge or retrain window. At ≤ 4M batch-cadence writes this is
manageable; it is the first thing that stops holding if the target ever grows
toward 30–40M (single-primary write throughput is what eventually forces
workloads off Postgres — see the Part II scale note).

### 12.2 Feature engineering: push the aggregates into SQL

All cross-row aggregates in §3.2 are natural SQL window/group queries:

```sql
-- e.g. mobile_application_count + mobile_unique_names in one pass
SELECT application_id,
       COUNT(*)        OVER (PARTITION BY mobile_no)      AS mobile_application_count,
       COUNT(DISTINCT applicant_name) -- via a grouped CTE join
       ...
```

- Per-row scalar features (age, ratios, name similarity, booleans) run in
  chunked Python (`fetchmany` batches of 100k) or as SQL expressions; the
  `difflib` name-similarity is the one per-row Python holdout (parallelise
  with a worker pool, or replace with `pg_trgm` similarity — validate
  equivalence first).
- District/state ranks and medians: `PERCENT_RANK() OVER (PARTITION BY
  district)`, `PERCENTILE_CONT(0.5)` per state.
- Result rows insert into `features` in chunks. Peak Python memory: one chunk,
  not the population — **< 1 GB instead of 20–30 GB**.

### 12.3 Scaling parameters must be persisted (correctness fix, not just scale)

Today's `MinMaxScaler.fit_transform` on the scored population leaks batch
statistics. At scale: fit min/max (and the log1p list) on the training
population, **store them in a `feature_scaling` table keyed by
`schema_version`**, and apply the stored parameters to every subsequent
batch/cohort. This also makes cohort preview scores comparable across batches.

### 12.4 Edge construction: cap the cliques

Replace "all pairs sharing a value" with a **hub-capped topology**:

- For each shared value with k members: if k ≤ K_CAP (e.g. 50), keep the
  clique (k² manageable); if k > K_CAP, connect members in a **star** to the
  highest-degree member (or a virtual hub node), preserving connectivity and
  degree signal at O(k) edges. The degree/count features (which carry most of
  the "how many share this" signal) are computed from raw group sizes anyway,
  so no information is lost to the tabular stream.
- Very-high-k values (an ISP NAT IP shared by 100k apps, a common father name)
  are **noise, not rings** — apply a frequency ceiling (drop edges for values
  above, e.g., the 99.9th percentile group size; keep the count feature). This
  is a structural/statistical cutoff, not a policy rule — derive the ceiling
  from the group-size distribution, not a hand-picked domain number
  (hard stop #1 compliant).
- Edge lists are generated by SQL self-joins on the indexed `identity_keys`
  columns, streamed into the graph builder — never materialised as a pandas
  cross-join.

Expected result: ~10–40M edges instead of billions — well within CPU PyG range.

### 12.5 Ego-graph serving from Postgres (replaces the .pt graph for the console)

The 3D ring / ego-graph endpoints need only a 1–2-hop neighbourhood. That is
two indexed lookups:

```sql
SELECT b.application_id FROM identity_keys a JOIN identity_keys b
  ON (b.mobile_no = a.mobile_no OR b.ip_address = a.ip_address OR ...)
WHERE a.application_id = $1 AND b.application_id <> $1;
```

Milliseconds per card, no 4M-node graph in API memory. (This is ADR-012 in
AGENTS.md Appendix F, now made concrete.)

## 13. Model layer at scale

### 13.1 Subspace IF, EVT, fusion — unchanged logic, chunked I/O

- IF: `fit` on a uniform sample (e.g. 500k rows is statistically ample for
  iForest), `score_samples` over the population in chunks. Minutes on 16 vCPU.
- EVT: operates on score vectors (3.5M floats ≈ 14 MB) — logic unchanged, and
  **reliability improves at this scale**. GPD shape estimation is known to be
  unstable at low exceedance counts: Hosking & Wallis (1987) show measurably
  worse bias/RMSE for probability-weighted-moment GPD estimators below ~500
  exceedances, with empirical stabilization reported above ~150–500. The tail
  sample at a 99.9th-percentile threshold grows from **~15 points (15k rows) to
  ~3,500 (3.5M rows)** — moving shape estimation from a known-unstable regime
  into an established-stable one, which should reduce how often the
  `EVT_SHAPE_MIN/MAX` rejection fires and falls back to empirical quantiles.
- Fusion: three-vector weighted sum — unchanged. Writes `scores` instead of CSV.

### 13.2 Hybrid GraphMCM: neighbor-sampled mini-batch training

The one real model change. Replace full-graph forward passes with **PyG
`NeighborLoader`**: sample e.g. [15, 10] neighbors per layer per relation,
batch 1024 seed nodes. Memory per batch is bounded regardless of graph size;
all losses (masked-feature, edge, LOE margin, DeepSVDD) compute per-batch
already.

**Retrain time: unvalidated projection.** The working estimate is **24–48 h
for a full retrain at 3.5M** on the 16 vCPU server (vs the current 8–16 h at
15k), acceptable for a once-or-twice-yearly cycle. However, **no published
CPU-only benchmark validates this at our node count and architecture.** The
closest primary source, DistGNN-MB (arXiv:2211.06385), benchmarks minibatch
GraphSAGE/GAT on x86 CPU at OGBN-Products/Papers100M scale but reports
relative speedups, not single-node wall-clock hours. Most other NeighborLoader
benchmarks are CPU-sample + GPU-train hybrids where host-to-device copy
consumes 60–80 % of per-batch time (arXiv:2106.06150) — that bottleneck does
not exist on a pure-CPU worker, so their throughput numbers do not transfer.
**Acceptance gate:** the migration-step-5 synthetic-1M test must record
epochs/hour; extrapolate linearly from that measurement before scheduling the
first real 3.5M retrain. Incremental fine-tune (MLP-only, RGCN frozen) stays
cheap: score-relevant subsets only, ~1–3 h (same caveat: confirm on the 1M
test).

**Scoring** (inference) streams the population through the same sampler —
no training graph in memory at once. Score direction, exports, and hard stops
(#2, #3) are untouched.

### 13.3 Dense-block detector

k-core + peeling on the IP relation only. The near-linear claim is sourced:
**FRAUDAR** (Hooi et al., KDD 2016, DOI 10.1145/2939672.2939747) reports
near-linear greedy peeling — priority-tree over vertex degrees giving
logarithmic-time minimum-degree retrieval per removal (the Charikar-2000
densest-subgraph approach) — validated on a real **1.47-billion-edge** Twitter
graph. k-core decomposition itself is **O(V+E)** (Batagelj & Zaveršnik).
**Design dependency, stated explicitly:** the §12.4 frequency ceiling (capping
hub-degree groups before edges are built) is what keeps the IP edge set in the
regime this literature was validated at — the ceiling and the peeling are one
design, not two independent choices. No detector-logic change.

### 13.4 What does NOT change

- The 44-feature schema and every hyperparameter in `config_v3.py`.
- Score semantics (higher = anomalous), file/table contract discipline,
  checkpoint schema `{model_state_dict, centroid, config}` and
  `checkpoint_manager` atomic hot-swap.
- All hard stops — especially: no rules, no embeddings out of the detector,
  human-gated self-training, programmatic-only exposure.

## 14. Kubernetes deployment on the 16 vCPU / 64 GB server

Extends the existing k3s plan (AGENTS.md ADR-011 / Appendix G) with Postgres:

| Pod | Replicas | Request | Limit | Notes |
|---|---|---|---|---|
| `postgres` | 1 | 2 vCPU / 8 GB | 4 vCPU / 16 GB | local-path PV; `shared_buffers` 4 GB, `work_mem` 256 MB (few concurrent queries, big sorts) |
| `nic-api` | 2 | 1 vCPU / 2 GB | 2 vCPU / 4 GB | smaller than before — no CSVs in memory; queue/card queries hit Postgres |
| `nic-worker` | **1 (fixed)** | 6 vCPU / 24 GB | 12 vCPU / 40 GB | training + scoring jobs; `torch.set_num_threads(8)` set in the Celery task wrapper only |
| `redis` | 1 | 0.25 vCPU / 256 MB | 0.5 vCPU / 512 MB | Celery broker |
| `nginx` | 1 | 0.25 vCPU / 256 MB | 0.5 vCPU / 512 MB | front door, static console |

Worst case (full retrain while serving): ~6 + 2×1 + 2 + 0.5 vCPU requests and
~24 + 4 + 8 + 1 GB ≈ 37 GB — inside 64 GB with headroom for page cache (which
Postgres loves). During the retrain window, API latency is unaffected because
readers hit Postgres, not the worker.

**Storage:** local-path PVs for `postgres-data`, `models/` (checkpoints), and
`outputs/cards` (generated HTML). Nightly `pg_dump` + checkpoint copy to
off-server storage is the minimum backup discipline.

**Migration order (each step independently shippable, `src/` model code
untouched until step 5):**

1. Stand up Postgres pod + schema; dual-write confirmed-fraud and pattern
   stores (JSON stays authoritative until parity verified).
2. Move `scores`/queue serving to Postgres (API reads); console unchanged.
3. Move ingestion: CSV upload → staging batch tables; add portal-sync writer.
4. SQL-pushdown feature engineering + persisted scaling params; validate
   feature-vector parity against the file pipeline **on the 15k dataset**
   (bit-for-bit where deterministic, tolerance elsewhere).
5. Hub-capped edge builder + NeighborLoader training path; validate on 15k
   (scores within the ±0.03–0.04 CUDA-noise floor / deterministic CPU run),
   then scale test on synthetic 1M rows before first real 3.5M run.

## 15. Open decisions to settle before implementation

1. **K_CAP and the group-size frequency ceiling** (§12.4) — derive from the
   real 3.5M group-size distribution; needs one profiling query after first
   ingest.
2. **NeighborLoader fan-out — ablate shape, not just magnitude.** [15, 10] is
   only a starting point. Fan-out is multiplicative across layers (a [15, 10]
   fan-out touches up to 150 nodes per seed, and per-edge-type sampling on
   heterogeneous graphs multiplies the same way — PyG NeighborLoader
   semantics), which practically caps depth at 2–3 layers (neighborhood
   explosion). Published evidence favors **front-loaded / adaptive fan-out**
   over flat: DistGNN (arXiv:2104.06700) and DAFOS (arXiv:2507.08845) both use
   uneven, degree-aware per-layer allocation — DAFOS reports a 12.6× speedup
   on Reddit and ogbn-products F1 73.78 % → 76.88 % vs flat fan-out. Ablate
   symmetric ([15, 15]) vs front-loaded ([25, 10]) on the 15k set against
   full-graph scores before trusting either at scale.
3. **`pg_trgm` vs `difflib`** for name similarity — equivalence check required;
   they are not identical metrics.
4. **Postgres HA** — single node is fine for this server; decide whether a
   warm standby is required by NIC ops policy.
5. **Batch cadence** — one 3.5M yearly batch vs rolling monthly cohorts changes
   the drift-check and retrain calendar (OPERATIONS_RUNBOOK §6 needs a rev
   once decided).

---

## 16. External references (verified 2026-07-21, external-review pass)

Citations below were located and link-verified during the external evidence
review of this document. Claims that could **not** be sourced are marked as
unvalidated projections at their point of use (notably the §13.2 retrain
wall-clock estimate).

| Claim (section) | Source |
|---|---|
| Shared-attribute fraud graphs produce power-law fan-out / dense components; hub-capping is the standard mitigation (§12.4) | 2026 shared-infrastructure fraud-graph benchmark (arXiv) |
| CPU minibatch GNN training at scale — closest available benchmark; reports relative speedups only (§13.2) | DistGNN-MB, arXiv:2211.06385 |
| CPU→GPU copy = 60–80 % of per-batch time in hybrid setups; does not transfer to pure-CPU (§13.2) | Global Neighbor Sampling, arXiv:2106.06150 |
| Multiplicative fan-out / neighborhood explosion; per-edge-type sampling on hetero graphs (§15 #2) | PyG NeighborLoader docs; Kumo.ai PyG production guide |
| Front-loaded / adaptive fan-out beats flat (12.6× speedup; F1 73.78→76.88 on ogbn-products) (§15 #2) | DistGNN arXiv:2104.06700; DAFOS arXiv:2507.08845 |
| GPD shape-estimator instability below ~500 exceedances; stabilization above ~150–500 (§13.1) | Hosking & Wallis 1987 (via ScienceDirect S0167947303000872); US8175830 |
| Near-linear greedy peeling, validated at 1.47B edges (§13.3) | FRAUDAR, Hooi et al., KDD 2016, DOI 10.1145/2939672.2939747 |
| k-core decomposition is O(V+E) (§13.3) | Batagelj & Zaveršnik |
| COPY: 10M rows in ~14 s vs ~9,000 s row-by-row; ~100k rows/s sustained (§12.1) | pganalyze benchmark; Tiger Data benchmark; CYBERTEC PostgreSQL 16 benchmark |
| Single-primary Postgres limits are write-side, not read-side (Part II scale note, §12.1) | OpenAI Postgres-scaling engineering account |
