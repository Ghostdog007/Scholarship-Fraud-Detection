-- NIC Fraud Detection — PostgreSQL system-of-record schema
-- Source design: docs/TECHNICAL_REFERENCE_AND_SCALING.md §11.1
-- Idempotent: safe to apply repeatedly (Gate 0 requirement, IMPLEMENTATION.md step 0).
-- All access goes through src/db/ (hard stop 14). No embedding columns, ever (hard stop 2).

-- ── Batches: every ingestion path lands rows under a batch ────────────────────
CREATE TABLE IF NOT EXISTS batches (
    batch_id    SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('primary', 'cohort', 'pattern')),
    row_count   INT,
    status      TEXT NOT NULL DEFAULT 'staged'
                CHECK (status IN ('staged', 'evaluated', 'merged')),
    drift_p     REAL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Raw ingested applications ─────────────────────────────────────────────────
-- application_id is TEXT: real portal IDs are alphanumeric (e.g.
-- 'AS202526000000139'). Corrected 2026-07-21 from the BIGINT in the original
-- TECHNICAL_REFERENCE §11.1 draft.
-- Typed raw columns are added at step 3 (ingestion) when the CSV schema is
-- bound; until then `raw` JSONB is the lossless record.
CREATE TABLE IF NOT EXISTS applications (
    application_id  TEXT PRIMARY KEY,
    batch_id        INT NOT NULL REFERENCES batches(batch_id),
    raw             JSONB NOT NULL,
    source          TEXT NOT NULL DEFAULT 'csv_upload'
                    CHECK (source IN ('csv_upload', 'portal_sync', 'pattern_csv', 'bulk_copy')),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_applications_batch ON applications (batch_id);

-- ── Identity keys: the 5 relations, normalised at ingest, indexed ─────────────
-- An "edge" is a shared value here; ego-graphs are indexed lookups (§12.5).
CREATE TABLE IF NOT EXISTS identity_keys (
    application_id    TEXT PRIMARY KEY REFERENCES applications(application_id),
    mobile_no         TEXT,
    ip_address        TEXT,
    father_name_norm  TEXT,
    mother_name_norm  TEXT,
    pincode           TEXT
);
CREATE INDEX IF NOT EXISTS idx_ik_mobile  ON identity_keys (mobile_no);
CREATE INDEX IF NOT EXISTS idx_ik_ip      ON identity_keys (ip_address);
CREATE INDEX IF NOT EXISTS idx_ik_father  ON identity_keys (father_name_norm);
CREATE INDEX IF NOT EXISTS idx_ik_mother  ON identity_keys (mother_name_norm);
CREATE INDEX IF NOT EXISTS idx_ik_pincode ON identity_keys (pincode);

-- ── Engineered features (the 44-dim model vector) ─────────────────────────────
CREATE TABLE IF NOT EXISTS features (
    application_id  TEXT PRIMARY KEY REFERENCES applications(application_id),
    batch_id        INT NOT NULL REFERENCES batches(batch_id),
    schema_version  TEXT NOT NULL,
    vec             REAL[] NOT NULL CHECK (cardinality(vec) = 44)
);
CREATE INDEX IF NOT EXISTS idx_features_batch ON features (batch_id);

-- ── Persisted scaling parameters (hard stop 11: fit once, never refit) ────────
CREATE TABLE IF NOT EXISTS feature_scaling (
    schema_version  TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    col_min         DOUBLE PRECISION NOT NULL,
    col_max         DOUBLE PRECISION NOT NULL,
    log1p           BOOLEAN NOT NULL DEFAULT FALSE,
    fitted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (schema_version, feature_name)
);

-- ── Scores (all components + fusion; higher = more anomalous, hard stop 3) ────
-- DOUBLE PRECISION (not REAL): the file outputs are float64 CSVs, and Gate 2
-- requires byte-identical endpoint payloads — float32 round-trip would change
-- the JSON representation of every score.
CREATE TABLE IF NOT EXISTS scores (
    application_id      TEXT NOT NULL REFERENCES applications(application_id),
    batch_id            INT NOT NULL REFERENCES batches(batch_id),
    model_version       TEXT NOT NULL,
    hybrid_anomaly_score DOUBLE PRECISION,
    feature_pred_error  DOUBLE PRECISION,
    edge_pred_error     DOUBLE PRECISION,
    subspace_if_score   DOUBLE PRECISION,
    group_scores        JSONB,   -- per-group subspace IF scores
    dense_block_ip      DOUBLE PRECISION,
    final_risk_score    DOUBLE PRECISION,
    label_source        TEXT,    -- risk_scores_v3.csv provenance column
    risk_bucket         TEXT CHECK (risk_bucket IN ('High', 'Medium', 'Low')),
    feature_errors      JSONB,   -- per-feature error vector (XAI)
    predicted_values    JSONB,   -- model-expected values (XAI)
    PRIMARY KEY (application_id, batch_id, model_version)
);
-- The review-queue query: top-N by risk within a batch.
CREATE INDEX IF NOT EXISTS idx_scores_queue
    ON scores (batch_id, final_risk_score DESC);

-- ── Confirmed fraud / false positives (supervisor hard labels) ────────────────
-- Mirrors data/processed/confirmed_fraud.json field-for-field (Gate 1 parity).
-- No FK to applications: labels exist for apps not yet ingested (until step 2).
-- cycle is TEXT (e.g. '2026H2'); confirmed_at is the store's date string.
CREATE TABLE IF NOT EXISTS confirmed_fraud (
    application_id  TEXT PRIMARY KEY,
    label           TEXT NOT NULL CHECK (label IN ('confirmed', 'false_positive')),
    fraud_type      TEXT,
    confirmed_by    TEXT,
    cycle           TEXT,
    feature_vec     REAL[] CHECK (feature_vec IS NULL OR cardinality(feature_vec) = 44),
    confirmed_at    TEXT,
    notes           TEXT
);

-- ── LOE fraud patterns (flagged rings; promoted -> topology exposure) ─────────
-- Mirrors data/processed/confirmed_fraud_graph_store.json field-for-field
-- (Gate 1 parity): pattern_id 'pat_<hex>', states are the store's lifecycle
-- (CONFIRMED -> SELECTED -> PROMOTED / REJECTED), subgraph + exposure JSONB.
CREATE TABLE IF NOT EXISTS loe_patterns (
    pattern_id      TEXT PRIMARY KEY,
    center_app_id   TEXT,
    fraud_type      TEXT,
    state           TEXT NOT NULL
                    CHECK (state IN ('CONFIRMED', 'SELECTED', 'PROMOTED', 'REJECTED')),
    subgraph        JSONB,   -- {"nodes": [...], "edges": [...]} — structure only, no embeddings
    exposure        JSONB,   -- promote() outcome: {"appended": bool, ...}
    confirmed_by    TEXT,
    notes           TEXT,
    created_at      TEXT,    -- store's ISO string, mirrored verbatim
    updated_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_loe_state ON loe_patterns (state);

-- ── EVT thresholds (the only numeric thresholds allowed, hard stop 1) ─────────
CREATE TABLE IF NOT EXISTS evt_thresholds (
    score_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    gpd_shape       DOUBLE PRECISION,
    gpd_scale       DOUBLE PRECISION,
    method          TEXT NOT NULL CHECK (method IN ('gpd', 'empirical_quantile')),
    fitted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (score_name, model_version)
);

-- ── Training runs (audit trail; replaces model_registry.json at cut-over) ─────
-- Mirrors outputs/model_registry.json run records field-for-field (Gate 1
-- parity): run_id is the registry's 12-char hex, run_type/cycle are free TEXT,
-- params/metrics/checkpoint are JSONB.
CREATE TABLE IF NOT EXISTS training_runs (
    run_id      TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,      -- registry's ISO timestamp, mirrored verbatim
    run_type    TEXT NOT NULL,
    cycle       TEXT,
    smoke_test  BOOLEAN NOT NULL DEFAULT FALSE,
    status      TEXT NOT NULL,
    params      JSONB,
    metrics     JSONB,
    checkpoint  JSONB
);

-- ── Schema migrations ledger (versioned migrations, hard stop 14) ─────────────
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INT PRIMARY KEY,
    filename    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
