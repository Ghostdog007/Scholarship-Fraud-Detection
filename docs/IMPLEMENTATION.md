# IMPLEMENTATION.md — V4-Scale Migration Plan

<!-- VERSION: 2.0 (V4-Scale rewrite, 2026-07-21). Supersedes the V4 locked-detection
     architecture plan, which is complete and summarised in HISTORY.md; pre-rewrite
     text in git history (`git show 9b772de:docs/IMPLEMENTATION.md` on main). -->

Five steps, strictly ordered. Each step is independently shippable, has an
acceptance gate on the 15k dataset, and leaves `main`-equivalent behavior
intact until its cut-over. **`src/` model modules are untouched until step 4.**
Design rationale for everything here: `TECHNICAL_REFERENCE_AND_SCALING.md`
Part II. Contract and hard stops: `AGENTS.md`.

Status legend: ☐ not started · ◐ in progress · ✅ gate passed

---

## Step 0 — Postgres stands up ☐

- `deploy/postgres/schema.sql`: the system-of-record schema
  (`applications`, `identity_keys` (5 indexed identity columns), `features`,
  `scores`, `confirmed_fraud`, `loe_patterns`, `batches`, `evt_thresholds`,
  `training_runs`, `feature_scaling`). Column definitions:
  `TECHNICAL_REFERENCE_AND_SCALING.md` §11.1.
- `src/db/` package: connection pool (`psycopg`), typed query functions,
  versioned migration runner. **All SQL lives here** (hard stop 14).
- docker-compose service + k8s manifest addition (`postgres` pod, local-path
  PV; sizing in §14 of the technical reference).
- Local dev: connect to the lead's local PostgreSQL install (connection
  parameters via env / `.env`, never hardcoded).

**Gate 0:** schema applies cleanly to an empty database twice (idempotent
migrations); `src/db/` round-trips one synthetic row per table.

## Step 1 — Dual-write the JSON stores ☐

- `confirmed_fraud_store.py`, `confirmed_fraud_graph_store.py`,
  `model_registry.py` gain a Postgres backend behind their **existing public
  interfaces**; writes go to both JSON and Postgres, reads stay on JSON.
- No handler or model-module changes.

**Gate 1:** replay the current JSON stores into Postgres; a comparison script
shows field-level parity (every record, every field). JSON remains
authoritative (hard stop 13).

## Step 2 — Serve reads from Postgres ☐

- Ingest the 15k scored outputs into `scores`; the review-queue,
  card-metadata, drift, and stats endpoints read from `src/db/` instead of
  CSV parses. Ego-graph/ring endpoints query `identity_keys` (indexed 1–2 hop
  lookup, technical reference §12.5) instead of loading the `.pt` graph.
- Console untouched; endpoint responses byte-comparable where deterministic.

**Gate 2:** for the full 15k population, every read endpoint returns
identical payloads from the Postgres path and the file path (automated diff
over the queue pages, N sampled cards, topology responses). Then flip reads
to Postgres; files still written.

## Step 3 — Ingestion lands in Postgres ☐

- Console CSV upload → `COPY` into a staging `batches` row
  (`status='staged'`); Evaluate scores it read-only; Decide's Merge flips
  `status='merged'`. **Frontend and endpoint signatures unchanged.**
- Pattern-CSV intake writes `applications` + `loe_patterns` the same way.
- Add the bulk/portal ingestion entry point (same staging path, no console).

**Gate 3:** upload → evaluate → decide flow produces the same console
behavior as on `main` for `frontend/sample_cohort.csv`, with rows landing in
Postgres instead of files.

## Step 4 — Feature engineering + graph at scale ☐

First model-module edits; one module at a time.

- `tabular_feature_engine_v3.py`: cross-row aggregates become SQL
  (window/group queries via `src/db/`), per-row scalars run chunked;
  **scaler params fit once and persisted** to `feature_scaling`
  (hard stop 11), applied — never refit — on scoring batches.
- `graph_builder_v3.py`: edges from SQL self-joins on `identity_keys`,
  **hub-capped** (clique ≤ K_CAP, star above, statistical group-size ceiling
  derived from the observed distribution — hard stop 1).

**Gate 4 (the big parity gate):** on the 15k dataset, the new path reproduces
`engineered_features_v3.csv` bit-for-bit where deterministic (tolerance only
where floating-point summation order differs), and graph degree/count
features match. Any discrepancy halts the step (quantitative claims protocol
#6).

## Step 5 — NeighborLoader training + scale test ☐

- `hybrid_graphmcm_v3.py`: training and scoring loops move to PyG
  `NeighborLoader` mini-batching. Model classes, losses, hyperparameters, and
  score exports unchanged. Fan-out per open decision #2 (ablate magnitude
  *and* shape — symmetric vs front-loaded — on 15k first).
- **Scale test on synthetic ~1M rows before any real 3.5M run:** record
  epochs/hour (this replaces the unvalidated 24–48 h projection), peak RSS,
  and edge counts under the chosen K_CAP.

**Gate 5:** (a) 15k scores from the sampled path match full-graph scores
within the ±0.03–0.04 noise floor (deterministic CPU comparison for anything
tighter); (b) the 1M synthetic run completes within the worker's memory
limit, with measured epochs/hour extrapolating to an acceptable 3.5M retrain
window. Measurements are recorded in `training_runs` (raw stdout preserved),
not restated from memory.

---

## After step 5

Cut-over review with the lead: retire dual-writes (JSON stores become
read-only archives), update `OPERATIONS_RUNBOOK.md` and `deploy/README.md`
for the Postgres-backed stack, and schedule the first real 3.5M ingest plus
the K_CAP profiling query (open decision #1).
