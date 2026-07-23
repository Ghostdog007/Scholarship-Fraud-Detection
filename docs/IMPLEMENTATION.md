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

## Step 0 — Postgres stands up ✅ (gate passed 2026-07-21)

> Gate evidence: `deploy/postgres/schema.sql` applied twice cleanly
> (idempotent) against local PostgreSQL 18.4; round-trip script passed 11/11
> checks (one row per table + 43-dim feature-vector rejection). Local dev DB:
> `nic_fraud` / role `nic_app`, credentials in git-ignored `.env`.

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

## Step 1 — Dual-write the JSON stores ✅ (gate passed 2026-07-21)

- `confirmed_fraud_store.py`, `confirmed_fraud_graph_store.py`,
  `model_registry.py` gain a Postgres backend behind their **existing public
  interfaces**; writes go to both JSON and Postgres, reads stay on JSON.
- No handler or model-module changes.

**Gate 1:** replay the current JSON stores into Postgres; a comparison script
shows field-level parity (every record, every field). JSON remains
authoritative (hard stop 13).

> Gate evidence: replay mirrored 49 confirmed + 1 false-positive + 1 pattern
> + 4 registry runs; field-level comparison passed with zero mismatches
> (feature_vec at float32 precision). Live add→PG-row→remove→PG-delete
> round-trip verified through the store's public API. Schema correction made
> during this step: `application_id` is **TEXT** (real IDs are alphanumeric,
> e.g. `AS202526000000139`); the three mirror tables track the JSON shapes
> verbatim — `deploy/postgres/schema.sql` is the authoritative DDL, and the
> §11.1 draft in TECHNICAL_REFERENCE (BIGINT) is superseded on these points.

## Step 2 — Serve reads from Postgres ✅ (gate passed 2026-07-21)

> Gate evidence: `src/db/ingest.py` mirrored the primary 15k batch
> (applications + identity_keys + features + scores). Parity harness passed:
> (1) full 15k `top-suspicious` ranking identical (score sequence equal;
> tie-groups compared as sets — 14,998 distinct scores); (2) fraud-store
> summary identical; (3) ego-graph neighbourhoods from the indexed
> `identity_keys` join equal the `.pt` graph adjacency for 298 sampled apps
> × 5 relations covering 15,419 edges. Reads flipped to Postgres
> (`NIC_READS_FROM_PG=0` is the escape hatch; handlers fall back to files on
> query failure). Deliberately still file-based, by design: reviewer-card /
> ring / topology **HTML builders** (they render from unchanged output files
> — parity holds trivially; the proven `ego_neighbors()` query replaces the
> `.pt` load at step 4 when the graph becomes hub-capped) and **drift**
> (compares staged cohort files; moves to Postgres with staging in step 3).
> Rerun `python -m src.db.ingest` after each pipeline run until cut-over.

- Ingest the 15k scored outputs into `scores`; the review-queue,
  card-metadata, drift, and stats endpoints read from `src/db/` instead of
  CSV parses. Ego-graph/ring endpoints query `identity_keys` (indexed 1–2 hop
  lookup, technical reference §12.5) instead of loading the `.pt` graph.
- Console untouched; endpoint responses byte-comparable where deterministic.

**Gate 2:** for the full 15k population, every read endpoint returns
identical payloads from the Postgres path and the file path (automated diff
over the queue pages, N sampled cards, topology responses). **Explicitly in
scope: reviewer/explanation cards, 3D identity rings, and ego-graphs must
render identically** — the ego-graph query change (indexed `identity_keys`
lookup replacing the in-memory `.pt` graph) must produce the same
neighbourhoods. Then flip reads to Postgres; files still written.

## Step 3 — Ingestion lands in Postgres ✅ (gate passed 2026-07-21)

> Gate evidence: full lifecycle through the real FastAPI app (TestClient)
> with `frontend/sample_cohort.csv` — 21/21 checks. Upload → staged batch in
> PG with **derived tables empty** (contract holds); Evaluate → identity_keys
> + features + preview scores populated (50/50/50) + drift_p recorded;
> console render parity — cohort queue, beautified reviewer card, 3D ring,
> ego-graph, and zip export all render; merge → status flips permanent and
> **refuses deletion**; remove-cohort endpoint deletes files AND the staged
> PG rows (50 apps + derived). Bulk/portal path = `src.db.ingest.
> stage_raw_csv()` (same staging, no console). Fixed en route:
> `batches.drift_p` REAL→DOUBLE PRECISION (KS p-values reach 1e-85 and
> underflow float32). Caveat: the Decide endpoint's Celery dispatch was not
> exercised in-process (no broker in the test harness); its PG merge logic
> was tested directly (`merge_batch`) and the file-merge code is unchanged.

- **Ingestion contract (lead-set):** senders deliver **raw-schema rows only**
  (the `data_for_ml_model.csv` shape) — no upstream preprocessing, ever. Raw
  lands in `applications` under a staged batch; `identity_keys` / `features` /
  `scores` are populated **per batch, only on admin-triggered actions**
  (Evaluate = read-only, Merge/retrain = permanent). Staged raw data is
  invisible to the model until the human gate fires.
- Console CSV upload → `COPY` into a staging `batches` row
  (`status='staged'`); Evaluate scores it read-only; Decide's Merge flips
  `status='merged'`. **Frontend and endpoint signatures unchanged.**
- Pattern-CSV intake writes `applications` + `loe_patterns` the same way.
- Add the bulk/portal ingestion entry point (same staging path, no console —
  same contract: raw rows in, nothing derived until the admin acts).

- **Cohort delete must clean Postgres too:** `POST /cohort/{name}/delete`
  today removes the staged files (which is where the on-the-fly cards/rings
  come from) plus the uploaded CSV, after an explicit console warning that
  all cohort outputs are discarded. Once cohort rows land in Postgres, the
  same endpoint must also delete that cohort's `applications` / `features` /
  `scores` rows (staged batches only — merged batches are permanent).

**Gate 3:** upload → evaluate → decide flow produces the same console
behavior as on `main` for `frontend/sample_cohort.csv`, with rows landing in
Postgres instead of files. **Full frontend parity is part of this gate:** the
evaluated cohort must render exactly as today — beautified reviewer cards,
3D identity rings, ego-graphs, exports — for cohort applications as well as
the base population (the user-visible console loses nothing).

## Step 4 — Feature engineering + graph at scale ✅ (gate passed 2026-07-21)

> Gate evidence (15k):
> **Features:** `build_base_pg()` (raw from PG JSONB via CSV-round-trip
> reconstruction, aggregates pushed to SQL) reproduces the canonical outputs
> **bit-for-bit — 63/63 nodeg columns and 44/44 final columns exact**, incl.
> the pandas-rank replication ((RANK+(ties−1)/2)/n) and PERCENTILE_CONT
> median. **Scaler persisted** (sklearn's exact scale_/min_ in
> `feature_scaling`); `apply_stored_scaling()` reproduces fit_transform
> within **1 ULP** (max abs diff 1.11e-16 — hex-verified single-rounding/FMA
> context difference in the identical x·scale+min expression; pipeline-path
> bit-parity is the binding check and is exact). Hard-stop-11 enforced:
> missing params raise, never refit.
> **Graph:** `build_graph_pg()` from `identity_keys` groups — degree features
> **5/5 bit-exact**, adjacency **identical on all 5 relations** (317k
> directed edges). **Hub-cap machinery verified but OFF by default** (1,657
> groups starred at k_cap=3 smoke; pincode 104k→10.7k undirected edges);
> K_CAP + ceiling remain open decision #1 (lead-owned, needs 3.5M
> profiling). Outputs land in separate `_pg` files until cut-over (hard
> stop 13); `main_v3.py` still runs the file path.

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

## Step 5 — NeighborLoader training + scale test ✅ (gates 5a + 5b passed 2026-07-21)

> **Fan-out ablation (open decision #2) — resolved by evidence, 2026-07-21**
> (deterministic CPU, frozen checkpoint, 15k, vs full-graph reference):
>
> | fanout | max abs Δ | mean | spearman | top-500 overlap | verdict |
> |---|---|---|---|---|---|
> | (15,15) | 0.414 | 0.020 | 0.957 | 71.0% | FAIL |
> | (25,10) | 0.359 | 0.021 | 0.950 | 73.4% | FAIL |
> | (15,10) | 0.435 | 0.024 | 0.951 | 69.4% | FAIL |
> | (50,50) | 0.241 | 0.004 | 0.997 | 94.4% | FAIL |
> | **(-1,-1) exact 2-hop, batched** | **0.00000** | 0.000 | **1.000** | **100%** | **PASS (exact)** |
>
> Truncating fan-outs are unusable on an uncapped graph (high-degree nodes
> lose most of their neighbourhood). **Exact-neighborhood batching is
> bit-equal to full-graph scoring for the 2-layer RGCN** — a 2-layer GNN
> only ever sees the 2-hop neighbourhood, so batching it exactly changes
> nothing. Paired with the hub-capped graph (bounded degree K_CAP → per-seed
> 2-hop ≤ 1+K+K² nodes), memory is bounded with **zero sampling noise**:
> fan-out magnitude/shape is moot. **Gate 5(a) passed in exact mode.**
> **ADOPTED — lead direction 2026-07-21: exact-neighborhood mode
> (fanout (-1,-1)) is the production sampled path. Open decision #2 closed.**
>
> **Gate 5(b) — synthetic-1M scale test, completed 2026-07-21** (raw stdout
> preserved in session; k_cap=50/ceiling=200 are TEST parameters, not the
> production K_CAP — open decision #1 unaffected). Laptop CPU, 8 threads
> (`torch.set_num_threads(8)`, the server config), exact-neighborhood
> ((-1,-1)) batched training/scoring, hub-capped graph via the production
> `_edges_from_groups`/`derive_group_ceiling` machinery, batch_size=512/2048.
>
> | Measurement | Value |
> |---|---|
> | Population | 1,000,000 nodes, 44 features |
> | Directed edges (hub-capped) | 17,562,098 (5 relations; e.g. pincode 4.92M directed after 1,109 groups starred + 100 ceiling-skipped out of 119,561 groups) |
> | Stage 1 epoch (measured, n=1) | 406 s |
> | Stage 2 epoch (measured, n=2) | 639 s, 550 s (mean 595 s) |
> | Scoring (full 1M) | 56 s |
> | Peak RSS | 3.53 GB |
> | Total wall time (this test) | 31.2 min |
>
> **Full-retrain extrapolation (80 S1 + 120 S2 epochs, config_v3 EPOCHS_STAGE1/2):**
> 80×406s + 120×595s = 28.9 h at 1M → **≈101 h (~4.2 days) at 3.5M** (linear
> row-count scaling). This **exceeds the retired 24–48 h guess by 2–4×** —
> report it as measured, not softened. Caveats, stated plainly:
> (1) laptop CPU, not the production 16 vCPU server — re-measure there before
> committing to a calendar; (2) batch_size=512/threading were not tuned —
> this is a baseline, not an optimized number; (3) edge count may grow
> **faster than linear** in row count as more rows collide per identity
> value at 3.5M — the ×3.5 scaling is an approximation, not a bound.
> **Memory is not the bottleneck**: 3.53 GB at 1M extrapolates to ~12 GB at
> 3.5M, comfortably inside the `nic-worker` 24–40 GB budget (Appendix-G-style
> sizing) — training time, not memory, is the real cut-over question.
> **Action for the lead:** a ~101 h full retrain is workable for a
> once-or-twice-yearly cycle only if scheduled with matching lead time in
> the operational calendar (batch cadence, open decision #5) — flag this
> before committing to a retrain window. Incremental fine-tune (MLP-only, 10
> epochs, RGCN frozen) is unaffected and stays cheap regardless.

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

**2026-07-23 — opt-in Postgres-sourced full retrain (pre-cut-over, proposed pending lead review):**
`main_v3.py`'s `build_base`/`build_graph` steps now branch on
`config_v3.DATA_SOURCE` (`NIC_DATA_SOURCE` env, default unchanged: `"file"`)
to call `build_base_pg()`/`build_graph_pg()` writing to the SAME canonical
paths, instead of the file-based builders — opt-in only, default behavior
for every existing caller is untouched. `POST /v3/training/full` gained a
`data_source` query param (default `"file"`) that threads through the Celery
task to the env var, so a full retrain can be dispatched via API reading
straight from Postgres, with no CSV involved, once data is staged + merged
(Intake -> Evaluate -> Decide, or Pattern queue -> Promote).

Fixed en route: `src/db/features.py:fetch_raw_frame()` previously sourced
its row set/order from the raw CSV's `application_id` list (a leftover
noted in its own docstring — "until cut-over adds an ingest_seq"), so any
batch merged into Postgres *after* the primary 15k would have been silently
dropped from a Postgres-sourced retrain. It now sources rows/order entirely
from Postgres (`ORDER BY batch_id, application_id` over every `status='merged'`
batch); only the static 136-column header still comes from the file (schema
names, not row data). Re-verified bit-exact against the canonical
`engineered_features_v3_nodeg.csv` on the 15k after the fix (`fetch_raw_frame`
row content identical when aligned on `application_id`; `build_base_pg()`
output max abs diff 0.0 aligned the same way — row order differs from the
file's, which is why alignment matters, but content does not).

**Not yet done — flagging, not closing hard stop 13:** this is opt-in, not a
default flip, and the `applications` staging path (`stage_raw_csv`) still
requires a CSV to land the raw rows in Postgres in the first place. A
CSV-free *ingestion* entry point (portal/DB push -> auto-preprocess) is a
separate, not-yet-scoped piece of work — this change only fixes the *read*
side (retrain sourcing) once data is already merged.

**2026-07-23, same day — the CSV-free ingestion entry point above is now
built:** `POST /v3/monitoring/push-dataset` — names a server-side CSV path
(not an inline row payload; deliberately, for scale), stages it in Postgres,
and auto-dispatches Evaluate as a background Celery task. Auto-preprocess
only, as scoped: Merge/retrain remain a separate, human-gated
`POST /v3/training/decision` call. See `README.md` changelog and
`docs/OPERATIONS_RUNBOOK.md` §5c for the operator/integrator steps. Still
requires *some* process to write the CSV to the shared `data/` volume —
this endpoint removes the console/browser step, not the file-write step.
