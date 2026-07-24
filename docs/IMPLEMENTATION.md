# IMPLEMENTATION.md — V4-Scale Migration Plan

<!-- VERSION: 2.0 (V4-Scale rewrite, 2026-07-21). Supersedes the V4 locked-detection
     architecture plan, which is complete and summarised in HISTORY.md; pre-rewrite
     text in git history (`git show 9b772de:docs/IMPLEMENTATION.md` on main). -->

This plan is five steps, and they're meant to be taken strictly in order.
Each one is independently shippable, each has an acceptance gate measured
against the 15k dataset, and each leaves `main`-equivalent behavior intact
right up until its own cut-over — nothing flips until its gate has actually
passed. **`src/` model modules stay untouched until step 4**; everything
before that is infrastructure underneath the existing code, not a rewrite of
it. If you want the design rationale behind any of this, it's in
`TECHNICAL_REFERENCE_AND_SCALING.md` Part II; the contract and hard stops
that govern it live in `AGENTS.md`.

Status legend: ☐ not started · ◐ in progress · ✅ gate passed

---

## Step 0 — Postgres stands up ✅ (gate passed 2026-07-21)

> Gate evidence: `deploy/postgres/schema.sql` applied twice cleanly
> (idempotent) against local PostgreSQL 18.4; round-trip script passed 11/11
> checks (one row per table + 43-dim feature-vector rejection). Local dev DB:
> `nic_fraud` / role `nic_app`, credentials in git-ignored `.env`.

This step is about giving the system a real system-of-record schema to grow
into. `deploy/postgres/schema.sql` defines it: `applications`,
`identity_keys` (5 indexed identity columns), `features`, `scores`,
`confirmed_fraud`, `loe_patterns`, `batches`, `evt_thresholds`,
`training_runs`, and `feature_scaling`. Column-level definitions for all of
that live in `TECHNICAL_REFERENCE_AND_SCALING.md` §11.1, so they're not
repeated here. Alongside the schema, a new `src/db/` package holds the
connection pool (`psycopg`), typed query functions, and a versioned
migration runner — **all SQL lives here**, per hard stop 14, nowhere else in
the codebase. On the infra side, this step also adds a docker-compose
`postgres` service and the corresponding k8s manifest addition (local-path
PV; sizing details are in §14 of the technical reference). For local dev,
you connect to the lead's local PostgreSQL install, with connection
parameters coming from env / `.env` — never hardcoded.

**Gate 0:** schema applies cleanly to an empty database twice (idempotent
migrations); `src/db/` round-trips one synthetic row per table.

## Step 1 — Dual-write the JSON stores ✅ (gate passed 2026-07-21)

This step teaches the existing JSON stores to also write to Postgres,
without changing anything about how they're used. `confirmed_fraud_store.py`,
`confirmed_fraud_graph_store.py`, and `model_registry.py` each gain a
Postgres backend behind their **existing public interfaces** — writes now go
to both JSON and Postgres, but reads stay on JSON for now, so nothing
downstream even notices the change. No handler or model-module code needed
to change for this.

**Gate 1:** replay the current JSON stores into Postgres; a comparison script
shows field-level parity (every record, every field). JSON remains
authoritative (hard stop 13).

> Gate evidence: replay mirrored 49 confirmed + 1 false-positive + 1 pattern
> + 4 registry runs; field-level comparison passed with zero mismatches
> (feature_vec at float32 precision). Live add→PG-row→remove→PG-delete
> round-trip verified through the store's public API. One schema correction
> came out of this step: `application_id` turned out to need to be **TEXT**,
> since real IDs are alphanumeric (e.g. `AS202526000000139`) — the three
> mirror tables track the JSON shapes verbatim, `deploy/postgres/schema.sql`
> is the authoritative DDL, and the §11.1 draft in TECHNICAL_REFERENCE
> (which had it as BIGINT) is superseded on this point.

## Step 2 — Serve reads from Postgres ✅ (gate passed 2026-07-21)

Here's where reads actually start flowing from Postgres instead of files.
The 15k scored outputs get ingested into `scores`, and the review-queue,
card-metadata, drift, and stats endpoints switch to reading from `src/db/`
instead of parsing CSVs. Ego-graph and ring endpoints move to querying
`identity_keys` directly (an indexed 1–2 hop lookup — see technical
reference §12.5) rather than loading the whole `.pt` graph into memory. The
console itself is untouched, and endpoint responses stay byte-comparable
wherever the underlying computation is deterministic.

**Gate 2:** for the full 15k population, every read endpoint returns
identical payloads from the Postgres path and the file path (automated diff
over the queue pages, N sampled cards, topology responses). **Explicitly in
scope: reviewer/explanation cards, 3D identity rings, and ego-graphs must
render identically** — the ego-graph query change (indexed `identity_keys`
lookup replacing the in-memory `.pt` graph) must produce the same
neighbourhoods. Then flip reads to Postgres; files still written.

> Gate evidence: `src/db/ingest.py` mirrored the primary 15k batch
> (applications + identity_keys + features + scores). The parity harness
> passed on three fronts: (1) full 15k `top-suspicious` ranking came out
> identical — score sequence equal, tie-groups compared as sets, across
> 14,998 distinct scores; (2) the fraud-store summary matched exactly; (3)
> ego-graph neighbourhoods from the indexed `identity_keys` join equalled
> the `.pt` graph adjacency for 298 sampled apps × 5 relations, covering
> 15,419 edges. Reads then flipped to Postgres (`NIC_READS_FROM_PG=0` is the
> escape hatch, and handlers fall back to files on query failure). Two
> things were deliberately left file-based rather than migrated here:
> reviewer-card / ring / topology **HTML builders**, because they render
> from unchanged output files and parity holds trivially there — the proven
> `ego_neighbors()` query is what replaces the `.pt` load at step 4, once
> the graph itself becomes hub-capped — and **drift**, which compares staged
> cohort files and moves to Postgres with staging in step 3. Until cut-over,
> `python -m src.db.ingest` needs to be rerun after each pipeline run.

## Step 3 — Ingestion lands in Postgres ✅ (gate passed 2026-07-21)

The ingestion contract for this step is lead-set and worth stating plainly:
senders deliver **raw-schema rows only** (the `data_for_ml_model.csv`
shape) — there is no upstream preprocessing, ever. Raw data lands in
`applications` under a staged batch, and `identity_keys` / `features` /
`scores` only get populated **per batch, on admin-triggered actions**
(Evaluate is read-only; Merge/retrain is what makes it permanent). In other
words, staged raw data stays invisible to the model until a human gate
fires.

Concretely: console CSV upload does a `COPY` into a staging `batches` row
(`status='staged'`), Evaluate scores it read-only, and Decide's Merge flips
`status='merged'` — with **frontend and endpoint signatures unchanged**
throughout. Pattern-CSV intake writes `applications` + `loe_patterns` the
same way, and the bulk/portal ingestion entry point added in this step
follows the identical staging path, just without a console in front of it —
same contract, raw rows in, nothing derived until the admin acts.

One consequence worth calling out: cohort delete has to clean Postgres too.
`POST /cohort/{name}/delete` today removes the staged files (which is where
the on-the-fly cards/rings are generated from) plus the uploaded CSV, after
an explicit console warning that all cohort outputs are being discarded.
Now that cohort rows also land in Postgres, that same endpoint has to delete
the cohort's `applications` / `features` / `scores` rows too — staged
batches only, since merged batches are permanent and not eligible for
deletion.

**Gate 3:** upload → evaluate → decide flow produces the same console
behavior as on `main` for `frontend/sample_cohort.csv`, with rows landing in
Postgres instead of files. **Full frontend parity is part of this gate:** the
evaluated cohort must render exactly as today — beautified reviewer cards,
3D identity rings, ego-graphs, exports — for cohort applications as well as
the base population (the user-visible console loses nothing).

> Gate evidence: the full lifecycle was run through the real FastAPI app
> (TestClient) with `frontend/sample_cohort.csv`, and all 21/21 checks
> passed. Upload produced a staged batch in PG with **derived tables
> empty** (the contract holds); Evaluate populated identity_keys + features
> + preview scores (50/50/50) and recorded drift_p; console render parity
> held across the cohort queue, the beautified reviewer card, the 3D ring,
> the ego-graph, and zip export; merge flipped status to permanent and
> correctly **refuses deletion**; and the remove-cohort endpoint deleted
> both the files and the staged PG rows (50 apps + derived). The
> bulk/portal path uses `src.db.ingest.stage_raw_csv()` — same staging,
> just no console in front of it. One fix landed en route:
> `batches.drift_p` needed to move from REAL to DOUBLE PRECISION, because KS
> p-values reach as low as 1e-85 and were underflowing float32. One caveat
> worth flagging: the Decide endpoint's Celery dispatch wasn't exercised
> in-process, since there's no broker in the test harness — its PG merge
> logic was instead tested directly via `merge_batch`, and the file-merge
> code itself is unchanged.

## Step 4 — Feature engineering + graph at scale ✅ (gate passed 2026-07-21)

This is the first step that touches model modules, and it's done one module
at a time. `tabular_feature_engine_v3.py` moves its cross-row aggregates
into SQL (window/group queries via `src/db/`), while per-row scalars run
chunked; **scaler params get fit once and persisted** to `feature_scaling`
(hard stop 11), then applied — never refit — on every scoring batch after
that. `graph_builder_v3.py` moves edge construction to SQL self-joins on
`identity_keys`, and gains **hub-capping**: cliques stay cliques up to
`K_CAP`, but beyond that they become a star, with the ceiling itself a
statistical group-size threshold derived from the observed distribution
rather than hand-picked (hard stop 1).

**Gate 4 (the big parity gate):** on the 15k dataset, the new path reproduces
`engineered_features_v3.csv` bit-for-bit where deterministic (tolerance only
where floating-point summation order differs), and graph degree/count
features match. Any discrepancy halts the step (quantitative claims protocol
#6).

> Gate evidence (15k) breaks into two halves. **Features:** `build_base_pg()`
> — which reads raw data from PG JSONB via CSV-round-trip reconstruction,
> with aggregates pushed down to SQL — reproduces the canonical outputs
> **bit-for-bit: 63/63 nodeg columns and 44/44 final columns exact**,
> including the pandas-rank replication ((RANK+(ties−1)/2)/n) and the
> PERCENTILE_CONT median. The **scaler is persisted** using sklearn's exact
> `scale_`/`min_` values in `feature_scaling`, and `apply_stored_scaling()`
> reproduces `fit_transform` within **1 ULP** (max abs diff 1.11e-16 —
> hex-verified as a single-rounding/FMA context difference in the otherwise
> identical `x·scale+min` expression; the pipeline-path bit-parity is the
> check that actually binds, and it's exact). Hard-stop-11 is enforced
> here too: missing params raise rather than silently refit. **Graph:**
> `build_graph_pg()`, built from `identity_keys` groups, reproduces degree
> features **5/5 bit-exact**, with adjacency **identical across all 5
> relations** (317k directed edges). The **hub-cap machinery is verified but
> left OFF by default** — a smoke test at k_cap=3 starred 1,657 groups and
> took pincode from 104k down to 10.7k undirected edges, which shows the
> mechanism works, but `K_CAP` and its ceiling remain open decision #1
> (lead-owned, needs 3.5M profiling before a real value gets picked).
> Outputs land in separate `_pg` files until cut-over, per hard stop 13;
> `main_v3.py` still runs the file path in the meantime.

## Step 5 — NeighborLoader training + scale test ✅ (gates 5a + 5b passed 2026-07-21)

This step has two parts: settling the fan-out question for mini-batch
training, and then proving the whole thing survives contact with a
million-row synthetic population before anyone risks a real 3.5M run.

**Fan-out ablation (open decision #2) — resolved by evidence, 2026-07-21.**
The comparison ran deterministically on CPU, against a frozen checkpoint, on
the 15k dataset, versus the full-graph reference:

| fanout | max abs Δ | mean | spearman | top-500 overlap | verdict |
|---|---|---|---|---|---|
| (15,15) | 0.414 | 0.020 | 0.957 | 71.0% | FAIL |
| (25,10) | 0.359 | 0.021 | 0.950 | 73.4% | FAIL |
| (15,10) | 0.435 | 0.024 | 0.951 | 69.4% | FAIL |
| (50,50) | 0.241 | 0.004 | 0.997 | 94.4% | FAIL |
| **(-1,-1) exact 2-hop, batched** | **0.00000** | 0.000 | **1.000** | **100%** | **PASS (exact)** |

The pattern is clear: truncating fan-outs just doesn't work on an uncapped
graph, because high-degree nodes lose most of their neighbourhood in the
process. **Exact-neighborhood batching, on the other hand, is bit-equal to
full-graph scoring for the 2-layer RGCN** — which makes sense once you
notice that a 2-layer GNN only ever sees the 2-hop neighbourhood in the
first place, so batching it exactly changes nothing about what the model
sees. Pair that with the hub-capped graph (bounded degree K_CAP, so each
seed's 2-hop neighbourhood is bounded at ≤ 1+K+K² nodes) and you get memory
that's bounded with **zero sampling noise** — which means the fan-out
magnitude/shape question that open decision #2 was tracking turns out to be
moot. **Gate 5(a) passed in exact mode.** **ADOPTED — lead direction
2026-07-21: exact-neighborhood mode (fanout (-1,-1)) is the production
sampled path. Open decision #2 is closed.**

**Gate 5(b) — synthetic-1M scale test, completed 2026-07-21** (raw stdout
preserved in session; note that k_cap=50/ceiling=200 were TEST parameters
for this run, not the production K_CAP — open decision #1 is unaffected by
them). The test ran on a laptop CPU, 8 threads (`torch.set_num_threads(8)`,
matching the server config), using exact-neighborhood ((-1,-1)) batched
training/scoring, a hub-capped graph built via the production
`_edges_from_groups`/`derive_group_ceiling` machinery, and batch sizes of
512/2048:

| Measurement | Value |
|---|---|
| Population | 1,000,000 nodes, 44 features |
| Directed edges (hub-capped) | 17,562,098 (5 relations; e.g. pincode 4.92M directed after 1,109 groups starred + 100 ceiling-skipped out of 119,561 groups) |
| Stage 1 epoch (measured, n=1) | 406 s |
| Stage 2 epoch (measured, n=2) | 639 s, 550 s (mean 595 s) |
| Scoring (full 1M) | 56 s |
| Peak RSS | 3.53 GB |
| Total wall time (this test) | 31.2 min |

**Full-retrain extrapolation** (80 S1 + 120 S2 epochs, per
`config_v3` `EPOCHS_STAGE1`/`EPOCHS_STAGE2`): 80×406s + 120×595s comes out
to 28.9 h at 1M, which extrapolates (linear row-count scaling) to
**≈101 h (~4.2 days) at 3.5M**. That **exceeds the retired 24–48 h guess by
2–4×**, and it should be reported as measured, not softened. Three caveats
apply, stated plainly rather than buried: (1) this ran on a laptop CPU, not
the production 16 vCPU server — it needs re-measuring there before anyone
commits to a calendar; (2) `batch_size=512/2048` and threading weren't
tuned, so this is a baseline number, not an optimized one; (3) edge count
may grow **faster than linear** in row count, since more rows collide per
identity value as the population grows to 3.5M — so the ×3.5 scaling is an
approximation, not a bound. On the other hand, **memory is clearly not the
bottleneck**: 3.53 GB at 1M extrapolates to roughly 12 GB at 3.5M, which sits
comfortably inside the `nic-worker` 24–40 GB budget (Appendix-G-style
sizing) — so training time, not memory, is the real cut-over question here.
**Action for the lead:** a ~101 h full retrain is workable for a
once-or-twice-yearly cycle, but only if it's scheduled with matching lead
time in the operational calendar (batch cadence, open decision #5) — this
should be flagged before anyone commits to a retrain window. The
incremental fine-tune path (MLP-only, 10 epochs, RGCN frozen) is unaffected
by any of this and stays cheap regardless.

In terms of the actual code change: `hybrid_graphmcm_v3.py`'s training and
scoring loops move to PyG `NeighborLoader` mini-batching, while model
classes, losses, hyperparameters, and score exports all stay unchanged.
Fan-out follows open decision #2 above (magnitude *and* shape — symmetric
vs front-loaded — were both ablated on 15k first). And critically, **the
scale test on synthetic ~1M rows happened before any real 3.5M run** — it's
what replaces the previously-unvalidated 24–48 h projection with measured
epochs/hour, peak RSS, and edge counts under the chosen K_CAP.

**Gate 5:** (a) 15k scores from the sampled path match full-graph scores
within the ±0.03–0.04 noise floor (deterministic CPU comparison for anything
tighter); (b) the 1M synthetic run completes within the worker's memory
limit, with measured epochs/hour extrapolating to an acceptable 3.5M retrain
window. Measurements are recorded in `training_runs` (raw stdout preserved),
not restated from memory.

---

## After step 5

Once step 5 closes, the next move is a cut-over review with the lead:
retire the dual-writes (the JSON stores become read-only archives), update
`OPERATIONS_RUNBOOK.md` and `deploy/README.md` for the Postgres-backed
stack, and schedule both the first real 3.5M ingest and the K_CAP profiling
query that open decision #1 has been waiting on.

**2026-07-23 — opt-in Postgres-sourced full retrain (pre-cut-over, proposed
pending lead review):** `main_v3.py`'s `build_base`/`build_graph` steps now
branch on `config_v3.DATA_SOURCE` (`NIC_DATA_SOURCE` env, default unchanged:
`"file"`) to call `build_base_pg()`/`build_graph_pg()`, writing to the SAME
canonical paths, instead of the file-based builders. This is opt-in only —
default behavior for every existing caller is untouched. `POST
/v3/training/full` gained a `data_source` query param (default `"file"`)
that threads through the Celery task to the env var, so a full retrain can
now be dispatched via API reading straight from Postgres, with no CSV
involved, once data has been staged and merged (Intake → Evaluate → Decide,
or Pattern queue → Promote).

One bug got fixed en route: `src/db/features.py:fetch_raw_frame()` had
previously sourced its row set/order from the raw CSV's `application_id`
list — a leftover its own docstring flagged as temporary ("until cut-over
adds an ingest_seq") — which meant any batch merged into Postgres *after*
the primary 15k would have been silently dropped from a Postgres-sourced
retrain. It now sources rows/order entirely from Postgres (`ORDER BY
batch_id, application_id` over every `status='merged'` batch); only the
static 136-column header still comes from the file, and that's schema
names, not row data. This was re-verified bit-exact against the canonical
`engineered_features_v3_nodeg.csv` on the 15k after the fix —
`fetch_raw_frame` row content is identical when aligned on
`application_id`, and `build_base_pg()` output has a max abs diff of 0.0
aligned the same way. (Row order differs from the file's, which is why
alignment matters, but content doesn't.)

**Not yet done — flagging, not closing hard stop 13:** this change is
opt-in, not a default flip, and the `applications` staging path
(`stage_raw_csv`) still requires a CSV to land the raw rows in Postgres in
the first place. A CSV-free *ingestion* entry point (portal/DB push →
auto-preprocess) is a separate, not-yet-scoped piece of work — this change
only fixes the *read* side (retrain sourcing) once data is already merged.

**2026-07-23, same day — the CSV-free ingestion entry point above is now
built:** `POST /v3/monitoring/push-dataset` names a server-side CSV path
(not an inline row payload — deliberately, for scale), stages it in
Postgres, and auto-dispatches Evaluate as a background Celery task. This is
auto-preprocess only, as scoped: Merge/retrain remain a separate,
human-gated `POST /v3/training/decision` call. See `README.md` changelog
and `docs/OPERATIONS_RUNBOOK.md` §5c for the operator/integrator steps.
Worth noting: this still requires *some* process to write the CSV to the
shared `data/` volume in the first place — this endpoint removes the
console/browser step, not the file-write step.
