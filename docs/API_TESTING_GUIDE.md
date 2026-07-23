# NIC Fraud Detection V3 — API Testing Guide

**Base URL:** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs` (run every endpoint interactively from the browser)

> **PowerShell note:** `curl` in PowerShell is an alias for `Invoke-WebRequest`. Always use `curl.exe` for curl commands, or use `Invoke-RestMethod` for JSON POST requests.

---

## 0. Start the Stack

```powershell
# First time (builds the Docker image — takes ~5 min)
docker compose up --build

# Every time after that
docker compose up -d

# Check all 3 containers are running
docker ps

# Stop everything when done
docker compose down
```

Expected containers:
| Name | Role |
|---|---|
| `nicfrauddetectionproject-redis-1` | Message broker |
| `nicfrauddetectionproject-nic-api-1` | FastAPI server (port 8000) |
| `nicfrauddetectionproject-nic-worker-1` | Celery worker (runs training jobs) |
| `nicfrauddetectionproject-nginx-1` | Front door — review console + API proxy (port 8080) |

> The review console (UI) is at `http://localhost:8080/`. Every endpoint below
> is also reachable through nginx at `http://localhost:8080/...` (same paths) —
> the direct `:8000` URLs here are for backend debugging.

---

## 1. Health Endpoints

### 1.1 Liveness — is the server up?

```powershell
curl.exe -s http://localhost:8000/health
```

**Expected:**
```json
{"status": "ok", "checkpoint_exists": true}
```

`checkpoint_exists: false` means `models/hybrid_graphmcm_v3.pth` is missing from the volume mount.

---

### 1.2 Readiness — do scored outputs exist?

```powershell
curl.exe -s http://localhost:8000/ready
```

**Expected (after a pipeline run):**
```json
{"status": "ready", "scores_exist": true}
```

**Expected (fresh install, no outputs yet):**
```json
{"status": "not_ready", "scores_exist": false}
```

---

## 2. Monitoring Endpoints

### 2.1 Fraud store summary

```powershell
curl.exe -s http://localhost:8000/v3/monitoring/fraud-store-summary
```

**Expected (fresh store):**
```json
{"n_confirmed": 0, "n_false_positives": 0, "by_fraud_type": {}}
```

**Expected (after submissions):**
```json
{"n_confirmed": 3, "n_false_positives": 1, "by_fraud_type": {"IP_CLUSTER": 2, "FEE_INFLATION": 1}}
```

---

### 2.2 Distribution drift check

```powershell
curl.exe -s http://localhost:8000/v3/monitoring/drift
```

**Expected (first run — no baseline yet):**
```json
{"p_value": 1.0, "recommendation": "first_run", "drift_detected": false}
```

**Expected (stable year-on-year):**
```json
{"p_value": 0.4231, "recommendation": "incremental", "drift_detected": false}
```

**Expected (significant drift detected):**
```json
{"p_value": 0.0043, "recommendation": "full_retrain", "drift_detected": true}
```

If `drift_detected: true` — stop and call the project lead before running any update. Do not proceed with incremental training.

---

### 2.2b Drift explanation — plain-English full-retrain rationale

Narrates the same drift decision for a human reviewer, using **only** numbers the
pipeline already computed (no new thresholds): the overall score-KS p-value vs
`DRIFT_KS_THRESHOLD`, plus the per-feature KS table. Feature counts are scoped to
the **44 model features** — the 24 dropped nominal identifiers (`v3_feature_schema.json`)
are excluded and reported separately, so the panel reflects what a retrain would
actually relearn.

```powershell
curl.exe -s "http://localhost:8000/v3/monitoring/drift-explain?top=12"
```

**Expected (shape):**
```json
{
  "recommendation": "full_retrain",
  "drift_detected": true,
  "p_value": 1.74e-26,
  "ks_threshold": 0.01,
  "overall_explanation": "The score distribution shifted significantly: KS p-value ... a full GPU retrain is recommended ...",
  "n_features": 44,
  "n_drifted": 0,
  "n_excluded_nonmodel": 24,
  "feature_summary": "None of the 44 model features drifted this cycle (24 non-model identifier columns excluded).",
  "top_features": [ { "feature": "...", "ks_stat": 0.0, "p_value": 1.0, "mean_shift": 0.0, "direction": "unchanged", "drifted": false, "explanation": "..." } ]
}
```

`n_features` is the count of model features present in the drift file; if
`feature_drift_v3.json` predates the 68→44 migration it still holds 68 entries,
but this endpoint reports only the 44 (the surplus lands in `n_excluded_nonmodel`).
Surfaced in the console's **admin → "Drift explanation"** panel.

---

### 2.3 Reviewer explanation cards — *why* an application is suspicious

Two HTML endpoints render the evidence for a single application. They are meant
to be **opened in a browser**, but `curl` is the way to fetch/save them from a
terminal or a headless server.

| Endpoint | Returns | Cost |
|---|---|---|
| `GET /v3/monitoring/{app_id}/card` | Interactive reviewer card: risk placement, ranked reason codes, per-field declared-vs-model-expected breakdown, closed-form fusion attribution (max fusion — single argmax DRIVER + margin over the next-highest detector, not a proportional split), the **model-traceability trail** (which model drove the score + fired which trigger), a supplementary Deep SAD center-distance signal when elevated (does not drive the fused score), a **RGCN relation-ablation** signal ("Neighbourhood-expectation driver" — which relation's removal would most improve this node's reconstruction fit; post-hoc, XAI-only, added 2026-07-22), EVT trigger sentences with the measured flagged rate (not just the target rate), and a lightweight identity ego-graph | Cheap — reads pre-computed `explanation_cards_v3.json` |
| `GET /v3/monitoring/{app_id}/ring` | Rotatable Plotly 3D identity ring (the deep-dive) — nodes coloured by risk, edges coloured/legend-grouped by relation with per-relation click-to-toggle / double-click-to-isolate (2026-07-22), gold diamond outline marking dense-block core members | **Lazy** — the ring is computed only on this request (i.e. when the card's "Examine full ring in 3D" link is clicked), never in batch |
| `GET /v3/monitoring/{app_id}/export` | Zip bundle for one flagged application: `<id>_scorecard.csv` (flat audit row incl. model-traceability summary) + `<id>_card.html` + `<id>_evidence.json` | Cheap — projects `explanation_cards_v3.json` |
| `GET /v3/monitoring/export/bulk` | Zip of **all** flagged applications: `manifest.csv` (one scorecard row each) + `cards/<id>.html` + `evidence/<id>.json` | Renders every card once — seconds for the top-N set |
| `GET /v3/monitoring/export/selected?ids=a,b,c` | Zip of a **reviewer-chosen subset**: `manifest.csv` + `cards/<id>.html` + **`rings/<id>.html`** (interactive 3D ring) + `evidence/<id>.json`; unknown ids listed in `_skipped.txt` | Renders one lazy Plotly ring per id (~5 MB each) — cap 200 ids |

Cards exist **only for flagged (suspicious) applications** — those that crossed an
EVT threshold, carry a self-training trigger, or hold a confirmed/pseudo label.
Everything else returns `404`.

**First, get a flagged application id** (the top-suspicious list is the easy source):
```powershell
curl.exe -s "http://localhost:8000/v3/monitoring/top-suspicious?n=5"
# copy an application_id from the output, e.g. GJ202526000221788
```

**Fetch the card and open it:**
```powershell
# Save to a file, then open in the default browser
curl.exe -s "http://localhost:8000/v3/monitoring/GJ202526000221788/card" -o card.html
Start-Process card.html

# Or open directly in the browser (no curl needed)
Start-Process "http://localhost:8000/v3/monitoring/GJ202526000221788/card"
```

**Fetch the 3D ring (this is the only call that triggers Plotly compute):**
```powershell
curl.exe -s "http://localhost:8000/v3/monitoring/GJ202526000221788/ring" -o ring.html
Start-Process ring.html
```

**Export one application (scorecard CSV + card + evidence JSON, zipped):**
```powershell
curl.exe -s "http://localhost:8000/v3/monitoring/GJ202526000221788/export" -o GJ202526000221788_export.zip
Expand-Archive GJ202526000221788_export.zip -DestinationPath .\one_export
```

**Export every flagged application at once (manifest + all cards + evidence):**
```powershell
curl.exe -s "http://localhost:8000/v3/monitoring/export/bulk" -o flagged_export.zip
Expand-Archive flagged_export.zip -DestinationPath .\bulk_export   # manifest.csv is the master list
```

**Export a chosen subset (from the queue's multi-select — adds the 3D ring per app):**
```powershell
curl.exe -s "http://localhost:8000/v3/monitoring/export/selected?ids=GJ202526000221788,GJ202526000239186" -o selected_export.zip
Expand-Archive selected_export.zip -DestinationPath .\selected_export   # cards/ + rings/ + evidence/ + manifest.csv
# 400 = no ids supplied or >200 ids; 404 = none of the ids have a card
```
The same bundles are available offline via the CLI, no server needed:
```powershell
.\.venv\Scripts\python.exe -m src.export_v3 --app-id GJ202526000221788   # -> outputs/exports/
.\.venv\Scripts\python.exe -m src.export_v3 --bulk
```

**Check the status codes without downloading the body:**
```powershell
# 200 = flagged application, card available
curl.exe -s -o NUL -w "card: %{http_code}`n" "http://localhost:8000/v3/monitoring/GJ202526000221788/card"

# 404 = application not flagged, or scores/cards not generated yet
curl.exe -s -o NUL -w "card: %{http_code}`n" "http://localhost:8000/v3/monitoring/NOT_A_REAL_ID/card"
```

**Expected:**
| Case | Status | Body |
|---|---|---|
| Flagged app | `200` | `text/html` reviewer card (~25 KB) |
| Flagged app, `/ring` | `200` | `text/html` Plotly page (self-contained) |
| Unflagged / unknown app | `404` | `{"detail": "No card for this application — not flagged, or scores/cards not yet generated"}` |
| App with no graph edges, `/ring` | `404` | `{"detail": "Application not in identity graph (no shared IP/mobile/name/pincode edges)"}` |
| Flagged app, `/export` | `200` | `application/zip` attachment (`<id>_export.zip`) |
| `/export/bulk` | `200` | `application/zip` attachment (`flagged_export_<ts>.zip`) |

> The card's inline ego-graph is the at-a-glance view; the **"Examine full ring in
> 3D"** link points at `/ring`, so Plotly cost is paid per view, not per batch —
> safe even when the flagged set is large. Cards are also written to disk on every
> full pipeline run (`outputs/cards/`, see the runbook) and logged to MLflow.

**One-click review loop.** When a card is served by the live API, its footer
buttons POST directly to the supervisor endpoints — no separate curl needed:
**⚑ Confirm fraud** → `POST /v3/supervisor/confirm-fraud` (writes the confirmed
store), **✓ Mark false positive** → `POST /v3/supervisor/mark-false-positive`, and
**↺ Undo label** → `POST /v3/supervisor/clear-label` (removes the label so the
application resets). The fraud-type is pre-selected from the card's own evidence.
See §5 for the raw calls those buttons make.

---

## 3. Model / Checkpoint Endpoints

### 3.1 Checkpoint info

```powershell
curl.exe -s http://localhost:8000/v3/model/checkpoint-info
```

**Expected:**
```json
{
  "exists": true,
  "size_mb": 0.71,
  "n_features": 44,
  "graph_emb_dim": 64,
  "n_edge_types": 5,
  "versioned_checkpoints": []
}
```

`n_features` must be 44 (68 minus the 24 dropped `IDENTIFIER_FEATURES`, adopted 2026-07-15), `graph_emb_dim` must be 64, `n_edge_types` must be 5. Any other value means a wrong checkpoint is loaded.

---

### 3.2 Rollback to a versioned checkpoint

Only run this if the current checkpoint is broken or PR-AUC regressed after an incremental update.

```powershell
# First check what versioned checkpoints exist
curl.exe -s http://localhost:8000/v3/model/checkpoint-info

# Then dispatch the rollback (replace the filename with one from versioned_checkpoints)
Invoke-RestMethod -Method POST "http://localhost:8000/v3/model/rollback" `
  -ContentType "application/json" `
  -Body '{"versioned_path": "models/checkpoints/hybrid_v3_2025-26_abc12345.pth"}'
```

**Expected:**
```json
{"job_id": "xxxxxxxx-...", "status": "pending", "message": "Rollback to hybrid_v3_2025-26_abc12345.pth queued"}
```

Poll the job using the `job_id` (see §4.3).

**404 if path not found:**
```json
{"detail": "Versioned checkpoint not found: models/checkpoints/hybrid_v3_...pth"}
```

---

## 4. Training Job Endpoints

All training endpoints return a `job_id` immediately. The actual work runs in the Celery worker in the background. Poll `GET /v3/training/jobs/{job_id}` to track progress.

### 4.1 Dispatch incremental update

Run this at the start of each yearly cycle (after submitting confirmed fraud from §5).

```powershell
# Real cycle
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/incremental?cycle=2025-26"

# Smoke test (2 epochs only — use to verify nothing crashes before committing)
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/incremental?cycle=2025-26&smoke_test=true"
```

**Expected:**
```json
{"job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "status": "pending", "message": "Incremental update queued for cycle '2025-26'"}
```

---

### 4.2 Dispatch full pipeline

Runs `main_v3.py` end-to-end (feature engineering → scoring → EVT → self-training → LightGBM → XAI). Takes 2–4 hours on CPU. Only needed when drift requires a full retrain.

```powershell
# Full run (reads data/raw/data_for_ml_model.csv — the default, unchanged)
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/full"

# Smoke test first (strongly recommended before committing hours)
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/full?smoke_test=true"

# Read straight from Postgres instead of the CSV (added 2026-07-23) — every
# MERGED batch (primary + any admin-merged cohorts/patterns), no CSV file
# needed. Data must already be staged + merged first (Intake -> Evaluate ->
# Decide, or Pattern queue -> Promote) — this only controls where the
# retrain reads FROM.
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/full?data_source=postgres"
```

**Expected:**
```json
{"job_id": "xxxxxxxx-...", "status": "pending", "message": "Full pipeline queued (source=file)"}
```

---

### 4.3 Poll job status

```powershell
# Replace the job_id with the one returned from any training dispatch
Invoke-RestMethod "http://localhost:8000/v3/training/jobs/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

**Possible statuses:**

| Status | Meaning |
|---|---|
| `pending` | Queued, worker hasn't started it yet |
| `running` | Worker is actively executing |
| `complete` | Finished successfully |
| `failed` | Error — check the `error` field |

**Expected (running):**
```json
{"job_id": "...", "status": "running", "result": null, "error": null}
```

**Expected (complete):**
```json
{"job_id": "...", "status": "complete", "result": {"status": "complete", "cycle": "2025-26", "smoke_test": false}, "error": null}
```

**Expected (failed):**
```json
{"job_id": "...", "status": "failed", "result": null, "error": "ModuleNotFoundError: ..."}
```

---

### 4.4 Upload a checkpoint from the GPU laptop

After full GPU training on the laptop, transfer the checkpoint via this endpoint instead of copying files manually.

```powershell
curl.exe -s -X POST http://localhost:8000/v3/training/upload-checkpoint `
  -F "file=@models/hybrid_graphmcm_v3.pth" `
  -F "cycle=2025-26" `
  -F "source_ref=abc123"
```

**Expected:**
```json
{"job_id": "...", "status": "pending", "message": "Checkpoint received and queued for validation"}
```

The checkpoint is validated (schema + dimension check) before going live. If validation fails, the live model is untouched.

---

### 4.5 Pull checkpoint via DVC

```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/pull-checkpoint"
```

**Expected:**
```json
{"job_id": "...", "status": "pending", "message": "DVC pull queued — will validate and hot-swap on completion"}
```

---

## 5. Supervisor Feedback Endpoints

### 5.0 Interactive Topology View

Open in browser (do not use curl) to see the interactive HTML visualization of the application's network context:
```text
http://localhost:8000/v3/monitoring/APP_2024_00123/topology
```

### 5.0b Pattern Lifecycle

List pending confirmed patterns:
```powershell
curl.exe -s http://localhost:8000/v3/supervisor/patterns
```

List EVERY flagged pattern across all sessions (the persistent "flagged history"
directory — all states, newest first):
```powershell
curl.exe -s http://localhost:8000/v3/supervisor/patterns/all
```

Soft IP-cluster coverage check — "has this application's cluster already been
flagged for LOE?" (read-only heuristic on the `shares_ip` edge; the console shows
it as a non-blocking warning banner the reviewer verifies via the 3D ring):
```powershell
curl.exe -s http://localhost:8000/v3/supervisor/patterns/coverage/APP_2024_00123
```
`covered` is true when the app is listed in, or shares an IP with a member of, any
non-rejected pattern; each match carries `in_exposure` (already in the LOE
topology-exposure set) and `n_shared_ip`. First call after a graph rebuild loads
the graph (~4 s); subsequent calls are cached until the graph mtime changes.

Delete flagged patterns from the store (flagged-history cleanup). Removes the
RECORD only — `removed_promoted` lists deleted ids that were promoted, whose
exposure/training effect persists until a rebuild. 404 only if none matched:
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/patterns/delete" `
  -ContentType "application/json" `
  -Body '{"pattern_ids": ["pat_xxxxxxxx"]}'
```

Export every flagged pattern (all states, all sessions) as a zip — manifest.csv
+ patterns/<pattern_id>.json (full record, matching the application export shape):
```powershell
curl.exe -s -o patterns_export.zip http://localhost:8000/v3/supervisor/patterns/export/bulk
```

Export a reviewer-chosen subset of flagged patterns (comma-separated pattern_ids,
console multi-select feeds this):
```powershell
curl.exe -s -o patterns_selected.zip "http://localhost:8000/v3/supervisor/patterns/export/selected?ids=pat_xxxxxxxx,pat_yyyyyyyy"
```
Unknown ids are skipped and listed in `_skipped.txt` inside the zip; 404 only if
none of the requested ids exist.

Confirm a pattern (saves subgraph):
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/patterns/confirm" `
  -ContentType "application/json" `
  -Body '{"application_id": "APP_2024_00123", "fraud_type": "IP_CLUSTER", "subgraph": {}, "confirmed_by": "investigator_name"}'
```

Promote a pattern to exposure and retrain. `mode` picks the retrain job:
`"incremental"` (default) fine-tunes the existing checkpoint but does NOT read
`synthetic_exposure_graph_v3.pt` — the promoted ring's real topology sits
unused until a full retrain runs. `"full_retrain"` reruns the full pipeline,
which trains the RGCN's Stage 1 LOE against that topology file directly — the
only path that actually teaches the model the ring's structure, not just its
feature values:
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/patterns/promote" `
  -ContentType "application/json" `
  -Body '{"pattern_ids": ["pat_xxxxxxxx"], "mode": "incremental", "smoke_test": true}'
```

Retrain directly from the flagged-history store — covers patterns already
PROMOTED (topology already appended; this just (re)runs training) as well as
any still CONFIRMED/SELECTED in the selection (promoted first, same as
`/patterns/promote`). Empty `pattern_ids` = every non-rejected pattern in the
store (the console's "Retrain all" button):
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/patterns/retrain" `
  -ContentType "application/json" `
  -Body '{"pattern_ids": [], "mode": "full_retrain", "smoke_test": true}'
```
`mode` must be `"incremental"` or `"full_retrain"` (422 otherwise). Response
reports `n_target`, `newly_promoted_pattern_ids` (patterns that weren't
PROMOTED yet and were promoted as part of this call), and the dispatched
`job_id` (poll via `GET /v3/training/jobs/{job_id}`).

### 5.1 Confirm a fraud case

Run this for every application investigators confirm as fraud after reviewing the XAI cards.

```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/confirm-fraud" `
  -ContentType "application/json" `
  -Body '{"application_id": "APP_2024_00123", "fraud_type": "IP_CLUSTER", "confirmed_by": "investigator_name", "notes": "25 applicants sharing one IP at same school", "cycle": "2024-25"}'
```

**Valid fraud types:** `IP_CLUSTER` · `FEE_INFLATION` · `INCOME_VIOLATION` · `NAME_COLLISION` · `CROSS_CHANNEL` · `OTHER`

**Expected:**
```json
{"status": "ok", "application_id": "APP_2024_00123", "fraud_type": "IP_CLUSTER", "n_confirmed": 1}
```

**422 if fraud_type is invalid:**
```json
{"detail": "fraud_type must be one of {'INCOME_VIOLATION', 'OTHER', ...}, got 'INVALID_TYPE'"}
```

**422 if application_id not found in engineered features:**
```json
{"detail": "application_id 'APP_FAKE' not found in data/processed/engineered_features_v3.csv"}
```

---

### 5.1b Confirm a batch, then optionally retrain (multi-select)

Labels a reviewer-selected batch in one call — each item is either a
confirmed-fraud (needs `fraud_type`) or a false-positive. Confirmed examples flow
into the same confirmed-fraud store the incremental trainer reads (3× weight,
RGCN frozen). Set `dispatch_retrain: true` to fire **one** human-gated
incremental fine-tune afterwards (this is the console's "Record + retrain"
action); `false` records the labels only. Per-item failures (bad `fraud_type`,
unknown id, unknown verdict) are collected in `errors` without aborting the batch.

```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/confirm-batch" `
  -ContentType "application/json" `
  -Body '{
    "items": [
      {"application_id": "GJ202526000239186", "verdict": "confirmed_fraud", "fraud_type": "IP_CLUSTER", "notes": "shared IP ring"},
      {"application_id": "GJ202526000221788", "verdict": "false_positive", "notes": "legit rural school"}
    ],
    "confirmed_by": "investigator_name",
    "cycle": "2025-26",
    "dispatch_retrain": false,
    "smoke_test": false
  }'
```

**Expected:**
```json
{"status": "ok", "n_recorded": 2, "recorded": [...], "errors": [],
 "n_confirmed": 1, "n_false_positives": 2, "retrain_dispatched": false, "job_id": null}
```

With `dispatch_retrain: true`, the response carries `retrain_dispatched: true` and
a `job_id` you poll at `GET /v3/training/jobs/{job_id}` (section 4.3). Recording
labels alone never advances training — the explicit retrain flag is the gate
(hard stop #5).

---

### 5.1c Bring in a brand-new fraud pattern (CSV → relational LOE)

For a **relationally-complex ring the model has never seen** that a supervisor
discovered independently. Upload the ring (full raw-schema rows, fresh
application_ids) via `/v3/monitoring/upload-dataset`, then:

**Test read-only** — does the current model already catch it? Merges the ring,
rebuilds features + the identity graph (so shared IP/name edges are real), scores
just the ring, restores everything. Takes ~1 min.
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/pattern/test" `
  -ContentType "application/json" -Body '{"dataset_path": "data/uploads/ring.csv"}'
```
```json
{"n_members": 6, "members": [{"application_id": "...", "hybrid_anomaly_score": 0.46}, ...],
 "mean_score": 0.44, "max_score": 0.47, "min_score": 0.39}
```

**Ingest as a relational pattern** — permanently merge the ring, extract its real
intra-ring subgraph, append it as a topology-exposure cluster, record it in both
confirmed stores, and (optionally) dispatch the human-gated retrain.
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/pattern/ingest" `
  -ContentType "application/json" -Body '{
    "dataset_path": "data/uploads/ring.csv", "fraud_type": "IP_CLUSTER",
    "confirmed_by": "lead_1", "notes": "landlord-address ring",
    "dispatch_retrain": true, "smoke_test": true }'
```
```json
{"status": "ok", "pattern_id": "pat_ab12cd34", "n_rows_added": 6, "n_members": 6,
 "topology_exposure": {"cluster_id": 50, "n_nodes": 6, "n_edges_directed": 60,
   "edges_per_relation": {"shares_ip": 30, "shares_mother_name": 30},
   "topology_exposure_total_nodes": 1299},
 "n_confirmed_recorded": 6, "retrain_dispatched": true, "job_id": "job_retrain_xxxxxxxx"}
```
`ingest` permanently mutates the dataset + exposure set; `test` restores
everything. 422 if the members share no identity attribute (no connected ring) or
the fraud_type is invalid.

---

### 5.2 Mark a false positive

Run this for every application that was flagged but investigators confirmed is legitimate.

```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/mark-false-positive" `
  -ContentType "application/json" `
  -Body '{"application_id": "APP_2024_00456", "confirmed_by": "investigator_name", "notes": "Legitimate rural school with shared broadband"}'
```

**Expected:**
```json
{"status": "ok", "application_id": "APP_2024_00456", "n_false_positives": 1}
```

---

### 5.3 Clear / undo a label (reset the review)

Removes an application from **both** the confirmed and false-positive stores, so
its state resets to unlabelled. Use it to correct a mis-click, or to re-run a
demo of the detection loop (label a topology → retrain → confirm it now scores as
fraud → **clear** → repeat). This is the button `↺ Undo label` on the card.

```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/clear-label" `
  -ContentType "application/json" `
  -Body '{"application_id": "APP_2024_00123"}'
```

**Expected (label existed):**
```json
{"status": "ok", "application_id": "APP_2024_00123",
 "removed_confirmed": true, "removed_false_positive": false,
 "n_confirmed": 0, "n_false_positives": 0}
```

**404 if the application had no label to clear:**
```json
{"detail": "No label to clear for 'APP_2024_00123' (not in confirmed or false-positive store)"}
```

> Clearing a label only edits the store JSON — it does **not** retrain. The change
> takes effect at the next incremental/full run (or `patterns/promote`), which
> rebuilds exposure from the current store.

---

## 6. End-to-End Workflow Test

Copy and run this block to exercise the full supervisor → training → monitoring cycle in one go.

```powershell
# 1. Check health
Write-Host "--- Health ---"
curl.exe -s http://localhost:8000/health

# 2. Check fraud store (should start empty)
Write-Host "`n--- Fraud Store (before) ---"
curl.exe -s http://localhost:8000/v3/monitoring/fraud-store-summary

# 3. Mark a false positive
Write-Host "`n--- Mark False Positive ---"
Invoke-RestMethod -Method POST "http://localhost:8000/v3/supervisor/mark-false-positive" `
  -ContentType "application/json" `
  -Body '{"application_id":"APP_TEST_001","confirmed_by":"tester","notes":"end-to-end test"}'

# 4. Check fraud store again (should show 1 FP)
Write-Host "`n--- Fraud Store (after) ---"
curl.exe -s http://localhost:8000/v3/monitoring/fraud-store-summary

# 5. Check checkpoint
Write-Host "`n--- Checkpoint Info ---"
curl.exe -s http://localhost:8000/v3/model/checkpoint-info

# 6. Check drift
Write-Host "`n--- Drift Check ---"
curl.exe -s http://localhost:8000/v3/monitoring/drift

# 7. Dispatch smoke-test incremental update
Write-Host "`n--- Dispatch Incremental (smoke test) ---"
$job = Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/incremental?cycle=test&smoke_test=true"
Write-Host "job_id: $($job.job_id)"

# 8. Poll until done (checks every 5 seconds, times out after 3 min)
Write-Host "`n--- Polling job status ---"
$deadline = (Get-Date).AddMinutes(3)
do {
    Start-Sleep -Seconds 5
    $r = Invoke-RestMethod "http://localhost:8000/v3/training/jobs/$($job.job_id)"
    Write-Host "  $((Get-Date).ToString('HH:mm:ss'))  status=$($r.status)"
} while ($r.status -in @("pending","running") -and (Get-Date) -lt $deadline)

Write-Host "`nFinal status: $($r.status)"
if ($r.error) { Write-Host "Error: $($r.error)" -ForegroundColor Red }
```

**Expected final output:**
```
Final status: complete
```

---

## 7. Check Worker Logs

If a job fails, check the Celery worker logs for the full traceback:

```powershell
docker logs nicfrauddetectionproject-nic-worker-1 --tail 50
```

Check the API server logs:

```powershell
docker logs nicfrauddetectionproject-nic-api-1 --tail 30
```

Stream logs live while a job runs:

```powershell
docker logs nicfrauddetectionproject-nic-worker-1 -f
```

Press `Ctrl+C` to stop streaming.

---

## 8. Common Errors

| Error | Cause | Fix |
|---|---|---|
| `connection refused` on port 8000 | Stack not running | `docker compose up -d` |
| `checkpoint_exists: false` | `.pth` file not in `models/` volume | Check `models/` directory has `hybrid_graphmcm_v3.pth` |
| job status `failed` + `ModuleNotFoundError` | Missing package in Docker image | `docker compose build` then `docker compose up -d` |
| `422 application_id not found` | App ID not in `engineered_features_v3.csv` | Run pipeline first so features exist, or check the app ID is correct |
| `422 fraud_type invalid` | Wrong fraud type string | Use one of: `IP_CLUSTER`, `FEE_INFLATION`, `INCOME_VIOLATION`, `NAME_COLLISION`, `CROSS_CHANNEL`, `OTHER` |
| `drift_detected: true` | Score distribution shifted | Stop — call project lead before proceeding |
| job stuck in `pending` forever | Worker container crashed | `docker logs nicfrauddetectionproject-nic-worker-1` to diagnose |

---

## 9. Simulate Data Drift → Human Review → Retrain → XAI

This section exercises 4 new endpoints added in ADR-014 (`docs/AGENTS.md`
Appendix F): `POST /v3/monitoring/evaluate-dataset`, `GET
/v3/monitoring/dataset-xai`, `GET /v3/monitoring/top-suspicious`, and `POST
/v3/training/decision`. Together they let a human evaluate an unseen
dataset, review drift + XAI, and explicitly choose **none / incremental /
full_retrain** — nothing trains automatically.

**This section was validated by calling the handler functions directly in
Python, not through the live Docker/Celery stack.** Before running it for
real: `docker compose up -d` (§0), then confirm `/health` and `/ready` both
return OK (§1) before starting at 9.1.

### 9.0 The simulated dataset

`data/raw/new_cohort_2026.csv` (600 rows) is generated by
`scripts/generate_drift_dataset.py`. It funnels applications through 3
synthetic institute IDs (instead of the normal spread across thousands of
institutes) with suspiciously round declared incomes (multiples of 50,000).
This pattern is deliberately **not** one of the 5 existing synthetic
exposure archetypes (IP concentration, mother-name collision, fee inflation,
age violation, income violation) — it's a genuine test of generalization,
per the archetype-expansion open item in `AGENTS.md` Appendix B/§11.

Regenerate it any time with:
```powershell
.venv\Scripts\python.exe scripts\generate_drift_dataset.py
```

### 9.1 Evaluate the new dataset (read-only — no training, no lasting change)

```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/monitoring/evaluate-dataset" `
  -ContentType "application/json" `
  -Body '{"dataset_path": "data/raw/new_cohort_2026.csv"}'
```

This temporarily merges the 600 rows into the raw CSV, rebuilds features +
graph, scores the new rows with the **current** checkpoint (no training),
then restores everything to exactly how it was — safe to call repeatedly.
Takes ~30-90s (feature engineering + graph rebuild over ~15,600 rows).

**Expected (measured on this exact dataset during implementation — your
p_value will vary slightly with checkpoint state, but drift_detected should
be `true`):**
```json
{"dataset_path": "data/raw/new_cohort_2026.csv", "n_rows": 600,
 "p_value": 2.52e-39, "recommendation": "full_retrain", "drift_detected": true,
 "staged_scores_path": "outputs/staged_scores_new_cohort_2026.csv"}
```
That p-value is not a typo — the institute-cluster + income-rounding pattern
is far outside the training distribution, so the KS test is maximally
confident. A p-value that small on real production data would itself be a
signal worth double-checking (e.g. a schema mismatch), not just "very
strong drift" — see §9.6.

### 9.1b Push a dataset in without going through the console (CSV-free portal/ETL path, added 2026-07-23)

`POST /v3/monitoring/push-dataset` is the scale-oriented alternative to
uploading through the browser: it takes a **server-side file path**, not an
inline row payload — deliberately, since 3-4M rows as JSON in a request body
doesn't hold up, but a path to a CSV a portal/ETL job already wrote into the
shared `data/` volume does. It schema-checks the same way as the console's
upload, stages the batch in Postgres, and — if the schema passes — dispatches
Evaluate as a **background Celery job** instead of blocking the request:

```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/monitoring/push-dataset" `
  -ContentType "application/json" `
  -Body '{"dataset_path": "data/uploads/new_cohort_2026.csv", "name": "portal_batch_2026_07_23"}'
```

**Expected (job dispatched immediately, doesn't wait for Evaluate to finish):**
```json
{"dataset_path": "data/uploads/new_cohort_2026.csv", "name": "portal_batch_2026_07_23",
 "n_rows": 600, "n_cols": 136, "schema_ok": true, "missing_columns": [], "extra_columns": [],
 "duplicate_ids": [], "staged_batch": {"batch_id": 7, "name": "portal_batch_2026_07_23", "status": "staged", "n_rows": 600, "n_id_conflicts": 0},
 "job_id": "xxxxxxxx-...", "message": "Staged and queued for auto-evaluate"}
```

Poll `job_id` the same way as any training job
(`GET /v3/training/jobs/{job_id}`). Once it completes, the batch is
reviewable in the console's cohort dataset switcher exactly like a
console-uploaded cohort — **nothing merges or retrains automatically**;
Merge/retrain still goes through `POST /v3/training/decision` (§9.3)
same as today. If `schema_ok` is `false`, `job_id` is `null` and nothing was
staged — fix the CSV and re-push.

### 9.2 Review XAI on the staged (not-yet-committed) data

```powershell
curl.exe -s "http://localhost:8000/v3/monitoring/dataset-xai?dataset_path=data/raw/new_cohort_2026.csv&top_n=20"
```

Returns the top-20 staged rows by `hybrid_anomaly_score` with per-feature
errors and a narrative — same shape as `explanation_cards_v3.json`, but
marked `[PREVIEW — pre-fusion, not yet in production scores]` since these
rows haven't been through EVT/self-training/fusion yet.

### 9.2b Cohort preview queue (review the evaluated cohort in the console)

After `evaluate-dataset`, the cohort is browsable read-only in the Review queue's
**Dataset** dropdown. The same data via API:

```powershell
# List evaluated cohorts (ring_available = merged-graph bundle persisted)
curl.exe -s http://localhost:8000/v3/monitoring/cohorts

# Ranked staged rows (pre-fusion hybrid_anomaly_score, higher = more anomalous)
curl.exe -s "http://localhost:8000/v3/monitoring/cohort/<name>/top-suspicious?n=50"

# Lightweight PREVIEW reviewer card (HTML) for one staged app -- Signal
# drivers tab now shows subspace IF / dense-block / a preview fusion score
# too (2026-07-22, computed over the cohort's merged population), not just
# the raw hybrid score; still no EVT triggers (those need the canonical
# population's fitted thresholds)
curl.exe -s http://localhost:8000/v3/monitoring/cohort/<name>/<app_id>/card

# 3D identity ring (HTML) rendered against the persisted merged base+cohort graph
curl.exe -s http://localhost:8000/v3/monitoring/cohort/<name>/<app_id>/ring

# Remove a cohort (demo reset) — deletes ONLY its staged files (+ uploaded CSV);
# base data and the downloadable sample CSV are untouched. 404 if none exist.
Invoke-RestMethod -Method POST "http://localhost:8000/v3/monitoring/cohort/<name>/delete"
```

`<name>` is the uploaded CSV's stem (e.g. `sample_cohort`). Rings need the bundle
`outputs/staged_graph_<name>.pt` + `staged_nodeorder_<name>.csv`, which
`evaluate-dataset` writes — cohorts evaluated before this feature show
`ring_available:false` and must be re-evaluated. Export / Flag-for-LOE stay
commit-gated (the apps aren't in the live graph until ingest).

### 9.3 Human decides — three ways to call the same endpoint

**(a) Do nothing** — just log the review:
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/decision" `
  -ContentType "application/json" `
  -Body '{"dataset_path": "data/raw/new_cohort_2026.csv", "action": "none", "cycle": "2026-cohort-A", "decided_by": "investigator_name"}'
```

**(b) Incremental fine-tune on the new data** — permanently merges the 600
rows into the raw CSV, rebuilds features/graph, then fine-tunes the existing
checkpoint (RGCN frozen unless ≥50 confirmed fraud):
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/decision" `
  -ContentType "application/json" `
  -Body '{"dataset_path": "data/raw/new_cohort_2026.csv", "action": "incremental", "cycle": "2026-cohort-A", "decided_by": "investigator_name", "smoke_test": true}'
```

**(c) Full retrain, combining previous + new data** — permanently merges the
600 rows into the raw CSV (old + new combined, 15,600 total), then dispatches
the full pipeline (`main_v3.py`), which rebuilds features/graph itself:
```powershell
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/decision" `
  -ContentType "application/json" `
  -Body '{"dataset_path": "data/raw/new_cohort_2026.csv", "action": "full_retrain", "cycle": "2026-cohort-A", "decided_by": "investigator_name", "smoke_test": true}'
```

**Pick (a), (b), or (c) — not more than one against the same dataset in the
same pass.** `merge_dataset_into_raw()` refuses to merge an `application_id`
that's already in the raw CSV, so calling `(b)` then `(c)` back-to-back will
`422` on the second call unless you reset in between (§9.5). `9.1`
(`evaluate-dataset`) is always safe to repeat — it self-restores.

**Expected (incremental/full_retrain):**
```json
{"status": "ok", "action": "incremental", "dataset_path": "data/raw/new_cohort_2026.csv",
 "job_id": "xxxxxxxx-...", "backup_dir": "data/backups/20260702_HHMMSS_decision_2026-cohort-A",
 "audit_log_path": "outputs/drift_audit_log.json"}
```

`backup_dir` holds a pre-merge snapshot of the raw/feature/graph files —
manually restorable if the decision needs to be undone (`Copy-Item` each
file back from that directory).

Every call — including `"none"` — appends one record to
`outputs/drift_audit_log.json`:
```powershell
Get-Content outputs\drift_audit_log.json | ConvertFrom-Json | Format-Table timestamp, action, decided_by, p_value, job_id
```

### 9.4 Poll the job, then test on the newer data

```powershell
Invoke-RestMethod "http://localhost:8000/v3/training/jobs/<job_id>"
```

Once `status: "complete"`, the new cohort's 600 applications are now part of
the canonical outputs — the same lookup method from earlier applies directly:

```powershell
# Any of the new applications, e.g. ZZ202627000900001..000900600
Get-Content outputs\risk_scores_v3.csv | Select-String "ZZ202627000900001"

# Top-20 suspicious across the WHOLE updated population (old + new), with XAI
curl.exe -s "http://localhost:8000/v3/monitoring/top-suspicious?n=20"
```

`top-suspicious` ranks `outputs/risk_scores_v3.csv` (by `risk_score_v3`, desc)
— the SAME file `xai_layer_v3` ranks to build the explanation cards — so every
row in the queue is guaranteed to have a reviewer card (for `n` ≤ the number of
cards generated). It falls back to `outputs/top_suspicious_v3.tsv` only if
`risk_scores_v3.csv` is absent. (Previously it read the TSV directly, which
could drift out of sync with the cards across partial runs and leave queued
rows with no card.) For richer per-application scores from an incremental
update, pull individual application scores from
`outputs/risk_scores_v3.csv` / `outputs/explanation_cards_v3.json` instead.

### 9.5 Reset between attempts

`(b)` and `(c)` both permanently modify `data/raw/data_for_ml_model.csv`
(15,000 → 15,600 rows) plus every derived file. To run the simulation again
from a clean slate:

```powershell
# Stop the stack so nothing writes mid-reset
docker compose down

# Raw CSV is git-tracked — this is the safest full reset
git checkout -- data/raw/data_for_ml_model.csv

# Rebuild the derived files from the clean raw CSV
docker compose up -d
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/full?smoke_test=true"
# poll until complete (§4.3), then re-run §9.1 onward
```

Alternatively, restore just the pre-merge snapshot instead of the full
pipeline rebuild — copy every file out of the `backup_dir` the `decision`
response returned back to its original path (`data/raw/`,
`data/processed/`, matching filenames) — faster, but skips re-validating
that the checkpoint is consistent with the restored data.

To reset only the audit trail and staged-preview files without touching the
raw data:
```powershell
Remove-Item outputs\drift_audit_log.json, outputs\staged_scores_*.csv, `
  outputs\staged_features_*.csv, outputs\staged_scores_meta_*.json -ErrorAction SilentlyContinue
```

### 9.6 What to actually look for

- **§9.1** `drift_detected: true` with a very small `p_value` confirms the
  model correctly recognizes the institute-cluster + income-rounding pattern
  as out-of-distribution — this is the core "does drift detection work"
  check.
- **§9.2** the XAI preview's `top_feature_errors` should be dominated by
  fields plausibly connected to the injected pattern (institution/verifier/
  temporal fields) rather than random noise — this is the "is the
  explanation actually informative" check, not just "does it return JSON."
- **§9.4** after `full_retrain`, re-run §9.1 against
  `data/raw/new_cohort_2026.csv` again (regenerate a *second* dataset with
  `scripts/generate_drift_dataset.py` first, editing `SEED` so the
  `application_id`s don't collide with the now-committed first batch) — the
  retrained model should show a smaller `p_value` gap / lower anomaly scores
  on the same fraud shape than the pre-retrain model did, since it has now
  seen that pattern. That comparison is the actual "did retraining help"
  evidence, not just "did the job finish."

---

## 10. Console build — model audit endpoints (MLflow replacement)

These back the "Model audit & deploy" console tab. All are also reachable via
nginx at `http://localhost:8080/...`.

### 10.1 Running-model status strip

```powershell
curl.exe -s http://localhost:8000/v3/model/stats
```

Returns the live checkpoint metadata, `latest_run`, `latest_incremental_metrics`
(PR-AUC etc.), scored-population size, confirmed/false-positive counts, and total
run count. Cheap — safe to poll.

### 10.2 Run history (replaces the MLflow run list)

```powershell
curl.exe -s "http://localhost:8000/v3/model/registry?limit=25"
```

Newest-first list of runs from `outputs/model_registry.json`. Each has
`run_type` (`incremental` | `full` | `checkpoint_swap`), `timestamp`, `cycle`,
`params`, `metrics`, `checkpoint`. Optional `&run_type=incremental` filter.

### 10.3 Upload a cohort CSV (browser intake, schema-validated)

```powershell
curl.exe -s -X POST http://localhost:8000/v3/monitoring/upload-dataset `
  -F "file=@data/raw/new_cohort_2026.csv"
```

**Expected:**
```json
{"dataset_path": "data/uploads/new_cohort_2026.csv", "filename": "new_cohort_2026.csv",
 "n_rows": 500, "n_cols": 136, "expected_cols": 136, "schema_ok": true,
 "missing_columns": [], "extra_columns": [], "duplicate_ids": []}
```

Saves the file under `data/uploads/` and validates its columns against the raw
schema **without mutating anything**. `schema_ok: false` (with `missing_columns`
populated) means the console blocks evaluate/retrain. Feed the returned
`dataset_path` into §9.1 (`evaluate-dataset`) or the `decision` endpoint.

> **Run history is written automatically** by the pipeline: incremental updates
> (`retraining_orchestrator`), full runs (`run_full_pipeline_task`), and every
> checkpoint swap (`validate_and_hotswap` — upload / dvc pull / rollback) each
> append a record. There is no MLflow server to start.
