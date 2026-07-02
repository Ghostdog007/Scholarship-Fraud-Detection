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
  "n_features": 68,
  "graph_emb_dim": 64,
  "n_edge_types": 5,
  "versioned_checkpoints": []
}
```

`n_features` must be 68, `graph_emb_dim` must be 64, `n_edge_types` must be 5. Any other value means a wrong checkpoint is loaded.

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
# Full run
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/full"

# Smoke test first (strongly recommended before committing hours)
Invoke-RestMethod -Method POST "http://localhost:8000/v3/training/full?smoke_test=true"
```

**Expected:**
```json
{"job_id": "xxxxxxxx-...", "status": "pending", "message": "Full pipeline queued"}
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
  -F "mlflow_run_id=abc123"
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
