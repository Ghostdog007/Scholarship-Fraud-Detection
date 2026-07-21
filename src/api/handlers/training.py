"""
Training job endpoints — ADR-005 + ADR-006.

POST /v3/training/incremental        → Celery async incremental update
POST /v3/training/full               → Celery async full pipeline
GET  /v3/training/jobs/{job_id}      → Poll Celery task status
POST /v3/training/upload-checkpoint  → Multipart .pth upload → validate_and_hotswap
POST /v3/training/pull-checkpoint    → dvc pull → validate_and_hotswap
POST /v3/training/decision           → human picks none|incremental|full_retrain after
                                        reviewing POST /v3/monitoring/evaluate-dataset
"""
import json
import shutil
import time
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import DecisionRequest, DecisionResponse, JobResponse, JobStatusResponse

log    = structlog.get_logger()
router = APIRouter()

_VALID_ACTIONS = {"none", "incremental", "full_retrain"}

_CELERY_STATE_MAP = {
    "PENDING": "pending",
    "STARTED": "running",
    "SUCCESS": "complete",
    "FAILURE": "failed",
    "RETRY":   "running",
    "REVOKED": "failed",
}


@router.post("/incremental", response_model=JobResponse)
def start_incremental(cycle: str = "unknown", smoke_test: bool = False):
    from src.api.tasks import run_incremental_task
    task = run_incremental_task.delay(cycle=cycle, smoke_test=smoke_test)
    log.info("training.incremental.queued", job_id=task.id, cycle=cycle, smoke_test=smoke_test)
    return JobResponse(
        job_id=task.id,
        status="pending",
        message=f"Incremental update queued for cycle '{cycle}'",
    )


@router.post("/full", response_model=JobResponse)
def start_full_pipeline(smoke_test: bool = False):
    from src.api.tasks import run_full_pipeline_task
    task = run_full_pipeline_task.delay(smoke_test=smoke_test)
    log.info("training.full.queued", job_id=task.id, smoke_test=smoke_test)
    return JobResponse(
        job_id=task.id,
        status="pending",
        message="Full pipeline queued" + (" (smoke test — 2 epochs)" if smoke_test else ""),
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    from celery.result import AsyncResult
    from src.api.tasks import celery_app

    result = AsyncResult(job_id, app=celery_app)
    status = _CELERY_STATE_MAP.get(result.state, result.state.lower())

    task_result = result.result if result.state == "SUCCESS" else None
    error       = str(result.result) if result.state == "FAILURE" else None

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        result=task_result,
        error=error,
    )


@router.post("/upload-checkpoint", response_model=JobResponse)
async def upload_checkpoint(
    file: UploadFile = File(...),
    cycle: str = Form("unknown"),
    source_ref: str = Form("none"),
):
    if not (file.filename or "").endswith(".pth"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a .pth checkpoint")

    temp_path = Path(f"models/incoming_{int(time.time())}_{uuid.uuid4().hex[:8]}.pth")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    from src.api.tasks import run_validate_checkpoint_task
    task = run_validate_checkpoint_task.delay(str(temp_path), cycle, source_ref)
    log.info("training.upload_checkpoint.queued", job_id=task.id, cycle=cycle, temp=str(temp_path))
    return JobResponse(
        job_id=task.id,
        status="pending",
        message="Checkpoint received and queued for validation",
    )


@router.post("/pull-checkpoint", response_model=JobResponse)
def pull_checkpoint():
    from src.api.tasks import run_pull_checkpoint_task
    task = run_pull_checkpoint_task.delay()
    log.info("training.pull_checkpoint.queued", job_id=task.id)
    return JobResponse(
        job_id=task.id,
        status="pending",
        message="DVC pull queued — will validate and hot-swap on completion",
    )


@router.post("/decision", response_model=DecisionResponse)
def training_decision(req: DecisionRequest):
    """
    Human-gated action after reviewing POST /v3/monitoring/evaluate-dataset
    (and GET /v3/monitoring/dataset-xai) for a given dataset_path.

    - "none"          -> logs the decision only, no file or model change.
    - "incremental"    -> merges dataset_path into the raw CSV (permanently),
                          rebuilds features + graph, then fine-tunes the
                          existing checkpoint (RGCN frozen unless >=50
                          confirmed fraud) via the existing incremental job.
    - "full_retrain"  -> merges dataset_path into the raw CSV (permanently,
                          old + new combined), then dispatches the existing
                          full-pipeline job, which rebuilds everything itself.

    Every call appends one record to outputs/drift_audit_log.json regardless
    of action, so "do nothing" decisions are as auditable as training ones.
    """
    from src.api import audit, dataset_ops

    if req.action not in _VALID_ACTIONS:
        raise HTTPException(status_code=422, detail=f"action must be one of {sorted(_VALID_ACTIONS)}, got '{req.action}'")

    dataset_path = Path(req.dataset_path)
    if req.action != "none" and not dataset_path.exists():
        raise HTTPException(status_code=422, detail=f"dataset_path not found: {req.dataset_path}")

    meta_path  = Path(f"outputs/staged_scores_meta_{dataset_path.stem}.json")
    prior_eval = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    job_id: str | None = None
    backup_dir: Path | None = None

    try:
        if req.action == "incremental":
            backup_dir = dataset_ops.backup_canonical_files(label=f"decision_{req.cycle}")
            dataset_ops.merge_dataset_into_raw(dataset_path)
            dataset_ops.rebuild_features_and_graph()

            from src.api.tasks import run_incremental_task
            task = run_incremental_task.delay(cycle=req.cycle, smoke_test=req.smoke_test)
            job_id = task.id

        elif req.action == "full_retrain":
            backup_dir = dataset_ops.backup_canonical_files(label=f"decision_{req.cycle}")
            dataset_ops.merge_dataset_into_raw(dataset_path)  # main_v3.py rebuilds features/graph itself

            from src.api.tasks import run_full_pipeline_task
            task = run_full_pipeline_task.delay(smoke_test=req.smoke_test)
            job_id = task.id
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 3: the human gate fired — mark the batch merged (permanent) in
    # Postgres. Best-effort; the file merge above is what's authoritative.
    if req.action in ("incremental", "full_retrain"):
        try:
            from src.db.ingest import batch_exists, merge_batch, stage_raw_csv
            if not batch_exists(dataset_path.stem):
                stage_raw_csv(dataset_path, name=dataset_path.stem)
            merge_batch(dataset_path.stem)
        except Exception as e:  # noqa: BLE001
            log.warning("training.decision.pg_merge_failed", error=str(e))

    record = audit.append_audit_record({
        "dataset_path":   str(dataset_path),
        "p_value":        prior_eval.get("p_value"),
        "recommendation": prior_eval.get("recommendation"),
        "action":         req.action,
        "cycle":          req.cycle,
        "decided_by":     req.decided_by,
        "job_id":         job_id,
        "backup_dir":     str(backup_dir) if backup_dir else None,
    })

    log.info("training.decision", action=req.action, dataset_path=str(dataset_path),
             job_id=job_id, decided_by=req.decided_by)

    return DecisionResponse(
        status="ok",
        action=req.action,
        dataset_path=str(dataset_path),
        job_id=job_id,
        backup_dir=str(backup_dir) if backup_dir else None,
        audit_log_path=str(audit.AUDIT_LOG),
    )
