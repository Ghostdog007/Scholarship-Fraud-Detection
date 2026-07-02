"""
Training job endpoints — ADR-005 + ADR-006.

POST /v3/training/incremental        → Celery async incremental update
POST /v3/training/full               → Celery async full pipeline
GET  /v3/training/jobs/{job_id}      → Poll Celery task status
POST /v3/training/upload-checkpoint  → Multipart .pth upload → validate_and_hotswap
POST /v3/training/pull-checkpoint    → dvc pull → validate_and_hotswap
"""
import shutil
import time
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import JobResponse, JobStatusResponse

log    = structlog.get_logger()
router = APIRouter()

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
    mlflow_run_id: str = Form("none"),
):
    if not (file.filename or "").endswith(".pth"):
        raise HTTPException(status_code=422, detail="Uploaded file must be a .pth checkpoint")

    temp_path = Path(f"models/incoming_{int(time.time())}_{uuid.uuid4().hex[:8]}.pth")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    from src.api.tasks import run_validate_checkpoint_task
    task = run_validate_checkpoint_task.delay(str(temp_path), cycle, mlflow_run_id)
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
