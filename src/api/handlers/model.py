"""
Model / checkpoint management endpoints — ADR-005 + ADR-008.

GET  /v3/model/checkpoint-info
POST /v3/model/rollback
"""
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas import CheckpointInfoResponse, JobResponse, RollbackRequest

log    = structlog.get_logger()
router = APIRouter()

_LIVE_CKPT = Path("models/hybrid_graphmcm_v3.pth")
_CKPT_DIR  = Path("models/checkpoints")


@router.get("/checkpoint-info", response_model=CheckpointInfoResponse)
def checkpoint_info():
    from src.checkpoint_manager import get_checkpoint_info
    info = get_checkpoint_info()
    log.info("model.checkpoint_info", **{k: v for k, v in info.items() if k != "versioned_checkpoints"})
    return CheckpointInfoResponse(**info)


@router.post("/rollback", response_model=JobResponse)
def rollback(req: RollbackRequest):
    versioned = Path(req.versioned_path)
    if not versioned.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Versioned checkpoint not found: {req.versioned_path}",
        )
    from src.api.tasks import run_rollback_task
    task = run_rollback_task.delay(req.versioned_path)
    log.info("model.rollback.queued", job_id=task.id, versioned_path=req.versioned_path)
    return JobResponse(
        job_id=task.id,
        status="pending",
        message=f"Rollback to {Path(req.versioned_path).name} queued",
    )
