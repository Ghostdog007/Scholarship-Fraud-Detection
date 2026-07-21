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


@router.get("/stats")
def model_stats():
    """Compact running-model snapshot for the console status strip (MLflow
    replacement). Cheap: reads the live checkpoint metadata, the local run
    registry, the confirmed-fraud store, and the scored-population size. Drift
    is intentionally excluded here (it's a heavier call the UI fetches
    separately via /v3/monitoring/drift)."""
    import pandas as pd
    from src.checkpoint_manager import get_checkpoint_info
    from src.model_registry import latest_run, list_runs
    from src.confirmed_fraud_store import load_confirmed, load_false_positive_ids

    # Step 2 (Gate 2 passed): scored-population count from Postgres, file
    # fallback. Import inside the try — a missing/broken db layer degrades
    # gracefully instead of 500ing.
    n_scored = 0
    try:
        from src.db import reads as db_reads
        if db_reads.reads_from_pg():
            n_scored = db_reads.n_scored()
    except Exception as e:  # noqa: BLE001
        log.warning("model.stats.pg_failed_falling_back", error=str(e))
    if n_scored == 0:
        scores = Path("outputs/risk_scores_v3.csv")
        n_scored = int(pd.read_csv(scores, usecols=["application_id"]).shape[0]) if scores.exists() else 0
    latest_incr = latest_run("incremental")

    stats = {
        "checkpoint":                  get_checkpoint_info(),
        "latest_run":                  latest_run(),
        "latest_incremental_metrics":  (latest_incr or {}).get("metrics", {}),
        "n_scored":                    n_scored,
        "n_confirmed":                 len(load_confirmed()),
        "n_false_positives":           len(load_false_positive_ids()),
        "n_runs":                      len(list_runs()),
    }
    log.info("model.stats", n_scored=n_scored, n_runs=stats["n_runs"])
    return stats


@router.get("/registry")
def model_registry_history(limit: int = 25, run_type: str | None = None):
    """Newest-first training/checkpoint run history from the local registry
    (outputs/model_registry.json) — the console's model-history table. Replaces
    the MLflow run list."""
    from src.model_registry import list_runs
    return {"runs": list_runs(limit=limit, run_type=run_type)}


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
