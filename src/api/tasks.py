"""
Celery task definitions — ADR-006.

All long-running operations (incremental update, full pipeline, checkpoint
validation, rollback) are dispatched as Celery tasks so API endpoints return
immediately with a job_id. Callers poll GET /v3/training/jobs/{job_id}.

Worker must run with concurrency=1. See celeryconfig.py and AGENTS.md hard
stop #16.
"""
import sys
import subprocess
from pathlib import Path

from celery import Celery
import celeryconfig

celery_app = Celery("nic_fraud")
celery_app.config_from_object(celeryconfig)


@celery_app.task(bind=True, name="tasks.run_incremental")
def run_incremental_task(self, cycle: str = "unknown", smoke_test: bool = False):
    from src.retraining_orchestrator import run_incremental_update
    run_incremental_update(smoke_test=smoke_test, cycle=cycle)
    return {"status": "complete", "cycle": cycle, "smoke_test": smoke_test}


@celery_app.task(bind=True, name="tasks.run_full_pipeline")
def run_full_pipeline_task(self, smoke_test: bool = False):
    args = [sys.executable, "main_v3.py"]
    if smoke_test:
        args.append("--smoke")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed (rc={result.returncode}): {result.stderr[-1000:]}"
        )
    # Record the full run in the local registry (MLflow replacement).
    from src.model_registry import log_run
    from pathlib import Path as _Path
    ckpt = _Path("models/hybrid_graphmcm_v3.pth")
    log_run(
        "full",
        cycle="full_pipeline",
        smoke_test=smoke_test,
        checkpoint={
            "path":    str(ckpt),
            "size_mb": round(ckpt.stat().st_size / 1e6, 2) if ckpt.exists() else None,
        },
    )
    return {
        "status": "complete",
        "smoke_test": smoke_test,
        "stdout_tail": result.stdout[-2000:],
    }


@celery_app.task(bind=True, name="tasks.run_validate_checkpoint")
def run_validate_checkpoint_task(
    self,
    temp_path: str,
    cycle: str = "unknown",
    source_ref: str = "none",
):
    from src.checkpoint_manager import validate_and_hotswap
    result = validate_and_hotswap(Path(temp_path), cycle=cycle, source_ref=source_ref)
    return {"status": "complete", **result}


@celery_app.task(bind=True, name="tasks.run_pull_checkpoint")
def run_pull_checkpoint_task(self):
    result = subprocess.run(
        ["dvc", "pull", "models/hybrid_graphmcm_v3.pth"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull failed: {result.stderr}")
    from src.checkpoint_manager import validate_and_hotswap
    validate_result = validate_and_hotswap(
        Path("models/hybrid_graphmcm_v3.pth"),
        cycle="dvc-pull",
    )
    return {"status": "complete", **validate_result}


@celery_app.task(bind=True, name="tasks.run_rollback")
def run_rollback_task(self, versioned_path: str):
    from src.checkpoint_manager import validate_and_hotswap
    result = validate_and_hotswap(Path(versioned_path), cycle="rollback")
    return {"status": "complete", "restored_from": versioned_path, **result}
