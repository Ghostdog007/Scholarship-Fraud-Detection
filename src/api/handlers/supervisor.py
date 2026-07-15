"""
Supervisor feedback endpoints — ADR-005.

POST /v3/supervisor/confirm-fraud
POST /v3/supervisor/mark-false-positive
GET  /v3/supervisor/patterns
POST /v3/supervisor/patterns/confirm
POST /v3/supervisor/patterns/promote
"""
import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ConfirmFraudRequest,
    FalsePositiveRequest,
    ClearLabelRequest,
    ConfirmPatternRequest,
    PromotePatternRequest
)

log    = structlog.get_logger()
router = APIRouter()


@router.post("/confirm-fraud")
def confirm_fraud(req: ConfirmFraudRequest):
    from src.confirmed_fraud_store import add_confirmed, load_confirmed
    try:
        add_confirmed(
            app_id=req.application_id,
            fraud_type=req.fraud_type,
            confirmed_by=req.confirmed_by,
            notes=req.notes,
            cycle=req.cycle,
        )
        n_confirmed = len(load_confirmed())
        log.info(
            "supervisor.confirm_fraud",
            app_id=req.application_id,
            fraud_type=req.fraud_type,
            n_confirmed=n_confirmed,
        )
        return {
            "status": "ok",
            "application_id": req.application_id,
            "fraud_type": req.fraud_type,
            "n_confirmed": n_confirmed,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/mark-false-positive")
def mark_false_positive(req: FalsePositiveRequest):
    from src.confirmed_fraud_store import add_false_positive, load_false_positive_ids
    try:
        add_false_positive(
            app_id=req.application_id,
            confirmed_by=req.confirmed_by,
            notes=req.notes,
        )
        n_fp = len(load_false_positive_ids())
        log.info(
            "supervisor.mark_false_positive",
            app_id=req.application_id,
            n_false_positives=n_fp,
        )
        return {
            "status": "ok",
            "application_id": req.application_id,
            "n_false_positives": n_fp,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/clear-label")
def clear_label(req: ClearLabelRequest):
    """Undo a supervisor label — removes the application from the confirmed and
    false-positive stores so its state resets. Enables re-demoing the detection
    loop (label a topology → retrain → detected → clear → repeat) and correcting
    mis-clicks. 404 if the application had no label to clear."""
    from src.confirmed_fraud_store import remove_label
    result = remove_label(req.application_id)
    if not (result["removed_confirmed"] or result["removed_false_positive"]):
        raise HTTPException(
            status_code=404,
            detail=f"No label to clear for '{req.application_id}' (not in confirmed or false-positive store)",
        )
    log.info("supervisor.clear_label", app_id=req.application_id, **result)
    return {"status": "ok", "application_id": req.application_id, **result}


@router.get("/patterns")
def list_patterns():
    from src.confirmed_fraud_graph_store import list_pending, count_pending
    pending = list_pending()
    return {
        "pending_count": len(pending),
        "patterns": pending
    }


@router.post("/patterns/confirm")
def confirm_pattern(req: ConfirmPatternRequest):
    from src.confirmed_fraud_graph_store import add_confirmed_pattern
    try:
        pid = add_confirmed_pattern(
            app_id=req.application_id,
            fraud_type=req.fraud_type,
            subgraph=req.subgraph,
            confirmed_by=req.confirmed_by,
            notes=req.notes
        )
        log.info("supervisor.confirm_pattern", pattern_id=pid, app_id=req.application_id)
        return {"status": "ok", "pattern_id": pid}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/patterns/promote")
def promote_patterns(req: PromotePatternRequest):
    from src.confirmed_fraud_graph_store import select, promote
    try:
        select(req.pattern_ids)
        promoted = promote(req.pattern_ids)
        log.info("supervisor.promote_patterns", n_promoted=len(promoted), pattern_ids=req.pattern_ids)
        
        # Trigger retrain. Two bugs fixed here:
        #   1. The old code imported `run_incremental_finetune`, which does not
        #      exist in src/api/tasks.py — the real task is `run_incremental_task`
        #      (name="tasks.run_incremental"), taking (cycle, smoke_test). The old
        #      import would have raised ImportError on the first promote call.
        #   2. The old code manufactured a `job_id` string but dispatched with
        #      `.delay()`, so Celery assigned a DIFFERENT id. GET
        #      /v3/training/jobs/{job_id} polls by Celery task id via
        #      AsyncResult(job_id), so the returned job_id could never be found.
        #
        # Fix: pin the Celery task_id to our job_id via apply_async(task_id=...),
        # so the id returned to the client IS the id Celery tracks. Promotion
        # retrains on the existing (now pattern-augmented) data, so this uses the
        # incremental orchestrator — no dataset_path needed.
        import uuid
        job_id = f"job_retrain_{uuid.uuid4().hex[:8]}"

        from src.api.tasks import run_incremental_task
        run_incremental_task.apply_async(
            kwargs=dict(cycle="pattern_promotion", smoke_test=req.smoke_test),
            task_id=job_id,
        )

        return {
            "status": "ok",
            "message": f"Promoted {len(promoted)} patterns. Retrain dispatched.",
            "job_id": job_id,
            "promoted_pattern_ids": [p["pattern_id"] for p in promoted]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
