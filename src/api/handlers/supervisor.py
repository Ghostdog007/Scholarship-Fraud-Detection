"""
Supervisor feedback endpoints — ADR-005.

POST /v3/supervisor/confirm-fraud
POST /v3/supervisor/mark-false-positive
"""
import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas import ConfirmFraudRequest, FalsePositiveRequest

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
