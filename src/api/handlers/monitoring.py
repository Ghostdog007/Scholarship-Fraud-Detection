"""
Monitoring endpoints — ADR-005.

GET /v3/monitoring/drift
GET /v3/monitoring/fraud-store-summary
"""
from collections import Counter

import structlog
from fastapi import APIRouter

from src.api.schemas import DriftCheckResponse, FraudStoreSummaryResponse

log    = structlog.get_logger()
router = APIRouter()


@router.get("/drift", response_model=DriftCheckResponse)
def check_drift():
    from src.retraining_orchestrator import _check_drift
    p_value, recommendation = _check_drift()
    drift_detected = recommendation == "full_retrain"
    log.info("monitoring.drift", p_value=p_value, recommendation=recommendation, drift_detected=drift_detected)
    return DriftCheckResponse(
        p_value=p_value,
        recommendation=recommendation,
        drift_detected=drift_detected,
    )


@router.get("/fraud-store-summary", response_model=FraudStoreSummaryResponse)
def fraud_store_summary():
    from src.confirmed_fraud_store import load_confirmed, load_false_positive_ids
    confirmed = load_confirmed()
    fp_ids    = load_false_positive_ids()
    by_type   = dict(Counter(r["fraud_type"] for r in confirmed))
    log.info(
        "monitoring.fraud_store_summary",
        n_confirmed=len(confirmed),
        n_false_positives=len(fp_ids),
    )
    return FraudStoreSummaryResponse(
        n_confirmed=len(confirmed),
        n_false_positives=len(fp_ids),
        by_fraud_type=by_type,
    )
