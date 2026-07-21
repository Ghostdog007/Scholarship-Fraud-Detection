"""
Pydantic request/response models for the V3 supervisor API (ADR-005).
"""
from typing import Any, Optional
from pydantic import BaseModel


# ── Supervisor feedback ───────────────────────────────────────────────────────

class ConfirmFraudRequest(BaseModel):
    application_id: str
    fraud_type: str       # must be one of confirmed_fraud_store.VALID_FRAUD_TYPES
    confirmed_by: str
    notes: str = ""
    cycle: str = ""


class FalsePositiveRequest(BaseModel):
    application_id: str
    confirmed_by: str
    notes: str = ""


class ClearLabelRequest(BaseModel):
    application_id: str


class ConfirmPatternRequest(BaseModel):
    application_id: str
    fraud_type: str
    subgraph: dict
    confirmed_by: str
    notes: str = ""


class PromotePatternRequest(BaseModel):
    pattern_ids: list[str]
    smoke_test: bool = False


class DeletePatternRequest(BaseModel):
    """Hard-delete flagged patterns from the store (flagged-history cleanup).
    Removes the RECORD only — it does not retrain or unwind any exposure cluster
    already written to synthetic_exposure_graph_v3.pt."""
    pattern_ids: list[str]


class PatternTestRequest(BaseModel):
    """Read-only detection test for a supervisor-uploaded fraud ring (already
    uploaded via /v3/monitoring/upload-dataset — pass its server-side path)."""
    dataset_path: str


class PatternIngestRequest(BaseModel):
    """Permanently ingest a supervisor ring as a relational pattern (topology
    exposure + confirmed stores), then optionally dispatch the human-gated
    incremental fine-tune."""
    dataset_path: str
    fraud_type: str           # one of confirmed_fraud_store.VALID_FRAUD_TYPES
    confirmed_by: str
    notes: str = ""
    cycle: str = ""
    dispatch_retrain: bool = False
    smoke_test: bool = False


class BatchLabelItem(BaseModel):
    application_id: str
    verdict: str              # "confirmed_fraud" | "false_positive"
    fraud_type: str = ""      # required when verdict == "confirmed_fraud"
    notes: str = ""


class ConfirmBatchRequest(BaseModel):
    """Label a reviewer-selected batch in one call, then optionally dispatch the
    human-gated incremental fine-tune. dispatch_retrain is the explicit human
    gate — recording labels never auto-advances training on its own."""
    items: list[BatchLabelItem]
    confirmed_by: str
    cycle: str = ""
    dispatch_retrain: bool = False
    smoke_test: bool = False


# ── Training jobs ─────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str = ""


class JobStatusResponse(BaseModel):
    job_id: str
    status: str           # pending | running | complete | failed
    result: Optional[Any] = None
    error: Optional[str] = None


# ── Monitoring ────────────────────────────────────────────────────────────────

class DriftCheckResponse(BaseModel):
    p_value: float
    recommendation: str   # incremental | full_retrain | first_run | no_scores
    drift_detected: bool


class FraudStoreSummaryResponse(BaseModel):
    n_confirmed: int
    n_false_positives: int
    by_fraud_type: dict[str, int]


# ── Drift simulation / human-gated retraining ────────────────────────────────

class EvaluateDatasetRequest(BaseModel):
    dataset_path: str   # e.g. "data/raw/new_cohort_2026.csv" — raw schema, unseen application_ids


class EvaluateDatasetResponse(BaseModel):
    dataset_path: str
    n_rows: int
    p_value: float
    recommendation: str    # incremental | full_retrain | first_run | no_scores
    drift_detected: bool
    staged_scores_path: str


class DecisionRequest(BaseModel):
    dataset_path: str
    action: str             # none | incremental | full_retrain
    cycle: str = "unknown"
    decided_by: str
    smoke_test: bool = False


class DecisionResponse(BaseModel):
    status: str
    action: str
    dataset_path: str
    job_id: Optional[str] = None
    backup_dir: Optional[str] = None
    audit_log_path: str


# ── Model / checkpoint ────────────────────────────────────────────────────────

class CheckpointInfoResponse(BaseModel):
    exists: bool
    size_mb: Optional[float] = None
    n_features: Optional[int] = None
    graph_emb_dim: Optional[int] = None
    n_edge_types: Optional[int] = None
    versioned_checkpoints: list[str] = []


class RollbackRequest(BaseModel):
    versioned_path: str   # e.g. "models/checkpoints/hybrid_v3_2025-26_abc12345.pth"
