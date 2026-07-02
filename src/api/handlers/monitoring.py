"""
Monitoring endpoints — ADR-005 (+ drift-simulation extension).

GET  /v3/monitoring/drift
GET  /v3/monitoring/fraud-store-summary
POST /v3/monitoring/evaluate-dataset   — score a new/unseen dataset read-only, check drift
GET  /v3/monitoring/dataset-xai        — XAI preview for the top-N staged rows
GET  /v3/monitoring/top-suspicious     — top-N suspicious apps from the last full run
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    DriftCheckResponse,
    EvaluateDatasetRequest,
    EvaluateDatasetResponse,
    FraudStoreSummaryResponse,
)
from src.config_v3 import DRIFT_KS_THRESHOLD

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


@router.post("/evaluate-dataset", response_model=EvaluateDatasetResponse)
def evaluate_dataset(req: EvaluateDatasetRequest):
    """
    Read-only: temporarily merges dataset_path into the raw CSV, rebuilds
    features + graph, scores the new rows with the CURRENT checkpoint (no
    training), then restores the canonical files. Leaves no lasting change —
    safe to call repeatedly. Writes a staged preview to outputs/staged_scores_*.
    """
    import numpy as np
    import shutil
    from scipy.stats import ks_2samp

    from src.api import dataset_ops

    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=422, detail=f"dataset_path not found: {req.dataset_path}")

    new_df  = pd.read_csv(dataset_path, low_memory=False)
    new_ids = set(new_df["application_id"].astype(str))

    backup_dir = dataset_ops.backup_canonical_files(label="eval")
    try:
        dataset_ops.merge_dataset_into_raw(dataset_path)
        dataset_ops.rebuild_features_and_graph()

        from src.api.inference import score_dataset_only
        staged_df = score_dataset_only(app_ids_to_return=new_ids)

        merged_features = pd.read_csv(dataset_ops.FINAL_CSV)
        staged_features = merged_features[
            merged_features["application_id"].astype(str).isin(new_ids)
        ]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        dataset_ops.restore_canonical_files(backup_dir)
        shutil.rmtree(backup_dir, ignore_errors=True)

    name = dataset_path.stem
    staged_scores_path   = Path(f"outputs/staged_scores_{name}.csv")
    staged_features_path = Path(f"outputs/staged_features_{name}.csv")
    staged_scores_path.parent.mkdir(parents=True, exist_ok=True)
    staged_df.to_csv(staged_scores_path, index=False)
    staged_features.to_csv(staged_features_path, index=False)

    baseline_path = Path("outputs/prev_cycle_scores_ks.json")
    if baseline_path.exists():
        baseline = np.array(json.loads(baseline_path.read_text())["scores"])
        stat, p  = ks_2samp(staged_df["hybrid_anomaly_score"].values, baseline)
        recommendation = "full_retrain" if p < DRIFT_KS_THRESHOLD else "incremental"
        drift_detected  = recommendation == "full_retrain"
    else:
        p, recommendation, drift_detected = 1.0, "first_run", False

    meta = {
        "dataset_path": str(dataset_path), "n_rows": len(staged_df),
        "p_value": float(p), "recommendation": recommendation, "drift_detected": drift_detected,
    }
    Path(f"outputs/staged_scores_meta_{name}.json").write_text(json.dumps(meta, indent=2))

    log.info("monitoring.evaluate_dataset", **meta)

    return EvaluateDatasetResponse(
        dataset_path=str(dataset_path),
        n_rows=len(staged_df),
        p_value=float(p),
        recommendation=recommendation,
        drift_detected=drift_detected,
        staged_scores_path=str(staged_scores_path),
    )


@router.get("/dataset-xai")
def dataset_xai(dataset_path: str, top_n: int = 20):
    """
    XAI preview for the top-N highest-scoring rows from the last
    /evaluate-dataset call on this dataset_path. Pre-fusion (hybrid model
    score only — no risk_score_v3/EVT triggers yet, since those require a
    committed pipeline run).
    """
    from src.xai_layer_v3 import _top_features, _narrative

    name = Path(dataset_path).stem
    staged_scores_path   = Path(f"outputs/staged_scores_{name}.csv")
    staged_features_path = Path(f"outputs/staged_features_{name}.csv")
    if not staged_scores_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"No staged scores for {dataset_path} — call POST /v3/monitoring/evaluate-dataset first",
        )

    scores_df   = pd.read_csv(staged_scores_path).sort_values("hybrid_anomaly_score", ascending=False).head(top_n)
    features_df = pd.read_csv(staged_features_path) if staged_features_path.exists() else pd.DataFrame()

    cards = []
    for _, row in scores_df.iterrows():
        per_feat = json.loads(row["per_feature_error_json"])
        predicted = (
            json.loads(row["per_feature_predicted_json"])
            if "per_feature_predicted_json" in scores_df.columns
            else None
        )
        actual_vals = {}
        if not features_df.empty:
            match = features_df[features_df["application_id"].astype(str) == str(row["application_id"])]
            if not match.empty:
                actual_vals = match.iloc[0].to_dict()

        top_feats = _top_features(per_feat, actual_vals, k=5, predicted=predicted)
        pseudo_card = {
            "risk_score_v3": float(row["hybrid_anomaly_score"]),
            "triggers": [],
            "top_graph_neighbors": [],
            "top_feature_errors": top_feats,
        }
        cards.append({
            "application_id":       str(row["application_id"]),
            "hybrid_anomaly_score": float(row["hybrid_anomaly_score"]),
            "top_feature_errors":   top_feats,
            "narrative":            _narrative(pseudo_card) + " [PREVIEW — pre-fusion, not yet in production scores]",
        })

    return {"dataset_path": dataset_path, "n_cards": len(cards), "cards": cards}


@router.get("/top-suspicious")
def top_suspicious(n: int = 20):
    """Top-N suspicious applications from the last full pipeline run (main_v3.py)."""
    tsv_path = Path("outputs/top_suspicious_v3.tsv")
    if not tsv_path.exists():
        raise HTTPException(status_code=404, detail="outputs/top_suspicious_v3.tsv not found — run a full pipeline first")
    df = pd.read_csv(tsv_path, sep="\t")
    return df.head(n).to_dict(orient="records")
