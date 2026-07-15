"""
Monitoring endpoints — ADR-005 (+ drift-simulation extension).

GET  /v3/monitoring/drift
GET  /v3/monitoring/fraud-store-summary
POST /v3/monitoring/evaluate-dataset   — score a new/unseen dataset read-only, check drift
GET  /v3/monitoring/dataset-xai        — XAI preview for the top-N staged rows
GET  /v3/monitoring/top-suspicious     — top-N suspicious apps from the last full run
GET  /v3/monitoring/{app_id}/topology  — interactive topology view (HTML)
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

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


@router.get("/drift-explain")
def drift_explain(top: int = 12):
    """Plain-English drift report for the full-retrain decision — surfaces ONLY
    numbers the pipeline already computed:

      * overall score-distribution KS (src.retraining_orchestrator._check_drift):
        p-value + recommendation, judged against config DRIFT_KS_THRESHOLD.
      * per-feature KS (outputs/feature_drift_v3.json): the features that shifted
        most between the previous and current cohort, with their mean shift.

    No thresholds are invented here (hard stop #1) — the only cutoff cited is the
    existing DRIFT_KS_THRESHOLD, and the per-feature 'drifted' flags are read
    straight from the file. This endpoint narrates; it never re-decides.
    """
    from src.retraining_orchestrator import _check_drift

    p_value, recommendation = _check_drift()
    drift_detected = recommendation == "full_retrain"

    # Overall sentence — faithful to _check_drift's own branches.
    if recommendation == "no_scores":
        overall_text = ("No scored cohort is staged yet, so there is nothing to compare - "
                        "run an evaluation first.")
    elif recommendation == "first_run":
        overall_text = ("This is the first scored cohort, so there is no previous cycle to "
                        "compare against - no drift can be measured yet.")
    elif drift_detected:
        overall_text = (f"The score distribution shifted significantly: KS p-value {p_value:.4g} "
                        f"is below the alert threshold of {DRIFT_KS_THRESHOLD} (smaller p = a "
                        f"bigger shift), so a full GPU retrain is recommended before the next "
                        f"incremental update.")
    else:
        overall_text = (f"The score distribution is stable: KS p-value {p_value:.4g} is at or "
                        f"above the alert threshold of {DRIFT_KS_THRESHOLD}, so an incremental "
                        f"fine-tune is sufficient - no full retrain needed.")

    # Per-feature detail from the file the incremental cycle writes. The drift
    # file is computed over ALL raw columns, but only the model's feature set
    # (44 numeric features after the 2026-07-15 noid drop) actually feeds the
    # detector — so restrict the narrative to those. The 24 nominal identifier
    # columns dropped from the model are excluded from the drift count; showing
    # them would misrepresent what a retrain would relearn.
    feat_path   = Path("outputs/feature_drift_v3.json")
    schema_path = Path("data/processed/v3_feature_schema.json")
    model_features: set[str] | None = None
    if schema_path.exists():
        model_features = set(json.loads(schema_path.read_text()).get("features", [])) or None

    features_total = 0
    n_drifted = 0
    n_excluded_nonmodel = 0
    drifted_feats: list[dict] = []
    if feat_path.exists():
        fd_all = json.loads(feat_path.read_text())
        # Keep only model features when the schema is available; otherwise fall
        # back to every column (older runs without a schema file).
        if model_features is not None:
            fd = {k: v for k, v in fd_all.items() if k in model_features}
            n_excluded_nonmodel = len(fd_all) - len(fd)
        else:
            fd = fd_all
        features_total = len(fd)
        n_drifted = sum(1 for s in fd.values() if s.get("drifted"))
        rows = []
        for name, s in fd.items():
            mean_curr = s.get("mean_curr")
            mean_prev = s.get("mean_prev")
            shift = None
            direction = "unchanged"
            if isinstance(mean_curr, (int, float)) and isinstance(mean_prev, (int, float)):
                shift = mean_curr - mean_prev
                direction = "higher" if shift > 0 else "lower" if shift < 0 else "unchanged"
            rows.append({
                "feature":   name,
                "ks_stat":   s.get("ks_stat"),
                "p_value":   s.get("p_value"),
                "mean_prev": mean_prev,
                "mean_curr": mean_curr,
                "mean_shift": shift,
                "direction": direction,
                "drifted":   bool(s.get("drifted", False)),
                "explanation": (
                    f"'{name}' shifted {direction} (mean {mean_prev} -> {mean_curr}); "
                    f"KS {s.get('ks_stat')}, p={s.get('p_value')}"
                    + (" - flagged as drifted." if s.get("drifted") else " - within normal variation.")
                ),
            })
        # Drifted first, then by KS statistic descending (largest shift first).
        rows.sort(key=lambda r: (not r["drifted"], -(r["ks_stat"] or 0.0)))
        drifted_feats = rows[:max(0, top)]

    excluded_note = (f" ({n_excluded_nonmodel} non-model identifier columns excluded)"
                     if n_excluded_nonmodel else "")
    if features_total == 0:
        feature_text = "No per-feature drift file yet - run an incremental cycle to populate it."
    elif n_drifted == 0:
        feature_text = (f"None of the {features_total} model features drifted this cycle{excluded_note}.")
    else:
        feature_text = (f"{n_drifted} of {features_total} model features drifted{excluded_note}; "
                        f"the largest shifts are listed below.")

    log.info("monitoring.drift_explain", p_value=p_value, recommendation=recommendation,
             n_features=features_total, n_drifted=n_drifted, n_excluded_nonmodel=n_excluded_nonmodel)
    return {
        "recommendation":  recommendation,
        "drift_detected":  drift_detected,
        "p_value":         p_value,
        "ks_threshold":    DRIFT_KS_THRESHOLD,
        "overall_explanation": overall_text,
        "n_features":      features_total,
        "n_drifted":       n_drifted,
        "n_excluded_nonmodel": n_excluded_nonmodel,
        "feature_summary": feature_text,
        "top_features":    drifted_feats,
    }


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


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Browser CSV intake for the deployment loop. Saves the uploaded cohort
    under data/uploads/, then validates its columns against the canonical raw
    schema (data/raw/data_for_ml_model.csv) WITHOUT mutating anything. Returns
    the server-side path (to feed /evaluate-dataset or /training/decision) plus
    a schema report. `schema_ok` is False when required columns are missing —
    the UI should block evaluate/retrain in that case.
    """
    import shutil
    import re
    from src.api import dataset_ops

    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload must be a .csv file")

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename).name)
    dest_dir  = Path("data/uploads")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest      = dest_dir / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if not dataset_ops.RAW_CSV.exists():
        raise HTTPException(status_code=500, detail=f"Canonical raw schema not found: {dataset_ops.RAW_CSV}")

    try:
        base_cols = list(pd.read_csv(dataset_ops.RAW_CSV, nrows=0).columns)
        new_df    = pd.read_csv(dest, low_memory=False)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")

    base_set, new_set = set(base_cols), set(new_df.columns)
    missing = sorted(base_set - new_set)   # required by raw schema, absent → hard fail
    extra   = sorted(new_set - base_set)   # unexpected extras → warn only

    duplicate_ids: list[str] = []
    if "application_id" in new_df.columns and "application_id" in base_cols:
        base_ids = set(pd.read_csv(dataset_ops.RAW_CSV, usecols=["application_id"])["application_id"].astype(str))
        duplicate_ids = sorted(set(new_df["application_id"].astype(str)) & base_ids)[:10]

    report = {
        "dataset_path":     str(dest),
        "filename":         safe_name,
        "n_rows":           int(len(new_df)),
        "n_cols":           int(new_df.shape[1]),
        "expected_cols":    len(base_cols),
        "schema_ok":        not missing,
        "missing_columns":  missing,
        "extra_columns":    extra,
        "duplicate_ids":    duplicate_ids,
    }
    log.info("monitoring.upload_dataset", filename=safe_name, n_rows=report["n_rows"],
             schema_ok=report["schema_ok"], n_missing=len(missing))
    return report


@router.get("/top-suspicious")
def top_suspicious(n: int = 20):
    """Top-N applications by fused risk, for the reviewer queue.

    Sourced from outputs/risk_scores_v3.csv — the SAME file xai_layer_v3 ranks
    to generate the explanation cards — so every queued row is guaranteed to
    have a card (for n <= the number of cards). The older top_suspicious_v3.tsv
    is written by main_v3.py in a separate step and can drift out of sync across
    partial runs, which left queued rows with no card (404); it is used only as
    a fallback when risk_scores_v3.csv is absent.

    Non-finite floats are sanitized to null globally by SafeJSONResponse.
    """
    risk_path = Path("outputs/risk_scores_v3.csv")
    if risk_path.exists():
        df = (pd.read_csv(risk_path)
                .sort_values("risk_score_v3", ascending=False)
                .head(n))
        return df.to_dict(orient="records")

    tsv_path = Path("outputs/top_suspicious_v3.tsv")
    if tsv_path.exists():
        return pd.read_csv(tsv_path, sep="\t").head(n).to_dict(orient="records")

    raise HTTPException(
        status_code=404,
        detail="No outputs/risk_scores_v3.csv or top_suspicious_v3.tsv — run a full pipeline first",
    )


@router.get("/export/bulk")
def export_bulk():
    """
    Bulk export of every flagged application as one downloadable zip:
    manifest.csv (one scorecard row per app) + cards/<id>.html + evidence/<id>.json.
    Built entirely from outputs/explanation_cards_v3.json — no recomputation.
    """
    from fastapi.responses import Response
    from src.export_v3 import build_bulk_export
    result = build_bulk_export()
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No explanation cards to export — run the pipeline (XAI layer) first",
        )
    filename, data = result
    log.info("monitoring.export_bulk", filename=filename, bytes=len(data))
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/selected")
def export_selected(ids: str = ""):
    """
    Export a reviewer-chosen subset of flagged applications as one zip:
    manifest.csv + cards/<id>.html + evidence/<id>.json + rings/<id>.html
    (the interactive 3D identity ring, self-contained). `ids` is a
    comma-separated list of application IDs (from the queue's multi-select).

    IDs with no card are skipped and listed in _skipped.txt; IDs with a card but
    no graph edges get card+evidence but no ring. Built from existing artefacts —
    the only render is the lazy Plotly ring. 404 if none of the IDs have a card.
    """
    from fastapi.responses import Response
    from src.export_v3 import build_selected_export

    id_list = [s.strip() for s in ids.split(",") if s.strip()]
    if not id_list:
        raise HTTPException(status_code=400, detail="No application IDs supplied (?ids=a,b,c)")
    # Each ring embeds Plotly (~5 MB); cap the batch so a stray request can't
    # build a multi-GB zip. Operational guard, not a domain threshold.
    if len(id_list) > 200:
        raise HTTPException(
            status_code=400,
            detail=f"Too many IDs ({len(id_list)}); export at most 200 at once, or use Export all flagged",
        )

    result = build_selected_export(id_list)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="None of the selected applications have a card (not flagged, or cards not generated)",
        )
    filename, data = result
    log.info("monitoring.export_selected", n_requested=len(id_list), filename=filename, bytes=len(data))
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{app_id}/export")
def export_application(app_id: str):
    """
    Single-application export zip: <id>_scorecard.csv (flat audit row incl. the
    model-traceability summary) + <id>_card.html (reviewer card) + <id>_evidence.json.
    404 if the application has no card (not flagged, or cards not generated).
    """
    from fastapi.responses import Response
    from src.export_v3 import build_single_export
    result = build_single_export(app_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No card for this application — not flagged, or scores/cards not yet generated",
        )
    filename, data = result
    log.info("monitoring.export_app", app_id=app_id, filename=filename, bytes=len(data))
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{app_id}/topology")
def get_topology(app_id: str):
    from fastapi.responses import HTMLResponse
    from src.topology_view import extract_ego, render_html
    ego = extract_ego(app_id, hops=1, node_cap=50)
    if not ego:
        raise HTTPException(status_code=404, detail="Ego graph not found or application_id missing")
    html_content = render_html(ego)
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/{app_id}/card")
def get_card(app_id: str):
    """
    Interactive reviewer card (HTML) for one flagged application: the evidence
    of WHY it is suspicious — risk placement, ranked reason codes, per-field
    declared-vs-model-expected breakdown, the closed-form fusion split, and a
    lightweight identity ego-graph. Cheap and on-demand; reads the pre-computed
    explanation_cards_v3.json. Its "Examine full ring in 3D" link points at
    /{app_id}/ring, where the expensive Plotly view is rendered lazily on click.
    """
    from fastapi.responses import HTMLResponse
    from src.xai_card_html_v3 import build_card_html
    html_content = build_card_html(app_id)
    if html_content is None:
        raise HTTPException(
            status_code=404,
            detail="No card for this application — not flagged, or scores/cards not yet generated",
        )
    log.info("monitoring.card", app_id=app_id)
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/{app_id}/ring")
def get_ring(app_id: str):
    """
    Rotatable Plotly 3D identity ring (HTML) for one application — the deep-dive
    view. LAZY BY DESIGN: the ring is computed only on this request (i.e. when a
    reviewer clicks the card's link), so Plotly cost is paid per view, never
    per batch. Returns 404 if the application has no typed edges in the graph.
    """
    from fastapi.responses import HTMLResponse
    from src.xai_card_html_v3 import build_ring_html
    html_content = build_ring_html(app_id)
    if html_content is None:
        raise HTTPException(
            status_code=404,
            detail="Application not in identity graph (no shared IP/mobile/name/pincode edges)",
        )
    log.info("monitoring.ring", app_id=app_id)
    return HTMLResponse(content=html_content, status_code=200)
