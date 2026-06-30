"""
retraining_orchestrator.py

Single entry point for all post-cycle model updates.

Usage (end of each yearly cycle, after human review is complete):

    # Incremental update (default — runs on CPU server)
    python -m src.retraining_orchestrator --cycle 2025-26

    # Force full retrain check (will alert if KS drift requires GPU)
    python -m src.retraining_orchestrator --cycle 2025-26 --check-drift

    # Smoke test (2 epochs, verify pipeline doesn't crash)
    python -m src.retraining_orchestrator --cycle 2025-26 --smoke-test

Decision logic:
  - KS test on current vs previous cycle score distribution
  - p < DRIFT_KS_THRESHOLD → alert: full GPU retrain recommended
  - p >= threshold         → run incremental fine-tune on server

Incremental update order:
  1. Build exposure tensor (real confirmed fraud + synthetic fallback)
  2. Fine-tune GraphMCM: freeze RGCN, 10 epochs Stage 2
  3. Refit Subspace IF on current cycle data
  4. Refit EVT on updated scores
  5. Run self-training (confirmed fraud already in store — bypasses EVT)
  6. Retrain LightGBM fusion with confirmed fraud at 3× sample weight

MLflow (ADR-002): every incremental update run is logged under the
  "nic-fraud-detection-v3-incremental" experiment.
  - Tags: cycle label, n_confirmed_fraud
  - Metrics: KS p-value, PR-AUC after update (from evaluate_model_v3)
  - Artifacts: checkpoint after update

Feature drift (ADR-009): per-feature KS test logged to
  outputs/feature_drift_v3.json after each incremental cycle.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config_v3 import DRIFT_KS_THRESHOLD

_PROJECT_ROOT  = Path(__file__).parent.parent.resolve()
_MLRUNS_DIR    = _PROJECT_ROOT / "mlruns"
_MLFLOW_DB_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"

SCORES_CSV        = Path("outputs/hybrid_scores_v3.csv")
PREV_KS_JSON      = Path("outputs/prev_cycle_scores_ks.json")
FEATURE_CSV       = Path("data/processed/engineered_features_v3.csv")
FEATURE_DRIFT_JSON = Path("outputs/feature_drift_v3.json")


def _check_drift() -> tuple[float, str]:
    """
    KS test: current hybrid_anomaly_score distribution vs previous cycle.
    Returns (p_value, recommendation).
    """
    if not SCORES_CSV.exists():
        return 1.0, "no_scores"
    if not PREV_KS_JSON.exists():
        print("[orchestrator] No previous cycle scores found — skipping drift check (first run).")
        return 1.0, "first_run"

    import pandas as pd
    curr_scores = pd.read_csv(SCORES_CSV)["hybrid_anomaly_score"].values
    prev_scores = np.array(json.loads(PREV_KS_JSON.read_text())["scores"])

    stat, p = ks_2samp(curr_scores, prev_scores)
    print(f"[orchestrator] KS test: stat={stat:.4f}  p={p:.4f}")

    if p < DRIFT_KS_THRESHOLD:
        return p, "full_retrain"
    return p, "incremental"


def _save_current_scores_for_ks() -> None:
    """Save current cycle scores so next year's KS test has a baseline."""
    if not SCORES_CSV.exists():
        return
    scores = pd.read_csv(SCORES_CSV)["hybrid_anomaly_score"].values.tolist()
    PREV_KS_JSON.write_text(json.dumps({"scores": scores}))
    print(f"[orchestrator] Saved {len(scores)} scores for next cycle KS baseline -> {PREV_KS_JSON}")


def _check_feature_drift() -> dict:
    """
    ADR-009: per-feature KS test against previous cycle feature distributions.

    Compares every one of the 68 engineered features against the previous
    cycle's saved distribution (outputs/prev_cycle_features_ks.json).
    Writes results to outputs/feature_drift_v3.json.

    Returns a summary dict: {feature_name: {ks_stat, p_value, drifted}}.
    Flags features where p < 0.01 as drifted. Alerts but does not block.
    """
    PREV_FEAT_JSON = Path("outputs/prev_cycle_features_ks.json")

    if not FEATURE_CSV.exists():
        print("[orchestrator] Feature CSV not found — skipping feature drift check.")
        return {}
    if not PREV_FEAT_JSON.exists():
        print("[orchestrator] No previous cycle feature snapshot — saving current as baseline.")
        curr_df = pd.read_csv(FEATURE_CSV)
        snapshot = {col: curr_df[col].tolist() for col in curr_df.columns if col != "application_id"}
        PREV_FEAT_JSON.write_text(json.dumps(snapshot))
        return {}

    curr_df  = pd.read_csv(FEATURE_CSV)
    prev_raw = json.loads(PREV_FEAT_JSON.read_text())

    drift_report: dict = {}
    drifted_features: list[str] = []

    for col in curr_df.columns:
        if col == "application_id" or col not in prev_raw:
            continue
        curr_vals = curr_df[col].dropna().values
        prev_vals = np.array(prev_raw[col])
        stat, p   = ks_2samp(curr_vals, prev_vals)
        drifted   = bool(p < 0.01)
        drift_report[col] = {
            "ks_stat":    round(float(stat), 6),
            "p_value":    round(float(p),    6),
            "mean_curr":  round(float(curr_vals.mean()), 6),
            "mean_prev":  round(float(prev_vals.mean()), 6),
            "drifted":    drifted,
        }
        if drifted:
            drifted_features.append(col)

    FEATURE_DRIFT_JSON.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_DRIFT_JSON.write_text(json.dumps(drift_report, indent=2))
    print(f"[orchestrator] Feature drift check: {len(drifted_features)}/{len(drift_report)} features drifted (p<0.01)")
    if drifted_features:
        print(f"[orchestrator] Drifted features: {drifted_features}")
        print(f"[orchestrator] Full drift report -> {FEATURE_DRIFT_JSON}")
    return drift_report


def _save_feature_snapshot() -> None:
    """Save current feature distributions as baseline for next cycle's drift check."""
    if not FEATURE_CSV.exists():
        return
    PREV_FEAT_JSON = Path("outputs/prev_cycle_features_ks.json")
    curr_df  = pd.read_csv(FEATURE_CSV)
    snapshot = {col: curr_df[col].tolist() for col in curr_df.columns if col != "application_id"}
    PREV_FEAT_JSON.write_text(json.dumps(snapshot))
    print(f"[orchestrator] Feature snapshot saved -> {PREV_FEAT_JSON}")


def run_incremental_update(smoke_test: bool = False, cycle: str = "unknown") -> None:
    """Full incremental update pipeline — safe to run on CPU server."""
    import torch
    from src.confirmed_fraud_store import get_exposure_tensor, summary as fraud_summary
    from src.confirmed_fraud_store import load_confirmed
    from src.hybrid_graphmcm_v3 import train_incremental
    from src.evt_scorer_v3 import run_evt
    from src.self_training_loop_v3 import run_self_training
    from src.fusion_classifier_v3 import run_fusion

    try:
        import mlflow
        mlflow_available = True
    except ImportError:
        mlflow_available = False

    def _body() -> dict:
        print("\n[orchestrator] ══ Incremental update pipeline starting ══")
        fraud_summary()

        # ADR-009: feature drift check before updating model
        drift_report = _check_feature_drift()

        EXPOSURE_PT = Path("data/processed/synthetic_exposure_set_v3.pt")
        synthetic   = torch.load(EXPOSURE_PT, weights_only=True)
        exposure, source = get_exposure_tensor(synthetic_fallback=synthetic)
        print(f"[orchestrator] Exposure source: {source}")

        n_confirmed = len(load_confirmed())
        freeze_rgcn = n_confirmed < 50
        print(f"[orchestrator] freeze_rgcn={freeze_rgcn} (confirmed={n_confirmed})")

        train_incremental(
            exposure_tensor=exposure,
            freeze_rgcn=freeze_rgcn,
            smoke_test=smoke_test,
        )

        try:
            from src.subspace_if_v3 import run_subspace_if
            run_subspace_if()
        except ImportError:
            print("[orchestrator] subspace_if_v3 not found — skipping subspace IF refit")

        run_evt()
        run_self_training(current_round=0)
        run_fusion()

        # Evaluate post-update PR-AUC
        metrics: dict = {}
        try:
            from src.evaluate_model_v3 import evaluate
            metrics = evaluate() or {}
        except Exception as e:
            print(f"[orchestrator] Post-update evaluation failed: {e}")

        _save_current_scores_for_ks()
        _save_feature_snapshot()

        print("[orchestrator] ══ Incremental update complete ══\n")
        return {"n_confirmed": n_confirmed, "freeze_rgcn": freeze_rgcn,
                "n_drifted_features": len([k for k, v in drift_report.items() if v.get("drifted")]),
                **metrics}

    if mlflow_available:
        import mlflow
        mlflow.set_tracking_uri(_MLFLOW_DB_URI)
        mlflow.set_experiment("nic-fraud-detection-v3-incremental")
        with mlflow.start_run(tags={"cycle": cycle, "smoke_test": str(smoke_test), "run_type": "incremental"}):
            result = _body()
            mlflow.log_param("cycle", cycle)
            mlflow.log_metric("n_confirmed_fraud", result.get("n_confirmed", 0))
            mlflow.log_metric("n_drifted_features", result.get("n_drifted_features", 0))
            eval_metrics = {k: v for k, v in result.items()
                            if k.startswith("pr_auc_") or k in ("score_retention", "n_categories_pass")}
            if eval_metrics:
                mlflow.log_metrics(eval_metrics)
            ckpt = Path("models/hybrid_graphmcm_v3.pth")
            if ckpt.exists():
                mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
            if FEATURE_DRIFT_JSON.exists():
                mlflow.log_artifact(str(FEATURE_DRIFT_JSON), artifact_path="drift")
            print("[orchestrator] MLflow run logged for incremental update.")
    else:
        _body()


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-cycle model update orchestrator")
    parser.add_argument("--cycle",       type=str,            help="Cycle label e.g. 2025-26")
    parser.add_argument("--check-drift", action="store_true", help="Run KS drift check before update")
    parser.add_argument("--smoke-test",  action="store_true", help="2-epoch smoke test")
    args = parser.parse_args()

    cycle = args.cycle or "unknown"

    if args.check_drift:
        p, recommendation = _check_drift()
        if recommendation == "full_retrain":
            print(
                f"\n[orchestrator] DRIFT DETECTED (p={p:.4f} < {DRIFT_KS_THRESHOLD})\n"
                f"  Score distribution has shifted significantly from the previous cycle.\n"
                f"  Recommendation: run full GPU retrain on the laptop before incremental update.\n"
                f"  To proceed with incremental anyway: rerun without --check-drift.\n"
            )
            return
        print(f"[orchestrator] Drift OK (p={p:.4f}) — proceeding with incremental update.")

    run_incremental_update(smoke_test=args.smoke_test, cycle=cycle)


if __name__ == "__main__":
    main()
