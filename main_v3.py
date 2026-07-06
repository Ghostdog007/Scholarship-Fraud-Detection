"""
main_v3.py — V3 Hybrid GraphMCM Pipeline Orchestrator

Build order enforced per AGENTS.md §8:
  1. feature_engine.build_base()            -> engineered_features_v3_nodeg.csv (63 features)
  2. graph_builder.build_graph()            -> identity_graph_v3.pt + degree_features_v3.csv
  3. feature_engine.add_degree_features()   -> engineered_features_v3.csv (68 features)
  4. synthetic_exposure_builder.build_exposure_set()
  5. hybrid_graphmcm.train()
  6. subspace_if.run_subspace_if()
  7. evt_scorer.run_evt()
  8. self_training_loop.run_self_training(round=0)
  9. fusion_classifier.run_fusion()
  10. xai_layer.run_xai()
  11. evaluate_model.evaluate()

HARD STOP: self-training round advancement requires Phase D PR-AUC check by project lead.
           This orchestrator always runs round=0. Do not modify to auto-advance.

MLflow (ADR-002): every full pipeline run is logged as an MLflow experiment run.
  - Params: all src/config_v3.py constants
  - Metrics: PR-AUC per category, score_retention, n_categories_pass
  - Artifacts: models/hybrid_graphmcm_v3.pth checkpoint
  Tracking URI: mlruns/ (local file-based, no server required).
  UI: mlflow ui  (open http://localhost:5000)
"""

import sys
import time
from pathlib import Path

# SQLite-backed MLflow store — survives any working-directory launch.
_PROJECT_ROOT  = Path(__file__).parent.resolve()
_MLRUNS_DIR    = _PROJECT_ROOT / "mlruns"          # kept for mlflow ui --backend-store-uri
_MLFLOW_DB_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"


PIPELINE_STEPS = [
    "build_base",
    "build_graph",
    "add_degree_features",
    "build_exposure_set",
    "train_hybrid",
    "subspace_if",
    "dense_block",
    "evt",
    "self_training",
    "fusion",
    "xai",
    "evaluate",
]


def _log_config_params(mlflow_run) -> None:
    """Log all config_v3 constants as MLflow params for reproducibility."""
    import mlflow
    import src.config_v3 as cfg

    param_names = [
        "N_FEATURES", "N_EDGE_TYPES", "MASK_NUM", "GRAPH_HIDDEN", "GRAPH_EMB_DIM",
        "MLP_HIDDEN", "Z_DIM", "LOE_MARGIN", "LAMBDA_EDGE", "LAMBDA_EXPOSURE",
        "EPOCHS_STAGE1", "EPOCHS_STAGE2", "LR", "BATCH_SIZE", "RANDOM_SEED",
        "INCREMENTAL_EPOCHS", "INCREMENTAL_LR", "CONFIRMED_WEIGHT",
        "DRIFT_KS_THRESHOLD", "MIN_SIGNALS_FOR_PROMOTION",
        "EVT_SHAPE_MIN", "EVT_SHAPE_MAX", "CENTROID_CLEAN_PERCENTILE",
    ]
    for name in param_names:
        val = getattr(cfg, name, None)
        if val is not None:
            mlflow.log_param(name, val)


def _log_top_suspicious(risk_path: Path, xai_path: Path, top_n: int = 20) -> None:
    """
    Build a compact TSV of the top-N suspicious applications (risk score + XAI
    narrative) and log it to MLflow as 'suspicious/top_suspicious.tsv'.

    This file appears on the MLflow Artifacts tab and gives an instant human-
    readable view of the highest-risk applications plus their explanations.
    """
    import json
    import csv
    import io
    import mlflow

    try:
        import pandas as pd
        raw = pd.read_csv(risk_path)
        score_col = "risk_score_v3" if "risk_score_v3" in raw.columns else "risk_score"
        risk_df = raw.sort_values(score_col, ascending=False).head(top_n)

        narratives: dict = {}
        if xai_path.exists():
            cards = json.loads(xai_path.read_text(encoding="utf-8"))
            narratives = {c["application_id"]: c for c in cards}

        buf = io.StringIO()
        writer = csv.writer(buf, delimiter="\t")
        writer.writerow([
            "rank", "application_id", "risk_score_v3",
            "hybrid_ae", "mcm_err", "top_errors", "triggers",
            "linked_applications", "review_status", "narrative",
        ])

        for rank, (_, row) in enumerate(risk_df.iterrows(), start=1):
            app_id = str(row["application_id"])
            card   = narratives.get(app_id, {})
            def _pct_note(e: dict) -> str:
                # 2026-07-03: cards no longer carry the hand-bucketed 'magnitude'
                # word — use the population value percentile instead.
                vp = e.get("value_percentile")
                if vp is None:
                    return ""
                return f" (>{vp:.1f} pct of applicants)" if vp >= 50.0 else f" (<{100.0 - vp:.1f} pct of applicants)"

            top_errors = "; ".join(
                f"{e.get('feature_label', e.get('feature','?'))}={e.get('error', 0.0):.3f}"
                + _pct_note(e)
                for e in card.get("top_feature_errors", [])[:3]
            )
            triggers = "; ".join(card.get("triggers", []))
            linked = "; ".join(
                f"{n.get('application_id','?')} via {n.get('edge_type','?')}"
                for n in card.get("top_graph_neighbors", [])
            )
            neighbor = linked
            narrative = card.get("narrative", "")
            writer.writerow([
                rank, app_id, f"{row[score_col]:.4f}",
                f"{card.get('hybrid_anomaly_score', 0.0):.4f}",
                f"{card.get('feature_pred_error', 0.0):.4f}",
                top_errors, triggers, neighbor,
                card.get("review_status", ""),
                narrative,
            ])

        tsv_path = Path("outputs/top_suspicious_v3.tsv")
        tsv_path.write_text(buf.getvalue(), encoding="utf-8")
        mlflow.log_artifact(str(tsv_path), artifact_path="suspicious")
        print(f"[main] Logged top-{top_n} suspicious applications -> MLflow artifact 'suspicious/top_suspicious_v3.tsv'")

        try:
            import src.topology_view as tv
            for _, row in risk_df.iterrows():
                app_id = str(row["application_id"])
                ego = tv.extract_ego(app_id, hops=1, node_cap=50)
                if ego and ego.get("nodes"):
                    svg_content = tv.render_svg(ego)
                    svg_path = Path(f"outputs/topology_{app_id}.svg")
                    svg_path.write_text(svg_content, encoding="utf-8")
                    mlflow.log_artifact(str(svg_path), artifact_path="topology")
                    svg_path.unlink(missing_ok=True)
            print(f"[main] Logged topology SVGs -> MLflow artifact 'topology/'")
        except Exception as svg_exc:
            print(f"[main] Warning: could not render SVGs: {svg_exc}")

    except Exception as exc:
        print(f"[main] Warning: could not build top-suspicious TSV: {exc}")


def run_pipeline(steps: list[str] | None = None, smoke_test: bool = False, cycle: str = "manual") -> None:
    try:
        import mlflow
        mlflow_available = True
    except ImportError:
        mlflow_available = False
        print("[main] mlflow not installed — running without experiment tracking")

    run_all = steps is None
    def should_run(name: str) -> bool:
        return run_all or name in (steps or [])

    t0 = time.time()

    def _pipeline_body():
        if should_run("build_base"):
            print("\n[main] Step 1: Feature engine -- build_base()")
            from src.tabular_feature_engine_v3 import build_base
            build_base()

        if should_run("build_graph"):
            print("\n[main] Step 2: Graph builder -- build_graph()")
            from src.graph_builder_v3 import build_graph
            build_graph()

        if should_run("add_degree_features"):
            print("\n[main] Step 3: Feature engine -- add_degree_features()")
            from src.tabular_feature_engine_v3 import add_degree_features
            add_degree_features()

        if should_run("build_exposure_set"):
            print("\n[main] Step 4: Synthetic exposure -- build_exposure_set()")
            from src.synthetic_exposure_builder_v3 import build_exposure_set
            build_exposure_set()
            # V4 (ADR-016): also build the connected-cluster topology exposure pack
            # so train() finds it when TOPO_EXPOSURE_ENABLED is on. Cheap; harmless
            # when topology exposure is disabled (the pack simply goes unused).
            from src.config_v3 import TOPO_EXPOSURE_ENABLED
            if TOPO_EXPOSURE_ENABLED:
                from src.synthetic_exposure_builder_v3 import build_topology_exposure
                build_topology_exposure()

        if should_run("train_hybrid"):
            print("\n[main] Step 5: Hybrid GraphMCM -- train()")
            from src.hybrid_graphmcm_v3 import train
            train(smoke_test=smoke_test)

        if should_run("subspace_if"):
            print("\n[main] Step 6: Subspace IF -- run_subspace_if()")
            from src.subspace_if_v3 import run_subspace_if
            run_subspace_if()

        if should_run("dense_block"):
            print("\n[main] Step 6b: Dense-block detector (IP) -- run_dense_block()")
            from src.dense_block_detector_v3 import run_dense_block
            run_dense_block()

        if should_run("evt"):
            print("\n[main] Step 7: EVT scorer -- run_evt()")
            from src.evt_scorer_v3 import run_evt
            run_evt()

        if should_run("self_training"):
            print("\n[main] Step 8: Self-training -- run_self_training(round=0)")
            from src.self_training_loop_v3 import run_self_training
            run_self_training(current_round=0)

        if should_run("fusion"):
            print("\n[main] Step 9: Fusion classifier -- run_fusion()")
            from src.fusion_classifier_v3 import run_fusion
            run_fusion()

        if should_run("xai"):
            print("\n[main] Step 10: XAI layer -- run_xai()")
            from src.xai_layer_v3 import run_xai
            run_xai()
            # Render the lightweight interactive reviewer cards for the flagged
            # (suspicious) applications only. ring_mode="api" keeps them simple —
            # the 3D Plotly ring is computed lazily by the API on link-click, so
            # nothing heavy is produced here even when frauds are many.
            try:
                from src.xai_card_html_v3 import render_cards
                n_cards = render_cards(suspicious_only=True, ring_mode="api")
                print(f"[main] Rendered {n_cards} suspicious reviewer card(s) -> outputs/cards/")
            except Exception as e:
                print(f"[main] card rendering skipped ({e})")

        metrics = {}
        if should_run("evaluate"):
            print("\n[main] Step 11: Evaluate -- evaluate()")
            from src.evaluate_model_v3 import evaluate
            metrics = evaluate() or {}

        return metrics

    def _log_artifacts_to_mlflow() -> None:
        """Log all output artifacts to the active MLflow run."""
        import mlflow

        ckpt = Path("models/hybrid_graphmcm_v3.pth")
        if ckpt.exists():
            mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")

        xai_path = Path("outputs/explanation_cards_v3.json")
        if xai_path.exists():
            mlflow.log_artifact(str(xai_path), artifact_path="xai")

        # Interactive reviewer-card gallery (simplistic; lazy 3D links resolve
        # against the live API). Tens of KB/card — no Plotly bundle logged.
        cards_dir = Path("outputs/cards")
        if cards_dir.exists():
            mlflow.log_artifacts(str(cards_dir), artifact_path="cards")

        risk_path = Path("outputs/risk_scores_v3.csv")
        if risk_path.exists():
            mlflow.log_artifact(str(risk_path), artifact_path="scores")
            _log_top_suspicious(risk_path, xai_path)

        evt_path = Path("outputs/evt_thresholds_v3.json")
        if evt_path.exists():
            mlflow.log_artifact(str(evt_path), artifact_path="thresholds")

        pseudo_path = Path("outputs/pseudo_labels_v3.json")
        if pseudo_path.exists():
            mlflow.log_artifact(str(pseudo_path), artifact_path="labels")

    if mlflow_available:
        import mlflow
        mlflow.set_tracking_uri(_MLFLOW_DB_URI)
        mlflow.set_experiment("nic-fraud-detection-v3")
        run_tag = "full" if run_all else f"steps:{','.join(steps or [])}"
        with mlflow.start_run(tags={"cycle": cycle, "smoke_test": str(smoke_test), "run_mode": run_tag}):
            _log_config_params(None)
            metrics = _pipeline_body()
            if metrics:
                mlflow.log_metrics(metrics)
            elapsed = time.time() - t0
            mlflow.log_metric("pipeline_duration_seconds", elapsed)
            _log_artifacts_to_mlflow()
            print(f"\n[main] MLflow run logged. View with: mlflow ui")
    else:
        metrics = _pipeline_body()

    elapsed = time.time() - t0
    print(f"\n[main] Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    cycle_arg  = next((a for a in sys.argv if a.startswith("--cycle=")),  None)
    steps_arg  = next((a for a in sys.argv if a.startswith("--steps=")),  None)
    cycle_label = cycle_arg.split("=", 1)[1] if cycle_arg else "manual"
    step_list   = steps_arg.split("=", 1)[1].split(",") if steps_arg else None

    if smoke:
        print("[main] Smoke-test mode: 2 epochs per stage")
    if step_list:
        print(f"[main] Step-mode: running only [{', '.join(step_list)}]")
    run_pipeline(steps=step_list, smoke_test=smoke, cycle=cycle_label)
