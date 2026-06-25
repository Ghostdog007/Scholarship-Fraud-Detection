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
"""

import sys
import time

PIPELINE_STEPS = [
    "build_base",
    "build_graph",
    "add_degree_features",
    "build_exposure_set",
    "train_hybrid",
    "subspace_if",
    "evt",
    "self_training",
    "fusion",
    "xai",
    "evaluate",
]


def run_pipeline(steps: list[str] | None = None, smoke_test: bool = False) -> None:
    run_all = steps is None
    def should_run(name: str) -> bool:
        return run_all or name in (steps or [])

    t0 = time.time()

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

    if should_run("train_hybrid"):
        print("\n[main] Step 5: Hybrid GraphMCM -- train()")
        from src.hybrid_graphmcm_v3 import train
        train(smoke_test=smoke_test)

    if should_run("subspace_if"):
        print("\n[main] Step 6: Subspace IF -- run_subspace_if()")
        from src.subspace_if_v3 import run_subspace_if
        run_subspace_if()

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

    if should_run("evaluate"):
        print("\n[main] Step 11: Evaluate -- evaluate()")
        from src.evaluate_model_v3 import evaluate
        evaluate()

    elapsed = time.time() - t0
    print(f"\n[main] Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        print("[main] Smoke-test mode: 2 epochs per stage")
    run_pipeline(smoke_test=smoke)
