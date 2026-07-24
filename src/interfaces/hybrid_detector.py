"""
Stable entry point for the Hybrid GraphMCM detector (RGCN, root_weight=False)
and its XAI-only relation ablation. Produces hybrid_anomaly_score = feature_
pred_error + 0.3*edge_pred_error. NOT the fusion layer by itself — see
src.interfaces.fusion.
Concrete implementation: src/hybrid_graphmcm_v3.py — import from here, not the
_v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.hybrid_graphmcm_v3 import (
    train,
    train_sampled,
    train_incremental,
    score_only,
    score_sampled,
    compute_score_frame,
    load_model_and_inputs,
    compute_relation_ablation,
    run_relation_ablation,
    run_attention_summary,
)
