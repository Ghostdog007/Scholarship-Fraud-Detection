"""
Read-only scoring for staged / not-yet-committed datasets.

Loads the *existing* trained checkpoint and scores whatever is currently at
the canonical FINAL_CSV/GRAPH_PT paths (caller is responsible for having
merged + rebuilt those first — see dataset_ops.py). Does NOT train, does NOT
overwrite the checkpoint or outputs/hybrid_scores_v3.csv. Mirrors the scoring
block of hybrid_graphmcm_v3.train() exactly (same normalization) so staged
scores are comparable to canonical ones.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config_v3 import LAMBDA_EDGE, N_EDGE_TYPES, N_FEATURES

FINAL_CSV   = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON = Path("data/processed/v3_feature_schema.json")
GRAPH_PT    = Path("data/processed/identity_graph_v3.pt")
MODEL_PTH   = Path("models/hybrid_graphmcm_v3.pth")


def score_dataset_only(app_ids_to_return: set[str] | None = None) -> pd.DataFrame:
    """
    Score every node currently in FINAL_CSV/GRAPH_PT with the existing checkpoint.
    If app_ids_to_return is given, the returned frame is filtered to just those
    rows — normalization is still computed over the full population, matching
    how the canonical pipeline would score them.
    """
    from src.hybrid_graphmcm_v3 import (
        HybridGraphMCM,
        _build_edge_index_and_types,
        _compute_isolated_mask,
    )

    if not MODEL_PTH.exists():
        raise FileNotFoundError(f"No checkpoint at {MODEL_PTH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schema    = json.loads(SCHEMA_JSON.read_text())
    features  = schema["features"]
    df        = pd.read_csv(FINAL_CSV)
    feat_cols = [c for c in df.columns if c != "application_id"]
    app_ids   = df["application_id"].astype(str).values

    x_all = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(device)
    data  = torch.load(GRAPH_PT, weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, device)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], device)

    ckpt  = torch.load(MODEL_PTH, weights_only=False, map_location=device)
    model = HybridGraphMCM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.centroid = ckpt["centroid"].to(device)
    model.eval()

    with torch.no_grad():
        pred_x, edge_prob, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)

        per_feat_err       = (pred_x - x_all).abs()
        feature_pred_error = per_feat_err.mean(dim=1)

        target = torch.zeros(x_all.shape[0], N_EDGE_TYPES, device=device)
        for rel_id, ei in enumerate(edge_index_list):
            if ei.shape[1] > 0:
                target[ei[0], rel_id] = 1.0
                target[ei[1], rel_id] = 1.0
        edge_pred_error = F.binary_cross_entropy(edge_prob, target, reduction="none").mean(dim=1)

        hybrid_anomaly_score = feature_pred_error + LAMBDA_EDGE * edge_pred_error

    def _norm(t: torch.Tensor) -> np.ndarray:
        v = t.cpu().numpy()
        lo, hi = v.min(), v.max()
        return ((v - lo) / (hi - lo + 1e-8)).astype(np.float32)

    per_feat_np = per_feat_err.cpu().numpy()
    per_feature_error_json = [
        json.dumps({features[j]: float(round(per_feat_np[i, j], 6)) for j in range(N_FEATURES)})
        for i in range(len(app_ids))
    ]

    out_df = pd.DataFrame({
        "application_id":         app_ids,
        "hybrid_anomaly_score":   _norm(hybrid_anomaly_score),
        "feature_pred_error":     _norm(feature_pred_error),
        "edge_pred_error":        _norm(edge_pred_error),
        "per_feature_error_json": per_feature_error_json,
    })

    if app_ids_to_return is not None:
        out_df = out_df[out_df["application_id"].isin(app_ids_to_return)].reset_index(drop=True)

    return out_df
