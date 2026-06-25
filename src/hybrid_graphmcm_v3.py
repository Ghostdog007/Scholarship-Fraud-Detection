"""
hybrid_graphmcm_v3.py

Hybrid GraphMCM: masked feature prediction informed by graph neighborhood context.

Feature stream: K=8 learned masks applied to 68-dim input features.
Graph stream:   RGCN encoder (5 typed edges, aggr='add', tanh) -> 64-dim h_N(i).
Concat [masked_x_i (68) ; h_N(i) (64)] -> MLP -> predicted x_i (68).
Edge predictor: MLP([x_i ; h_N(i)]) -> sigmoid -> (5,) one prob per edge type.

Training:
  Stage 1 (EPOCHS_STAGE1): graph-side LOE warm-start only.
             Synthetic exposure nodes pushed away from normal centroid.
  Stage 2 (EPOCHS_STAGE2): free joint reconstruction.
             Feature prediction loss + edge prediction loss + LOE off.

Isolated nodes: trainable isolated_embedding nn.Parameter (GRAPH_EMB_DIM,).
Outputs: outputs/hybrid_scores_v3.csv, models/hybrid_graphmcm_v3.pth
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from src.config_v3 import (
    BATCH_SIZE,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
    GRAPH_EMB_DIM,
    GRAPH_HIDDEN,
    LAMBDA_EDGE,
    LAMBDA_EXPOSURE,
    LOE_MARGIN,
    LR,
    MASK_NUM,
    MLP_HIDDEN,
    N_EDGE_TYPES,
    N_FEATURES,
    RANDOM_SEED,
    Z_DIM,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FINAL_CSV   = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON = Path("data/processed/v3_feature_schema.json")
GRAPH_PT    = Path("data/processed/identity_graph_v3.pt")
EXPOSURE_PT = Path("data/processed/synthetic_exposure_set_v3.pt")
OUT_CSV     = Path("outputs/hybrid_scores_v3.csv")
MODEL_PTH   = Path("models/hybrid_graphmcm_v3.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DimensionError(Exception):
    pass


def _check_shape(tensor: torch.Tensor, expected: tuple, name: str) -> None:
    if tensor.shape[1:] != torch.Size(expected[1:]):
        raise DimensionError(f"{name}: expected shape[1:]={expected[1:]}, got {tensor.shape[1:]}")


# ---------------------------------------------------------------------------
# RGCN Encoder
# ---------------------------------------------------------------------------

class RGCNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = RGCNConv(N_FEATURES, GRAPH_HIDDEN, num_relations=N_EDGE_TYPES, aggr="add")
        self.conv2 = RGCNConv(GRAPH_HIDDEN, GRAPH_EMB_DIM, num_relations=N_EDGE_TYPES, aggr="add")

    def forward(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        edge_type_tensor: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=x.device)
        h = torch.tanh(self.conv1(x, edge_index, edge_type_tensor))
        h = torch.tanh(self.conv2(h, edge_index, edge_type_tensor))
        return h  # (N, GRAPH_EMB_DIM)


# ---------------------------------------------------------------------------
# Hybrid GraphMCM
# ---------------------------------------------------------------------------

class HybridGraphMCM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Learned masks: (MASK_NUM, N_FEATURES) — softmax-normalised per mask
        self.mask_logits = nn.Parameter(torch.randn(MASK_NUM, N_FEATURES))

        # Graph encoder
        self.rgcn = RGCNEncoder()

        # Isolated node embedding (trainable, not zero)
        self.isolated_embedding = nn.Parameter(torch.randn(GRAPH_EMB_DIM))

        # MLP: concat(masked_x, h_N) -> predicted_x
        concat_dim = N_FEATURES + GRAPH_EMB_DIM  # 132
        self.predictor = nn.Sequential(
            nn.Linear(concat_dim, MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN, Z_DIM),
            nn.ReLU(),
            nn.Linear(Z_DIM, N_FEATURES),
        )

        # Edge predictor: (x_i || h_N) -> (N_EDGE_TYPES,)
        self.edge_predictor = nn.Sequential(
            nn.Linear(concat_dim, MLP_HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN // 2, N_EDGE_TYPES),
            nn.Sigmoid(),
        )

        # DeepSVDD centroid (not a parameter — updated during init)
        self.register_buffer("centroid", torch.zeros(GRAPH_EMB_DIM))
        self.centroid_initialized = False

    def _apply_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Average over K soft-masked versions of x."""
        masks = torch.softmax(self.mask_logits, dim=1)  # (K, N_FEATURES)
        masked = x.unsqueeze(0) * masks.unsqueeze(1)    # (K, B, N_FEATURES)
        return masked.mean(dim=0)                        # (B, N_FEATURES)

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        edge_type_tensor: torch.Tensor,
        isolated_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.rgcn(x, edge_index_list, edge_type_tensor)
        iso_emb = self.isolated_embedding.unsqueeze(0).expand(h.shape[0], -1)
        mask_exp = isolated_mask.unsqueeze(1).expand_as(h)
        return torch.where(mask_exp, iso_emb, h)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        edge_type_tensor: torch.Tensor,
        isolated_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_shape(x, (None, N_FEATURES), "x")

        masked_x = self._apply_masks(x)
        h_n      = self.encode_graph(x, edge_index_list, edge_type_tensor, isolated_mask)

        _check_shape(h_n, (None, GRAPH_EMB_DIM), "h_N(i)")

        concat   = torch.cat([masked_x, h_n], dim=1)  # (N, 132)
        pred_x   = self.predictor(concat)              # (N, N_FEATURES)
        edge_prob = self.edge_predictor(concat)        # (N, N_EDGE_TYPES)

        return pred_x, edge_prob, h_n, concat

    @torch.no_grad()
    def init_centroid(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        edge_type_tensor: torch.Tensor,
        isolated_mask: torch.Tensor,
    ) -> None:
        self.eval()
        h = self.encode_graph(x, edge_index_list, edge_type_tensor, isolated_mask)
        self.centroid = h.mean(dim=0).detach()
        self.centroid_initialized = True
        print(f"[hybrid] Centroid initialised, norm={self.centroid.norm():.4f}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_edge_index_and_types(data, device: torch.device) -> tuple[list, torch.Tensor]:
    edge_index_list = []
    edge_type_ids   = []
    for rel_id, edge_type in enumerate(data.edge_types):
        ei = data[edge_type].edge_index.to(device)
        if ei.shape[1] > 0:
            edge_index_list.append(ei)
            edge_type_ids.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long, device=device))

    if edge_index_list:
        edge_type_tensor = torch.cat(edge_type_ids)
    else:
        edge_type_tensor = torch.zeros(0, dtype=torch.long, device=device)

    return edge_index_list, edge_type_tensor


def _compute_isolated_mask(edge_index_list: list, n_nodes: int, device: torch.device) -> torch.Tensor:
    has_edge = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ei in edge_index_list:
        has_edge[ei.unique()] = True
    return ~has_edge


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _feature_pred_loss(pred_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_x, x)


def _edge_pred_loss(
    edge_prob: torch.Tensor,
    edge_index_list: list[torch.Tensor],
    n_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    target = torch.zeros(n_nodes, N_EDGE_TYPES, device=device)
    for rel_id, ei in enumerate(edge_index_list):
        if ei.shape[1] > 0:
            target[ei[0], rel_id] = 1.0
            target[ei[1], rel_id] = 1.0
    return F.binary_cross_entropy(edge_prob, target)


def _loe_loss(
    h_synth: torch.Tensor,
    centroid: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    dist = torch.norm(h_synth - centroid.unsqueeze(0), dim=1)
    exposure = torch.exp(-torch.sqrt(dist + 1e-8))
    return lam * exposure.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_synth_h(
    model: HybridGraphMCM,
    x_synth: torch.Tensor,
    edge_index_list: list,
    edge_type_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    n_synth = x_synth.shape[0]
    synth_isolated = torch.ones(n_synth, dtype=torch.bool, device=device)
    # Synthetic nodes have no graph edges — use isolated_embedding for all
    h_synth = model.encode_graph(x_synth, [], torch.zeros(0, dtype=torch.long, device=device), synth_isolated)
    return h_synth


def train(smoke_test: bool = False) -> None:
    print(f"[hybrid] Device: {DEVICE}")

    schema   = json.loads(SCHEMA_JSON.read_text())
    features = schema["features"]
    if len(features) != N_FEATURES:
        raise DimensionError(f"Schema has {len(features)} features, expected {N_FEATURES}")

    df = pd.read_csv(FINAL_CSV)
    feat_cols = [c for c in df.columns if c != "application_id"]
    app_ids   = df["application_id"].values

    x_all = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(DEVICE)
    _check_shape(x_all, (None, N_FEATURES), "x_all")

    data         = torch.load(GRAPH_PT, weights_only=False)
    x_synth      = torch.load(EXPOSURE_PT, weights_only=True).to(DEVICE)

    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], DEVICE)
    print(f"[hybrid] Isolated nodes: {isolated_mask.sum().item()} / {x_all.shape[0]}")

    model = HybridGraphMCM().to(DEVICE)
    model.init_centroid(x_all, edge_index_list, edge_type_tensor, isolated_mask)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epochs_s1 = 2 if smoke_test else EPOCHS_STAGE1
    epochs_s2 = 2 if smoke_test else EPOCHS_STAGE2

    # ---- Stage 1: Graph LOE warm-start ----
    print(f"[hybrid] Stage 1: {epochs_s1} epochs (graph LOE warm-start) ...")
    model.train()
    for epoch in range(epochs_s1):
        lam_t = LAMBDA_EXPOSURE * (1.0 - epoch / epochs_s1)

        optimizer.zero_grad()
        _, _, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)
        svdd_loss = torch.norm(h_n - model.centroid.unsqueeze(0), dim=1).mean()

        h_synth = _get_synth_h(model, x_synth, edge_index_list, edge_type_tensor, DEVICE)
        loe     = _loe_loss(h_synth, model.centroid, lam_t)

        loss = svdd_loss + loe
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs_s1 // 5) == 0 or smoke_test:
            print(f"  S1 epoch {epoch+1}/{epochs_s1} | svdd={svdd_loss.item():.4f} loe={loe.item():.4f} lam={lam_t:.3f}")

    # ---- Stage 2: Free joint reconstruction ----
    print(f"[hybrid] Stage 2: {epochs_s2} epochs (free joint reconstruction) ...")
    model.train()
    for epoch in range(epochs_s2):
        optimizer.zero_grad()

        pred_x, edge_prob, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)

        feat_loss = _feature_pred_loss(pred_x, x_all)
        edge_loss = _edge_pred_loss(edge_prob, edge_index_list, x_all.shape[0], DEVICE)
        loss      = feat_loss + LAMBDA_EDGE * edge_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs_s2 // 5) == 0 or smoke_test:
            print(f"  S2 epoch {epoch+1}/{epochs_s2} | feat={feat_loss.item():.4f} edge={edge_loss.item():.4f}")

    # ---- Scoring ----
    print("[hybrid] Scoring all nodes ...")
    model.eval()
    with torch.no_grad():
        pred_x, edge_prob, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)

        per_feat_err = (pred_x - x_all).abs()                    # (N, 68)
        feature_pred_error = per_feat_err.mean(dim=1)            # (N,)

        target = torch.zeros(x_all.shape[0], N_EDGE_TYPES, device=DEVICE)
        for rel_id, ei in enumerate(edge_index_list):
            if ei.shape[1] > 0:
                target[ei[0], rel_id] = 1.0
                target[ei[1], rel_id] = 1.0
        edge_pred_error = F.binary_cross_entropy(edge_prob, target, reduction="none").mean(dim=1)

        hybrid_anomaly_score = feature_pred_error + LAMBDA_EDGE * edge_pred_error

    # Normalise scores to [0, 1]
    def _norm(t: torch.Tensor) -> np.ndarray:
        v = t.cpu().numpy()
        lo, hi = v.min(), v.max()
        return ((v - lo) / (hi - lo + 1e-8)).astype(np.float32)

    hybrid_scores  = _norm(hybrid_anomaly_score)
    feat_err_arr   = _norm(feature_pred_error)
    edge_err_arr   = _norm(edge_pred_error)
    per_feat_np    = per_feat_err.cpu().numpy()

    per_feature_error_json = [
        json.dumps({features[j]: float(round(per_feat_np[i, j], 6)) for j in range(N_FEATURES)})
        for i in range(len(app_ids))
    ]

    out_df = pd.DataFrame({
        "application_id":      app_ids,
        "hybrid_anomaly_score": hybrid_scores,
        "feature_pred_error":   feat_err_arr,
        "edge_pred_error":      edge_err_arr,
        "per_feature_error_json": per_feature_error_json,
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[hybrid] Scores saved -> {OUT_CSV}")
    print(f"[hybrid] hybrid_anomaly_score range: [{hybrid_scores.min():.4f}, {hybrid_scores.max():.4f}]")

    # Save checkpoint
    centroid_val = model.centroid.cpu()
    MODEL_PTH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "centroid": centroid_val,
            "config": {
                "N_FEATURES": N_FEATURES,
                "GRAPH_EMB_DIM": GRAPH_EMB_DIM,
                "GRAPH_HIDDEN": GRAPH_HIDDEN,
                "MLP_HIDDEN": MLP_HIDDEN,
                "Z_DIM": Z_DIM,
                "MASK_NUM": MASK_NUM,
                "N_EDGE_TYPES": N_EDGE_TYPES,
            },
            "feature_names": features,
        },
        MODEL_PTH,
    )
    print(f"[hybrid] Checkpoint saved -> {MODEL_PTH}")


if __name__ == "__main__":
    train(smoke_test=False)
