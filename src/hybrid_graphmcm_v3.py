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

_BFP = "S2FuaXNoayBTaGFybWEgfCBOU1VUIHwgQmF0Y2ggMjAyNyB8IHNvbGUgYXV0aG9yLCBOSUMgRnJhdWQgRGV0ZWN0aW9uIFByb2plY3Q="

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GATConv

from src.config_v3 import (
    BATCH_SIZE,
    CENTROID_CLEAN_PERCENTILE,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
    GRAPH_EMB_DIM,
    GRAPH_HIDDEN,
    INCREMENTAL_EPOCHS,
    INCREMENTAL_LR,
    LAMBDA_EDGE,
    LAMBDA_EDGE_SCORE,
    LAMBDA_EXPOSURE,
    LOE_MARGIN,
    LOE_STAGE2_WEIGHT,
    LR,
    MASK_NUM,
    MLP_HIDDEN,
    EDGE_TYPES,
    N_EDGE_TYPES,
    N_FEATURES,
    RANDOM_SEED,
    Z_DIM,
    ENCODER_ARCH,
    ARCH_VERSION,
    ATTN_HEADS,
    ATTN_LEAKY_SLOPE,
    SEMANTIC_ATTN_HIDDEN,
    TOPO_EXPOSURE_ENABLED,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FINAL_CSV   = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON = Path("data/processed/v3_feature_schema.json")
GRAPH_PT    = Path("data/processed/identity_graph_v3.pt")
EXPOSURE_PT = Path("data/processed/synthetic_exposure_set_v3.pt")
TOPO_PT     = Path("data/processed/synthetic_exposure_graph_v3.pt")  # module global so the ablation can redirect it
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
        # root_weight=False (2026-07-22): RGCNConv defaults to root_weight=True,
        # which adds a learned self-transform of each node's OWN unmasked
        # features directly into h_n -- independent of the MCM masking in
        # _apply_masks(). That leaks self-signal around the intentional mask
        # for every connected node (isolated nodes are unaffected; they're
        # already overridden by isolated_embedding). Disabling it makes h_n
        # pure multi-relation neighbor aggregation, which is the actual MCM
        # contract. Validated on stress_testing_1: overall PR-AUC 0.153->0.201,
        # mobile-ring 0.029->0.078, IP-ring 0.032->0.055, no low-degree
        # regression (3-5 degree bucket 0.193->0.385, 6+ bucket 0.153->0.201).
        self.conv1 = RGCNConv(N_FEATURES, GRAPH_HIDDEN, num_relations=N_EDGE_TYPES, aggr="add", root_weight=False)
        self.conv2 = RGCNConv(GRAPH_HIDDEN, GRAPH_EMB_DIM, num_relations=N_EDGE_TYPES, aggr="add", root_weight=False)
        self.last_beta_r = torch.ones(N_EDGE_TYPES) / N_EDGE_TYPES

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

    def top_alpha(self, node_idx: int, k: int) -> list:
        return []


class HANEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_dict = nn.ModuleDict({
            str(r): GATConv(N_FEATURES, GRAPH_HIDDEN // ATTN_HEADS, heads=ATTN_HEADS, 
                            negative_slope=ATTN_LEAKY_SLOPE, concat=True, add_self_loops=True)
            for r in range(N_EDGE_TYPES)
        })
        self.conv2_dict = nn.ModuleDict({
            str(r): GATConv(GRAPH_HIDDEN, GRAPH_EMB_DIM // ATTN_HEADS, heads=ATTN_HEADS, 
                            negative_slope=ATTN_LEAKY_SLOPE, concat=True, add_self_loops=True)
            for r in range(N_EDGE_TYPES)
        })
        self.semantic_attention = nn.Sequential(
            nn.Linear(GRAPH_EMB_DIM, SEMANTIC_ATTN_HIDDEN),
            nn.Tanh(),
            nn.Linear(SEMANTIC_ATTN_HIDDEN, 1, bias=False)
        )
        self.last_beta_r = None
        self.last_alpha = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        edge_type_tensor: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        # Reconstruct per-relation edges from edge_type_tensor rather than
        # indexing edge_index_list positionally: callers may pass a list of only
        # the non-empty relations (from _build_edge_index_and_types), so position
        # r is NOT relation r. edge_type_tensor aligns with cat(edge_index_list)
        # for every caller in this codebase, so masking on it is the safe key.
        if edge_index_list:
            edge_index_all = torch.cat(edge_index_list, dim=1)
        else:
            edge_index_all = torch.zeros((2, 0), dtype=torch.long, device=device)

        z_r_list = []
        alpha_dict = {}

        for r in range(N_EDGE_TYPES):
            if edge_type_tensor.numel() > 0:
                ei = edge_index_all[:, edge_type_tensor == r]
            else:
                ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            h1 = F.elu(self.conv1_dict[str(r)](x, ei))
            out = self.conv2_dict[str(r)](h1, ei, return_attention_weights=True)
            if isinstance(out, tuple):
                h2, (ei_out, alpha) = out
                alpha_dict[r] = (ei_out.detach().cpu(), alpha.detach().cpu())
            else:
                h2 = out
                alpha_dict[r] = None
            z_r_list.append(h2)

        self.last_alpha = alpha_dict

        w_r_list = []
        for r in range(N_EDGE_TYPES):
            mean_z_r = z_r_list[r].mean(dim=0, keepdim=True)
            w_r_list.append(self.semantic_attention(mean_z_r))

        w = torch.cat(w_r_list, dim=1)
        beta_r = torch.softmax(w, dim=1)
        self.last_beta_r = beta_r.squeeze(0).detach().cpu()

        h = x.new_zeros(x.shape[0], GRAPH_EMB_DIM)
        for r in range(N_EDGE_TYPES):
            h = h + beta_r[0, r] * z_r_list[r]

        return h

    def top_alpha(self, node_idx: int, k: int) -> list:
        res = []
        if self.last_alpha is None:
            return res
        for r, data in self.last_alpha.items():
            if data is None:
                continue
            ei, alpha = data
            if alpha.dim() > 1:
                alpha = alpha.mean(dim=1)
            mask = ei[1] == node_idx
            src_nodes = ei[0][mask]
            weights = alpha[mask]
            for src, w in zip(src_nodes, weights):
                if src.item() != node_idx:
                    res.append({"neighbor_idx": src.item(), "relation": r, "weight": float(w.item())})
        res.sort(key=lambda x: x["weight"], reverse=True)
        return res[:k]


# ---------------------------------------------------------------------------
# Hybrid GraphMCM
# ---------------------------------------------------------------------------

class HybridGraphMCM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Learned masks: (MASK_NUM, N_FEATURES) — softmax-normalised per mask
        self.mask_logits = nn.Parameter(torch.randn(MASK_NUM, N_FEATURES))

        # Graph encoder
        self.encoder = RGCNEncoder() if ENCODER_ARCH == "rgcn" else HANEncoder()

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
        h = self.encoder(x, edge_index_list, edge_type_tensor)
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

    @property
    def last_beta_r(self):
        return self.encoder.last_beta_r

    def top_alpha(self, node_idx: int, k: int) -> list:
        return self.encoder.top_alpha(node_idx, k)

    @torch.no_grad()
    def attention_summary(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        edge_type_tensor: torch.Tensor,
        isolated_mask: torch.Tensor,
        app_ids: np.ndarray,
    ) -> pd.DataFrame:
        import math
        n_nodes = x.shape[0]
        h = self.encoder(x, edge_index_list, edge_type_tensor)

        beta_r = np.full((n_nodes, N_EDGE_TYPES), 0.2, dtype=np.float32)
        alpha_entropy = np.zeros((n_nodes, N_EDGE_TYPES), dtype=np.float32)
        alpha_top1 = np.zeros(n_nodes, dtype=np.float32)

        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=x.device)

        if edge_index.shape[1] > 0:
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            rel = edge_type_tensor.cpu().numpy()

            h_src = h[edge_index[0]]
            h_dst = h[edge_index[1]]
            e_ij = (h_src * h_dst).sum(dim=1).cpu().numpy()

            df_edges = pd.DataFrame({'src': src, 'dst': dst, 'rel': rel, 'e': e_ij})

            def softmax_group(x):
                x = x - np.max(x)
                e_x = np.exp(x)
                return e_x / e_x.sum()

            df_edges['a'] = df_edges.groupby(['dst', 'rel'])['e'].transform(softmax_group)

            def norm_entropy(a):
                if len(a) <= 1:
                    return 0.0
                ent = -np.sum(a * np.log(a + 1e-9))
                return ent / math.log(len(a))

            entropy_df = df_edges.groupby(['dst', 'rel'])['a'].apply(norm_entropy).reset_index()
            for _, row in entropy_df.iterrows():
                alpha_entropy[int(row['dst']), int(row['rel'])] = row['a']

            top1_df = df_edges.groupby('dst')['a'].max().reset_index()
            for _, row in top1_df.iterrows():
                alpha_top1[int(row['dst'])] = row['a']

            energy_df = df_edges.groupby(['dst', 'rel'])['e'].sum().reset_index()
            node_energies = np.zeros((n_nodes, N_EDGE_TYPES), dtype=np.float32)
            for _, row in energy_df.iterrows():
                node_energies[int(row['dst']), int(row['rel'])] = row['e']

            e_max = np.max(node_energies, axis=1, keepdims=True)
            exp_e = np.exp(node_energies - e_max)
            beta_r = exp_e / np.sum(exp_e, axis=1, keepdims=True)

            iso = isolated_mask.cpu().numpy()
            beta_r[iso] = 0.2
            alpha_entropy[iso] = 0.0
            alpha_top1[iso] = 0.0

        df = pd.DataFrame({
            "application_id": app_ids,
            **{f"beta_{r}": beta_r[:, r] for r in range(N_EDGE_TYPES)},
            **{f"alpha_entropy_{r}": alpha_entropy[:, r] for r in range(N_EDGE_TYPES)},
            "alpha_top1": alpha_top1
        })
        return df

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
        # Contamination-aware centroid: exclude top (100 - CENTROID_CLEAN_PERCENTILE)%
        # of nodes by embedding norm before computing the mean. These are the most
        # anomalous embeddings and likely include fraud — including them shifts the
        # centroid toward fraud, causing DeepSVDD hypersphere inflation.
        norms = h.norm(dim=1)
        cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
        clean_mask = norms <= cutoff
        n_excluded = int((~clean_mask).sum().item())
        self.centroid = h[clean_mask].mean(dim=0).detach()
        self.centroid_initialized = True
        print(f"[hybrid] Centroid initialised (excluded {n_excluded} high-norm nodes, "
              f"{CENTROID_CLEAN_PERCENTILE}th pct cutoff={cutoff:.4f}), "
              f"norm={self.centroid.norm():.4f}")


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


def _derive_loe_margin(h_real: torch.Tensor, centroid: torch.Tensor, percentile: float = 75.0) -> torch.Tensor:
    """Data-derived LOE margin (added 2026-07-22, see README changelog) —
    replaces the fixed LOE_MARGIN=2.0 constant, which was found (by direct
    measurement, not assumption) to be ~3x SMALLER than even the REAL
    population's own median embedding-to-centroid distance at this embedding
    dimensionality (GRAPH_EMB_DIM=64): real dist-to-centroid median ~5.9 vs a
    margin of 2.0. That meant exposure embeddings were already past the
    "margin" before training even started, in both the old exp(-sqrt(dist))
    formula and a fixed hinge — the constant was never calibrated to the
    embedding space's actual scale. Deriving the margin from the CURRENT
    epoch's real embedding distribution (same principle as
    CENTROID_CLEAN_PERCENTILE — percentile of an observed distribution, not a
    hand-picked number, hard stop #1) makes the target self-calibrating:
    "push exposure embeddings out beyond what's typical for real data,"
    whatever scale the network currently lives at."""
    with torch.no_grad():
        real_dist = torch.norm(h_real - centroid.unsqueeze(0), dim=1)
        return torch.quantile(real_dist, percentile / 100.0)


def _loe_loss(
    h_synth: torch.Tensor,
    centroid: torch.Tensor,
    lam: float,
    margin: torch.Tensor | float = LOE_MARGIN,
) -> torch.Tensor:
    """Hinge/margin loss (changed 2026-07-22, see README changelog): pushes
    exposure embeddings at least `margin` away from the centroid. Replaces
    exp(-sqrt(dist)), which saturates to ~0 the moment dist exceeds a handful
    of units — found via prototyping to vanish within the first few epochs of
    training regardless of epoch budget, so it was only ever a nudge at
    initialization, not a real training signal. A hinge has a genuine,
    non-vanishing gradient anywhere inside the margin. `margin` defaults to
    the LOCKED constant for backward compatibility but every production call
    site now passes a data-derived margin from _derive_loe_margin — see there
    for why the fixed constant doesn't work at this embedding scale."""
    dist = torch.norm(h_synth - centroid.unsqueeze(0), dim=1)
    exposure = torch.clamp(margin - dist, min=0.0)
    return lam * exposure.mean()


# ---------------------------------------------------------------------------
# Shared inference — single source of truth for the hybrid_scores_v3.csv schema
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_score_frame(
    model: "HybridGraphMCM",
    x_all: torch.Tensor,
    edge_index_list: list,
    edge_type_tensor: torch.Tensor,
    isolated_mask: torch.Tensor,
    app_ids: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    """
    Score every node and return the full hybrid_scores_v3.csv frame.

    Used by train(), train_incremental() (via _score_and_save) and the API's
    read-only staged scoring (src/api/inference.py) so the schema — including
    per_feature_predicted_json — cannot drift between the three paths.

    per_feature_predicted_json holds the model's predicted (scaled) value for
    each feature given the rest of the application and its neighborhood. The
    XAI layer pairs it with the actual value to state expected-vs-actual with
    direction. Predictions are exported as-is (the decoder head is a plain
    Linear layer, so values may fall slightly outside [0, 1]).
    """
    model.eval()
    pred_x, edge_prob, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)

    per_feat_err       = (pred_x - x_all).abs()          # (N, N_FEATURES)
    feature_pred_error = per_feat_err.mean(dim=1)        # (N,)

    target = torch.zeros(x_all.shape[0], N_EDGE_TYPES, device=x_all.device)
    for rel_id, ei in enumerate(edge_index_list):
        if ei.shape[1] > 0:
            target[ei[0], rel_id] = 1.0
            target[ei[1], rel_id] = 1.0
    edge_pred_error = F.binary_cross_entropy(edge_prob, target, reduction="none").mean(dim=1)

    # LAMBDA_EDGE_SCORE (0.0), not LAMBDA_EDGE (0.3, still the training-loss
    # weight) -- see config_v3.py comment, README changelog 2026-07-23.
    # edge_pred_error is still computed and returned below for XAI use; it is
    # excluded from the ranking score itself.
    hybrid_anomaly_score = feature_pred_error + LAMBDA_EDGE_SCORE * edge_pred_error

    def _norm(t: torch.Tensor) -> np.ndarray:
        v = t.cpu().numpy()
        lo, hi = v.min(), v.max()
        return ((v - lo) / (hi - lo + 1e-8)).astype(np.float32)

    per_feat_np = per_feat_err.cpu().numpy()
    pred_np     = pred_x.cpu().numpy()

    per_feature_error_json = [
        json.dumps({features[j]: float(round(per_feat_np[i, j], 6)) for j in range(N_FEATURES)})
        for i in range(len(app_ids))
    ]
    per_feature_predicted_json = [
        json.dumps({features[j]: float(round(pred_np[i, j], 6)) for j in range(N_FEATURES)})
        for i in range(len(app_ids))
    ]

    return pd.DataFrame({
        "application_id":             app_ids,
        "hybrid_anomaly_score":       _norm(hybrid_anomaly_score),
        "feature_pred_error":         _norm(feature_pred_error),
        "edge_pred_error":            _norm(edge_pred_error),
        "per_feature_error_json":     per_feature_error_json,
        "per_feature_predicted_json": per_feature_predicted_json,
    })


def load_model_and_inputs(device: torch.device = DEVICE):
    """
    Load the existing checkpoint plus canonical inputs for inference-only
    scoring. Returns (model, x_all, edge_index_list, edge_type_tensor,
    isolated_mask, app_ids, features). No training, no writes.
    """
    if not MODEL_PTH.exists():
        raise FileNotFoundError(f"No checkpoint at {MODEL_PTH}. Run full train() first.")

    schema    = json.loads(SCHEMA_JSON.read_text())
    features  = schema["features"]
    if len(features) != N_FEATURES:
        raise DimensionError(f"Schema has {len(features)} features, expected {N_FEATURES}")

    df        = pd.read_csv(FINAL_CSV)
    feat_cols = [c for c in df.columns if c != "application_id"]
    app_ids   = df["application_id"].values

    x_all = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(device)
    _check_shape(x_all, (None, N_FEATURES), "x_all")

    data = torch.load(GRAPH_PT, weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, device)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], device)

    ckpt  = torch.load(MODEL_PTH, weights_only=False, map_location=device)
    model = HybridGraphMCM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.centroid = ckpt["centroid"].to(device)

    return model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features


def score_only() -> None:
    """
    Regenerate outputs/hybrid_scores_v3.csv from the existing checkpoint.
    No training, no checkpoint write. Added alongside the
    per_feature_predicted_json export (2026-07-02) so the schema change does
    not force a retrain — the same weights produce identical scores plus the
    new column.
    """
    print(f"[hybrid] score_only() | device={DEVICE}")
    model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features = load_model_and_inputs()
    print(f"[hybrid] Isolated nodes: {isolated_mask.sum().item()} / {x_all.shape[0]}")

    out_df = compute_score_frame(model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    scores = out_df["hybrid_anomaly_score"].values
    print(f"[hybrid] Scores saved -> {OUT_CSV}")
    print(f"[hybrid] hybrid_anomaly_score range: [{scores.min():.4f}, {scores.max():.4f}]")


ABLATION_CSV = Path("outputs/relation_ablation_v3.csv")


@torch.no_grad()
def compute_relation_ablation() -> pd.DataFrame:
    """
    Post-hoc, per-relation neighbourhood-importance for the LOCKED RGCN
    encoder. HAN has a true learned per-relation attention (encoder.last_beta_r,
    model.top_alpha) but HAN is not production (regressed -0.091, 3-seed
    ablation; see config_v3/README 2026-07-22) -- RGCN has no such mechanism,
    so "expected this based on neighbours" had no way to say WHICH relation
    drove that expectation. This fills that gap without touching the locked
    architecture: for each of the N_EDGE_TYPES relations, re-run inference
    with that relation's edges removed (masked via edge_type_tensor, not
    edge_index_list position -- callers may pass only the non-empty relations,
    so position r is NOT relation r) and compare feature_pred_error to the
    full-graph baseline.

    ablation_delta_<relation> = baseline_error - ablated_error, per node.
    Positive: removing this relation's neighbours made the node fit BETTER
    (looked more normal) -- i.e. that relation's neighbourhood context is why
    the model's expectation for this node looked anomalous. Negative/zero:
    that relation wasn't driving the anomaly (or its neighbours were making
    the fit better, not worse).

    5 extra full-graph forward passes total (once per relation, not once per
    flagged application) -- cheap, no retraining. Only the scalar per-node
    delta leaves the model (hard stop 2: no embeddings, ever).

    Writes/returns: application_id, ablation_delta_<relation> for each of the
    5 edge types -> outputs/relation_ablation_v3.csv.
    """
    print("[hybrid] compute_relation_ablation() starting ...")
    model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features = load_model_and_inputs()
    model.eval()
    device = x_all.device

    def _feature_pred_error(eil: list, ett: torch.Tensor, iso_mask: torch.Tensor) -> torch.Tensor:
        pred_x, _, _, _ = model(x_all, eil, ett, iso_mask)
        return (pred_x - x_all).abs().mean(dim=1)

    edge_index_all = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=device)
    baseline_err = _feature_pred_error(edge_index_list, edge_type_tensor, isolated_mask)

    out = {"application_id": app_ids}
    for r, rel_name in enumerate(EDGE_TYPES):
        if edge_type_tensor.numel() > 0:
            keep = edge_type_tensor != r
        else:
            keep = torch.zeros(0, dtype=torch.bool, device=device)
        ablated_ei  = edge_index_all[:, keep]
        ablated_ett = edge_type_tensor[keep]
        ablated_eil = [ablated_ei] if ablated_ei.shape[1] > 0 else []
        ablated_iso = _compute_isolated_mask(ablated_eil, x_all.shape[0], device)
        ablated_err = _feature_pred_error(ablated_eil, ablated_ett, ablated_iso)
        delta = (baseline_err - ablated_err).cpu().numpy()
        out[f"ablation_delta_{rel_name}"] = delta
        print(f"[hybrid]   {rel_name}: mean delta={delta.mean():.6f}, "
              f"{(delta > 0).sum()}/{len(delta)} nodes positively driven by this relation")

    df = pd.DataFrame(out)
    ABLATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABLATION_CSV, index=False)
    print(f"[hybrid] Relation ablation saved -> {ABLATION_CSV}")
    return df


def run_relation_ablation() -> None:
    compute_relation_ablation()


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


def _get_synth_h_topology(
    model: HybridGraphMCM,
    topo_pack: dict,
    device: torch.device,
) -> torch.Tensor:
    x_topo = topo_pack["x"].to(device)
    edge_index = topo_pack["edge_index"].to(device)
    edge_type = topo_pack["edge_type"].to(device)
    
    n_synth = x_topo.shape[0]
    # Group edges by relation AND build an edge_type tensor that aligns with the
    # concatenation order of edge_index_list. encode_graph -> RGCN does
    # cat(edge_index_list) and labels each edge with edge_type_grouped[i]; passing
    # the pack's original (ungrouped) edge_type here would misalign relation
    # labels and corrupt the RGCN topology-exposure signal (config-2).
    edge_index_list = []
    grouped_types = []
    for r in range(N_EDGE_TYPES):
        mask = edge_type == r
        if mask.any():
            ei_r = edge_index[:, mask]
            edge_index_list.append(ei_r)
            grouped_types.append(torch.full((ei_r.shape[1],), r, dtype=torch.long, device=device))
        else:
            edge_index_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))

    edge_type_grouped = (torch.cat(grouped_types) if grouped_types
                         else torch.zeros(0, dtype=torch.long, device=device))
    isolated_mask = _compute_isolated_mask(edge_index_list, n_synth, device)
    h_synth = model.encode_graph(x_topo, edge_index_list, edge_type_grouped, isolated_mask)
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
    topo_pack    = None
    if TOPO_EXPOSURE_ENABLED:
        topo_pack = torch.load(TOPO_PT, weights_only=False)

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

        if TOPO_EXPOSURE_ENABLED and topo_pack is not None:
            h_synth = _get_synth_h_topology(model, topo_pack, DEVICE)
        else:
            h_synth = _get_synth_h(model, x_synth, edge_index_list, edge_type_tensor, DEVICE)

        margin  = _derive_loe_margin(h_n, model.centroid)
        loe     = _loe_loss(h_synth, model.centroid, lam_t, margin)

        loss = svdd_loss + loe
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs_s1 // 5) == 0 or smoke_test:
            print(f"  S1 epoch {epoch+1}/{epochs_s1} | svdd={svdd_loss.item():.4f} loe={loe.item():.4f} "
                  f"lam={lam_t:.3f} margin={margin.item():.3f} std={h_synth.std(0).mean().item():.4f}")

    # ---- Stage 2: Free joint reconstruction + a PERSISTENT (non-decaying) LOE
    # term (added 2026-07-22 — see README changelog). Previously Stage 2 had no
    # exposure term at all, so whatever separation Stage 1 built could be freely
    # re-absorbed by 120 epochs of unconstrained reconstruction (dense synthetic
    # cliques reconstruct too easily — MAR critique — and nothing stopped Stage 2
    # re-learning to reconstruct them well). LOE_STAGE2_WEIGHT is fixed, not
    # decayed, so the signal survives to the end of training.
    print(f"[hybrid] Stage 2: {epochs_s2} epochs (free joint reconstruction + persistent LOE) ...")
    model.train()
    for epoch in range(epochs_s2):
        optimizer.zero_grad()

        pred_x, edge_prob, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)

        feat_loss = _feature_pred_loss(pred_x, x_all)
        edge_loss = _edge_pred_loss(edge_prob, edge_index_list, x_all.shape[0], DEVICE)

        if TOPO_EXPOSURE_ENABLED and topo_pack is not None:
            h_synth_s2 = _get_synth_h_topology(model, topo_pack, DEVICE)
        else:
            h_synth_s2 = _get_synth_h(model, x_synth, edge_index_list, edge_type_tensor, DEVICE)
        margin_s2 = _derive_loe_margin(h_n, model.centroid)
        loe_s2 = _loe_loss(h_synth_s2, model.centroid, LOE_STAGE2_WEIGHT, margin_s2)

        loss = feat_loss + LAMBDA_EDGE * edge_loss + loe_s2

        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs_s2 // 5) == 0 or smoke_test:
            print(f"  S2 epoch {epoch+1}/{epochs_s2} | feat={feat_loss.item():.4f} "
                  f"edge={edge_loss.item():.4f} loe={loe_s2.item():.4f} margin={margin_s2.item():.3f}")

    # ---- Scoring ----
    print("[hybrid] Scoring all nodes ...")
    out_df = compute_score_frame(model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features)
    hybrid_scores = out_df["hybrid_anomaly_score"].values

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[hybrid] Scores saved -> {OUT_CSV}")
    print(f"[hybrid] hybrid_anomaly_score range: [{hybrid_scores.min():.4f}, {hybrid_scores.max():.4f}]")

    # Isolated-node diagnostic: report what fraction of top-100 are degree-0 nodes.
    # A high fraction here means the model is primarily detecting isolated nodes, not
    # relational fraud — this is the forensically-clean-node detection boundary.
    iso_np = isolated_mask.cpu().numpy()
    top100_idx = np.argsort(hybrid_scores)[-100:]
    iso_in_top100 = iso_np[top100_idx].sum()
    print(f"[hybrid] Isolated-node diagnostic: {iso_in_top100}/100 top-scoring nodes have degree=0 "
          f"(detection for these relies on subspace IF, not graph signal)")

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
                "ARCH_VERSION": ARCH_VERSION,
            },
            "feature_names": features,
        },
        MODEL_PTH,
    )
    print(f"[hybrid] Checkpoint saved -> {MODEL_PTH}")


# ---------------------------------------------------------------------------
# V4-Scale step 5: NeighborLoader mini-batch paths
#
# Model classes, losses, and hyperparameters are UNCHANGED — these are
# alternative training/scoring loops that bound memory per batch regardless
# of graph size. Full-graph paths above remain the default until cut-over.
# Edge-prediction TARGETS always come from the node's GLOBAL relation
# presence (not the sampled subgraph), so sampling noise affects only the
# neighbourhood embedding, never the target.
# ---------------------------------------------------------------------------

def _global_edge_target(data, n_nodes: int) -> torch.Tensor:
    """(N, N_EDGE_TYPES) 1.0 where the node has >=1 edge of that relation in
    the FULL graph — the same target _edge_pred_loss builds, precomputed."""
    target = torch.zeros(n_nodes, N_EDGE_TYPES)
    for rel_id, edge_type in enumerate(data.edge_types):
        ei = data[edge_type].edge_index
        if ei.shape[1] > 0:
            target[ei[0], rel_id] = 1.0
            target[ei[1], rel_id] = 1.0
    return target


def _make_neighbor_loader(data, fanout: tuple[int, int], batch_size: int,
                          shuffle: bool, seed: int = RANDOM_SEED):
    from torch_geometric.loader import NeighborLoader
    torch.manual_seed(seed)
    num_neighbors = {et: list(fanout) for et in data.edge_types}
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes="application",
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _batch_inputs(batch, data, device: torch.device, iso_global: torch.Tensor):
    """Per-batch tensors in the exact shape the model's forward expects.
    Relation ids follow enumerate(data.edge_types) — the same ordering
    _build_edge_index_and_types uses, so β_r indices stay consistent."""
    x = batch["application"].x.to(device)
    n_id = batch["application"].n_id
    edge_index_list, type_ids = [], []
    for rel_id, edge_type in enumerate(data.edge_types):
        ei = batch[edge_type].edge_index.to(device)
        if ei.shape[1] > 0:
            edge_index_list.append(ei)
            type_ids.append(torch.full((ei.shape[1],), rel_id, dtype=torch.long, device=device))
    edge_type_tensor = (torch.cat(type_ids) if type_ids
                        else torch.zeros(0, dtype=torch.long, device=device))
    iso = iso_global[n_id].to(device)
    n_seed = batch["application"].batch_size
    return x, edge_index_list, edge_type_tensor, iso, n_id, n_seed


@torch.no_grad()
def score_sampled(fanout: tuple[int, int] = (25, 10), batch_size: int = 1024,
                  device: torch.device = DEVICE, seed: int = RANDOM_SEED,
                  data=None, x_override=None, model=None) -> pd.DataFrame:
    """hybrid_scores frame via NeighborLoader — same schema/normalisation as
    compute_score_frame. Deterministic for a fixed seed on CPU."""
    if model is None:
        model, _, _, _, _, app_ids, features = load_model_and_inputs(device)
        data = torch.load(GRAPH_PT, weights_only=False)
    else:
        schema = json.loads(SCHEMA_JSON.read_text())
        features = schema["features"]
        df = pd.read_csv(FINAL_CSV)
        app_ids = df["application_id"].values
    model.eval()

    # The stored graph carries the 63-dim pre-drop node features; the model
    # takes the 44-dim FINAL_CSV matrix (same override the full-graph path
    # does in load_model_and_inputs).
    if x_override is None:
        _df = pd.read_csv(FINAL_CSV)
        _cols = [c for c in _df.columns if c != "application_id"]
        x_override = torch.tensor(_df[_cols].values, dtype=torch.float32)
    data["application"].x = x_override
    n_nodes = data["application"].x.shape[0]
    full_eil, _ = _build_edge_index_and_types(data, torch.device("cpu"))
    iso_global = _compute_isolated_mask(full_eil, n_nodes, torch.device("cpu"))
    target_global = _global_edge_target(data, n_nodes)

    pred_all = torch.zeros(n_nodes, N_FEATURES)
    prob_all = torch.zeros(n_nodes, N_EDGE_TYPES)
    x_seed_all = torch.zeros(n_nodes, N_FEATURES)

    loader = _make_neighbor_loader(data, fanout, batch_size, shuffle=False, seed=seed)
    for batch in loader:
        x, eil, ett, iso, n_id, n_seed = _batch_inputs(batch, data, device, iso_global)
        pred_x, edge_prob, _, _ = model(x, eil, ett, iso)
        seed_ids = n_id[:n_seed]
        pred_all[seed_ids] = pred_x[:n_seed].cpu()
        prob_all[seed_ids] = edge_prob[:n_seed].cpu()
        x_seed_all[seed_ids] = x[:n_seed].cpu()

    per_feat_err = (pred_all - x_seed_all).abs()
    feature_pred_error = per_feat_err.mean(dim=1)
    edge_pred_error = F.binary_cross_entropy(
        prob_all, target_global, reduction="none").mean(dim=1)
    # LAMBDA_EDGE_SCORE, not LAMBDA_EDGE -- see compute_score_frame() above.
    hybrid = feature_pred_error + LAMBDA_EDGE_SCORE * edge_pred_error

    def _norm(t: torch.Tensor) -> np.ndarray:
        v = t.numpy()
        lo, hi = v.min(), v.max()
        return ((v - lo) / (hi - lo + 1e-8)).astype(np.float32)

    return pd.DataFrame({
        "application_id": app_ids,
        "hybrid_anomaly_score": _norm(hybrid),
        "feature_pred_error": _norm(feature_pred_error),
        "edge_pred_error": _norm(edge_pred_error),
    })


def train_sampled(data, x_synth, topo_pack, fanout: tuple[int, int] = (25, 10),
                  batch_size: int = 1024, epochs_s1: int = EPOCHS_STAGE1,
                  epochs_s2: int = EPOCHS_STAGE2, device: torch.device = DEVICE,
                  seed: int = RANDOM_SEED, log_every: int = 1) -> "HybridGraphMCM":
    """Two-stage training via NeighborLoader. Same objectives as train():
    Stage 1 = SVDD compactness + LOE margin (exposure forward is full-batch —
    the exposure set is small by construction); Stage 2 = feature MSE +
    λ_edge·edge BCE, computed on SEED nodes only with global targets.
    Returns the trained model; the CALLER decides where the checkpoint goes
    (scale tests must never touch the live path — hard stop 9)."""
    import time as _time
    torch.manual_seed(seed)
    n_nodes = data["application"].x.shape[0]
    full_eil, _ = _build_edge_index_and_types(data, torch.device("cpu"))
    iso_global = _compute_isolated_mask(full_eil, n_nodes, torch.device("cpu"))
    target_global = _global_edge_target(data, n_nodes)

    model = HybridGraphMCM().to(device)

    # Centroid init from sampled embeddings (batched, no full-graph pass).
    loader = _make_neighbor_loader(data, fanout, batch_size, shuffle=False, seed=seed)
    hs = []
    with torch.no_grad():
        for batch in loader:
            x, eil, ett, iso, n_id, n_seed = _batch_inputs(batch, data, device, iso_global)
            h = model.encode_graph(x, eil, ett, iso)[:n_seed]
            hs.append(h.cpu())
    h_all = torch.cat(hs)
    norms = h_all.norm(dim=1)
    cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
    model.centroid = h_all[norms <= cutoff].mean(dim=0).to(device)
    model.centroid_initialized = True
    print(f"[hybrid] sampled centroid init: norm={model.centroid.norm():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x_synth = x_synth.to(device) if x_synth is not None else None

    epoch_times: list[float] = []
    model.train()
    for stage, n_epochs in (("S1", epochs_s1), ("S2", epochs_s2)):
        for epoch in range(n_epochs):
            t0 = _time.time()
            lam_t = LAMBDA_EXPOSURE * (1.0 - epoch / max(n_epochs, 1))
            loader = _make_neighbor_loader(data, fanout, batch_size,
                                           shuffle=True, seed=seed + epoch)
            agg = {"loss": 0.0, "n": 0}
            for batch in loader:
                x, eil, ett, iso, n_id, n_seed = _batch_inputs(batch, data, device, iso_global)
                optimizer.zero_grad()
                if stage == "S1":
                    h_n = model.encode_graph(x, eil, ett, iso)[:n_seed]
                    svdd = torch.norm(h_n - model.centroid.unsqueeze(0), dim=1).mean()
                    if TOPO_EXPOSURE_ENABLED and topo_pack is not None:
                        h_synth = _get_synth_h_topology(model, topo_pack, device)
                    else:
                        h_synth = _get_synth_h(model, x_synth, eil, ett, device)
                    margin = _derive_loe_margin(h_n, model.centroid)
                    loss = svdd + _loe_loss(h_synth, model.centroid, lam_t, margin)
                else:
                    pred_x, edge_prob, h_n, _ = model(x, eil, ett, iso)
                    tgt = target_global[n_id[:n_seed]].to(device)
                    feat_loss = F.mse_loss(pred_x[:n_seed], x[:n_seed])
                    edge_loss = F.binary_cross_entropy(edge_prob[:n_seed], tgt)
                    # Persistent Stage 2 LOE (mirrors train(), added 2026-07-22)
                    if TOPO_EXPOSURE_ENABLED and topo_pack is not None:
                        h_synth_s2 = _get_synth_h_topology(model, topo_pack, device)
                    else:
                        h_synth_s2 = _get_synth_h(model, x_synth, eil, ett, device)
                    margin_s2 = _derive_loe_margin(h_n[:n_seed], model.centroid)
                    loss = (feat_loss + LAMBDA_EDGE * edge_loss
                            + _loe_loss(h_synth_s2, model.centroid, LOE_STAGE2_WEIGHT, margin_s2))
                loss.backward()
                optimizer.step()
                agg["loss"] += float(loss.item())
                agg["n"] += 1
            dt = _time.time() - t0
            epoch_times.append(dt)
            if (epoch + 1) % log_every == 0:
                print(f"  {stage} epoch {epoch+1}/{n_epochs} | "
                      f"mean_loss={agg['loss']/max(agg['n'],1):.4f} | {dt:.1f}s")
    model._sampled_epoch_times = epoch_times  # scale-test telemetry
    return model


def train_incremental(
    exposure_tensor: torch.Tensor | None = None,
    epochs: int = INCREMENTAL_EPOCHS,
    lr: float = INCREMENTAL_LR,
    freeze_rgcn: bool = True,
    smoke_test: bool = False,
) -> None:
    """
    CPU fine-tune for yearly incremental update.

    Loads existing checkpoint, freezes the RGCN encoder (preserves graph
    topology knowledge), and runs Stage 2 reconstruction loss for `epochs`
    epochs on the current year's graph + feature data.

    exposure_tensor: confirmed fraud + synthetic tensors from confirmed_fraud_store.
                     If None, loads the default synthetic exposure set.
    freeze_rgcn:     True = only update MLP prediction head (recommended for CPU).
                     False = update all weights (use when confirmed fraud > 50).
    """
    print(f"[hybrid] train_incremental() | epochs={epochs} lr={lr} freeze_rgcn={freeze_rgcn} device={DEVICE}")

    if not MODEL_PTH.exists():
        raise FileNotFoundError(f"No checkpoint at {MODEL_PTH}. Run full train() first.")

    schema    = json.loads(SCHEMA_JSON.read_text())
    features  = schema["features"]
    df        = pd.read_csv(FINAL_CSV)
    feat_cols = [c for c in df.columns if c != "application_id"]
    app_ids   = df["application_id"].values

    x_all = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(DEVICE)
    data  = torch.load(GRAPH_PT, weights_only=False)

    if exposure_tensor is None:
        exposure_tensor = torch.load(EXPOSURE_PT, weights_only=True)
    x_exposure = exposure_tensor.to(DEVICE)

    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], DEVICE)

    # Load existing weights
    ckpt  = torch.load(MODEL_PTH, weights_only=False, map_location=DEVICE)
    model = HybridGraphMCM().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.centroid = ckpt["centroid"].to(DEVICE)
    print(f"[hybrid] Loaded checkpoint from {MODEL_PTH}")

    if freeze_rgcn:
        print("[hybrid] Freezing encoder ...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("[hybrid] RGCN encoder frozen — updating predictor + edge_predictor + mask_logits only")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    run_epochs = 2 if smoke_test else epochs
    model.train()
    print(f"[hybrid] Incremental Stage 2: {run_epochs} epochs ...")
    for epoch in range(run_epochs):
        optimizer.zero_grad()

        pred_x, edge_prob, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)
        feat_loss = _feature_pred_loss(pred_x, x_all)
        edge_loss = _edge_pred_loss(edge_prob, edge_index_list, x_all.shape[0], DEVICE)

        # LOE on confirmed/synthetic exposure — keeps fraudster embeddings away from centroid
        h_exp  = _get_synth_h(model, x_exposure, edge_index_list, edge_type_tensor, DEVICE)
        margin = _derive_loe_margin(h_n, model.centroid)
        loe    = _loe_loss(h_exp, model.centroid, LAMBDA_EXPOSURE * 0.5, margin)  # half weight during fine-tune

        loss = feat_loss + LAMBDA_EDGE * edge_loss + loe
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, run_epochs // 5) == 0 or smoke_test:
            print(f"  epoch {epoch+1}/{run_epochs} | feat={feat_loss.item():.4f} edge={edge_loss.item():.4f} loe={loe.item():.4f}")

    # Re-score and save checkpoint (overwrites — old checkpoint backed up)
    _backup_checkpoint()
    _score_and_save(model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features)
    print("[hybrid] Incremental update complete.")


def _backup_checkpoint() -> None:
    if MODEL_PTH.exists():
        backup = MODEL_PTH.with_suffix(".pth.bak")
        import shutil
        shutil.copy2(MODEL_PTH, backup)
        print(f"[hybrid] Previous checkpoint backed up -> {backup}")


def _score_and_save(
    model: HybridGraphMCM,
    x_all: torch.Tensor,
    edge_index_list: list,
    edge_type_tensor: torch.Tensor,
    isolated_mask: torch.Tensor,
    app_ids: np.ndarray,
    features: list[str],
) -> None:
    """Shared scoring + checkpoint save logic used by train() and train_incremental()."""
    out_df = compute_score_frame(model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, features)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[hybrid] Scores saved -> {OUT_CSV}")

    MODEL_PTH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "centroid":         model.centroid.cpu(),
            "config": {
                "N_FEATURES": N_FEATURES, "GRAPH_EMB_DIM": GRAPH_EMB_DIM,
                "GRAPH_HIDDEN": GRAPH_HIDDEN, "MLP_HIDDEN": MLP_HIDDEN,
                "Z_DIM": Z_DIM, "MASK_NUM": MASK_NUM, "N_EDGE_TYPES": N_EDGE_TYPES,
                "ARCH_VERSION": ARCH_VERSION,
            },
            "feature_names": features,
        },
        MODEL_PTH,
    )
    print(f"[hybrid] Checkpoint saved -> {MODEL_PTH}")


def run_attention_summary() -> None:
    print("[attention_summary] Loading model and inputs...")
    model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids, _ = load_model_and_inputs(DEVICE)
    model.eval()

    print("[attention_summary] Computing attention summary...")
    df = model.attention_summary(x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids)

    out_path = Path("outputs/attention_summary_v3.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[attention_summary] Saved to {out_path}")


if __name__ == "__main__":
    import sys
    if "--score-only" in sys.argv:
        score_only()
    elif "--attention-summary" in sys.argv:
        run_attention_summary()
    else:
        train(smoke_test=False)
