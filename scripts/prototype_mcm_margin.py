"""
Prototype: keep Hybrid GraphMCM's philosophy ("predict a node's features from
its neighborhood context") but change what counts as anomalous.

Today: hybrid_anomaly_score = raw prediction error (low error -> "normal").
Problem: a synthetic clique makes the prediction task TRIVIALLY easy (every
neighbor is a near-duplicate), so low error there means "this was an easy
guess," not "this applicant is normal" -- the MAR critique.

This prototype keeps the exact same mechanism (masked-feature stream + RGCN
graph stream + predictor), but scores on a MARGIN instead of raw error:

  error_real   = prediction error using the node's REAL neighbors
  error_random = prediction error using a RANDOMLY SWAPPED same-size set of
                 non-neighbor nodes standing in for its neighborhood
  margin_score = error_random - error_real

A normal applicant: real neighbors help a little more than random ones would
-- a modest, healthy gap. A synthetic-ring member: real neighbors predict it
almost perfectly (they're near-duplicates) while random nodes predict it no
better than chance -- an abnormally LARGE gap. Still "predict from context";
only the anomaly rule changes.

Crucially, this keeps a topology-exposure hook: the SAME synthetic_exposure_
graph_v3.pt file (which promoted confirmed-pattern rings get appended to)
plugs into an LOE-style warm start here exactly as it does in production --
neither dense-block nor the contrastive prototype have any such mechanism.
This script demonstrates that, not just claims it.

Simplifications vs. the real Hybrid GraphMCM (transparent, not hidden):
  - single stochastic feature mask per step, not 8 learned fixed masks
  - no edge_pred_error term -- feature reconstruction only, since that's
    where the neighbor-swap margin idea is being tested
  - one combined training loop (LOE warm-start + reconstruction jointly,
    LOE weight decayed to a small floor rather than dropped to zero), not
    the full Stage-1/Stage-2 split -- fewer epochs than the real 80+120

Writes: outputs/stress_testing_1_v3_stats.json
"""
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score

from src.config_v3 import EDGE_TYPES, RANDOM_SEED, CENTROID_CLEAN_PERCENTILE, LOE_MARGIN
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NAME = "stress_testing_1"
STAGED_GRAPH     = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_v2_full_scores.csv")   # has old hybrid + new contrastive already
GT_CSV           = Path(f"data/uploads/{NAME}_ground_truth.csv")
NODEG_CSV        = Path("data/processed/engineered_features_v3_nodeg.csv")
SCHEMA_JSON      = Path("data/processed/v3_feature_schema.json")
TOPO_PT          = Path("data/processed/synthetic_exposure_graph_v3.pt")

OUT_STATS = Path(f"outputs/{NAME}_v3_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EMB_DIM, MLP_HIDDEN = 64, 32, 128  # trimmed for the 8GB GPU, same reason as prototype_v2_components.py
N_EPOCHS = 150
FEAT_MASK_P = 0.25
LOE_FLOOR = 0.15   # LOE weight decays toward this, never fully to zero (tests the "Stage 2 undoes Stage 1" hypothesis)


def build_full_44dim_x(data, nodeorder: pd.DataFrame) -> torch.Tensor:
    nodeg_cols = [c for c in pd.read_csv(NODEG_CSV, nrows=0).columns if c != "application_id"]
    schema_features = json.loads(SCHEMA_JSON.read_text())["features"]
    x63 = data["application"].x
    n_nodes = x63.shape[0]
    degree_cols = [f"degree_{et}" for et in EDGE_TYPES]
    degree_mat = torch.zeros((n_nodes, len(degree_cols)), dtype=torch.float32)
    for r, et in enumerate(EDGE_TYPES):
        ei = data["application", et, "application"].edge_index
        if ei.numel() == 0:
            continue
        idx, counts = torch.unique(ei.reshape(-1), return_counts=True)
        degree_mat[idx, r] = counts.float() / 2.0
    cols = []
    for name in schema_features:
        if name in nodeg_cols:
            cols.append(x63[:, nodeg_cols.index(name)])
        elif name in degree_cols:
            cols.append(degree_mat[:, degree_cols.index(name)])
        else:
            raise ValueError(f"Schema feature '{name}' not found: {name}")
    x44 = torch.stack(cols, dim=1)
    assert x44.shape[1] == 44
    return x44


class MCMMarginModel(nn.Module):
    """Masked-feature stream + RGCN graph stream + predictor, mirroring
    Hybrid GraphMCM's mechanism (simplified, see module docstring)."""

    def __init__(self, in_dim: int, n_relations: int):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.zeros(in_dim))
        self.conv1 = RGCNConv(in_dim, HIDDEN, n_relations)
        self.conv2 = RGCNConv(HIDDEN, EMB_DIM, n_relations)
        self.predictor = nn.Sequential(
            nn.Linear(in_dim + EMB_DIM, MLP_HIDDEN), nn.ReLU(),
            nn.Linear(MLP_HIDDEN, in_dim),
        )

    def encode_graph(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        return self.conv2(h, edge_index, edge_type)

    def forward(self, x, edge_index, edge_type, h_neighborhood=None):
        """h_neighborhood: precomputed neighbor embedding to use instead of
        self-encoding (lets the margin scorer swap in a random neighborhood
        for the SAME node's masked features without re-running the encoder
        on a hypothetical rewired graph)."""
        keep_prob = torch.sigmoid(self.mask_logits)
        mask = (torch.rand_like(keep_prob) < keep_prob).float()
        x_masked = x * mask.unsqueeze(0)
        if h_neighborhood is None:
            h_neighborhood = self.encode_graph(x, edge_index, edge_type)
        pred_x = self.predictor(torch.cat([x_masked, h_neighborhood], dim=1))
        return pred_x, h_neighborhood


def loe_loss(h_synth: torch.Tensor, centroid: torch.Tensor, lam: float) -> torch.Tensor:
    dist = torch.norm(h_synth - centroid.unsqueeze(0), dim=1)
    exposure = torch.exp(-torch.sqrt(dist + 1e-8))
    return lam * exposure.mean()


def get_synth_h_topology(model: MCMMarginModel, topo_pack: dict) -> torch.Tensor:
    x_topo = topo_pack["x"]
    edge_index = topo_pack["edge_index"]
    edge_type = topo_pack["edge_type"]
    return model.encode_graph(x_topo, edge_index, edge_type)


def main() -> None:
    print(f"[mcm_margin] Loading staged artifacts for '{NAME}' ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    prev = pd.read_csv(FULL_SCORES_CSV)
    prev["application_id"] = prev["application_id"].astype(str)
    cohort_ids = set(prev["application_id"])

    x44 = build_full_44dim_x(data, nodeorder)
    # NOT z-score normalized: the 44 engineered features are already MinMax
    # scaled by the feature engine (project convention), and so is
    # synthetic_exposure_graph_v3.pt's x -- z-scoring only x_real here would
    # put real and topology-exposure features on different scales and break
    # the LOE comparison.
    x_norm = x44.to(DEVICE)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
    n_nodes = x_norm.shape[0]

    topo_available = TOPO_PT.exists()
    if topo_available:
        topo_pack_raw = torch.load(TOPO_PT, weights_only=False)
        topo_pack = {k: v.to(DEVICE) for k, v in topo_pack_raw.items()}
        print(f"[mcm_margin] Topology-exposure file found: {topo_pack['x'].shape[0]} synthetic/promoted nodes "
              f"across {int(topo_pack_raw['cluster_id'].max().item())+1} clusters -- will be used for LOE warm-start.")
    else:
        print("[mcm_margin] WARNING: no topology-exposure file found -- training without LOE warm-start.")

    model = MCMMarginModel(x_norm.shape[1], len(EDGE_TYPES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"[mcm_margin] training {N_EPOCHS} epochs (masked-feature reconstruction + LOE warm-start, "
          f"LOE weight decaying to a floor of {LOE_FLOOR} instead of zero), device={DEVICE} ...")
    model.train()
    for epoch in range(N_EPOCHS):
        pred_x, h_n = model(x_norm, edge_index, edge_type_tensor)
        recon_loss = F.mse_loss(pred_x, x_norm)

        lam = LOE_FLOOR + (1.0 - LOE_FLOOR) * max(0.0, 1.0 - epoch / (N_EPOCHS * 0.6))
        loe_term = torch.tensor(0.0, device=DEVICE)
        if topo_available:
            with torch.no_grad():
                norms = h_n.norm(dim=1)
                cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
                centroid = h_n[norms <= cutoff].mean(dim=0)
            h_synth = get_synth_h_topology(model, topo_pack)
            loe_term = loe_loss(h_synth, centroid, lam)

        loss = recon_loss + loe_term
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCHS}  recon={recon_loss.item():.4f}  loe={float(loe_term):.4f}  lam={lam:.3f}")

    # ── scoring: error_real vs error_random-swap margin ─────────────────────
    print("[mcm_margin] Scoring: error_real vs error_random-swap margin ...")
    model.eval()
    with torch.no_grad():
        h_real = model.encode_graph(x_norm, edge_index, edge_type_tensor)
        pred_real, _ = model(x_norm, edge_index, edge_type_tensor, h_neighborhood=h_real)
        error_real = ((pred_real - x_norm) ** 2).mean(dim=1)

        gen = torch.Generator().manual_seed(RANDOM_SEED + 1)
        rand_perm = torch.randperm(n_nodes, generator=gen).to(DEVICE)
        h_random = h_real[rand_perm]  # swap in a random OTHER node's neighborhood embedding
        pred_random, _ = model(x_norm, edge_index, edge_type_tensor, h_neighborhood=h_random)
        error_random = ((pred_random - x_norm) ** 2).mean(dim=1)

        margin_score = (error_random - error_real).cpu().numpy()
        error_real_score = error_real.cpu().numpy()

    df_scores = pd.DataFrame({
        "application_id": app_ids_merged,
        "mcm_margin_score": margin_score,
        "mcm_error_real_score": error_real_score,
    })
    df_scores["application_id"] = df_scores["application_id"].astype(str)
    df_scores = df_scores[df_scores["application_id"].isin(cohort_ids)]

    df = prev.merge(df_scores, on="application_id", how="left")
    out_csv = Path(f"outputs/{NAME}_v3_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[mcm_margin] Wrote {out_csv}")

    y = df["is_fraud"].astype(int).values
    stats: dict = {"overall": {
        "hybrid_old (RGCN-MCM reconstruction, production)": float(average_precision_score(y, df["hybrid_anomaly_score"].values)),
        "contrastive_new (GRACE-style + redundancy)":        float(average_precision_score(y, df["contrastive_score"].values)),
        "mcm_error_real (this prototype, raw error, no margin)": float(average_precision_score(y, df["mcm_error_real_score"].values)),
        "mcm_margin (this prototype, real-vs-random-neighbor gap)": float(average_precision_score(y, df["mcm_margin_score"].values)),
    }}

    neg = df[df["fraud_type"] == "NONE"]
    stats["per_fraud_type"] = {}
    for ft in sorted(df["fraud_type"].unique()):
        if ft == "NONE":
            continue
        pos = df[df["fraud_type"] == ft]
        sub = pd.concat([pos, neg])
        yy = (sub["fraud_type"] == ft).astype(int).values
        stats["per_fraud_type"][ft] = {
            "n": int(len(pos)),
            "hybrid_old":       float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "mcm_error_real":   float(average_precision_score(yy, sub["mcm_error_real_score"].values)),
            "mcm_margin":       float(average_precision_score(yy, sub["mcm_margin_score"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[mcm_margin] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
