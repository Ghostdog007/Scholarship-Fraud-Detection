"""
Prototype: Deep SAD-style semi-supervised distance objective + per-cluster
prototypes from topology exposure, as a SEPARATE new signal alongside (not
replacing) Hybrid GraphMCM's reconstruction stream.

Rationale (see README changelog / conversation): reconstruction and LOE
compete for the same embedding space -- the better LOE works, the more it
fights reconstruction's pull toward "reconstruct everything well," which is
the likely cause of the mobile-ring regression after the LOE margin fix.
Deep SAD (Ruff et al., ICLR 2020) sidesteps this entirely: no reconstruction
term, unlabeled real data is pulled toward a center, LABELED exposure
examples (topology exposure's synthetic archetypes + promoted confirmed
patterns) are pushed away with an inverted-distance term. Extended here with
PER-CLUSTER prototypes (topology exposure already carries cluster_id) so
known fraud archetypes form distinguishable prototype points rather than one
undifferentiated "away from normal" mass -- closer to AnomalyGFM / few-shot
graph anomaly detection's prototype-matching framing.

Two output scores, both new, neither replacing hybrid_anomaly_score:
  center_dist_score      : distance from the normal center (higher = more anomalous)
  prototype_match_score  : inverse distance to the NEAREST known-archetype
                           prototype (higher = more anomalous = "recognizes
                           this pattern", the literal "memory" signal)

Writes: outputs/stress_testing_1_deepsad_stats.json
"""
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score

from src.config_v3 import EDGE_TYPES, RANDOM_SEED, CENTROID_CLEAN_PERCENTILE
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NAME = "stress_testing_1"
STAGED_GRAPH     = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_afterfix_full_scores.csv")  # has subspace/dense/hybrid/risk + ground truth
NODEG_CSV        = Path("data/processed/engineered_features_v3_nodeg.csv")
SCHEMA_JSON      = Path("data/processed/v3_feature_schema.json")
TOPO_PT          = Path("data/processed/synthetic_exposure_graph_v3.pt")

OUT_STATS = Path(f"outputs/{NAME}_deepsad_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EMB_DIM = 64, 32
N_EPOCHS = 150
ETA = 1.0            # Deep SAD inverted-distance weight for labeled anomalies
LAMBDA_COMPACT = 0.5  # per-cluster prototype compactness weight


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


class DeepSADEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, n_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, HIDDEN, n_relations)
        self.conv2 = RGCNConv(HIDDEN, EMB_DIM, n_relations)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        return self.conv2(h, edge_index, edge_type)


def main() -> None:
    print(f"[deepsad] Loading staged artifacts for '{NAME}' ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    prev = pd.read_csv(FULL_SCORES_CSV)
    prev["application_id"] = prev["application_id"].astype(str)
    cohort_ids = set(prev["application_id"])

    x44 = build_full_44dim_x(data, nodeorder).to(DEVICE)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=DEVICE)

    topo_pack_raw = torch.load(TOPO_PT, weights_only=False)
    topo_pack = {k: v.to(DEVICE) for k, v in topo_pack_raw.items()}
    cluster_id = topo_pack["cluster_id"]
    n_clusters = int(cluster_id.max().item()) + 1
    print(f"[deepsad] Topology exposure: {topo_pack['x'].shape[0]} nodes across {n_clusters} clusters")

    model = DeepSADEncoder(x44.shape[1], len(EDGE_TYPES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize center from an untrained forward pass (same convention as
    # production's init_centroid: exclude the highest-norm 5% to avoid the
    # center drifting toward outliers before training starts).
    with torch.no_grad():
        h0 = model(x44, edge_index, edge_type_tensor)
        norms0 = h0.norm(dim=1)
        cutoff0 = torch.quantile(norms0, CENTROID_CLEAN_PERCENTILE / 100.0)
        center = h0[norms0 <= cutoff0].mean(dim=0).detach()

    print(f"[deepsad] training {N_EPOCHS} epochs (Deep SAD distance + per-cluster prototype compactness) ...")
    model.train()
    for epoch in range(N_EPOCHS):
        h_real = model(x44, edge_index, edge_type_tensor)
        h_synth = model(topo_pack["x"], topo_pack["edge_index"], topo_pack["edge_type"])

        dist_real = ((h_real - center) ** 2).sum(dim=1)
        loss_normal = dist_real.mean()

        dist_synth = ((h_synth - center) ** 2).sum(dim=1)
        loss_anomaly = (ETA / (dist_synth + 1e-6)).mean()  # Deep SAD inverted-distance term

        # Per-cluster prototype compactness: known archetypes should form
        # distinguishable, tight clusters, not one diffuse "away from normal" mass.
        loss_compact = torch.tensor(0.0, device=DEVICE)
        for k in range(n_clusters):
            mask = cluster_id == k
            if mask.sum() < 2:
                continue
            proto_k = h_synth[mask].mean(dim=0).detach()
            loss_compact = loss_compact + ((h_synth[mask] - proto_k) ** 2).sum(dim=1).mean()
        loss_compact = loss_compact / max(n_clusters, 1)

        loss = loss_normal + loss_anomaly + LAMBDA_COMPACT * loss_compact
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Recompute center periodically (EMA-free, simple refresh) so it tracks
        # the evolving embedding space rather than staying fixed at init.
        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                norms = h_real.norm(dim=1)
                cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
                center = h_real[norms <= cutoff].mean(dim=0).detach()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCHS}  normal={loss_normal.item():.4f} "
                  f"anomaly={loss_anomaly.item():.4f} compact={loss_compact.item():.4f}")

    print("[deepsad] Scoring: center-distance + nearest-prototype-match ...")
    model.eval()
    with torch.no_grad():
        h_real_final = model(x44, edge_index, edge_type_tensor)
        h_synth_final = model(topo_pack["x"], topo_pack["edge_index"], topo_pack["edge_type"])

        norms = h_real_final.norm(dim=1)
        cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
        center = h_real_final[norms <= cutoff].mean(dim=0)

        center_dist = ((h_real_final - center) ** 2).sum(dim=1).sqrt()

        prototypes = []
        for k in range(n_clusters):
            mask = cluster_id == k
            if mask.sum() == 0:
                continue
            prototypes.append(h_synth_final[mask].mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)  # (K, emb_dim)

        # Distance from every real node to every prototype -> nearest match
        d = torch.cdist(h_real_final, prototypes)  # (N, K)
        nearest_proto_dist = d.min(dim=1).values
        prototype_match_score = 1.0 / (nearest_proto_dist + 1e-3)

    df_scores = pd.DataFrame({
        "application_id": app_ids_merged,
        "center_dist_score": center_dist.cpu().numpy(),
        "prototype_match_score": prototype_match_score.cpu().numpy(),
    })
    df_scores["application_id"] = df_scores["application_id"].astype(str)
    df_scores = df_scores[df_scores["application_id"].isin(cohort_ids)]

    df = prev.merge(df_scores, on="application_id", how="left")
    out_csv = Path(f"outputs/{NAME}_deepsad_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[deepsad] Wrote {out_csv}")

    def _minmax(x):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-9)

    df["deepsad_max"] = np.maximum.reduce([_minmax(df["center_dist_score"].values),
                                           _minmax(df["prototype_match_score"].values)])

    y = df["is_fraud"].astype(int).values
    stats: dict = {"overall": {
        "hybrid_reconstruction (unchanged, after LOE fix)": float(average_precision_score(y, df["hybrid_anomaly_score"].values)),
        "center_dist_score (new, Deep SAD distance-from-normal)":      float(average_precision_score(y, df["center_dist_score"].values)),
        "prototype_match_score (new, nearest-known-archetype match)":  float(average_precision_score(y, df["prototype_match_score"].values)),
        "deepsad_max (new, max of the two above)":                     float(average_precision_score(y, df["deepsad_max"].values)),
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
            "hybrid_reconstruction": float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "center_dist":           float(average_precision_score(yy, sub["center_dist_score"].values)),
            "prototype_match":       float(average_precision_score(yy, sub["prototype_match_score"].values)),
            "deepsad_max":           float(average_precision_score(yy, sub["deepsad_max"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[deepsad] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
