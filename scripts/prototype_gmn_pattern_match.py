"""
Prototype: Graph Matching Network (GMN)-style pattern matching, as a SEPARATE
new signal alongside (not replacing) Hybrid GraphMCM's reconstruction stream
or the Deep SAD center-distance prototype.

Rationale (see conversation / README changelog): the Deep SAD per-cluster
prototype test (prototype_deepsad_prototypes.py) showed nearest-prototype
matching underperforms (0.116 overall PR-AUC, near-random on most
categories) because collapsing each exposure cluster to a single mean vector
and only pulling members toward that mean (never pushing different clusters
apart) leaves prototypes too close together in embedding space to
discriminate. Graph Matching Networks (Li et al., ICML 2019) replace
"nearest mean" with cross-graph ATTENTION: a query node attends over every
node in a reference pattern's subgraph individually, producing a matched
context vector, and a (query, matched-context, difference) feature decides
the match score. This keeps each reference cluster's internal structure
distinguishable instead of collapsing it to one point.

Scoped down from the full Li et al. formulation for tractability on a
single 8GB GPU at 50k-node scale: attention is computed once at the final
embedding layer (not propagated through every message-passing round), and
each of the ~50 exposure clusters is treated as one reference pattern
subgraph. This is a real cross-graph-attention matching mechanism, not a
literal reproduction of the paper.

Training signal (no real labels used, same convention as LOE/Deep SAD):
each exposure cluster's own synthetic nodes should match well against the
REST of their own cluster (leave-one-out) and poorly against other
clusters' nodes and a sampled background of real nodes. Real nodes are
scored only at inference time.

Output score:
  gmn_match_score : max over all known cluster patterns of the cross-graph
                     match score (higher = matches a known fraud archetype
                     more strongly)

Writes: outputs/stress_testing_1_gmn_stats.json
"""
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score

from src.config_v3 import EDGE_TYPES, RANDOM_SEED
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NAME = "stress_testing_1"
STAGED_GRAPH     = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_afterfix_full_scores.csv")
NODEG_CSV        = Path("data/processed/engineered_features_v3_nodeg.csv")
SCHEMA_JSON      = Path("data/processed/v3_feature_schema.json")
TOPO_PT          = Path("data/processed/synthetic_exposure_graph_v3.pt")

OUT_STATS = Path(f"outputs/{NAME}_gmn_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EMB_DIM = 64, 32
N_EPOCHS = 150
NEG_CLUSTERS_PER_STEP = 4   # other-cluster negatives sampled per positive cluster per epoch
NEG_BACKGROUND = 64         # random real-node negatives sampled per positive cluster per epoch


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


class SharedEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, n_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, HIDDEN, n_relations)
        self.conv2 = RGCNConv(HIDDEN, EMB_DIM, n_relations)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        return self.conv2(h, edge_index, edge_type)


class MatchHead(torch.nn.Module):
    """Cross-graph attention + (query, matched-context, difference) -> match logit."""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.scale = math.sqrt(emb_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 3, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, 1),
        )

    def forward(self, h_query: torch.Tensor, h_ref: torch.Tensor) -> torch.Tensor:
        # h_query: (Q, d), h_ref: (R, d) -> (Q,) match logits
        sim = (h_query @ h_ref.t()) / self.scale       # (Q, R)
        attn = torch.softmax(sim, dim=1)
        matched = attn @ h_ref                          # (Q, d)
        feat = torch.cat([h_query, matched, h_query - matched], dim=1)
        return self.mlp(feat).squeeze(-1)


def main() -> None:
    print(f"[gmn] Loading staged artifacts for '{NAME}' ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    prev = pd.read_csv(FULL_SCORES_CSV)
    prev["application_id"] = prev["application_id"].astype(str)
    cohort_ids = set(prev["application_id"])

    x44 = build_full_44dim_x(data, nodeorder).to(DEVICE)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
    n_real = x44.shape[0]

    topo_pack_raw = torch.load(TOPO_PT, weights_only=False)
    topo_pack = {k: v.to(DEVICE) for k, v in topo_pack_raw.items()}
    cluster_id = topo_pack["cluster_id"]
    n_clusters = int(cluster_id.max().item()) + 1
    cluster_members = [torch.where(cluster_id == k)[0] for k in range(n_clusters)]
    cluster_members = [c for c in cluster_members if c.numel() >= 4]  # need enough for leave-one-out split
    n_clusters = len(cluster_members)
    print(f"[gmn] Topology exposure: {topo_pack['x'].shape[0]} nodes across {n_clusters} usable clusters (>=4 members)")

    encoder = SharedEncoder(x44.shape[1], len(EDGE_TYPES)).to(DEVICE)
    head = MatchHead(EMB_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)

    print(f"[gmn] training {N_EPOCHS} epochs (cross-graph attention matching) ...")
    encoder.train()
    head.train()
    for epoch in range(N_EPOCHS):
        h_synth = encoder(topo_pack["x"], topo_pack["edge_index"], topo_pack["edge_type"])
        # Background real-node embeddings recomputed each epoch too (encoder is shared/trained)
        h_real = encoder(x44, edge_index, edge_type_tensor)

        pos_losses = []
        neg_losses = []
        other_cluster_order = np.random.permutation(n_clusters)
        for ci, members in enumerate(cluster_members):
            m = members.numel()
            perm = members[torch.randperm(m, device=DEVICE)]
            half = m // 2
            query_idx, ref_idx = perm[:half], perm[half:]
            if query_idx.numel() == 0 or ref_idx.numel() == 0:
                continue
            h_q = h_synth[query_idx]
            h_r = h_synth[ref_idx]

            pos_score = head(h_q, h_r)
            pos_losses.append(F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)))

            # Negatives: other clusters' nodes
            other_ids = [j for j in other_cluster_order if j != ci][:NEG_CLUSTERS_PER_STEP]
            if other_ids:
                neg_ref_idx = torch.cat([cluster_members[j] for j in other_ids])
                h_neg_ref = h_synth[neg_ref_idx]
                neg_score = head(h_q, h_neg_ref)
                neg_losses.append(F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score)))

            # Negatives: random real background nodes (assumed mostly clean, same
            # convention as centroid clean-percentile elsewhere in this project)
            bg_idx = torch.randint(0, n_real, (NEG_BACKGROUND,), device=DEVICE)
            h_bg = h_real[bg_idx]
            neg_bg_score = head(h_q, h_bg)
            neg_losses.append(F.binary_cross_entropy_with_logits(neg_bg_score, torch.zeros_like(neg_bg_score)))

        loss = torch.stack(pos_losses).mean() + torch.stack(neg_losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCHS}  loss={loss.item():.4f} "
                  f"pos={torch.stack(pos_losses).mean().item():.4f} neg={torch.stack(neg_losses).mean().item():.4f}")

    print("[gmn] Scoring: max cross-graph match score over all known patterns ...")
    encoder.eval()
    head.eval()
    with torch.no_grad():
        h_synth_final = encoder(topo_pack["x"], topo_pack["edge_index"], topo_pack["edge_type"])
        h_real_final = encoder(x44, edge_index, edge_type_tensor)

        best_score = torch.full((n_real,), -1e9, device=DEVICE)
        for members in cluster_members:
            h_ref = h_synth_final[members]
            score_k = head(h_real_final, h_ref)  # (n_real,)
            best_score = torch.maximum(best_score, score_k)

    df_scores = pd.DataFrame({
        "application_id": app_ids_merged,
        "gmn_match_score": best_score.cpu().numpy(),
    })
    df_scores["application_id"] = df_scores["application_id"].astype(str)
    df_scores = df_scores[df_scores["application_id"].isin(cohort_ids)]

    df = prev.merge(df_scores, on="application_id", how="left")
    out_csv = Path(f"outputs/{NAME}_gmn_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[gmn] Wrote {out_csv}")

    y = df["is_fraud"].astype(int).values
    stats: dict = {"overall": {
        "hybrid_reconstruction (unchanged, after LOE fix)": float(average_precision_score(y, df["hybrid_anomaly_score"].values)),
        "gmn_match_score (new, cross-graph attention pattern match)": float(average_precision_score(y, df["gmn_match_score"].values)),
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
            "gmn_match_score": float(average_precision_score(yy, sub["gmn_match_score"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[gmn] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
