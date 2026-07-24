"""
Prototype: UniGAD-inspired subgraph-level anomaly scoring (Tang et al.,
"UniGAD: Unifying Multi-level Graph Anomaly Detection", NeurIPS 2024).

Rationale: GMN (prototype_gmn_pattern_match.py) scored a node by attending
over a FIXED reference cluster's full node set and came back statistically
indistinguishable from the current reconstruction baseline. UniGAD's core
idea is different: instead of a fixed-radius neighborhood or a fixed
reference set, LEARN which local subgraph around a node is the
anomaly-relevant one (their MRQSampler, via a Rayleigh-quotient / spectral
energy objective), then score that extracted subgraph jointly with the
node itself (their GraphStitch fusion).

Scoped down for tractability on one 8GB GPU / 50k-node graph, and honestly
labeled as such:
  - Ego-subgraphs are 1-hop, capped at MAX_NEIGHBORS members (a soft
    version of the project's existing hub-cap idea), not the paper's
    general k-hop candidate pool.
  - The Laplacian quadratic form is computed over a STAR topology
    (center-to-neighbor edges only), not the full induced subgraph
    Laplacian (neighbor-to-neighbor edges are not included) -- this is a
    real simplification, not the paper's exact MRQSampler.
  - Node importance scores (which neighbors matter) are a single learned
    MLP head (SAGPool-style), not the paper's trained sampler network.
  - GraphStitch is approximated by a single fusion MLP over
    [center embedding ; soft-pooled subgraph embedding ; spectral energy
    scalar] rather than a shared multi-task backbone across node/edge/
    subgraph levels.

This keeps the CORE mechanism -- learned subgraph-member selection +
spectral (Laplacian quadratic) signal + fused node/subgraph scoring --
while staying tractable. Trained on stress_testing_1's staged artifacts
only; production is untouched.

Training signal (same convention as every other prototype this session --
no real fraud labels used at train time, only topology exposure's
synthetic archetypes):
  positives : each exposure cluster's own real topo edges define genuine
              ring ego-subgraphs (center = a cluster member, neighbors =
              its real cluster-mates via topo_pack edges)
  negatives : real (production) nodes' own 1-hop ego-subgraphs, assumed
              mostly non-fraud (same "clean background" convention used by
              DeepSVDD centroid / Deep SAD elsewhere in this project)

Output score: unigad_subgraph_score (higher = more anomalous)

Writes: outputs/stress_testing_1_unigad_stats.json
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

OUT_STATS = Path(f"outputs/{NAME}_unigad_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EMB_DIM = 64, 32
N_EPOCHS = 150
MAX_NEIGHBORS = 25          # ego-subgraph cap, mirrors the project's hub-cap concept
POS_CENTERS_PER_EPOCH = 128 # sampled cluster-member centers per epoch
NEG_CENTERS_PER_EPOCH = 128 # sampled real background centers per epoch
SCORE_BATCH = 2000          # batch size for full-population inference pass


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


def build_csr_adjacency(edge_index: torch.Tensor, n_nodes: int):
    """Undirected adjacency as CSR-style (offsets, neighbor array), CPU numpy."""
    ei = edge_index.cpu().numpy()
    if ei.shape[1] == 0:
        return np.zeros(n_nodes + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    src = np.concatenate([ei[0], ei[1]])
    dst = np.concatenate([ei[1], ei[0]])
    order = np.argsort(src, kind="stable")
    sorted_src = src[order]
    sorted_dst = dst[order]
    counts = np.bincount(sorted_src, minlength=n_nodes)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return offsets, sorted_dst


def sample_neighbors(offsets: np.ndarray, neighbors: np.ndarray, node: int, cap: int) -> np.ndarray:
    lo, hi = offsets[node], offsets[node + 1]
    nbrs = neighbors[lo:hi]
    nbrs = nbrs[nbrs != node]
    if nbrs.shape[0] > cap:
        nbrs = np.random.choice(nbrs, size=cap, replace=False)
    return nbrs


class SharedEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, n_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, HIDDEN, n_relations)
        self.conv2 = RGCNConv(HIDDEN, EMB_DIM, n_relations)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        return self.conv2(h, edge_index, edge_type)


class SubgraphScorer(torch.nn.Module):
    """MRQSampler-lite (learned member importance + star Laplacian energy)
    fused with GraphStitch-lite (single head over center/pooled/energy)."""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.importance = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim // 2), torch.nn.ReLU(),
            torch.nn.Linear(emb_dim // 2, 1),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2 + 1, emb_dim), torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, 1),
        )

    def score_one(self, z_center: torch.Tensor, z_neighbors: torch.Tensor) -> torch.Tensor:
        # z_center: (d,), z_neighbors: (m, d) -> scalar logit
        if z_neighbors.shape[0] == 0:
            g = z_center
            energy = torch.zeros((), device=z_center.device)
        else:
            s = torch.sigmoid(self.importance(z_neighbors)).squeeze(-1)      # (m,)
            g = (s.unsqueeze(-1) * z_neighbors).sum(0) / (s.sum() + 1e-6)     # pooled subgraph readout
            diff2 = ((z_neighbors - z_center.unsqueeze(0)) ** 2).sum(-1)      # (m,) star Laplacian terms
            energy = (s * diff2).sum()                                        # spectral (quadratic-form) energy
        feat = torch.cat([z_center, g, energy.unsqueeze(0)], dim=0)
        return self.head(feat).squeeze(-1)

    def score_batch(self, z_all: torch.Tensor, centers: list[int], offsets, neighbors, cap: int) -> torch.Tensor:
        logits = []
        for c in centers:
            nbrs = sample_neighbors(offsets, neighbors, c, cap)
            z_c = z_all[c]
            z_n = z_all[nbrs] if nbrs.shape[0] > 0 else z_all.new_zeros((0, z_all.shape[1]))
            logits.append(self.score_one(z_c, z_n))
        return torch.stack(logits)


def main() -> None:
    print(f"[unigad] Device: {DEVICE}")
    print(f"[unigad] Loading staged artifacts for '{NAME}' ...")
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
    real_offsets, real_neighbors = build_csr_adjacency(edge_index, n_real)

    topo_pack_raw = torch.load(TOPO_PT, weights_only=False)
    topo_pack = {k: v.to(DEVICE) for k, v in topo_pack_raw.items()}
    cluster_id = topo_pack["cluster_id"]
    n_synth = topo_pack["x"].shape[0]
    topo_offsets, topo_neighbors = build_csr_adjacency(topo_pack["edge_index"], n_synth)
    cluster_np = cluster_id.cpu().numpy()
    valid_centers = np.where((topo_offsets[1:] - topo_offsets[:-1]) > 0)[0]
    print(f"[unigad] Topology exposure: {n_synth} nodes, {len(valid_centers)} with real degree > 0")

    encoder = SharedEncoder(x44.shape[1], len(EDGE_TYPES)).to(DEVICE)
    scorer = SubgraphScorer(EMB_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(scorer.parameters()), lr=1e-3)

    print(f"[unigad] training {N_EPOCHS} epochs (learned subgraph selection + spectral energy + fused scoring) ...")
    encoder.train()
    scorer.train()
    for epoch in range(N_EPOCHS):
        z_real = encoder(x44, edge_index, edge_type_tensor)
        z_synth = encoder(topo_pack["x"], topo_pack["edge_index"], topo_pack["edge_type"])

        pos_centers = np.random.choice(valid_centers, size=min(POS_CENTERS_PER_EPOCH, len(valid_centers)), replace=False)
        neg_centers = np.random.choice(n_real, size=NEG_CENTERS_PER_EPOCH, replace=False)

        pos_logits = scorer.score_batch(z_synth, pos_centers.tolist(), topo_offsets, topo_neighbors, MAX_NEIGHBORS)
        neg_logits = scorer.score_batch(z_real, neg_centers.tolist(), real_offsets, real_neighbors, MAX_NEIGHBORS)

        loss = (F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCHS}  loss={loss.item():.4f} "
                  f"pos_logit_mean={pos_logits.mean().item():.3f} neg_logit_mean={neg_logits.mean().item():.3f}")

    print("[unigad] Scoring all real nodes (batched) ...")
    encoder.eval()
    scorer.eval()
    with torch.no_grad():
        z_real_final = encoder(x44, edge_index, edge_type_tensor)
        all_scores = torch.zeros(n_real, device=DEVICE)
        for start in range(0, n_real, SCORE_BATCH):
            end = min(start + SCORE_BATCH, n_real)
            centers = list(range(start, end))
            logits = scorer.score_batch(z_real_final, centers, real_offsets, real_neighbors, MAX_NEIGHBORS)
            all_scores[start:end] = torch.sigmoid(logits)
            if start % (SCORE_BATCH * 5) == 0:
                print(f"  scored {end}/{n_real}")

    df_scores = pd.DataFrame({
        "application_id": app_ids_merged,
        "unigad_subgraph_score": all_scores.cpu().numpy(),
    })
    df_scores["application_id"] = df_scores["application_id"].astype(str)
    df_scores = df_scores[df_scores["application_id"].isin(cohort_ids)]

    df = prev.merge(df_scores, on="application_id", how="left")
    out_csv = Path(f"outputs/{NAME}_unigad_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[unigad] Wrote {out_csv}")

    y = df["is_fraud"].astype(int).values
    stats: dict = {"overall": {
        "hybrid_reconstruction (unchanged, after LOE fix)": float(average_precision_score(y, df["hybrid_anomaly_score"].values)),
        "unigad_subgraph_score (new, learned-subgraph + spectral energy)": float(average_precision_score(y, df["unigad_subgraph_score"].values)),
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
            "unigad_subgraph_score": float(average_precision_score(yy, sub["unigad_subgraph_score"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[unigad] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
