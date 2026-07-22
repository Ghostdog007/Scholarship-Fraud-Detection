"""
Prototype comparison, run READ-ONLY against the stress_testing_1 staged
artifacts. Does NOT touch any locked/production module
(hybrid_graphmcm_v3.py, dense_block_detector_v3.py, fusion_classifier_v3.py
are only ever imported for their pure scoring functions, never modified).

Three prototypes, each compared against its predecessor on the SAME
ground-truth-labeled cohort:

  1. Dense-block extended to shares_mobile + shares_pincode (not just
     shares_ip), each relation scored independently exactly like the locked
     detector, then MAX-combined across relations before it would reach
     fusion -- mirrors subspace_if_v3's own group-max pattern instead of
     adding two more fusion weights.

  2. A contrastive graph encoder replacing Hybrid GraphMCM's reconstruction
     objective. NOTE on design: textbook CoLA (node vs. its own 1-hop
     subgraph agreement) would NOT catch our injected rings -- a ring member
     agrees perfectly with its (near-identical) neighbors by construction, so
     CoLA-style local agreement would score it as NORMAL, reproducing the
     exact reconstruction blind spot we're trying to escape. Instead: a
     GRACE/DGI-style contrastive encoder (two augmented views, InfoNCE
     pretraining) produces embeddings, and the anomaly score is GROUP-LEVEL
     embedding redundancy -- how suspiciously similar a node's embedding is
     to its real relational neighbors' embeddings. A synthetic clique
     produces near-duplicate embeddings across many members; a normal
     heterogeneous neighborhood does not, even though it's just as densely
     connected. This is an explicit adaptation, not literal CoLA.

  3. Two label-independent fusion alternatives to the locked weighted-sum:
     plain max, and Borda-style rank aggregation (average rank across
     detectors, converted back to a descending score) -- both compared
     against weighted-sum and against each component alone.

Writes: outputs/stress_testing_1_v2_stats.json
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

from src.config_v3 import EDGE_TYPES, RANDOM_SEED, DENSE_BLOCK_CAMOUFLAGE_C, DENSE_BLOCK_KCORE_PREFILTER
from src.dense_block_detector_v3 import dense_block_scores, _k_core_prune, _charikar_peeling
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types
from src.fusion_classifier_v3 import score_level_fusion, _minmax

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NAME = "stress_testing_1"
STAGED_GRAPH     = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_full_scores.csv")   # from stress_test_1_analysis.py
GT_CSV           = Path(f"data/uploads/{NAME}_ground_truth.csv")
NODEG_CSV        = Path("data/processed/engineered_features_v3_nodeg.csv")
SCHEMA_JSON      = Path("data/processed/v3_feature_schema.json")

OUT_STATS = Path(f"outputs/{NAME}_v2_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EMB_DIM = 64, 32   # trimmed from 128/64 -- two full-graph (7.1M-edge) forward passes per step OOM'd an 8GB GPU at 128/64
N_EPOCHS = 120
NEG_BATCH = 4096
FEAT_MASK_P = 0.2
EDGE_DROP_P = 0.2


# ── build the true 44-dim FINAL-schema feature matrix for ALL merged nodes ──
def build_full_44dim_x(data, nodeorder: pd.DataFrame) -> torch.Tensor:
    """data['application'].x is the 63-col pre-degree NODEG matrix (identifiers
    still present, no degree cols). The real 44-dim model input = 39 of those
    63 columns (by name, per the schema) + 5 degree columns computed from the
    graph's own edges. Column NAMES/order come from the current (unmerged)
    NODEG_CSV header -- build_graph() writes the graph's x in that same
    column order regardless of row count, so this is safe."""
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
        degree_mat[idx, r] = counts.float() / 2.0  # each undirected edge counted twice

    cols = []
    for name in schema_features:
        if name in nodeg_cols:
            cols.append(x63[:, nodeg_cols.index(name)])
        elif name in degree_cols:
            cols.append(degree_mat[:, degree_cols.index(name)])
        else:
            raise ValueError(f"Schema feature '{name}' not found in nodeg columns or degree columns")
    x44 = torch.stack(cols, dim=1)
    assert x44.shape[1] == 44, f"expected 44 columns, got {x44.shape[1]}"
    return x44


# ── 1. Extended dense-block (mobile + ip + pincode), max-combined ──────────
def extended_dense_block(data, app_ids_merged: np.ndarray) -> pd.DataFrame:
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    from src.config_v3 import DENSE_BLOCK_RELATIONS as _unused  # documents what we're overriding
    relations = [0, 1, 4]  # shares_mobile, shares_ip, shares_pincode

    import src.dense_block_detector_v3 as dbd
    orig = dbd.DENSE_BLOCK_RELATIONS
    dbd.DENSE_BLOCK_RELATIONS = relations
    try:
        df = dense_block_scores(edge_index_list, edge_type_tensor, len(app_ids_merged), app_ids_merged)
    finally:
        dbd.DENSE_BLOCK_RELATIONS = orig

    rel_cols = [f"dense_block_score_{EDGE_TYPES[r].replace('shares_', '')}" for r in relations]
    df["dense_block_score_relational"] = df[rel_cols].max(axis=1)
    return df[["application_id"] + rel_cols + ["dense_block_score_relational"]]


# ── 2. Contrastive RGCN encoder + embedding-redundancy anomaly score ───────
class ContrastiveRGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int, n_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden, n_relations)
        self.conv2 = RGCNConv(hidden, emb_dim, n_relations)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        h = self.conv2(h, edge_index, edge_type)
        return h


def _augment(x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, gen: torch.Generator):
    feat_mask = (torch.rand(x.shape[1], generator=gen) > FEAT_MASK_P).float().to(x.device)
    x_aug = x * feat_mask.unsqueeze(0)
    keep = (torch.rand(edge_index.shape[1], generator=gen) > EDGE_DROP_P).to(edge_index.device)
    return x_aug, edge_index[:, keep], edge_type[keep]


def info_nce(h1: torch.Tensor, h2: torch.Tensor, batch_idx: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    a = F.normalize(h1[batch_idx], dim=1)
    b = F.normalize(h2[batch_idx], dim=1)
    sim = a @ b.t() / temp
    labels = torch.arange(len(batch_idx), device=a.device)
    return F.cross_entropy(sim, labels)


def train_contrastive_encoder(x44: torch.Tensor, data) -> torch.Tensor:
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=DEVICE)

    x_norm = (x44 - x44.mean(0, keepdim=True)) / (x44.std(0, keepdim=True) + 1e-6)
    x_norm = x_norm.to(DEVICE)
    model = ContrastiveRGCN(x_norm.shape[1], HIDDEN, EMB_DIM, len(EDGE_TYPES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_nodes = x_norm.shape[0]
    gen = torch.Generator().manual_seed(RANDOM_SEED)  # CPU generator; index tensors moved to DEVICE explicitly

    print(f"[contrastive] training {N_EPOCHS} epochs on {n_nodes} nodes, {edge_index.shape[1]} directed edges, device={DEVICE} ...")
    model.train()
    for epoch in range(N_EPOCHS):
        x1, ei1, et1 = _augment(x_norm, edge_index, edge_type_tensor, gen)
        x2, ei2, et2 = _augment(x_norm, edge_index, edge_type_tensor, gen)
        h1 = model(x1, ei1, et1)
        h2 = model(x2, ei2, et2)

        batch_idx = torch.randperm(n_nodes, generator=gen)[:NEG_BATCH].to(DEVICE)
        loss = 0.5 * (info_nce(h1, h2, batch_idx) + info_nce(h2, h1, batch_idx))

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCHS}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        h_final = model(x_norm, edge_index, edge_type_tensor)
    return h_final.cpu()  # move back to CPU -- downstream scoring is a per-node Python loop, GPU would only add sync overhead there


def embedding_redundancy_score(h_final: torch.Tensor, data) -> np.ndarray:
    """Per node: mean cosine similarity to its REAL relational neighbors
    (union across all 5 relations). High = suspiciously uniform local
    neighborhood (a synthetic clique); isolated nodes score 0, same
    isolated-node convention used elsewhere in this codebase."""
    n_nodes = h_final.shape[0]
    h_norm = F.normalize(h_final, dim=1)
    adj: dict[int, set[int]] = {}
    for et in EDGE_TYPES:
        ei = data["application", et, "application"].edge_index
        if ei.numel() == 0:
            continue
        src, dst = ei[0].tolist(), ei[1].tolist()
        for u, v in zip(src, dst):
            if u == v:
                continue
            adj.setdefault(u, set()).add(v)

    score = np.zeros(n_nodes, dtype=np.float32)
    for u, nbrs in adj.items():
        nbr_idx = torch.tensor(list(nbrs), dtype=torch.long)
        sims = (h_norm[u].unsqueeze(0) * h_norm[nbr_idx]).sum(dim=1)
        score[u] = sims.mean().item()
    return score


# ── 3. Fusion alternatives ──────────────────────────────────────────────────
def borda_fusion(components: list[np.ndarray]) -> np.ndarray:
    """Average rank across detectors (higher raw score = better rank),
    normalised back to [0,1] so higher = more anomalous, consistent with the
    rest of the codebase's convention."""
    ranks = [pd.Series(c).rank(method="average").values for c in components]
    avg_rank = np.mean(ranks, axis=0)
    return _minmax(avg_rank)


def plain_max_fusion(components: list[np.ndarray]) -> np.ndarray:
    normed = [_minmax(c) for c in components]
    return np.maximum.reduce(normed)


def main() -> None:
    print(f"[prototype_v2] Loading staged artifacts for '{NAME}' ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    full = pd.read_csv(FULL_SCORES_CSV)
    full["application_id"] = full["application_id"].astype(str)
    cohort_ids = set(full["application_id"])

    print("[prototype_v2] Reconstructing the true 44-dim FINAL feature matrix for all merged nodes ...")
    x44 = build_full_44dim_x(data, nodeorder)

    # ── 1. extended dense-block: reuse the already-computed result from the
    # first prototype_v2 run (outputs/stress_testing_1_v2_full_scores.csv) --
    # deterministic Charikar peeling, nothing changed, no need to redo the
    # ~8min recompute just to get a fresh contrastive epoch count ──────────
    print("[prototype_v2] Reusing previously computed extended dense-block scores ...")
    prev_v2 = pd.read_csv(Path(f"outputs/{NAME}_v2_full_scores.csv"))
    prev_v2["application_id"] = prev_v2["application_id"].astype(str)
    dense_ext = prev_v2[["application_id", "dense_block_score_relational"]].copy()
    dense_ext = dense_ext[dense_ext["application_id"].isin(cohort_ids)]

    # ── 2. contrastive encoder + redundancy score ────────────────────────
    h_final = train_contrastive_encoder(x44, data)
    print("[prototype_v2] Computing embedding-redundancy anomaly score ...")
    redundancy = embedding_redundancy_score(h_final, data)
    contrastive_df = pd.DataFrame({
        "application_id": app_ids_merged,
        "contrastive_score": redundancy,
    })
    contrastive_df["application_id"] = contrastive_df["application_id"].astype(str)
    contrastive_df = contrastive_df[contrastive_df["application_id"].isin(cohort_ids)]

    # ── merge everything ──────────────────────────────────────────────────
    df = (full
          .merge(dense_ext[["application_id", "dense_block_score_relational"]], on="application_id", how="left")
          .merge(contrastive_df, on="application_id", how="left"))
    df["dense_block_score_relational"] = df["dense_block_score_relational"].fillna(0.0)
    df["contrastive_score"] = df["contrastive_score"].fillna(0.0)

    # ── 3. fusion alternatives, OLD components (subspace + dense_ip + hybrid) ─
    s = df["subspace_if_score"].values
    d_old = df["dense_block_score_ip"].values
    h_old = df["hybrid_anomaly_score"].values
    d_new = df["dense_block_score_relational"].values
    c_new = df["contrastive_score"].values

    df["fusion_weighted_sum_old"] = score_level_fusion(s, d_old, h_old)  # locked, unchanged
    df["fusion_max_old"]          = plain_max_fusion([s, d_old, h_old])
    df["fusion_borda_old"]        = borda_fusion([s, d_old, h_old])

    df["fusion_weighted_sum_new"] = score_level_fusion(s, d_new, c_new)  # same weights, new components
    df["fusion_max_new"]          = plain_max_fusion([s, d_new, c_new])
    df["fusion_borda_new"]        = borda_fusion([s, d_new, c_new])

    out_csv = Path(f"outputs/{NAME}_v2_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[prototype_v2] Wrote {out_csv}")

    # ── stats ────────────────────────────────────────────────────────────
    y = df["is_fraud"].astype(int).values
    stats: dict = {}

    def pr(col):
        return float(average_precision_score(y, df[col].values))

    stats["overall"] = {
        "subspace_if (unchanged)":            pr("subspace_if_score"),
        "dense_block_ip (old, ip-only)":       pr("dense_block_score_ip"),
        "dense_block_relational (new, max of mobile/ip/pincode)": pr("dense_block_score_relational"),
        "hybrid_anomaly_score (old, RGCN-MCM reconstruction)":    pr("hybrid_anomaly_score"),
        "contrastive_score (new, GRACE-style + redundancy)":      pr("contrastive_score"),
        "risk_score_v3 (locked weighted-sum, old components)":    pr("risk_score_v3"),
        "fusion_weighted_sum_old":  pr("fusion_weighted_sum_old"),
        "fusion_max_old":           pr("fusion_max_old"),
        "fusion_borda_old":         pr("fusion_borda_old"),
        "fusion_weighted_sum_new":  pr("fusion_weighted_sum_new"),
        "fusion_max_new":           pr("fusion_max_new"),
        "fusion_borda_new":         pr("fusion_borda_new"),
    }

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
            "dense_ip_old":          float(average_precision_score(yy, sub["dense_block_score_ip"].values)),
            "dense_relational_new":  float(average_precision_score(yy, sub["dense_block_score_relational"].values)),
            "hybrid_old":            float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "contrastive_new":       float(average_precision_score(yy, sub["contrastive_score"].values)),
            "fusion_weighted_sum_old": float(average_precision_score(yy, sub["fusion_weighted_sum_old"].values)),
            "fusion_borda_new":      float(average_precision_score(yy, sub["fusion_borda_new"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[prototype_v2] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
