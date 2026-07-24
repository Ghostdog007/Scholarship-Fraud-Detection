"""
Prototype: CARE-GNN-style similarity-aware neighbor selector, layered on top
of the Deep SAD center-distance objective (the strongest relational signal
found this session: 0.201 overall / 0.093 mobile-ring on stress_testing_1).

Rationale (CARE-GNN, Dou et al., CIKM 2020): fraud nodes camouflage by
connecting to normal-looking neighbors, diluting the aggregated signal.
CARE-GNN filters WHICH neighbors get aggregated per relation, via a
per-relation similarity threshold p_r that self-adjusts using label
feedback each epoch (their "RL" module -- in practice a simple feedback
rule, reproduced as such here, not full policy-gradient RL).

This prototype is the "self-learning with examples" component the user
asked for, with two properties requested explicitly:
  - it learns what to distrust using LOE / topology exposure (the only
    labeled examples available at scale, same convention as every other
    prototype this session -- no real confirmed fraud used)
  - DORMANCY: if a relation doesn't see enough exposure examples in a
    given epoch to trust the feedback signal, its threshold is frozen at
    a fully-permissive value (admits all neighbors, i.e. behaves exactly
    like plain RGCN aggregation for that relation) instead of updating on
    noise. Only relations that see >= MIN_EXAMPLES exposure nodes ever
    "activate" and start filtering.

Architecture:
  FilteredRGCNEncoder : 2-layer RGCN, but edges are filtered per relation
    by cosine similarity of the CURRENT layer's endpoint embeddings against
    a per-relation threshold p_r (a non-gradient buffer, not a learned
    parameter -- updated by the feedback rule below, not backprop).
  Training objective    : Deep SAD center-pull / exposure-push (reused
    from prototype_deepsad_prototypes.py, the best objective found so far).
  Threshold update      : once per epoch, per relation:
    - if exposure nodes touching that relation this epoch < MIN_EXAMPLES:
        stay dormant (p_r frozen at DORMANT_VALUE = -1.0, admits everyone)
    - else:
        activate (if not already), compute separation reward = mean
        center-distance of that relation's exposure nodes minus mean
        center-distance of a real background sample; if reward improved
        vs last epoch, loosen p_r (admit more neighbors); if it got worse,
        tighten p_r (admit fewer, trust only closest neighbors)

Output score: care_filtered_center_dist (higher = more anomalous)

Writes: outputs/stress_testing_1_care_stats.json
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
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_afterfix_full_scores.csv")
DEEPSAD_CSV      = Path(f"outputs/{NAME}_deepsad_full_scores.csv")
NODEG_CSV        = Path("data/processed/engineered_features_v3_nodeg.csv")
SCHEMA_JSON      = Path("data/processed/v3_feature_schema.json")
TOPO_PT          = Path("data/processed/synthetic_exposure_graph_v3.pt")

OUT_STATS = Path(f"outputs/{NAME}_care_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EMB_DIM = 64, 32
N_EPOCHS = 150
ETA = 1.0
MIN_EXAMPLES = 20          # exposure nodes touching a relation, per epoch, to trust feedback
DORMANT_VALUE = -1.0       # cosine similarity floor -> admits every edge (no filtering)
ACTIVE_INIT = 0.0          # threshold a relation starts at on first activation
STEP = 0.03                # per-epoch threshold adjustment size
P_MIN, P_MAX = -0.9, 0.9   # clamp range once active


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


def filter_edges(x: torch.Tensor, edge_index_list: list[torch.Tensor], p_r: torch.Tensor):
    """Keep only edges whose endpoint similarity (under current x) clears
    that relation's threshold. p_r is a plain tensor, no gradient."""
    filtered_ei, filtered_et = [], []
    for r, ei in enumerate(edge_index_list):
        if ei.numel() == 0:
            continue
        src, dst = ei[0], ei[1]
        with torch.no_grad():
            sim = F.cosine_similarity(x[src], x[dst], dim=1)
            mask = sim >= p_r[r]
        kept = ei[:, mask]
        if kept.numel() == 0:
            continue
        filtered_ei.append(kept)
        filtered_et.append(torch.full((kept.shape[1],), r, dtype=torch.long, device=x.device))
    if filtered_ei:
        return torch.cat(filtered_ei, dim=1), torch.cat(filtered_et)
    return torch.zeros((2, 0), dtype=torch.long, device=x.device), torch.zeros((0,), dtype=torch.long, device=x.device)


class FilteredRGCNEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, n_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, HIDDEN, n_relations, root_weight=False)
        self.conv2 = RGCNConv(HIDDEN, EMB_DIM, n_relations, root_weight=False)

    def forward(self, x, edge_index_list, p_r):
        ei1, et1 = filter_edges(x, edge_index_list, p_r)
        h1 = F.relu(self.conv1(x, ei1, et1))
        ei2, et2 = filter_edges(h1, edge_index_list, p_r)
        h2 = self.conv2(h1, ei2, et2)
        return h2, ei1  # ei1 returned for diagnostics only


def main() -> None:
    print(f"[care] Device: {DEVICE}")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    prev = pd.read_csv(FULL_SCORES_CSV)
    prev["application_id"] = prev["application_id"].astype(str)
    cohort_ids = set(prev["application_id"])
    deepsad_ref = pd.read_csv(DEEPSAD_CSV)[["application_id", "center_dist_score"]]
    deepsad_ref["application_id"] = deepsad_ref["application_id"].astype(str)

    x44 = build_full_44dim_x(data, nodeorder).to(DEVICE)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    n_real = x44.shape[0]
    n_relations = len(EDGE_TYPES)

    topo_pack_raw = torch.load(TOPO_PT, weights_only=False)
    topo_pack = {k: v.to(DEVICE) for k, v in topo_pack_raw.items()}
    topo_edge_index_list = []
    for r in range(n_relations):
        m = topo_pack["edge_type"] == r
        topo_edge_index_list.append(topo_pack["edge_index"][:, m])
    # Per-relation exposure node membership (which synth nodes touch relation r at all)
    relation_members = []
    for r in range(n_relations):
        ei = topo_edge_index_list[r]
        members = torch.unique(ei.reshape(-1)) if ei.numel() > 0 else torch.zeros(0, dtype=torch.long, device=DEVICE)
        relation_members.append(members)
        print(f"[care] relation {EDGE_TYPES[r]}: {members.numel()} exposure nodes touch it")

    encoder = FilteredRGCNEncoder(x44.shape[1], n_relations).to(DEVICE)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    p_r = torch.full((n_relations,), DORMANT_VALUE, device=DEVICE)   # start fully dormant
    activated = [False] * n_relations
    last_reward = [None] * n_relations

    with torch.no_grad():
        h0, _ = encoder(x44, edge_index_list, p_r)
        norms0 = h0.norm(dim=1)
        cutoff0 = torch.quantile(norms0, CENTROID_CLEAN_PERCENTILE / 100.0)
        center = h0[norms0 <= cutoff0].mean(dim=0).detach()

    print(f"[care] training {N_EPOCHS} epochs (filtered RGCN + Deep SAD objective + self-adjusting neighbor filter) ...")
    encoder.train()
    for epoch in range(N_EPOCHS):
        h_real, _ = encoder(x44, edge_index_list, p_r)
        h_synth, _ = encoder(topo_pack["x"], topo_edge_index_list, p_r)

        dist_real = ((h_real - center) ** 2).sum(dim=1)
        loss_normal = dist_real.mean()
        dist_synth = ((h_synth - center) ** 2).sum(dim=1)
        loss_anomaly = (ETA / (dist_synth + 1e-6)).mean()
        loss = loss_normal + loss_anomaly

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                norms = h_real.norm(dim=1)
                cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
                center = h_real[norms <= cutoff].mean(dim=0).detach()

        # --- Self-adjusting per-relation neighbor filter (non-gradient) ---
        with torch.no_grad():
            bg_idx = torch.randint(0, n_real, (512,), device=DEVICE)
            bg_dist_mean = ((h_real[bg_idx] - center) ** 2).sum(dim=1).sqrt().mean().item()
            for r in range(n_relations):
                members = relation_members[r]
                if members.numel() < MIN_EXAMPLES:
                    continue  # not enough signal this epoch -- stays dormant
                if not activated[r]:
                    p_r[r] = ACTIVE_INIT
                    activated[r] = True
                    print(f"  [epoch {epoch+1}] relation '{EDGE_TYPES[r]}' ACTIVATED "
                          f"({members.numel()} exposure nodes >= {MIN_EXAMPLES})")
                exp_dist_mean = ((h_synth[members] - center) ** 2).sum(dim=1).sqrt().mean().item()
                reward = exp_dist_mean - bg_dist_mean  # want exposure far, background close
                if last_reward[r] is not None:
                    if reward > last_reward[r]:
                        p_r[r] = max(P_MIN, p_r[r].item() - STEP)   # improving -> admit more
                    else:
                        p_r[r] = min(P_MAX, p_r[r].item() + STEP)   # worse -> filter harder
                last_reward[r] = reward

        if (epoch + 1) % 10 == 0:
            p_str = ", ".join(f"{EDGE_TYPES[r]}={p_r[r].item():.2f}" for r in range(n_relations))
            print(f"  epoch {epoch+1}/{N_EPOCHS} loss={loss.item():.4f} | p_r: {p_str}")

    print(f"[care] Final thresholds: " + ", ".join(f"{EDGE_TYPES[r]}={p_r[r].item():.3f} (active={activated[r]})" for r in range(n_relations)))

    print("[care] Scoring all real nodes ...")
    encoder.eval()
    with torch.no_grad():
        h_real_final, _ = encoder(x44, edge_index_list, p_r)
        norms = h_real_final.norm(dim=1)
        cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
        center_final = h_real_final[norms <= cutoff].mean(dim=0)
        care_score = ((h_real_final - center_final) ** 2).sum(dim=1).sqrt()

    df_scores = pd.DataFrame({
        "application_id": app_ids_merged,
        "care_filtered_center_dist": care_score.cpu().numpy(),
    })
    df_scores["application_id"] = df_scores["application_id"].astype(str)
    df_scores = df_scores[df_scores["application_id"].isin(cohort_ids)]

    df = prev.merge(df_scores, on="application_id", how="left").merge(deepsad_ref, on="application_id", how="left")
    out_csv = Path(f"outputs/{NAME}_care_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[care] Wrote {out_csv}")

    y = df["is_fraud"].astype(int).values
    stats: dict = {
        "final_thresholds": {EDGE_TYPES[r]: {"p_r": float(p_r[r].item()), "activated": activated[r]} for r in range(n_relations)},
        "overall": {
            "hybrid_reconstruction (prod baseline, after LOE fix + root_weight fix)": float(average_precision_score(y, df["hybrid_anomaly_score"].values)),
            "center_dist_score (Deep SAD, no neighbor filtering)": float(average_precision_score(y, df["center_dist_score"].values)),
            "care_filtered_center_dist (new, Deep SAD + self-adjusting neighbor filter)": float(average_precision_score(y, df["care_filtered_center_dist"].values)),
        },
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
            "hybrid_reconstruction": float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "center_dist_score": float(average_precision_score(yy, sub["center_dist_score"].values)),
            "care_filtered_center_dist": float(average_precision_score(yy, sub["care_filtered_center_dist"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[care] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
