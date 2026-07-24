"""
Prototype: RGCNConv(root_weight=False) for Hybrid GraphMCM's graph encoder.

Rationale: RGCNConv defaults to root_weight=True, so h_n (the graph-context
embedding fed to the feature predictor alongside masked_x) contains a direct
learned self-transform of the node's own TRUE unmasked features, in addition
to neighbor aggregation. That leaks self-signal around the intentional
masking in _apply_masks(), undermining the "predict me from my neighborhood"
MCM contract for every connected node (isolated nodes are unaffected --
they already get h_n overridden by the trainable isolated_embedding).
Disabling root_weight forces h_n to be pure multi-relation neighbor
aggregation.

This is the closest-to-production ablation run this session: it reuses the
REAL HybridGraphMCM class, REAL two-stage LOE training loop (Stage 1 warm
start + Stage 2 persistent LOE, current fixed-margin formula), and REAL
compute_score_frame -- the only change is root_weight=False on both RGCNConv
layers, patched in after model construction. Trained on stress_testing_1's
staged artifacts only; production checkpoint/data are untouched.

Writes: outputs/stress_testing_1_rootweightoff_stats.json
"""
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score

from src.config_v3 import (
    EDGE_TYPES, RANDOM_SEED, N_FEATURES, GRAPH_HIDDEN, GRAPH_EMB_DIM,
    LR, EPOCHS_STAGE1, EPOCHS_STAGE2, LAMBDA_EXPOSURE, LAMBDA_EDGE,
    LOE_STAGE2_WEIGHT,
)
from src.hybrid_graphmcm_v3 import (
    HybridGraphMCM, _build_edge_index_and_types, _compute_isolated_mask,
    _derive_loe_margin, _loe_loss, _get_synth_h_topology,
    _feature_pred_loss, _edge_pred_loss, compute_score_frame, DEVICE,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NAME = "stress_testing_1"
STAGED_GRAPH     = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_afterfix_full_scores.csv")
NODEG_CSV        = Path("data/processed/engineered_features_v3_nodeg.csv")
SCHEMA_JSON      = Path("data/processed/v3_feature_schema.json")
TOPO_PT          = Path("data/processed/synthetic_exposure_graph_v3.pt")

OUT_STATS = Path(f"outputs/{NAME}_rootweightoff_stats.json")


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


def main() -> None:
    print(f"[rootweightoff] Device: {DEVICE}")
    print(f"[rootweightoff] Loading staged artifacts for '{NAME}' ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids = nodeorder["application_id"].astype(str).values
    prev = pd.read_csv(FULL_SCORES_CSV)
    prev["application_id"] = prev["application_id"].astype(str)
    cohort_ids = set(prev["application_id"])

    x_all = build_full_44dim_x(data, nodeorder).to(DEVICE)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], DEVICE)
    print(f"[rootweightoff] Isolated nodes: {isolated_mask.sum().item()} / {x_all.shape[0]}")

    topo_pack = torch.load(TOPO_PT, weights_only=False)

    model = HybridGraphMCM().to(DEVICE)

    # --- The one architectural change: strip the self/root transform out of
    # both RGCN layers so h_n is pure multi-relation neighbor aggregation.
    n_edge_types = len(EDGE_TYPES)
    model.encoder.conv1 = RGCNConv(N_FEATURES, GRAPH_HIDDEN, num_relations=n_edge_types,
                                    aggr="add", root_weight=False).to(DEVICE)
    model.encoder.conv2 = RGCNConv(GRAPH_HIDDEN, GRAPH_EMB_DIM, num_relations=n_edge_types,
                                    aggr="add", root_weight=False).to(DEVICE)
    print("[rootweightoff] Patched RGCNConv layers with root_weight=False")

    model.init_centroid(x_all, edge_index_list, edge_type_tensor, isolated_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[rootweightoff] Stage 1: {EPOCHS_STAGE1} epochs (graph LOE warm-start) ...")
    model.train()
    for epoch in range(EPOCHS_STAGE1):
        lam_t = LAMBDA_EXPOSURE * (1.0 - epoch / EPOCHS_STAGE1)
        optimizer.zero_grad()
        _, _, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)
        svdd_loss = torch.norm(h_n - model.centroid.unsqueeze(0), dim=1).mean()
        h_synth = _get_synth_h_topology(model, topo_pack, DEVICE)
        margin = _derive_loe_margin(h_n, model.centroid)
        loe = _loe_loss(h_synth, model.centroid, lam_t, margin)
        loss = svdd_loss + loe
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(1, EPOCHS_STAGE1 // 5) == 0:
            print(f"  S1 epoch {epoch+1}/{EPOCHS_STAGE1} | svdd={svdd_loss.item():.4f} loe={loe.item():.4f} "
                  f"lam={lam_t:.3f} margin={margin.item():.3f}")

    print(f"[rootweightoff] Stage 2: {EPOCHS_STAGE2} epochs (free joint reconstruction + persistent LOE) ...")
    model.train()
    for epoch in range(EPOCHS_STAGE2):
        optimizer.zero_grad()
        pred_x, edge_prob, h_n, _ = model(x_all, edge_index_list, edge_type_tensor, isolated_mask)
        feat_loss = _feature_pred_loss(pred_x, x_all)
        edge_loss = _edge_pred_loss(edge_prob, edge_index_list, x_all.shape[0], DEVICE)
        h_synth_s2 = _get_synth_h_topology(model, topo_pack, DEVICE)
        margin_s2 = _derive_loe_margin(h_n, model.centroid)
        loe_s2 = _loe_loss(h_synth_s2, model.centroid, LOE_STAGE2_WEIGHT, margin_s2)
        loss = feat_loss + LAMBDA_EDGE * edge_loss + loe_s2
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(1, EPOCHS_STAGE2 // 5) == 0:
            print(f"  S2 epoch {epoch+1}/{EPOCHS_STAGE2} | feat={feat_loss.item():.4f} "
                  f"edge={edge_loss.item():.4f} loe={loe_s2.item():.4f} margin={margin_s2.item():.3f}")

    print("[rootweightoff] Scoring all nodes ...")
    schema_features = json.loads(SCHEMA_JSON.read_text())["features"]
    out_df = compute_score_frame(model, x_all, edge_index_list, edge_type_tensor, isolated_mask,
                                  app_ids, schema_features)
    out_df["application_id"] = out_df["application_id"].astype(str)
    out_df = out_df[out_df["application_id"].isin(cohort_ids)]
    out_df = out_df.rename(columns={
        "hybrid_anomaly_score": "hybrid_anomaly_score_rootweightoff",
        "feature_pred_error": "feature_pred_error_rootweightoff",
    })
    total_degree = torch.zeros(x_all.shape[0], device=DEVICE)
    for ei in edge_index_list:
        if ei.numel() > 0:
            idx, counts = torch.unique(ei.reshape(-1), return_counts=True)
            total_degree[idx] += counts.float() / 2.0
    # total_degree is indexed by node order (== app_ids order)
    deg_df = pd.DataFrame({"application_id": app_ids.astype(str), "_total_degree": total_degree.cpu().numpy()})
    out_df = out_df.merge(deg_df, on="application_id", how="left")

    keep_cols = ["application_id", "hybrid_anomaly_score_rootweightoff", "feature_pred_error_rootweightoff", "_total_degree"]
    df = prev.merge(out_df[keep_cols], on="application_id", how="left")

    out_csv = Path(f"outputs/{NAME}_rootweightoff_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[rootweightoff] Wrote {out_csv}")

    y = df["is_fraud"].astype(int).values
    stats: dict = {"overall": {
        "hybrid_reconstruction (current prod, root_weight=True, after LOE fix)": float(average_precision_score(y, df["hybrid_anomaly_score"].values)),
        "hybrid_reconstruction (root_weight=False, this prototype)": float(average_precision_score(y, df["hybrid_anomaly_score_rootweightoff"].values)),
        "feature_pred_error (root_weight=False, isolates reconstruction-only signal)": float(average_precision_score(y, df["feature_pred_error_rootweightoff"].values)),
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
            "hybrid_reconstruction_prod": float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "hybrid_reconstruction_rootweightoff": float(average_precision_score(yy, sub["hybrid_anomaly_score_rootweightoff"].values)),
        }

    # Per-degree breakdown (root_weight=False costs low-degree nodes their
    # self-transform crutch -- explicitly worth checking, not just the aggregate).
    total_degree = df["_total_degree"].fillna(0)
    bins = pd.cut(total_degree, bins=[-0.1, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", "6+"])
    stats["per_degree_bucket"] = {}
    for b in bins.cat.categories:
        m = (bins == b).values
        if m.sum() < 20:
            continue
        yy = y[m]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        stats["per_degree_bucket"][str(b)] = {
            "n": int(m.sum()),
            "hybrid_reconstruction_prod": float(average_precision_score(yy, df.loc[m, "hybrid_anomaly_score"].values)),
            "hybrid_reconstruction_rootweightoff": float(average_precision_score(yy, df.loc[m, "hybrid_anomaly_score_rootweightoff"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[rootweightoff] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
