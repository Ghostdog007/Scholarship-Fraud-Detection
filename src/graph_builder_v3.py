"""
graph_builder_v3.py

Reads engineered_features_v3_nodeg.csv (63 features, no degree cols).
Builds a PyG HeteroData identity graph with 5 typed edge types.
Computes per-node degree for each edge type.
Writes:
  data/processed/identity_graph_v3.pt
  data/processed/degree_features_v3.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.config_v3 import EDGE_TYPES, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)

NODEG_CSV   = Path("data/processed/engineered_features_v3_nodeg.csv")
GRAPH_PT    = Path("data/processed/identity_graph_v3.pt")
DEGREE_CSV  = Path("data/processed/degree_features_v3.csv")
RAW_CSV     = Path("data/raw/data_for_ml_model.csv")

# Columns in raw CSV used to build each edge type
EDGE_RAW_COLS = {
    "shares_mobile":      "mobile_no",
    "shares_ip":          "ip_address",
    "shares_father_name": "father_name",
    "shares_mother_name": "mother_name",
    "shares_pincode":     "permanent_pincode",
}


def _build_edges(raw_df: pd.DataFrame, col: str) -> tuple[list[int], list[int]]:
    """Return (src, dst) index lists for all pairs sharing the same value in col."""
    valid = raw_df[[col]].copy()
    valid[col] = valid[col].astype(str).str.lower().str.strip()
    valid = valid[valid[col].notna() & (valid[col] != "") & (valid[col] != "nan")]

    src_list, dst_list = [], []
    for _, group in valid.groupby(col):
        idxs = group.index.tolist()
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                src_list.append(idxs[i])
                dst_list.append(idxs[j])
                src_list.append(idxs[j])
                dst_list.append(idxs[i])
    return src_list, dst_list


def build_graph() -> None:
    print("[graph_builder] build_graph() starting ...")

    feat_df = pd.read_csv(NODEG_CSV)
    raw_df  = pd.read_csv(RAW_CSV, low_memory=False)

    app_ids   = feat_df["application_id"].values
    feat_cols = [c for c in feat_df.columns if c != "application_id"]
    x = torch.tensor(feat_df[feat_cols].values, dtype=torch.float32)

    # Build original integer row index for raw_df aligned to feat_df order
    # raw_df rows are aligned by position to feat_df (same 15k fresh applicants)
    raw_df = raw_df.reset_index(drop=True)

    data = HeteroData()
    data["application"].x = x

    degree_dict = {f"degree_{et}": np.zeros(len(feat_df), dtype=np.float32) for et in EDGE_TYPES}

    for edge_type, raw_col in EDGE_RAW_COLS.items():
        if raw_col not in raw_df.columns:
            print(f"[graph_builder] WARNING: raw column '{raw_col}' not found -- skipping {edge_type}")
            data["application", edge_type, "application"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            continue

        src, dst = _build_edges(raw_df, raw_col)

        if len(src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)

        data["application", edge_type, "application"].edge_index = edge_index

        deg_col = f"degree_{edge_type}"
        if len(src) > 0:
            unique, counts = np.unique(src + dst, return_counts=True)
            for node_idx, cnt in zip(unique, counts):
                degree_dict[deg_col][node_idx] += cnt / 2  # each edge counted twice

        n_edges = edge_index.shape[1] // 2  # undirected count
        print(f"[graph_builder]   {edge_type}: {n_edges} undirected edges")

    GRAPH_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, GRAPH_PT)
    print(f"[graph_builder] Graph saved -> {GRAPH_PT}")
    print(f"[graph_builder] Nodes: {x.shape[0]}, Feature dim: {x.shape[1]}")

    deg_df = pd.DataFrame(degree_dict)
    deg_df.insert(0, "application_id", app_ids)
    deg_df.to_csv(DEGREE_CSV, index=False)
    print(f"[graph_builder] Degree features saved -> {DEGREE_CSV}")

    isolated = (deg_df[[f"degree_{et}" for et in EDGE_TYPES]].sum(axis=1) == 0).sum()
    print(f"[graph_builder] Isolated nodes (degree=0 across all edge types): {isolated} ({100*isolated/len(feat_df):.1f}%)")


if __name__ == "__main__":
    build_graph()
