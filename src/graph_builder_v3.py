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


# ── Step 4 (V4-Scale): SQL-sourced, hub-capped edge construction ─────────────
# Groups come from Postgres identity_keys (normalisation proven identical to
# _build_edges in Gate 2). Capping policy:
#   * group size <= k_cap  -> full clique (unchanged signal)
#   * group size >  k_cap  -> star to the group's first member by node index
#                             (deterministic hub; O(k) edges instead of O(k^2))
#   * group size >  ceiling -> edges skipped entirely (mega-groups are shared
#                             infrastructure noise; count features keep the
#                             size signal). The ceiling is derived from the
#                             observed group-size distribution at the given
#                             percentile — never a hand-picked domain number
#                             (hard stop 1).
# BOTH CAPS DEFAULT OFF (None): K_CAP is open decision #1 (lead-owned,
# needs 3.5M profiling). With caps off this reproduces _build_edges exactly.

IDENTITY_COLS = {
    "shares_mobile":      "mobile_no",
    "shares_ip":          "ip_address",
    "shares_father_name": "father_name_norm",
    "shares_mother_name": "mother_name_norm",
    "shares_pincode":     "pincode",
}


def _edges_from_groups(groups: list[tuple[str, list[str]]],
                       node_index: dict[str, int],
                       k_cap: int | None = None,
                       ceiling: int | None = None) -> tuple[list[int], list[int], dict]:
    """Build (src, dst) bidirectional index lists from shared-value groups."""
    src_list: list[int] = []
    dst_list: list[int] = []
    stats = {"clique_groups": 0, "star_groups": 0, "ceiling_skipped": 0}
    for _value, members in groups:
        idxs = sorted(node_index[m] for m in members if m in node_index)
        k = len(idxs)
        if k < 2:
            continue
        if ceiling is not None and k > ceiling:
            stats["ceiling_skipped"] += 1
            continue
        if k_cap is not None and k > k_cap:
            hub = idxs[0]
            stats["star_groups"] += 1
            for j in idxs[1:]:
                src_list.extend((hub, j))
                dst_list.extend((j, hub))
        else:
            stats["clique_groups"] += 1
            for a in range(k):
                for b in range(a + 1, k):
                    src_list.extend((idxs[a], idxs[b]))
                    dst_list.extend((idxs[b], idxs[a]))
    return src_list, dst_list, stats


def derive_group_ceiling(sizes: list[int], percentile: float = 99.9) -> int:
    """Statistical ceiling from the observed group-size distribution
    (hard stop 1 — derived, not hand-picked)."""
    if not sizes:
        return 0
    return int(np.percentile(np.array(sizes, dtype=np.float64), percentile))


def build_graph_pg(out_graph: Path | None = None, out_degree: Path | None = None,
                   k_cap: int | None = None, ceiling: int | None = None) -> dict:
    """Scale-path build_graph (step 4): identical node features/order, edges
    from Postgres shared-value groups with optional hub-capping. Writes to
    SEPARATE files until cut-over (hard stop 13). Returns per-relation stats."""
    print(f"[graph_builder] build_graph_pg() starting (k_cap={k_cap}, ceiling={ceiling}) ...")
    from src.db.features import edge_groups

    feat_df = pd.read_csv(NODEG_CSV)
    app_ids = feat_df["application_id"].astype(str).tolist()
    node_index = {a: i for i, a in enumerate(app_ids)}
    feat_cols = [c for c in feat_df.columns if c != "application_id"]
    x = torch.tensor(feat_df[feat_cols].values, dtype=torch.float32)

    data = HeteroData()
    data["application"].x = x
    degree_dict = {f"degree_{et}": np.zeros(len(feat_df), dtype=np.float32) for et in EDGE_TYPES}
    all_stats: dict = {}

    for edge_type, id_col in IDENTITY_COLS.items():
        groups = edge_groups(id_col)
        src, dst, stats = _edges_from_groups(groups, node_index, k_cap=k_cap, ceiling=ceiling)
        edge_index = (torch.tensor([src, dst], dtype=torch.long)
                      if src else torch.zeros((2, 0), dtype=torch.long))
        data["application", edge_type, "application"].edge_index = edge_index

        deg_col = f"degree_{edge_type}"
        if src:
            unique, counts = np.unique(src + dst, return_counts=True)
            for node_idx, cnt in zip(unique, counts):
                degree_dict[deg_col][node_idx] += cnt / 2
        stats["n_undirected_edges"] = edge_index.shape[1] // 2
        stats["n_groups"] = len(groups)
        all_stats[edge_type] = stats
        print(f"[graph_builder]   {edge_type}: {stats['n_undirected_edges']} undirected edges "
              f"({stats['clique_groups']} clique / {stats['star_groups']} star / "
              f"{stats['ceiling_skipped']} skipped groups)")

    out_graph = out_graph or Path("data/processed/identity_graph_v3_pg.pt")
    out_degree = out_degree or Path("data/processed/degree_features_v3_pg.csv")
    out_graph.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_graph)
    deg_df = pd.DataFrame(degree_dict)
    deg_df.insert(0, "application_id", feat_df["application_id"].values)
    deg_df.to_csv(out_degree, index=False)
    print(f"[graph_builder] build_graph_pg() done -> {out_graph}, {out_degree}")
    return all_stats


if __name__ == "__main__":
    build_graph()
