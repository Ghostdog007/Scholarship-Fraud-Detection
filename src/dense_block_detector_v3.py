"""
dense_block_detector_v3.py

T1 Dense-Block Detector
Uses Charikar greedy peeling with camouflage-resistant column weighting to assign
a dense-block suspicion score to every node, per relation.
Deterministic (stable tie-break by node id).
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.config_v3 import (
    DENSE_BLOCK_ENABLED, DENSE_BLOCK_RELATIONS, DENSE_BLOCK_KCORE_PREFILTER,
    DENSE_BLOCK_CAMOUFLAGE_C
)

OUT_CSV = Path("outputs/dense_block_scores_v3.csv")


def _k_core_prune(adj: dict, k: int = 2) -> set:
    """Returns the set of nodes in the k-core of the graph."""
    degrees = {u: len(nbrs) for u, nbrs in adj.items()}
    nodes = set(adj.keys())
    
    while True:
        to_remove = [u for u in nodes if degrees[u] < k]
        if not to_remove:
            break
        for u in to_remove:
            nodes.remove(u)
            for v in adj[u]:
                if v in nodes:
                    degrees[v] -= 1
                    
    return nodes


def _charikar_peeling(adj: dict, nodes: set, node_w: dict) -> dict:
    """
    Greedy peeling algorithm.
    Returns a dict mapping node -> density of the densest subgraph it belonged to.
    """
    if not nodes:
        return {}
        
    # Active graph state
    active = set(nodes)
    
    # Precompute edge weights
    # W(u, v) = w[u] * w[v]
    # Keep track of weighted degree in active graph
    active_w_deg = {u: 0.0 for u in active}
    active_edges = 0.0
    
    for u in active:
        for v in adj[u]:
            if v in active and u < v:
                ew = node_w[u] * node_w[v]
                active_w_deg[u] += ew
                active_w_deg[v] += ew
                active_edges += ew
                
    # Record peeling sequence
    peel_seq = []
    densities = []
    
    while active:
        density = active_edges / len(active)
        densities.append(density)
        
        # Find node with min weighted degree. Tie-break by node id for determinism.
        min_u = None
        min_deg = float('inf')
        for u in active:
            d = active_w_deg[u]
            if d < min_deg or (d == min_deg and (min_u is None or u < min_u)):
                min_deg = d
                min_u = u
                
        peel_seq.append(min_u)
        active.remove(min_u)
        
        for v in adj[min_u]:
            if v in active:
                ew = node_w[min_u] * node_w[v]
                active_w_deg[v] -= ew
                active_edges -= ew
                
    # Densest block a node belongs to = max density over the subgraphs that still
    # contained it. A node peeled at step i was present in subgraphs 0..i, so its
    # score is the PREFIX max of densities[0..i]. (A backward/suffix max would
    # wrongly hand peripheral nodes the dense core's density and starve the core,
    # inverting the signal — verified on triangle+pendant.)
    max_density = 0.0
    node_densities = {}
    for i in range(len(peel_seq)):
        u = peel_seq[i]
        if densities[i] > max_density:
            max_density = densities[i]
        node_densities[u] = max_density

    return node_densities


def dense_block_scores(
    edge_index_list: list[torch.Tensor],
    edge_type_tensor: torch.Tensor,
    n_nodes: int,
    app_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Returns DataFrame: application_id, dense_block_score_<relation_name> for each
    active DENSE_BLOCK_RELATIONS (score = camouflage-weighted density of the
    densest block the node belongs to, min-max normalised to [0,1] per relation),
    PLUS dense_block_score_relational — DENSE_BLOCK_RELATION_WEIGHTS applied to
    each relation's own normalised score, then max-combined across relations.
    Max (not sum/average) so whichever relation is actually elevated for a given
    node dominates, rather than being diluted by the other two; weighted (not
    equal) so IP — the dominant real fraud vector for this population — can't be
    outranked by ordinary, non-fraud density in mobile/pincode elsewhere in the
    population (see config_v3 comment + README changelog 2026-07-22).
    """
    from src.config_v3 import EDGE_TYPES, DENSE_BLOCK_RELATION_WEIGHTS

    col_names = []
    scores = {}
    for r in DENSE_BLOCK_RELATIONS:
        rel_name = EDGE_TYPES[r].replace("shares_", "")
        col = f"dense_block_score_{rel_name}"
        col_names.append(col)
        scores[col] = np.zeros(n_nodes, dtype=np.float32)
    
    for r in DENSE_BLOCK_RELATIONS:
        rel_name = EDGE_TYPES[r].replace("shares_", "")
        col = f"dense_block_score_{rel_name}"
        
        ei = edge_index_list[r]
        if ei.shape[1] == 0:
            continue
            
        # Build undirected adjacency
        adj = {}
        ei_np = ei.cpu().numpy()
        for i in range(ei_np.shape[1]):
            u, v = int(ei_np[0, i]), int(ei_np[1, i])
            if u != v:
                if u not in adj: adj[u] = set()
                if v not in adj: adj[v] = set()
                adj[u].add(v)
                adj[v].add(u)
                
        if not adj:
            continue
            
        # Global degree for camouflage weighting
        global_deg = {u: len(nbrs) for u, nbrs in adj.items()}
        node_w = {u: 1.0 / np.log(global_deg[u] + DENSE_BLOCK_CAMOUFLAGE_C) for u in adj}
        
        # Pre-filter (k-core)
        if DENSE_BLOCK_KCORE_PREFILTER:
            nodes = _k_core_prune(adj, k=2)
        else:
            nodes = set(adj.keys())
            
        # Peel
        node_densities = _charikar_peeling(adj, nodes, node_w)
        
        # Min-max normalize per relation
        if node_densities:
            d_vals = list(node_densities.values())
            min_d, max_d = min(d_vals), max(d_vals)
            
            arr = scores[col]
            for u, d in node_densities.items():
                if max_d > min_d:
                    arr[u] = (d - min_d) / (max_d - min_d)
                else:
                    arr[u] = 1.0 if d > 0 else 0.0
                    
    df = pd.DataFrame({"application_id": app_ids})
    for col in col_names:
        df[col] = scores[col]

    weighted = [
        DENSE_BLOCK_RELATION_WEIGHTS.get(r, 1.0) * scores[f"dense_block_score_{EDGE_TYPES[r].replace('shares_', '')}"]
        for r in DENSE_BLOCK_RELATIONS
    ]
    df["dense_block_score_relational"] = np.maximum.reduce(weighted) if weighted else np.zeros(n_nodes, dtype=np.float32)

    return df


def run_dense_block():
    print("[dense_block] run_dense_block() starting ...")
    if not DENSE_BLOCK_ENABLED:
        print("[dense_block] DENSE_BLOCK_ENABLED is False. Exiting.")
        return
        
    from src.hybrid_graphmcm_v3 import _build_edge_index_and_types
    import json
    
    final_csv = Path("data/processed/engineered_features_v3.csv")
    df = pd.read_csv(final_csv)
    app_ids = df["application_id"].values
    n_nodes = len(app_ids)
    
    data = torch.load(Path("data/processed/identity_graph_v3.pt"), weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, torch.device("cpu"))
    
    out_df = dense_block_scores(edge_index_list, edge_type_tensor, n_nodes, app_ids)
    
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[dense_block] Saved {len(out_df)} scores -> {OUT_CSV}")


if __name__ == "__main__":
    run_dense_block()
