"""
ring_candidate_v3.py

Track K: T8a Ring Candidates
Generates candidate subgraphs using per-relation connected components
and ego-expansion around top suspicious nodes.
"""
import torch
import numpy as np
from src.config_v3 import RING_MIN_SIZE, RANDOM_SEED

def _get_subgraph_edges(node_set: set, full_edges: list) -> list:
    """Extracts all edges among the given set of nodes from the full edge list."""
    # full_edges is a list of (src, dst, rel)
    # We must remap node indices to 0..len(node_set)-1 for the subgraph representation?
    # Wait, the spec for ring_fingerprint_v3 says:
    # "edges: list of tuples (src_idx, dst_idx, rel) where src/dst are 0-indexed in the ring"
    # Actually, my fingerprint implementation just uses the node indices directly to index into arrays of size N.
    # Ah, in fingerprint I did:
    # `n_nodes = len(subgraph["node_ids"])`
    # `degrees = [0.0] * n_nodes` -> this means src/dst must be 0-indexed in the subgraph!
    
    node_list = sorted(list(node_set))
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    subgraph_edges = []
    for u, v, r in full_edges:
        if u in node_set and v in node_set:
            subgraph_edges.append((node_to_idx[u], node_to_idx[v], r))
            
    return node_list, subgraph_edges

def generate_candidates(
    edge_index_list: list[torch.Tensor],
    edge_type_tensor: torch.Tensor,
    x_all: torch.Tensor,
    hybrid_scores: torch.Tensor,
    min_size: int = RING_MIN_SIZE
) -> list[dict]:
    """
    Generates candidate rings.
    """
    # 0. Build base structures
    full_edges = []
    rel_edges = {r: [] for r in range(5)}
    adj = {}
    
    if edge_index_list:
        ei = torch.cat(edge_index_list, dim=1).cpu().numpy()
        et = edge_type_tensor.cpu().numpy()
        for i in range(ei.shape[1]):
            u, v, r = int(ei[0, i]), int(ei[1, i]), int(et[i])
            full_edges.append((u, v, r))
            rel_edges[r].append((u, v))
            
            if u not in adj: adj[u] = set()
            if v not in adj: adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
            
    candidates = []
    seen_node_sets = set()
    
    def add_candidate(node_set: set):
        if len(node_set) < min_size:
            return
        frozen = frozenset(node_set)
        if frozen in seen_node_sets:
            return
        seen_node_sets.add(frozen)
        
        node_list, sg_edges = _get_subgraph_edges(node_set, full_edges)
        
        sg_x = x_all[node_list]
        sg_scores = hybrid_scores[node_list]
        
        candidates.append({
            "node_ids": node_list,
            "x": sg_x,
            "edges": sg_edges,
            "scores": sg_scores
        })
        
    # 1. Per-relation connected components
    for r in range(5):
        if not rel_edges[r]:
            continue
            
        rel_adj = {}
        for u, v in rel_edges[r]:
            if u not in rel_adj: rel_adj[u] = set()
            if v not in rel_adj: rel_adj[v] = set()
            rel_adj[u].add(v)
            rel_adj[v].add(u)
            
        visited = set()
        for node in rel_adj:
            if node not in visited:
                comp = set()
                queue = [node]
                while queue:
                    curr = queue.pop(0)
                    if curr not in comp:
                        comp.add(curr)
                        visited.add(curr)
                        queue.extend(rel_adj.get(curr, []))
                add_candidate(comp)
            
    # 2. Ego expansion around suspicious nodes
    if len(hybrid_scores) > 0:
        # Top-N (e.g. 500)
        n_top = min(500, len(hybrid_scores))
        scores_np = hybrid_scores.cpu().numpy()
        top_indices = np.argsort(scores_np)[::-1][:n_top]
        
        for u in top_indices:
            u = int(u)
            node_set = {u}
            if u in adj:
                node_set.update(adj[u])
            add_candidate(node_set)
            
    # Sort for determinism
    candidates.sort(key=lambda c: sorted(c["node_ids"]))
    
    return candidates
