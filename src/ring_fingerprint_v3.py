"""
ring_fingerprint_v3.py

Track J: T8b Ring Fingerprint
Computes a pure, deterministic structural fingerprint for a candidate subgraph.
No raw embeddings (h_N) are used. Only public inputs (x, edges, anomaly scores).
"""
import numpy as np

# Total D calculation:
# size (3)
# per-relation density (5)
# relation composition (5)
# degree-sequence moments (4)
# motif counts (3)
# bipartite-ness (1)
# spectral signature (3)
# node-feature aggregates: 3 * 68 = 204
# member anomaly aggregates: 3
FINGERPRINT_DIM = 231

def fingerprint(subgraph: dict) -> np.ndarray:
    """
    Computes fingerprint array for a subgraph.
    subgraph keys:
      - 'node_ids': list[int]
      - 'x': np.ndarray or torch.Tensor of shape (n, 68)
      - 'edges': list of tuples (src_idx, dst_idx, rel) where src/dst are 0-indexed in the ring
      - 'scores': np.ndarray or list of floats of shape (n,)
    """
    n_nodes = len(subgraph["node_ids"])
    n_edges = len(subgraph["edges"])
    
    # 1. Size
    ratio = n_edges / n_nodes if n_nodes > 0 else 0.0
    size_feats = [float(n_nodes), float(n_edges), float(ratio)]
    
    # 2 & 3. Per-relation density and composition
    rel_counts = [0.0] * 5
    for src, dst, rel in subgraph["edges"]:
        if 0 <= rel < 5:
            rel_counts[rel] += 1.0
            
    max_possible_edges = n_nodes * (n_nodes - 1)
    if max_possible_edges > 0:
        rel_density = [c / max_possible_edges for c in rel_counts]
    else:
        rel_density = [0.0] * 5
        
    if n_edges > 0:
        rel_comp = [c / n_edges for c in rel_counts]
    else:
        rel_comp = [0.0] * 5
        
    # 4. Degree-sequence moments
    degrees = [0.0] * n_nodes
    for src, dst, rel in subgraph["edges"]:
        degrees[src] += 1
        degrees[dst] += 1
        
    if n_nodes > 0:
        deg_sum = float(np.sum(degrees))
        deg_max = float(np.max(degrees))
        deg_std = float(np.std(degrees))
        deg_mean = float(np.mean(degrees))
    else:
        deg_sum = deg_max = deg_std = deg_mean = 0.0
        
    deg_moments = [deg_sum, deg_max, deg_std, deg_mean]
    
    # 5. Motif counts (triangle count, leaf nodes, star hub)
    # Undirected adjacency for structural motifs
    adj = {i: set() for i in range(n_nodes)}
    for src, dst, rel in subgraph["edges"]:
        adj[src].add(dst)
        adj[dst].add(src)
        
    n_leaves = sum(1 for d in degrees if d == 1)
    
    triangle_count = 0
    for i in range(n_nodes):
        for j in adj[i]:
            if j > i:
                for k in adj[j]:
                    if k > j and i in adj[k]:
                        triangle_count += 1
                        
    motif_counts = [float(triangle_count), float(n_leaves), deg_max]
    
    # 6. Bipartite-ness (2-coloring conflict ratio)
    conflict_ratio = 0.0
    if n_nodes > 0 and n_edges > 0:
        colors = {}
        conflicts = 0
        def bfs_color(start):
            nonlocal conflicts
            colors[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if v not in colors:
                        colors[v] = 1 - colors[u]
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        conflicts += 1
                        
        for i in range(n_nodes):
            if i not in colors:
                bfs_color(i)
                
        # Each undirected edge conflict is counted twice in BFS traversal of adjacency list
        conflict_ratio = (conflicts / 2) / n_edges
        
    bipartite = [float(conflict_ratio)]
    
    # 7. Spectral signature
    K = 3
    spectral_k = [0.0] * K
    if n_nodes > 0:
        # Build normalized Laplacian
        deg_matrix = np.diag(degrees)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for u in adj:
            for v in adj[u]:
                adj_matrix[u, v] = 1.0
                
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(n_nodes) - np.dot(D_inv_sqrt, np.dot(adj_matrix, D_inv_sqrt))
        
        eigenvalues = np.linalg.eigvalsh(L)
        # Sort descending or ascending? Laplacian eigenvalues are >= 0. Usually sort ascending, skip 0.
        eigenvalues = np.sort(eigenvalues)
        
        for i in range(min(n_nodes, K)):
            spectral_k[i] = float(eigenvalues[i])
            
    # 8. Node-feature aggregates
    x = subgraph.get("x")
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if x.shape[0] > 0:
        x_sum = np.sum(x, axis=0)
        x_max = np.max(x, axis=0)
        x_std = np.std(x, axis=0)
        node_feats = np.concatenate([x_sum, x_max, x_std]).tolist()
    else:
        node_feats = [0.0] * 204
        
    # 9. Member anomaly aggregates
    scores = subgraph.get("scores", [])
    if hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
    elif not isinstance(scores, np.ndarray):
        scores = np.array(scores)
        
    if len(scores) > 0:
        score_sum = float(np.sum(scores))
        score_max = float(np.max(scores))
        score_mean = float(np.mean(scores))
    else:
        score_sum = score_max = score_mean = 0.0
        
    score_feats = [score_sum, score_max, score_mean]
    
    # Combine
    out = np.array(
        size_feats + rel_density + rel_comp + deg_moments + motif_counts + bipartite + spectral_k + node_feats + score_feats,
        dtype=np.float32
    )
    
    assert len(out) == FINGERPRINT_DIM, f"Fingerprint dim {len(out)} != {FINGERPRINT_DIM}"
    return out

def _test():
    # Unit test: clique != star, iso cliques ==
    import torch
    x = torch.zeros((5, 68))
    scores = np.zeros(5)
    
    clique_edges = []
    for i in range(5):
        for j in range(i+1, 5):
            clique_edges.append((i, j, 0))
            clique_edges.append((j, i, 0))
            
    star_edges = []
    for i in range(1, 5):
        star_edges.append((0, i, 0))
        star_edges.append((i, 0, 0))
        
    clique_sg = {"node_ids": [0,1,2,3,4], "x": x, "edges": clique_edges, "scores": scores}
    star_sg = {"node_ids": [0,1,2,3,4], "x": x, "edges": star_edges, "scores": scores}
    
    fp_clique = fingerprint(clique_sg)
    fp_star = fingerprint(star_sg)
    
    assert not np.allclose(fp_clique, fp_star), "Clique and star fingerprints should differ"
    assert np.allclose(fp_clique, fingerprint(clique_sg)), "Determinism failed"
    print("Ring fingerprint tests passed!")

if __name__ == "__main__":
    _test()
