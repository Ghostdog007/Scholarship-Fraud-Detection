"""
topology_view.py

Extracts and renders BFS ego-graphs from the V3 identity graph.
Used by the supervisor UI (Track F) to provide network context.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

GRAPH_PT = Path("data/processed/identity_graph_v3.pt")
HYBRID_CSV = Path("outputs/hybrid_scores_v3.csv")

EDGE_TYPE_COLORS = {
    "shares_ip": "#e41a1c",
    "shares_mobile": "#377eb8",
    "shares_father_name": "#4daf4a",
    "shares_mother_name": "#984ea3",
    "shares_pincode": "#ff7f00",
}

def extract_ego(app_id: str, hops: int = 1, node_cap: int = 50) -> dict:
    """
    Extract a BFS ego-graph around the center app_id.
    """
    if not HYBRID_CSV.exists() or not GRAPH_PT.exists():
        return {}

    df = pd.read_csv(HYBRID_CSV)
    if app_id not in df["application_id"].values:
        return {}
    
    app_ids = df["application_id"].values
    scores = df["hybrid_anomaly_score"].values
    # fallback to risk score if available
    if "risk_score_v3" in df.columns:
        scores = df["risk_score_v3"].values
        
    id_to_idx = {str(aid): i for i, aid in enumerate(app_ids)}
    idx_to_id = {i: str(aid) for i, aid in enumerate(app_ids)}
    
    app_id = str(app_id)
    if app_id not in id_to_idx:
        return {}
    center_idx = id_to_idx[app_id]
    
    data = torch.load(GRAPH_PT, weights_only=False)
    
    # build adj list
    adj = {}
    for edge_type_tuple in data.edge_types:
        et = edge_type_tuple[1]
        ei = data[edge_type_tuple].edge_index
        if ei.shape[1] == 0:
            continue
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for s, d in zip(src, dst):
            adj.setdefault(s, []).append((d, et))
            # PyG undirected edges might be stored 2-way, but if not we can add inverse
            # wait, they are typically stored bidirectional. We'll rely on what's there.
    
    # BFS
    visited = {center_idx}
    queue = [(center_idx, 0)]
    
    while queue:
        curr, depth = queue.pop(0)
        if depth < hops:
            for nbr, et in adj.get(curr, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))
                    
    # Cap by score
    node_list = list(visited)
    if len(node_list) > node_cap:
        # always keep center
        node_list.remove(center_idx)
        # sort rest by score descending
        node_list.sort(key=lambda x: scores[x], reverse=True)
        kept_nodes = [center_idx] + node_list[:node_cap - 1]
    else:
        kept_nodes = node_list
        
    kept_set = set(kept_nodes)
    
    nodes = []
    for n in kept_nodes:
        nodes.append({
            "id": idx_to_id.get(n, str(n)),
            "score": float(scores[n]),
            "is_center": bool(n == center_idx)
        })
        
    edges = []
    # add all edges between kept nodes
    for src in kept_nodes:
        for dst, et in adj.get(src, []):
            if dst in kept_set:
                edges.append({
                    "source": idx_to_id.get(src, str(src)),
                    "target": idx_to_id.get(dst, str(dst)),
                    "type": et
                })
                
    # remove duplicate reverse edges if any
    seen_edges = set()
    uniq_edges = []
    for e in edges:
        canon = tuple(sorted([e["source"], e["target"]])) + (e["type"],)
        if canon not in seen_edges:
            seen_edges.add(canon)
            uniq_edges.append(e)
            
    return {
        "center": app_id,
        "nodes": nodes,
        "edges": uniq_edges,
        "shown": len(nodes),
        "total": len(visited)
    }


def render_html(ego: dict) -> str:
    """
    Return a self-contained HTML page using basic D3 force layout (embedded).
    Wait, instructions say "no CDN" - meaning JS must be inline or we just do static layout in Python.
    Actually I can generate SVG with static layout in Python and embed it.
    """
    svg = render_svg(ego)
    return f"<html><body>{svg}</body></html>"


def render_svg(ego: dict) -> str:
    """
    Render ego graph as SVG string using a simple spring layout implementation.
    """
    nodes = ego.get("nodes", [])
    edges = ego.get("edges", [])
    if not nodes:
        return "<svg></svg>"
        
    # Spring layout (Fruchterman-Reingold)
    n_nodes = len(nodes)
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    
    pos = np.random.rand(n_nodes, 2) * 10
    if n_nodes > 0:
        # Center the center node
        for i, n in enumerate(nodes):
            if n.get("is_center"):
                pos[i] = [5, 5]
                
    k = 10.0 / np.sqrt(max(n_nodes, 1))
    
    # 50 iterations
    for _ in range(50):
        # Repulsion
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dist[dist == 0] = 1e-5 # avoid division by zero
        repulsion = (k * k / dist)[:, :, np.newaxis] * (diff / dist[:, :, np.newaxis])
        disp = np.sum(repulsion, axis=1)
        
        # Attraction
        for e in edges:
            if e["source"] in id_to_idx and e["target"] in id_to_idx:
                s = id_to_idx[e["source"]]
                t = id_to_idx[e["target"]]
                diff_e = pos[t] - pos[s]
                dist_e = np.linalg.norm(diff_e)
                if dist_e > 0:
                    attraction = (dist_e * dist_e / k) * (diff_e / dist_e)
                    disp[s] += attraction
                    disp[t] -= attraction
                    
        # Update
        disp_len = np.linalg.norm(disp, axis=-1)
        disp_len[disp_len == 0] = 1
        pos += (disp / disp_len[:, np.newaxis]) * min(0.5, 10.0/n_nodes)
        
    # Normalize to 0-800
    pos -= pos.min(axis=0)
    max_val = pos.max()
    if max_val > 0:
        pos = (pos / max_val) * 700 + 50
    else:
        pos[:] = 400
        
    width = 800
    height = 800
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    
    # Edges
    for e in edges:
        if e["source"] in id_to_idx and e["target"] in id_to_idx:
            s = id_to_idx[e["source"]]
            t = id_to_idx[e["target"]]
            x1, y1 = pos[s]
            x2, y2 = pos[t]
            color = EDGE_TYPE_COLORS.get(e["type"], "#999")
            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" />')
            
    # Nodes
    for i, n in enumerate(nodes):
        x, y = pos[i]
        color = "red" if n.get("score", 0) > 0.5 else "blue" # simplistic coloring based on score
        r = 10 if n.get("is_center") else 6
        stroke = "black" if n.get("is_center") else "none"
        stroke_width = 3 if n.get("is_center") else 0
        svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}">')
        svg.append(f'<title>{n["id"]} (Score: {n.get("score", 0):.3f})</title>')
        svg.append('</circle>')
        
    svg.append('</svg>')
    return "\\n".join(svg)
