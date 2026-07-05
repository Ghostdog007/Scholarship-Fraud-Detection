"""
graph_viz_v3.py

Interactive 3D visualization of flagged fraud-ring node configurations.

Read-only. Consumes public inputs only (identity graph structure, application
ids, and hybrid_anomaly_score) — never raw embeddings (hard stop #2). Generates
candidate rings, ranks them by member risk, and writes one self-contained,
offline, rotatable Plotly HTML per ring to outputs/viz/.

Node color   = hybrid_anomaly_score (risk).
Edge color   = relation type (shares_mobile / shares_ip / ...).
Hover        = application_id, risk, in-ring degree.

Run:
  python -m src.graph_viz_v3            # top 10 flagged rings
  python -m src.graph_viz_v3 --top-k 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
import plotly.graph_objects as go

from src.config_v3 import EDGE_TYPES, RANDOM_SEED
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types, DEVICE
from src.ring_candidate_v3 import generate_candidates

FINAL_CSV  = Path("data/processed/engineered_features_v3.csv")
GRAPH_PT   = Path("data/processed/identity_graph_v3.pt")
HYBRID_CSV = Path("outputs/hybrid_scores_v3.csv")
OUT_DIR    = Path("outputs/viz")

# One distinct colour per relation type (index aligns with EDGE_TYPES).
RELATION_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _load_inputs():
    df = pd.read_csv(FINAL_CSV)
    feat_cols = [c for c in df.columns if c != "application_id"]
    app_ids = df["application_id"].values
    x_all = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(DEVICE)

    data = torch.load(GRAPH_PT, weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)

    hybrid = pd.read_csv(HYBRID_CSV)[["application_id", "hybrid_anomaly_score"]]
    # align risk to node order via application_id
    risk_map = dict(zip(hybrid["application_id"], hybrid["hybrid_anomaly_score"]))
    risk = np.array([risk_map.get(a, 0.0) for a in app_ids], dtype=np.float32)
    risk_t = torch.tensor(risk, dtype=torch.float32, device=DEVICE)
    return app_ids, x_all, edge_index_list, edge_type_tensor, risk, risk_t


def _rank_rings(candidates: list[dict]) -> list[dict]:
    """Rank by mean member risk (desc), tie-break by size (desc)."""
    def key(sg):
        s = sg["scores"]
        s = s.cpu().numpy() if hasattr(s, "cpu") else np.asarray(s)
        return (float(s.mean()), len(sg["node_ids"]))
    return sorted(candidates, key=key, reverse=True)


def _figure_for_ring(sg: dict, app_ids: np.ndarray, rank: int) -> go.Figure:
    node_ids = list(sg["node_ids"])           # global indices, sorted
    n = len(node_ids)
    scores = sg["scores"]
    scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)

    # Undirected graph over LOCAL indices; dedupe edges, remember relation.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_rel = {}
    for u, v, r in sg["edges"]:
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        G.add_edge(*pair)
        # if multiple relations connect a pair, keep the first seen (stable)
        edge_rel.setdefault(pair, int(r))

    pos = nx.spring_layout(G, dim=3, seed=RANDOM_SEED)
    degree = dict(G.degree())

    traces = []
    # ---- edges, grouped by relation so each relation is a legend entry ----
    for rel_id, rel_name in enumerate(EDGE_TYPES):
        xs, ys, zs = [], [], []
        for pair, r in edge_rel.items():
            if r != rel_id:
                continue
            a, b = pair
            xs += [pos[a][0], pos[b][0], None]
            ys += [pos[a][1], pos[b][1], None]
            zs += [pos[a][2], pos[b][2], None]
        if not xs:
            continue
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=RELATION_COLORS[rel_id % len(RELATION_COLORS)], width=4),
            name=rel_name, hoverinfo="none",
        ))

    # ---- nodes ----
    nx_, ny_, nz_ = zip(*[pos[i] for i in range(n)])
    hover = [
        f"application_id: {app_ids[node_ids[i]]}<br>"
        f"risk: {scores[i]:.4f}<br>"
        f"in-ring degree: {degree.get(i, 0)}"
        for i in range(n)
    ]
    traces.append(go.Scatter3d(
        x=nx_, y=ny_, z=nz_, mode="markers",
        marker=dict(
            size=6, color=scores, colorscale="YlOrRd", cmin=0.0, cmax=1.0,
            colorbar=dict(title="risk"), line=dict(color="#333", width=0.5),
        ),
        text=hover, hoverinfo="text", name="applications",
    ))

    mean_risk = float(scores.mean())
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(f"Flagged ring #{rank}  |  {n} nodes, {G.number_of_edges()} edges  |  "
               f"mean risk {mean_risk:.3f}"),
        scene=dict(xaxis=dict(showticklabels=False, title=""),
                   yaxis=dict(showticklabels=False, title=""),
                   zaxis=dict(showticklabels=False, title="")),
        margin=dict(l=0, r=0, t=40, b=0), showlegend=True,
    )
    return fig


def render_flagged_rings(top_k: int = 10) -> None:
    print(f"[graph_viz] Loading inputs | device={DEVICE}")
    app_ids, x_all, eil, ett, risk, risk_t = _load_inputs()

    print("[graph_viz] Generating candidate rings ...")
    candidates = generate_candidates(eil, ett, x_all, risk_t)
    print(f"[graph_viz] {len(candidates)} candidates | ranking by mean member risk")
    ranked = _rank_rings(candidates)[:top_k]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rank, sg in enumerate(ranked, start=1):
        fig = _figure_for_ring(sg, app_ids, rank)
        out = OUT_DIR / f"flagged_ring_{rank:02d}.html"
        fig.write_html(str(out), include_plotlyjs=True, full_html=True)
        s = sg["scores"]; s = s.cpu().numpy() if hasattr(s, "cpu") else np.asarray(s)
        print(f"[graph_viz]  #{rank:2d}  n={len(sg['node_ids']):3d}  "
              f"mean_risk={float(s.mean()):.3f}  -> {out}")
    print(f"[graph_viz] Done. Open any HTML in a browser and drag to rotate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    render_flagged_rings(top_k=ap.parse_args().top_k)
