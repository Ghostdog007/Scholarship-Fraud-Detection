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

# ---- palette -----------------------------------------------------------------
# Design: one loud element (nodes = risk), everything else quiet. Edges are
# muted-but-distinct by relation; nodes carry the vivid risk colour. Tuned to
# read on a dark ground.
BG_COLOR    = "#0e1117"   # deep charcoal ground
PANEL_COLOR = "#161b22"   # legend / hover panel
GRID_COLOR  = "rgba(255,255,255,0.05)"
FG_COLOR    = "#c9d1d9"   # primary text
MUTED_COLOR = "#8b949e"   # secondary text
FONT_FAMILY = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"

# One distinct colour per relation type (index aligns with EDGE_TYPES).
# shares_ip (index 1) is the V4-load-bearing relation — given the hottest hue.
RELATION_COLORS = ["#4cc9f0", "#f72585", "#4ade80", "#b57bff", "#ffb703"]

# Risk colourscale: calm teal (low) → hot red (high), legible on dark.
RISK_SCALE = [
    [0.00, "#2c7da0"],
    [0.35, "#90be6d"],
    [0.60, "#f4a261"],
    [0.85, "#e85d04"],
    [1.00, "#e5383b"],
]


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

    # More iterations + wider scale → cleaner separation of core vs periphery.
    pos = nx.spring_layout(G, dim=3, seed=RANDOM_SEED, iterations=200, scale=2.0)
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
        col = RELATION_COLORS[rel_id % len(RELATION_COLORS)]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=col, width=2.5),
            opacity=0.45,                      # quiet edges; nodes stay loud
            name=rel_name.replace("shares_", ""), hoverinfo="none",
        ))

    # ---- nodes: size ∝ in-ring degree so hubs pop; colour = risk ----
    nx_, ny_, nz_ = zip(*[pos[i] for i in range(n)])
    degs = np.array([degree.get(i, 0) for i in range(n)], dtype=float)
    dmax = degs.max() if degs.max() > 0 else 1.0
    sizes = 9.0 + 15.0 * (degs / dmax)          # 9 (leaf) → 24 (hub)
    hover = [
        f"<b>application_id</b>  {app_ids[node_ids[i]]}<br>"
        f"<b>risk</b>  {scores[i]:.4f}<br>"
        f"<b>in-ring degree</b>  {int(degs[i])}"
        for i in range(n)
    ]
    traces.append(go.Scatter3d(
        x=nx_, y=ny_, z=nz_, mode="markers",
        marker=dict(
            size=sizes, color=scores, colorscale=RISK_SCALE, cmin=0.0, cmax=1.0,
            opacity=0.95,
            line=dict(color="rgba(14,17,23,0.9)", width=1.2),   # dark rim = glow
            colorbar=dict(
                title=dict(text="risk", font=dict(color=FG_COLOR, size=13)),
                tickfont=dict(color=MUTED_COLOR, size=11),
                thickness=14, len=0.55, x=0.98, outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
        ),
        text=hover, hoverinfo="text",
        name="applications", showlegend=False,
    ))

    mean_risk = float(scores.mean())
    axis = dict(showticklabels=False, title="", showspikes=False,
                showbackground=True, backgroundcolor=BG_COLOR,
                gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"<b>Flagged ring #{rank}</b>"
                  f"<span style='color:{MUTED_COLOR}'>"
                  f"    {n} nodes · {G.number_of_edges()} edges · "
                  f"mean risk {mean_risk:.3f}</span>"),
            font=dict(family=FONT_FAMILY, size=18, color=FG_COLOR),
            x=0.02, xanchor="left", y=0.97,
        ),
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(family=FONT_FAMILY, color=FG_COLOR),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                   bgcolor=BG_COLOR,
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))),
        hoverlabel=dict(bgcolor=PANEL_COLOR, bordercolor="rgba(255,255,255,0.12)",
                        font=dict(family=FONT_FAMILY, color=FG_COLOR, size=12)),
        legend=dict(
            title=dict(text="relation", font=dict(color=MUTED_COLOR, size=12)),
            bgcolor="rgba(22,27,34,0.75)", bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1, font=dict(color=FG_COLOR, size=12),
            x=0.01, y=0.5, itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=46, b=0), showlegend=True,
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
