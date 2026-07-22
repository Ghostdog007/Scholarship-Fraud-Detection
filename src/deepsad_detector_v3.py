"""
deepsad_detector_v3.py

Deep SAD center-distance signal (Ruff et al., ICLR 2020) — a semi-supervised
anomaly detector trained on topology exposure's synthetic archetypes, kept
architecturally SEPARATE from Hybrid GraphMCM (own encoder, own objective, own
checkpoint). No reconstruction loss: real nodes are pulled toward a learned
"normal" center, topology exposure's synthetic nodes are pushed away via an
inverted-distance term. This sidesteps the MAR failure mode (dense synthetic
cliques reconstruct too easily under a reconstruction objective) because there
is no reconstruction objective here to fight.

Validated on stress_testing_1 (see outputs/stress_testing_1_deepsad_stats.json
and config_v3.py's DEEPSAD_* block for the full rationale/numbers): the
strongest single relational signal found this session (0.201 overall PR-AUC,
0.093 mobile-ring, 0.050 IP-ring vs hybrid_reconstruction's 0.153 / 0.029 / 0.032).

Deliberately NOT part of fusion_classifier_v3.py's locked score-level fusion —
this is a supplementary signal surfaced on XAI cards only (see xai_layer_v3.py),
not a driver of final_risk_score. Promoting it into fusion is a separate,
not-yet-made decision.

A companion per-cluster "nearest known archetype" prototype-match mechanism was
also tested and explicitly rejected (near-random, no inter-prototype separation
term) — not implemented here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from src.config_v3 import (
    N_FEATURES, N_EDGE_TYPES, EDGE_TYPES, RANDOM_SEED, CENTROID_CLEAN_PERCENTILE,
    TOPO_EXPOSURE_ENABLED, DEEPSAD_ENABLED, DEEPSAD_HIDDEN, DEEPSAD_EMB_DIM,
    DEEPSAD_EPOCHS, DEEPSAD_ETA, DEEPSAD_LR, DEEPSAD_CENTER_REFRESH_EVERY,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FINAL_CSV = Path("data/processed/engineered_features_v3.csv")
GRAPH_PT  = Path("data/processed/identity_graph_v3.pt")
TOPO_PT   = Path("data/processed/synthetic_exposure_graph_v3.pt")

MODEL_PTH = Path("models/deepsad_v3.pth")
OUT_CSV   = Path("outputs/deepsad_scores_v3.csv")


class DeepSADEncoder(torch.nn.Module):
    """Own 2-layer RGCN, NOT shared with Hybrid GraphMCM. root_weight left at
    its RGCNConv default (True) -- that combination is what was validated on
    stress_testing_1; root_weight=False was validated for Hybrid GraphMCM's
    encoder separately and has not been re-tested here."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = RGCNConv(N_FEATURES, DEEPSAD_HIDDEN, N_EDGE_TYPES)
        self.conv2 = RGCNConv(DEEPSAD_HIDDEN, DEEPSAD_EMB_DIM, N_EDGE_TYPES)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_type))
        return self.conv2(h, edge_index, edge_type)


def _init_center(model: DeepSADEncoder, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h0 = model(x, edge_index, edge_type)
        norms0 = h0.norm(dim=1)
        cutoff0 = torch.quantile(norms0, CENTROID_CLEAN_PERCENTILE / 100.0)
        return h0[norms0 <= cutoff0].mean(dim=0).detach()


def train() -> None:
    print(f"[deepsad] Device: {DEVICE}")
    if not DEEPSAD_ENABLED:
        print("[deepsad] DEEPSAD_ENABLED is False. Exiting.")
        return
    if not TOPO_EXPOSURE_ENABLED or not TOPO_PT.exists():
        print("[deepsad] TOPO_EXPOSURE_ENABLED is False or topology exposure pack "
              "missing -- Deep SAD has no exposure signal to push against. Exiting.")
        return

    from src.hybrid_graphmcm_v3 import _build_edge_index_and_types

    df = pd.read_csv(FINAL_CSV)
    feat_cols = [c for c in df.columns if c != "application_id"]
    app_ids = df["application_id"].values
    x_all = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(DEVICE)
    assert x_all.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {x_all.shape[1]}"

    data = torch.load(GRAPH_PT, weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=DEVICE)

    topo_pack_raw = torch.load(TOPO_PT, weights_only=False)
    topo_pack = {k: v.to(DEVICE) for k, v in topo_pack_raw.items()}

    model = DeepSADEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=DEEPSAD_LR)
    center = _init_center(model, x_all, edge_index, edge_type_tensor)

    print(f"[deepsad] training {DEEPSAD_EPOCHS} epochs (center-pull / exposure-push, no reconstruction) ...")
    model.train()
    for epoch in range(DEEPSAD_EPOCHS):
        h_real = model(x_all, edge_index, edge_type_tensor)
        h_synth = model(topo_pack["x"], topo_pack["edge_index"], topo_pack["edge_type"])

        dist_real = ((h_real - center) ** 2).sum(dim=1)
        loss_normal = dist_real.mean()
        dist_synth = ((h_synth - center) ** 2).sum(dim=1)
        loss_anomaly = (DEEPSAD_ETA / (dist_synth + 1e-6)).mean()
        loss = loss_normal + loss_anomaly

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % DEEPSAD_CENTER_REFRESH_EVERY == 0:
            with torch.no_grad():
                norms = h_real.norm(dim=1)
                cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
                center = h_real[norms <= cutoff].mean(dim=0).detach()

        if (epoch + 1) % max(1, DEEPSAD_EPOCHS // 10) == 0:
            print(f"  epoch {epoch+1}/{DEEPSAD_EPOCHS} | normal={loss_normal.item():.4f} anomaly={loss_anomaly.item():.4f}")

    print("[deepsad] Scoring all nodes ...")
    model.eval()
    with torch.no_grad():
        h_final = model(x_all, edge_index, edge_type_tensor)
        norms = h_final.norm(dim=1)
        cutoff = torch.quantile(norms, CENTROID_CLEAN_PERCENTILE / 100.0)
        center_final = h_final[norms <= cutoff].mean(dim=0)
        dist = ((h_final - center_final) ** 2).sum(dim=1).sqrt().cpu().numpy()

    lo, hi = dist.min(), dist.max()
    center_dist_score = ((dist - lo) / (hi - lo + 1e-8)).astype(np.float32)

    out_df = pd.DataFrame({"application_id": app_ids, "center_dist_score": center_dist_score})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[deepsad] Scores saved -> {OUT_CSV}")
    print(f"[deepsad] center_dist_score range (pre-normalize): [{dist.min():.4f}, {dist.max():.4f}]")

    MODEL_PTH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "center": center_final.cpu(),
            "config": {
                "N_FEATURES": N_FEATURES,
                "DEEPSAD_HIDDEN": DEEPSAD_HIDDEN,
                "DEEPSAD_EMB_DIM": DEEPSAD_EMB_DIM,
                "N_EDGE_TYPES": N_EDGE_TYPES,
                "ARCH_VERSION": "deepsad_v1",
            },
        },
        MODEL_PTH,
    )
    print(f"[deepsad] Checkpoint saved -> {MODEL_PTH}")


def run_deepsad() -> None:
    train()


if __name__ == "__main__":
    run_deepsad()
