"""
Prototype: Latent Outlier Exposure (Qiu et al., ICML 2022) applied to the
Deep SAD center-distance objective — NOTE the acronym collision with this
project's own "Learning-from-Only-Exposure" (LOE); they are different
mechanisms solving different problems (see README changelog 2026-07-22).

Rationale: production Deep SAD (deepsad_detector_v3.py) and Hybrid GraphMCM's
centroid init both assume the REAL population is mostly clean, using a fixed
heuristic (CENTROID_CLEAN_PERCENTILE=95 -- exclude the top 5% by embedding
norm from the center calculation, once per refresh). That's a one-shot
exclusion, not a learned one, and a documented project weakness: "DeepSVDD
centroid: if fraud dominates the population, hypersphere silently inflates."

Qiu et al.'s fix: jointly infer, via block-coordinate updates, which
unlabeled training points are likely secretly anomalous, WHILE training --
alternating between (a) re-estimating a latent anomalous/normal label per
real node from the current embeddings and (b) updating the model treating
presumed-anomalous real nodes as exposure examples (pushed away), not just
excluded from the pull term.

Scoped-down, honestly labeled implementation:
  - Contamination ratio alpha = 0.05 (matches CENTROID_CLEAN_PERCENTILE's
    existing 95th-pct convention, not independently tuned here).
  - Latent-label re-estimation happens every DEEPSAD_CENTER_REFRESH_EVERY
    epochs (matching the existing center-refresh cadence), not every step --
    a practical block-coordinate schedule, not the paper's exact algorithm.
  - Presumed-anomalous real nodes get an inverted-distance PUSH term (like
    the known synthetic exposure nodes), weighted lower (LATENT_ETA) than
    the known exposure nodes (DEEPSAD_ETA) to reflect that the label is a
    guess, not ground truth.

Compares against the current production Deep SAD (center_dist_score,
0.201 overall / 0.093 mobile-ring on stress_testing_1) to see whether
contamination-aware center estimation actually helps on data that HAS real
injected fraud sitting in the "real" population (stress_testing_1's ground
truth is exactly this scenario).

Writes: outputs/stress_testing_1_loe_qiu_stats.json
"""
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score

from src.config_v3 import EDGE_TYPES, RANDOM_SEED, CENTROID_CLEAN_PERCENTILE
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NAME = "stress_testing_1"
STAGED_GRAPH     = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_afterfix_full_scores.csv")
DEEPSAD_CSV      = Path(f"outputs/{NAME}_deepsad_full_scores.csv")
NODEG_CSV        = Path("data/processed/engineered_features_v3_nodeg.csv")
SCHEMA_JSON      = Path("data/processed/v3_feature_schema.json")
TOPO_PT          = Path("data/processed/synthetic_exposure_graph_v3.pt")

OUT_STATS = Path(f"outputs/{NAME}_loe_qiu_alpha015_stats.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, EMB_DIM = 64, 32
N_EPOCHS = 150
ETA = 1.0                  # known synthetic-exposure push weight (matches production DEEPSAD_ETA)
LATENT_ETA = 0.3           # latent (guessed) real-contamination push weight -- lower, it's a guess
ALPHA = 0.15               # contamination ratio -- matched to stress_testing_1's actual ~15% injected
                            # fraud rate (first run used 0.05, borrowed from production's
                            # CENTROID_CLEAN_PERCENTILE convention, and regressed -- see README/chat)
RELABEL_EVERY = 25         # epochs between latent-label re-estimation (matches center-refresh cadence)


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


class SharedEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, n_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, HIDDEN, n_relations)
        self.conv2 = RGCNConv(HIDDEN, EMB_DIM, n_relations)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type))
        return self.conv2(h, edge_index, edge_type)


def main() -> None:
    print(f"[loe_qiu] Device: {DEVICE}")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    prev = pd.read_csv(FULL_SCORES_CSV)
    prev["application_id"] = prev["application_id"].astype(str)
    cohort_ids = set(prev["application_id"])
    deepsad_ref = pd.read_csv(DEEPSAD_CSV)[["application_id", "center_dist_score"]]
    deepsad_ref["application_id"] = deepsad_ref["application_id"].astype(str)
    deepsad_ref = deepsad_ref.rename(columns={"center_dist_score": "center_dist_score_prod"})

    x44 = build_full_44dim_x(data, nodeorder).to(DEVICE)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
    n_real = x44.shape[0]
    n_latent = max(1, int(ALPHA * n_real))

    topo_pack_raw = torch.load(TOPO_PT, weights_only=False)
    topo_pack = {k: v.to(DEVICE) for k, v in topo_pack_raw.items()}

    encoder = SharedEncoder(x44.shape[1], len(EDGE_TYPES)).to(DEVICE)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    with torch.no_grad():
        h0 = encoder(x44, edge_index, edge_type_tensor)
        norms0 = h0.norm(dim=1)
        cutoff0 = torch.quantile(norms0, CENTROID_CLEAN_PERCENTILE / 100.0)
        center = h0[norms0 <= cutoff0].mean(dim=0).detach()

    latent_mask = torch.zeros(n_real, dtype=torch.bool, device=DEVICE)  # presumed-anomalous real nodes
    n_relabels = 0

    print(f"[loe_qiu] training {N_EPOCHS} epochs (Deep SAD + latent contamination relabeling, "
          f"alpha={ALPHA}, relabel every {RELABEL_EVERY} epochs, n_latent={n_latent}) ...")
    encoder.train()
    for epoch in range(N_EPOCHS):
        h_real = encoder(x44, edge_index, edge_type_tensor)
        h_synth = encoder(topo_pack["x"], topo_pack["edge_index"], topo_pack["edge_type"])

        dist_real = ((h_real - center) ** 2).sum(dim=1)

        normal_mask = ~latent_mask
        loss_normal = dist_real[normal_mask].mean() if normal_mask.any() else dist_real.mean()

        loss_synth_anomaly = (ETA / (((h_synth - center) ** 2).sum(dim=1) + 1e-6)).mean()

        if latent_mask.any():
            loss_latent_anomaly = (LATENT_ETA / (dist_real[latent_mask] + 1e-6)).mean()
        else:
            loss_latent_anomaly = torch.zeros((), device=DEVICE)

        loss = loss_normal + loss_synth_anomaly + loss_latent_anomaly

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Block-coordinate step: re-estimate latent contamination labels from
        # current embeddings, then recompute center excluding them (same
        # exclusion the production heuristic does, but refreshed jointly with
        # the labels instead of using a single static percentile cutoff).
        if (epoch + 1) % RELABEL_EVERY == 0:
            with torch.no_grad():
                d = ((h_real - center) ** 2).sum(dim=1)
                topk = torch.topk(d, n_latent).indices
                latent_mask = torch.zeros(n_real, dtype=torch.bool, device=DEVICE)
                latent_mask[topk] = True
                n_relabels += 1
                center = h_real[~latent_mask].mean(dim=0).detach()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCHS} loss={loss.item():.4f} "
                  f"normal={loss_normal.item():.4f} synth={loss_synth_anomaly.item():.4f} "
                  f"latent={loss_latent_anomaly.item():.4f} n_flagged_latent={int(latent_mask.sum())}")

    print(f"[loe_qiu] Scoring all real nodes ({n_relabels} relabel rounds happened) ...")
    encoder.eval()
    with torch.no_grad():
        h_real_final = encoder(x44, edge_index, edge_type_tensor)
        d = ((h_real_final - center) ** 2).sum(dim=1)
        topk = torch.topk(d, n_latent).indices
        final_latent_mask = torch.zeros(n_real, dtype=torch.bool, device=DEVICE)
        final_latent_mask[topk] = True
        center_final = h_real_final[~final_latent_mask].mean(dim=0)
        care_score = ((h_real_final - center_final) ** 2).sum(dim=1).sqrt()

    df_scores = pd.DataFrame({
        "application_id": app_ids_merged,
        "center_dist_score_loe_qiu": care_score.cpu().numpy(),
    })
    df_scores["application_id"] = df_scores["application_id"].astype(str)
    df_scores = df_scores[df_scores["application_id"].isin(cohort_ids)]

    df = prev.merge(df_scores, on="application_id", how="left").merge(deepsad_ref, on="application_id", how="left")
    out_csv = Path(f"outputs/{NAME}_loe_qiu_alpha015_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[loe_qiu] Wrote {out_csv}")

    y = df["is_fraud"].astype(int).values
    stats: dict = {
        "config": {"alpha": ALPHA, "relabel_every": RELABEL_EVERY, "latent_eta": LATENT_ETA, "n_relabel_rounds": n_relabels},
        "overall": {
            "hybrid_reconstruction (prod baseline, LOE fix + root_weight fix)": float(average_precision_score(y, df["hybrid_anomaly_score"].values)),
            "center_dist_score_prod (Deep SAD, static clean-percentile center)": float(average_precision_score(y, df["center_dist_score_prod"].values)),
            "center_dist_score_loe_qiu (new, contamination-aware iterative relabeling)": float(average_precision_score(y, df["center_dist_score_loe_qiu"].values)),
        },
    }

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
            "hybrid_reconstruction": float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "center_dist_score_prod": float(average_precision_score(yy, sub["center_dist_score_prod"].values)),
            "center_dist_score_loe_qiu": float(average_precision_score(yy, sub["center_dist_score_loe_qiu"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[loe_qiu] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
