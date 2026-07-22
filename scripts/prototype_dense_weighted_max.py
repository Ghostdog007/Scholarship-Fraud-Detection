"""
Fix for the IP-ring regression found in the extended dense-block prototype.

Root cause: max(ip, mobile, pincode) is mathematically the same whether you
combine within the detector or later inside a max-based fusion (max is
associative) -- so a NONE (valid) row whose mobile or pincode component is
elevated by ordinary, non-fraud density (a real shared family phone, a real
dense district pincode) can outrank a true IP_CLUSTER member in the SAME
combined column. Equal-weight max has no way to prefer the relation you trust
more.

Fix: a PRIORITY-WEIGHTED max -- max(w_ip*ip, w_mobile*mobile, w_pincode*pincode)
with w_ip dominant. This keeps the "let the strongest relation through, don't
dilute by summing" principle (still not a weighted-sum), but protects IP's
specialization from being casually outcompeted by the newly-added, noisier
relations, matching the operational priority that most real fraud runs
through IP.

Recomputes the 3 per-relation dense-block scores directly (deterministic
Charikar peeling, no training -- fast) rather than re-running the full v2
prototype, then sweeps a few weight settings and reports concrete PR-AUC.

Writes: outputs/stress_testing_1_v2b_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.config_v3 import EDGE_TYPES
from src.dense_block_detector_v3 import dense_block_scores
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types
import src.dense_block_detector_v3 as dbd

NAME = "stress_testing_1"
STAGED_GRAPH     = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
FULL_SCORES_CSV  = Path(f"outputs/{NAME}_v2_full_scores.csv")
OUT_STATS        = Path(f"outputs/{NAME}_v2b_stats.json")


def main() -> None:
    print(f"[dense_weighted] Loading merged graph for '{NAME}' ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    prev = pd.read_csv(FULL_SCORES_CSV)
    prev["application_id"] = prev["application_id"].astype(str)
    cohort_ids = set(prev["application_id"])

    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, torch.device("cpu"))
    relations = [0, 1, 4]  # shares_mobile, shares_ip, shares_pincode

    orig = dbd.DENSE_BLOCK_RELATIONS
    dbd.DENSE_BLOCK_RELATIONS = relations
    try:
        rel_df = dense_block_scores(edge_index_list, edge_type_tensor, len(app_ids_merged), app_ids_merged)
    finally:
        dbd.DENSE_BLOCK_RELATIONS = orig

    rel_df["application_id"] = rel_df["application_id"].astype(str)
    rel_df = rel_df[rel_df["application_id"].isin(cohort_ids)]
    prev_clean = prev.drop(columns=["dense_block_score_ip"], errors="ignore")
    df = prev_clean.merge(rel_df[["application_id", "dense_block_score_mobile",
                                   "dense_block_score_ip", "dense_block_score_pincode"]],
                           on="application_id", how="left").fillna(0.0)

    weight_settings = {
        "equal_max (previous prototype)":       (1.0, 1.0, 1.0),
        "ip_priority_moderate (1.0/0.6/0.4)":    (0.6, 1.0, 0.4),
        "ip_priority_strong (1.0/0.3/0.2)":      (0.3, 1.0, 0.2),
        "ip_only (production, unchanged)":       (0.0, 1.0, 0.0),
    }

    y = df["is_fraud"].astype(int).values
    stats = {"weight_settings_(w_mobile, w_ip, w_pincode)": {}, "per_fraud_type": {}}

    for label, (w_m, w_i, w_p) in weight_settings.items():
        col = f"combo__{label}"
        df[col] = np.maximum.reduce([
            w_m * df["dense_block_score_mobile"].values,
            w_i * df["dense_block_score_ip"].values,
            w_p * df["dense_block_score_pincode"].values,
        ])
        stats["weight_settings_(w_mobile, w_ip, w_pincode)"][label] = {
            "weights": [w_m, w_i, w_p],
            "overall_pr_auc": float(average_precision_score(y, df[col].values)),
        }

    neg = df[df["fraud_type"] == "NONE"]  # snapshotted AFTER combo__ columns exist
    for ft in ["IP_CLUSTER", "MOBILE_CLUSTER", "PINCODE_CLUSTER"]:
        pos = df[df["fraud_type"] == ft]
        sub = pd.concat([pos, neg])
        yy = (sub["fraud_type"] == ft).astype(int).values
        stats["per_fraud_type"][ft] = {
            label: float(average_precision_score(yy, sub[f"combo__{label}"].values))
            for label in weight_settings
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[dense_weighted] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
