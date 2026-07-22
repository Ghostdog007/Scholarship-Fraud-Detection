"""
Confirm no regression after dropping shares_pincode from the dense-block gate
(config_v3.DENSE_BLOCK_RELATIONS: [0,1,4] -> [0,1], weights {0:0.3,1:1.0}).

Reuses the already-computed subspace_if_score / hybrid_anomaly_score / ground
truth from outputs/stress_testing_1_afterfix_full_scores.csv (the current
production-equivalent baseline: root_weight=False + old 3-relation dense-block
+ max fusion, overall risk_score_v3 PR-AUC 0.4182). Only dense_block_score_relational
is recomputed, using the live dense_block_scores() with the NEW config (mobile+ip
only) against the same staged merged graph, then re-fused with the unchanged
locked max-fusion formula.

Writes: outputs/stress_testing_1_nopincode_full_scores.csv,
        outputs/stress_testing_1_nopincode_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.config_v3 import DENSE_BLOCK_RELATIONS, DENSE_BLOCK_RELATION_WEIGHTS
from src.dense_block_detector_v3 import dense_block_scores
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types

NAME = "stress_testing_1"
BASELINE_CSV = Path(f"outputs/{NAME}_afterfix_full_scores.csv")
STAGED_GRAPH = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")

OUT_CSV = Path(f"outputs/{NAME}_nopincode_full_scores.csv")
OUT_STATS = Path(f"outputs/{NAME}_nopincode_stats.json")


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def main() -> None:
    print(f"[nopincode] DENSE_BLOCK_RELATIONS={DENSE_BLOCK_RELATIONS} "
          f"DENSE_BLOCK_RELATION_WEIGHTS={DENSE_BLOCK_RELATION_WEIGHTS}")
    assert DENSE_BLOCK_RELATIONS == [0, 1], "config not updated as expected"

    base = pd.read_csv(BASELINE_CSV)
    base["application_id"] = base["application_id"].astype(str)
    cohort_ids = set(base["application_id"])

    print("[nopincode] Recomputing dense-block scores (mobile+ip only) on merged graph ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, torch.device("cpu"))
    nodeorder = pd.read_csv(STAGED_NODEORDER)
    app_ids_merged = nodeorder["application_id"].astype(str).values
    dense_df = dense_block_scores(edge_index_list, edge_type_tensor, len(app_ids_merged), app_ids_merged)
    dense_df["application_id"] = dense_df["application_id"].astype(str)
    dense_df = dense_df[dense_df["application_id"].isin(cohort_ids)]

    merged = base.drop(columns=[c for c in base.columns if c.startswith("dense_block_score")]) \
                 .merge(dense_df, on="application_id", how="left")
    merged["dense_block_score_relational"] = merged["dense_block_score_relational"].fillna(0.0)

    s = _minmax(merged["subspace_if_score"].values)
    d = _minmax(merged["dense_block_score_relational"].values)
    h = _minmax(merged["hybrid_anomaly_score"].values)
    merged["risk_score_v3"] = _minmax(np.maximum.reduce([s, d, h]))

    merged.to_csv(OUT_CSV, index=False)
    print(f"[nopincode] Wrote {OUT_CSV}")

    y = merged["is_fraud"].astype(int).values
    stats: dict = {"n_rows": len(merged), "n_fraud": int(y.sum())}
    for col in ["subspace_if_score", "dense_block_score_relational", "hybrid_anomaly_score", "risk_score_v3"]:
        stats[col] = {"pr_auc": float(average_precision_score(y, merged[col].values))}

    stats["per_fraud_type"] = {}
    neg = merged[merged["fraud_type"] == "NONE"]
    for ft in sorted(merged["fraud_type"].unique()):
        if ft == "NONE":
            continue
        pos = merged[merged["fraud_type"] == ft]
        sub = pd.concat([pos, neg])
        yy = (sub["fraud_type"] == ft).astype(int).values
        stats["per_fraud_type"][ft] = {
            "n": int(len(pos)),
            "risk_score_v3": float(average_precision_score(yy, sub["risk_score_v3"].values)),
            "hybrid": float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "subspace": float(average_precision_score(yy, sub["subspace_if_score"].values)),
            "dense_relational": float(average_precision_score(yy, sub["dense_block_score_relational"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[nopincode] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
