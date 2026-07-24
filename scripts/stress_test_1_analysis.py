"""
Full-fusion stats for the stress_testing_1 cohort (50,000 applications).

POST /v3/monitoring/evaluate-dataset only stages the pre-fusion
hybrid_anomaly_score for a cohort (see export_v3.py's staged-bundle README:
"PRE-FUSION preview... NOT the committed fused risk_score_v3"). To honestly
report the model's actual bifurcation ability we need all three detectors —
subspace_if_score, dense_block_score_ip, hybrid_anomaly_score — combined
through the SAME locked formula fusion_classifier_v3.score_level_fusion()
uses in production. This script reproduces that, read-only, from the staged
artifacts evaluate-dataset already wrote (no canonical file is touched):

  outputs/staged_features_stress_testing_1.csv   44-dim engineered features (cohort only)
  outputs/staged_graph_stress_testing_1.pt       merged identity graph (base 15k + cohort)
  outputs/staged_nodeorder_stress_testing_1.csv  node order for the merged graph
  outputs/staged_scores_stress_testing_1.csv     pre-fusion hybrid_anomaly_score (cohort only)

subspace_if_score is refit on the cohort's own 44 features (matches
subspace_if_v3.run_subspace_if()'s per-run IsolationForest exactly, just
pointed at the cohort file instead of the canonical one — same code path,
different population). dense_block_score_ip is computed on the REAL merged
graph via dense_block_detector_v3.dense_block_scores(), then subset to the
cohort's node indices, so any IP edges that bridge into the base 15k
population are represented correctly.

Writes: outputs/stress_testing_1_full_scores.csv, outputs/stress_testing_1_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config_v3 import RANDOM_SEED, SUBSPACE_GROUPS, FUSION_W_SUBSPACE, FUSION_W_DENSE_IP, FUSION_W_HYBRID
from src.fusion_classifier_v3 import score_level_fusion
from src.dense_block_detector_v3 import dense_block_scores
from src.hybrid_graphmcm_v3 import _build_edge_index_and_types

NAME = "stress_testing_1"
STAGED_FEATURES = Path(f"outputs/staged_features_{NAME}.csv")
STAGED_GRAPH    = Path(f"outputs/staged_graph_{NAME}.pt")
STAGED_NODEORDER = Path(f"outputs/staged_nodeorder_{NAME}.csv")
STAGED_SCORES   = Path(f"outputs/staged_scores_{NAME}.csv")
GT_CSV          = Path(f"data/uploads/{NAME}_ground_truth.csv")

OUT_SCORES = Path(f"outputs/{NAME}_full_scores.csv")
OUT_STATS  = Path(f"outputs/{NAME}_stats.json")


def run_subspace_if_on(df: pd.DataFrame) -> np.ndarray:
    """Same logic as subspace_if_v3.run_subspace_if(), against an arbitrary
    features frame instead of the canonical FINAL_CSV."""
    group_arrays = {}
    for group_name, group_cols in SUBSPACE_GROUPS.items():
        available = [c for c in group_cols if c in df.columns]
        if not available:
            group_arrays[group_name] = np.zeros(len(df))
            continue
        X = df[available].values
        clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_SEED)
        clf.fit(X)
        raw = -clf.decision_function(X)
        lo, hi = raw.min(), raw.max()
        group_arrays[group_name] = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    return np.stack(list(group_arrays.values()), axis=1).max(axis=1)


def main() -> None:
    print(f"[stress_analysis] Loading staged artifacts for '{NAME}' ...")
    feats     = pd.read_csv(STAGED_FEATURES)
    hybrid_df = pd.read_csv(STAGED_SCORES, usecols=["application_id", "hybrid_anomaly_score"])
    gt        = pd.read_csv(GT_CSV, dtype={"application_id": str})
    nodeorder = pd.read_csv(STAGED_NODEORDER)

    cohort_ids = set(feats["application_id"].astype(str))
    print(f"[stress_analysis] Cohort rows: {len(feats)}  |  merged graph nodes: {len(nodeorder)}")

    # ── subspace IF, refit on the cohort's own 44 features ──────────────────
    print("[stress_analysis] Fitting subspace IF on cohort features ...")
    subspace_score = run_subspace_if_on(feats)
    subspace_df = pd.DataFrame({
        "application_id": feats["application_id"].astype(str).values,
        "subspace_if_score": subspace_score,
    })

    # ── dense-block (shares_ip only, per DENSE_BLOCK_RELATIONS), on the REAL
    #    merged graph (base 15k + cohort), subset back to the cohort ────────
    print("[stress_analysis] Computing dense-block IP scores on the merged graph ...")
    data = torch.load(STAGED_GRAPH, weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, torch.device("cpu"))
    app_ids_merged = nodeorder["application_id"].astype(str).values
    dense_df = dense_block_scores(edge_index_list, edge_type_tensor, len(app_ids_merged), app_ids_merged)
    dense_df["application_id"] = dense_df["application_id"].astype(str)
    dense_df = dense_df[dense_df["application_id"].isin(cohort_ids)].rename(
        columns={"dense_block_score_ip": "dense_block_score_ip"})

    # ── merge all three + apply the LOCKED fusion formula, unchanged ────────
    hybrid_df["application_id"] = hybrid_df["application_id"].astype(str)
    merged = (hybrid_df
              .merge(subspace_df, on="application_id", how="inner")
              .merge(dense_df[["application_id", "dense_block_score_ip"]], on="application_id", how="left"))
    merged["dense_block_score_ip"] = merged["dense_block_score_ip"].fillna(0.0)

    print(f"[stress_analysis] fusion weights: subspace={FUSION_W_SUBSPACE} dense_ip={FUSION_W_DENSE_IP} hybrid={FUSION_W_HYBRID}")
    merged["risk_score_v3"] = score_level_fusion(
        merged["subspace_if_score"].values,
        merged["dense_block_score_ip"].values,
        merged["hybrid_anomaly_score"].values,
    )

    df = merged.merge(gt, on="application_id", how="inner")
    assert len(df) == len(feats), f"row count mismatch after merge: {len(df)} vs {len(feats)}"
    df.to_csv(OUT_SCORES, index=False)
    print(f"[stress_analysis] Wrote fused scores -> {OUT_SCORES}")

    # ── stats ────────────────────────────────────────────────────────────────
    y = df["is_fraud"].astype(int).values
    stats: dict = {"n_rows": len(df), "n_fraud": int(y.sum()), "base_rate": float(y.mean())}

    for score_col in ["subspace_if_score", "dense_block_score_ip", "hybrid_anomaly_score", "risk_score_v3"]:
        s = df[score_col].values
        stats[score_col] = {
            "pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            "mean_fraud": float(df.loc[y == 1, score_col].mean()),
            "mean_valid": float(df.loc[y == 0, score_col].mean()),
        }

    stats["per_fraud_type"] = {}
    neg = df[df["fraud_type"] == "NONE"]
    for ft in sorted(df["fraud_type"].unique()):
        if ft == "NONE":
            continue
        pos = df[df["fraud_type"] == ft]
        sub = pd.concat([pos, neg])
        yy = (sub["fraud_type"] == ft).astype(int).values
        stats["per_fraud_type"][ft] = {
            "n": int(len(pos)),
            "risk_score_v3_pr_auc": float(average_precision_score(yy, sub["risk_score_v3"].values)),
            "hybrid_only_pr_auc": float(average_precision_score(yy, sub["hybrid_anomaly_score"].values)),
            "subspace_only_pr_auc": float(average_precision_score(yy, sub["subspace_if_score"].values)),
            "dense_ip_only_pr_auc": float(average_precision_score(yy, sub["dense_block_score_ip"].values)),
        }

    df_sorted = df.sort_values("risk_score_v3", ascending=False).reset_index(drop=True)
    stats["top_n_coverage"] = {}
    for topn in [500, 1000, 2500, 5000, 7500]:
        top = df_sorted.head(topn)
        n_fraud_in_top = int(top["is_fraud"].sum())
        stats["top_n_coverage"][str(topn)] = {
            "n_fraud_captured": n_fraud_in_top,
            "recall": n_fraud_in_top / int(y.sum()),
            "precision": n_fraud_in_top / topn,
        }

    rank = df_sorted.reset_index().set_index("application_id")["index"]
    stats["ring_recall"] = {}
    for rt in ["IP_CLUSTER", "MOBILE_CLUSTER", "PINCODE_CLUSTER"]:
        sub = df[(df["fraud_type"] == rt) & (df["ring_id"] != "")]
        ring_stats = []
        for rid, g in sub.groupby("ring_id"):
            ranks = rank.loc[g["application_id"]].values
            ring_stats.append({"size": len(g), "min_rank": int(ranks.min()), "frac_in_top5000": float((ranks < 5000).mean())})
        if ring_stats:
            rs = pd.DataFrame(ring_stats)
            stats["ring_recall"][rt] = {
                "n_rings": len(rs),
                "median_min_rank": float(rs["min_rank"].median()),
                "mean_frac_ring_in_top5000": float(rs["frac_in_top5000"].mean()),
            }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[stress_analysis] Wrote stats -> {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
