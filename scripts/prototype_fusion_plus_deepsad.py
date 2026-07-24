"""
Prototype: does adding Deep SAD's center_dist_score as a 4th input to the
locked max-fusion actually improve overall detection on stress_testing_1?

Current locked fusion (fusion_classifier_v3.py):
  risk = minmax( max( minmax(subspace), minmax(dense_relational), minmax(hybrid) ) )

Candidate:
  risk = minmax( max( minmax(subspace), minmax(dense_relational), minmax(hybrid),
                       minmax(center_dist_score) ) )

Pure re-scoring of already-computed columns (no retraining) -- CPU, instant.
Uses outputs/stress_testing_1_deepsad_full_scores.csv, which already carries
subspace_if_score, dense_block_score_relational, hybrid_anomaly_score,
center_dist_score, and ground truth (is_fraud/fraud_type) in one file.

Writes: outputs/stress_testing_1_fusion4_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

NAME = "stress_testing_1"
IN_CSV = Path(f"outputs/{NAME}_deepsad_full_scores.csv")
OUT_STATS = Path(f"outputs/{NAME}_fusion4_stats.json")


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    y = df["is_fraud"].astype(int).values

    s = _minmax(df["subspace_if_score"].values)
    d = _minmax(df["dense_block_score_relational"].values)
    h = _minmax(df["hybrid_anomaly_score"].values)
    c = _minmax(df["center_dist_score"].values)

    fusion3 = _minmax(np.maximum.reduce([s, d, h]))                 # current locked fusion
    fusion4 = _minmax(np.maximum.reduce([s, d, h, c]))               # + Deep SAD

    df["fusion3_score"] = fusion3
    df["fusion4_score"] = fusion4

    # Which detector actually drove fusion4, to see how often Deep SAD wins
    stacked = np.stack([s, d, h, c], axis=1)
    keys = ["subspace", "dense_relational", "hybrid", "deepsad"]
    driver4 = np.array(keys)[np.argmax(stacked, axis=1)]
    driver_counts = {k: int((driver4 == k).sum()) for k in keys}

    out_csv = Path(f"outputs/{NAME}_fusion4_full_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"[fusion4] Wrote {out_csv}")

    stats: dict = {
        "driver_win_counts_in_fusion4": driver_counts,
        "overall": {
            "fusion3 (LOCKED, current prod: max(subspace, dense_relational, hybrid))": float(average_precision_score(y, fusion3)),
            "fusion4 (candidate: + center_dist_score)": float(average_precision_score(y, fusion4)),
        },
    }

    neg = df[df["fraud_type"] == "NONE"]
    stats["per_fraud_type"] = {}
    for ft in sorted(df["fraud_type"].unique()):
        if ft == "NONE":
            continue
        pos = df[df["fraud_type"] == ft]
        sub_idx = pd.concat([pos, neg]).index
        yy = (df.loc[sub_idx, "fraud_type"] == ft).astype(int).values
        stats["per_fraud_type"][ft] = {
            "n": int(len(pos)),
            "fusion3": float(average_precision_score(yy, df.loc[sub_idx, "fusion3_score"].values)),
            "fusion4": float(average_precision_score(yy, df.loc[sub_idx, "fusion4_score"].values)),
        }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[fusion4] Wrote {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
