"""
subspace_if_v3.py

Fits one IsolationForest per feature group (financial / identity / network).
No training data required — IF fits on the real data directly.
Scores are anomaly scores: higher = more anomalous (consistent with V3 convention).
Writes: outputs/subspace_if_scores_v3.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config_v3 import RANDOM_SEED, SUBSPACE_GROUPS

FINAL_CSV   = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON = Path("data/processed/v3_feature_schema.json")
OUT_CSV     = Path("outputs/subspace_if_scores_v3.csv")


def compute_subspace_if_scores(df: pd.DataFrame, features: set[str] | None = None) -> pd.DataFrame:
    """
    Pure scoring function — single source of truth, refit on whatever `df`
    is passed (the canonical population for run_subspace_if(), or a merged
    base+cohort population for a read-only cohort preview; see
    src/api/handlers/monitoring.py::evaluate_dataset()). No file I/O.

    Returns: application_id, subspace_if_score, group_scores_json.
    """
    if features is None:
        schema = json.loads(SCHEMA_JSON.read_text())
        features = set(schema["features"])

    group_score_arrays = {}
    for group_name, group_cols in SUBSPACE_GROUPS.items():
        available = [c for c in group_cols if c in df.columns and c in features]
        if not available:
            print(f"[subspace_if] WARNING: no available columns for group '{group_name}' -- skipping")
            group_score_arrays[group_name] = np.zeros(len(df))
            continue

        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            print(f"[subspace_if] WARNING: group '{group_name}' missing cols: {missing}")

        X = df[available].values
        clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_SEED)
        clf.fit(X)

        # decision_function returns higher = more normal; negate so higher = more anomalous
        raw_scores = clf.decision_function(X)
        anomaly_scores = -raw_scores

        # MinMax scale to [0, 1]
        lo, hi = anomaly_scores.min(), anomaly_scores.max()
        if hi > lo:
            anomaly_scores = (anomaly_scores - lo) / (hi - lo)
        else:
            anomaly_scores = np.zeros_like(anomaly_scores)

        group_score_arrays[group_name] = anomaly_scores
        print(f"[subspace_if]   group '{group_name}': {len(available)} features, score range [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")

    # Aggregate: max across groups (captures the worst-offending group)
    stacked = np.stack(list(group_score_arrays.values()), axis=1)
    subspace_if_score = stacked.max(axis=1)

    # Per-row JSON with each group's score
    group_scores_json = [
        json.dumps({g: float(round(group_score_arrays[g][i], 6)) for g in group_score_arrays})
        for i in range(len(df))
    ]

    return pd.DataFrame({
        "application_id":    df["application_id"].values,
        "subspace_if_score": subspace_if_score.astype(np.float32),
        "group_scores_json": group_scores_json,
    })


def run_subspace_if() -> None:
    print("[subspace_if] run_subspace_if() starting ...")
    df = pd.read_csv(FINAL_CSV)
    out_df = compute_subspace_if_scores(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[subspace_if] Saved {len(out_df)} scores -> {OUT_CSV}")
    print(f"[subspace_if] Score range: [{out_df['subspace_if_score'].min():.4f}, {out_df['subspace_if_score'].max():.4f}]")


if __name__ == "__main__":
    run_subspace_if()
