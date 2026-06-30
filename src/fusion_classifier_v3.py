"""
fusion_classifier_v3.py

LightGBM fusion: 2 anomaly scores -> 1 continuous risk score in [0, 1].
Inputs: hybrid_anomaly_score, subspace_if_score (scalar only -- no raw embeddings).
Labels: pseudo_labels_v3.json (EVT-confirmed + self-training promoted positives).
Writes: outputs/risk_scores_v3.csv
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config_v3 import CONFIRMED_WEIGHT, RANDOM_SEED

HYBRID_CSV   = Path("outputs/hybrid_scores_v3.csv")
SUBSPACE_CSV = Path("outputs/subspace_if_scores_v3.csv")
LABELS_JSON  = Path("outputs/pseudo_labels_v3.json")
OUT_CSV      = Path("outputs/risk_scores_v3.csv")

FEATURE_COLS = ["hybrid_anomaly_score", "feature_pred_error", "edge_pred_error", "subspace_if_score"]

LGB_PARAMS = {
    "objective":       "binary",
    "metric":          "average_precision",
    "num_leaves":      31,
    "learning_rate":   0.05,
    "n_estimators":    200,
    "min_child_samples": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "random_state":    RANDOM_SEED,
    "verbose":         -1,
}


def run_fusion() -> None:
    print("[fusion] run_fusion() starting ...")

    hybrid_df   = pd.read_csv(HYBRID_CSV)
    subspace_df = pd.read_csv(SUBSPACE_CSV)[["application_id", "subspace_if_score"]]
    labels_data = json.loads(LABELS_JSON.read_text())

    merged = hybrid_df.merge(subspace_df, on="application_id")

    positive_ids   = {r["application_id"] for r in labels_data["positive_set"]}
    # source field: "confirmed" = hard label from supervisor, "evt_pseudo" = EVT-promoted
    source_map     = {r["application_id"]: r.get("source", "evt_pseudo") for r in labels_data["positive_set"]}
    label_source_map = {r["application_id"]: f"round_{r['round']}" for r in labels_data["positive_set"]}

    merged["label"]        = merged["application_id"].isin(positive_ids).astype(int)
    merged["label_source"] = merged["application_id"].map(label_source_map).fillna("negative")
    merged["label_src"]    = merged["application_id"].map(source_map).fillna("negative")

    # Sample weights: confirmed fraud = CONFIRMED_WEIGHT, pseudo = 1.0, negative = 1.0
    # False positives confirmed by supervisor carry weight 0 (excluded via label=0 already)
    merged["sample_weight"] = merged["label_src"].map({
        "confirmed":  CONFIRMED_WEIGHT,
        "evt_pseudo": 1.0,
        "negative":   1.0,
    }).fillna(1.0)

    n_pos       = merged["label"].sum()
    n_confirmed = (merged["label_src"] == "confirmed").sum()
    n_pseudo    = (merged["label_src"] == "evt_pseudo").sum()
    n_neg       = (merged["label"] == 0).sum()
    print(f"[fusion] Positives: {n_pos} (confirmed={n_confirmed} pseudo={n_pseudo}) | Negatives: {n_neg}")
    print(f"[fusion] Confirmed fraud sample weight: {CONFIRMED_WEIGHT}×")

    if n_pos < 5:
        print("[fusion] WARNING: fewer than 5 positives -- classifier will be unreliable. Run self_training first.")

    available_cols = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[available_cols].values
    y = merged["label"].values

    # Cross-validated predict_proba to avoid train-set overfitting
    risk_scores = np.zeros(len(merged), dtype=np.float32)

    sample_weights = merged["sample_weight"].values

    if n_pos >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            clf.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
            risk_scores[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
            print(f"[fusion]   fold {fold+1}/5 done")
    else:
        # Not enough positives: use raw hybrid score as fallback
        print("[fusion] Fallback: using hybrid_anomaly_score as risk_score_v3")
        hybrid_col = merged["hybrid_anomaly_score"].values
        lo, hi = hybrid_col.min(), hybrid_col.max()
        risk_scores = ((hybrid_col - lo) / (hi - lo + 1e-8)).astype(np.float32)

    out_df = pd.DataFrame({
        "application_id": merged["application_id"].values,
        "risk_score_v3":  risk_scores,
        "label_source":   merged["label_source"].values,
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[fusion] Saved {len(out_df)} risk scores -> {OUT_CSV}")
    print(f"[fusion] risk_score_v3 range: [{risk_scores.min():.4f}, {risk_scores.max():.4f}]")


if __name__ == "__main__":
    run_fusion()
