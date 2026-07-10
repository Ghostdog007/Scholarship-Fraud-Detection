"""
fusion_classifier_v3.py

V4 FINAL — weighted SCORE-LEVEL fusion (locked). Replaces the 14-positive LightGBM,
which destroyed the raw detector signals (subspace INCOME 0.966->0.315, RGCN IP
0.51->0.169; see docs/AGENTS.md H.8).

risk_score_v3 = minmax( W_SUBSPACE*subspace_if_score
                      + W_DENSE_IP *dense_block_score_ip
                      + W_HYBRID   *hybrid_anomaly_score )

each component min-max normalised to [0,1] first. No labels used in the combine
(label-independent) — subspace is the tabular backbone, dense-block-IP the IP
specialist, RGCN the relational/topology signal. All scalar inputs only.

Inputs : outputs/hybrid_scores_v3.csv, outputs/subspace_if_scores_v3.csv,
         outputs/dense_block_scores_v3.csv
Writes : outputs/risk_scores_v3.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_v3 import (
    FUSION_W_SUBSPACE, FUSION_W_DENSE_IP, FUSION_W_HYBRID, DENSE_BLOCK_ENABLED,
)

HYBRID_CSV   = Path("outputs/hybrid_scores_v3.csv")
SUBSPACE_CSV = Path("outputs/subspace_if_scores_v3.csv")
DENSE_CSV    = Path("outputs/dense_block_scores_v3.csv")
LABELS_JSON  = Path("outputs/pseudo_labels_v3.json")
OUT_CSV      = Path("outputs/risk_scores_v3.csv")


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def score_level_fusion(
    subspace_if_score: np.ndarray,
    dense_block_score_ip: np.ndarray,
    hybrid_anomaly_score: np.ndarray,
) -> np.ndarray:
    """
    The LOCKED weighted score-level fusion (single source of truth).

    risk = minmax( W_SUBSPACE*minmax(subspace)
                 + W_DENSE_IP *minmax(dense_ip)
                 + W_HYBRID   *minmax(hybrid) )

    All inputs higher = more anomalous (hard stop #3). Label-independent: no
    learned gate can bury a strong raw signal. Used by both run_fusion (production)
    and compare_architectures_v3 (validation) so the two never drift.
    """
    s = _minmax(subspace_if_score)
    d = _minmax(dense_block_score_ip)
    h = _minmax(hybrid_anomaly_score)
    combined = FUSION_W_SUBSPACE * s + FUSION_W_DENSE_IP * d + FUSION_W_HYBRID * h
    return _minmax(combined)


def run_fusion() -> None:
    print("[fusion] score-level fusion (V4 final) ...")

    hybrid_df   = pd.read_csv(HYBRID_CSV)[["application_id", "hybrid_anomaly_score"]]
    subspace_df = pd.read_csv(SUBSPACE_CSV)[["application_id", "subspace_if_score"]]
    merged = hybrid_df.merge(subspace_df, on="application_id")

    if DENSE_BLOCK_ENABLED and DENSE_CSV.exists():
        dense_df = pd.read_csv(DENSE_CSV)
        ip_col = "dense_block_score_ip" if "dense_block_score_ip" in dense_df.columns else None
        if ip_col:
            merged = merged.merge(dense_df[["application_id", ip_col]], on="application_id", how="left")
            merged["dense_block_score_ip"] = merged["dense_block_score_ip"].fillna(0.0)
        else:
            merged["dense_block_score_ip"] = 0.0
    else:
        merged["dense_block_score_ip"] = 0.0
        print("[fusion] dense-block disabled/absent -> IP specialist contributes 0")

    risk = score_level_fusion(
        merged["subspace_if_score"].values,
        merged["dense_block_score_ip"].values,
        merged["hybrid_anomaly_score"].values,
    ).astype(np.float32)

    print(f"[fusion] weights: subspace={FUSION_W_SUBSPACE} dense_ip={FUSION_W_DENSE_IP} hybrid={FUSION_W_HYBRID}")

    # label_source is metadata only (not used in the combine)
    label_source = pd.Series("negative", index=merged.index)
    if LABELS_JSON.exists():
        labels_data = json.loads(LABELS_JSON.read_text())
        ls = {r["application_id"]: f"round_{r['round']}" for r in labels_data["positive_set"]}
        label_source = merged["application_id"].map(ls).fillna("negative")

    out_df = pd.DataFrame({
        "application_id": merged["application_id"].values,
        "risk_score_v3":  risk,
        "label_source":   label_source.values,
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[fusion] Saved {len(out_df)} risk scores -> {OUT_CSV}")
    print(f"[fusion] risk_score_v3 range: [{risk.min():.4f}, {risk.max():.4f}]")


if __name__ == "__main__":
    run_fusion()
