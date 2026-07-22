"""
fusion_classifier_v3.py

V4.1 FINAL — max SCORE-LEVEL fusion (locked, changed 2026-07-22). Originally an
additive weighted-sum; replaced with an unweighted max after the stress_testing_1
ablation showed the sum diluting whichever detector actually found the fraud with
near-random noise from the other two on every category tested (e.g. mobile-ring:
subspace alone 0.674 PR-AUC vs the old weighted-sum fused 0.349). Overall PR-AUC on
that ablation: 0.403 (weighted-sum) -> 0.447 (max). See README changelog and
outputs/stress_testing_1_v2_stats.json / _v2b_stats.json for the full numbers.
The original LightGBM replacement rationale (docs/AGENTS.md H.8: destroyed subspace
INCOME 0.966->0.315, RGCN IP 0.51->0.169) still holds for why the combiner is not
LEARNED — that risk is about fitting weights to sparse labels, not about which
fixed combination FUNCTION (sum vs max) is used.

risk_score_v3 = minmax( max( minmax(subspace_if_score),
                             minmax(dense_block_score_relational),
                             minmax(hybrid_anomaly_score) ) )

each component min-max normalised to [0,1] first, THEN maxed — no weights: max has
no per-component weight to tune, it takes whichever raw signal is strongest for
that application. No labels used in the combine (label-independent) — subspace is
the tabular backbone, dense-block-relational the mobile/IP/pincode structural
specialist (IP-priority weighted internally, see dense_block_detector_v3), RGCN the
relational/topology signal. All scalar inputs only.

Inputs : outputs/hybrid_scores_v3.csv, outputs/subspace_if_scores_v3.csv,
         outputs/dense_block_scores_v3.csv
Writes : outputs/risk_scores_v3.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_v3 import DENSE_BLOCK_ENABLED

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
    dense_block_score_relational: np.ndarray,
    hybrid_anomaly_score: np.ndarray,
) -> np.ndarray:
    """
    The LOCKED max score-level fusion (single source of truth).

    risk = minmax( max( minmax(subspace), minmax(dense_relational), minmax(hybrid) ) )

    All inputs higher = more anomalous (hard stop #3). Label-independent: no
    learned gate can bury a strong raw signal — max is stricter about this than
    the old sum was, since it preserves the single strongest detector's value
    exactly regardless of the other two. Used by both run_fusion (production)
    and compare_architectures_v3 (validation) so the two never drift.
    """
    s = _minmax(subspace_if_score)
    d = _minmax(dense_block_score_relational)
    h = _minmax(hybrid_anomaly_score)
    combined = np.maximum.reduce([s, d, h])
    return _minmax(combined)


def run_fusion() -> None:
    print("[fusion] score-level fusion (V4 final) ...")

    hybrid_df   = pd.read_csv(HYBRID_CSV)[["application_id", "hybrid_anomaly_score"]]
    subspace_df = pd.read_csv(SUBSPACE_CSV)[["application_id", "subspace_if_score"]]
    merged = hybrid_df.merge(subspace_df, on="application_id")

    if DENSE_BLOCK_ENABLED and DENSE_CSV.exists():
        dense_df = pd.read_csv(DENSE_CSV)
        rel_col = "dense_block_score_relational" if "dense_block_score_relational" in dense_df.columns else None
        if rel_col:
            merged = merged.merge(dense_df[["application_id", rel_col]], on="application_id", how="left")
            merged["dense_block_score_relational"] = merged["dense_block_score_relational"].fillna(0.0)
        else:
            merged["dense_block_score_relational"] = 0.0
    else:
        merged["dense_block_score_relational"] = 0.0
        print("[fusion] dense-block disabled/absent -> relational specialist contributes 0")

    risk = score_level_fusion(
        merged["subspace_if_score"].values,
        merged["dense_block_score_relational"].values,
        merged["hybrid_anomaly_score"].values,
    ).astype(np.float32)

    print("[fusion] combine: max(minmax(subspace), minmax(dense_relational), minmax(hybrid))")

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
