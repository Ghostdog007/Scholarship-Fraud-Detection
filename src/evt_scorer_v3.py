"""
evt_scorer_v3.py

Fits a Generalized Pareto Distribution (GPD) to the extreme right tail of each
anomaly score distribution using Peak Over Threshold (POT).
Derives statistically-grounded thresholds -- no domain-set numeric cutoffs.

Scores fitted:
  - hybrid_anomaly_score       (general anomalies)
  - subspace_if_score          (aggregate IF)
  - subspace_if_financial      (income / fee fraud)
  - subspace_if_identity       (name collision rings)
  - subspace_if_network        (IP concentration rings)
  - edge_pred_error            (cross-channel fraud rings)

Writes: outputs/evt_thresholds_v3.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from src.config_v3 import EVT_SHAPE_MAX, EVT_SHAPE_MIN, RANDOM_SEED

np.random.seed(RANDOM_SEED)

HYBRID_CSV   = Path("outputs/hybrid_scores_v3.csv")
SUBSPACE_CSV = Path("outputs/subspace_if_scores_v3.csv")
OUT_JSON     = Path("outputs/evt_thresholds_v3.json")

Q            = 0.002   # false-positive rate (only human-set value in this module)
U_PERCENTILE = 95      # POT: default initial threshold percentile

# Per-signal overrides — identity IF flags too many noise cases at 95th percentile
U_PERCENTILE_OVERRIDES: dict[str, int] = {
    "subspace_if_identity": 97,
}


def _fit_evt(scores: np.ndarray, score_name: str, u_percentile: int = U_PERCENTILE) -> dict:
    # Jitter: smooth discrete cluster spikes (e.g. identical IP-cluster scores)
    # before fitting GPD. Sigma=0.001 is negligible relative to score range [0,1]
    # but breaks exact ties that violate GPD smoothness assumptions.
    rng    = np.random.default_rng(seed=42)
    scores = scores + rng.normal(0, 0.001, size=scores.shape)
    scores = np.clip(scores, 0.0, None)

    u            = np.percentile(scores, u_percentile)
    exceedances  = scores[scores > u] - u

    # Discreteness warning: if >50% of tail values are identical, distribution
    # likely has a spike that will produce a poorly conditioned GPD fit.
    if len(exceedances) > 0:
        top_val_freq = (exceedances == exceedances.max()).mean()
        if top_val_freq > 0.5:
            print(f"[evt] WARNING: '{score_name}' tail is highly discrete "
                  f"({top_val_freq:.0%} of exceedances share the max value) — GPD fit may be unreliable")

    if len(exceedances) < 10:
        print(f"[evt] WARNING: too few exceedances ({len(exceedances)}) for '{score_name}' -- percentile fallback")
        threshold = np.percentile(scores, 100 * (1 - Q))
        return {
            "u": float(u), "scale": None, "shape": None,
            "threshold": float(threshold), "method": "percentile_fallback",
            "n_exceedances": len(exceedances),
        }

    shape, _loc, scale = genpareto.fit(exceedances, floc=0)

    # Shape validity check: ξ outside [-0.5, 1.0] indicates the distribution
    # violates GPD regularity assumptions. Fall back to empirical quantile.
    if not (EVT_SHAPE_MIN <= shape <= EVT_SHAPE_MAX):
        print(f"[evt] WARNING: '{score_name}' GPD shape={shape:.4f} outside valid range "
              f"[{EVT_SHAPE_MIN}, {EVT_SHAPE_MAX}] — falling back to empirical quantile")
        threshold = np.percentile(scores, 100 * (1 - Q))
        return {
            "u": float(u), "scale": float(scale), "shape": float(shape),
            "threshold": float(threshold), "method": "empirical_fallback_bad_shape",
            "n_exceedances": int(len(exceedances)),
        }

    n_total  = len(scores)
    n_exceed = len(exceedances)
    if abs(shape) < 1e-6:
        threshold = u + scale * np.log(n_total * Q / n_exceed)
    else:
        threshold = u + (scale / shape) * ((n_total * Q / n_exceed) ** (-shape) - 1)

    threshold = float(np.clip(threshold, scores.min(), scores.max()))
    n_flagged = int((scores >= threshold).sum())
    print(f"[evt]   '{score_name}': u={u:.4f}, scale={scale:.4f}, shape={shape:.4f}, "
          f"threshold={threshold:.4f}, flagged={n_flagged}")

    return {
        "u": float(u), "scale": float(scale), "shape": float(shape),
        "threshold": threshold, "method": "POT-GPD",
        "n_exceedances": int(n_exceed), "n_flagged": n_flagged, "q": Q,
    }


def _extract_group_scores(subspace_df: pd.DataFrame, group: str) -> np.ndarray:
    """Parse group_scores_json column and return the score for one group."""
    return np.array([
        json.loads(row).get(group, 0.0)
        for row in subspace_df["group_scores_json"]
    ], dtype=np.float64)


def run_evt() -> None:
    print("[evt] run_evt() starting ...")

    hybrid_df   = pd.read_csv(HYBRID_CSV)
    subspace_df = pd.read_csv(SUBSPACE_CSV)

    thresholds = {}

    # --- existing scores ---
    thresholds["hybrid"] = _fit_evt(
        hybrid_df["hybrid_anomaly_score"].values.astype(np.float64),
        "hybrid_anomaly_score",
    )
    thresholds["subspace_if"] = _fit_evt(
        subspace_df["subspace_if_score"].values.astype(np.float64),
        "subspace_if_score",
    )

    # --- per-group subspace IF (new) ---
    for group in ("financial", "identity", "network"):
        scores    = _extract_group_scores(subspace_df, group)
        key       = f"subspace_if_{group}"
        u_pct     = U_PERCENTILE_OVERRIDES.get(key, U_PERCENTILE)
        thresholds[key] = _fit_evt(scores, key, u_percentile=u_pct)

    # --- edge prediction error (new) ---
    thresholds["edge_pred_error"] = _fit_evt(
        hybrid_df["edge_pred_error"].values.astype(np.float64),
        "edge_pred_error",
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(thresholds, indent=2))
    print(f"[evt] Thresholds saved -> {OUT_JSON}")
    print(f"[evt] Total signals fitted: {len(thresholds)}")


if __name__ == "__main__":
    run_evt()
