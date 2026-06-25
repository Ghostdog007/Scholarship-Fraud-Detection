"""
self_training_loop_v3.py

Promotes records to pseudo-positive labels using per-signal EVT tail agreement.

Round 0: UNION of 5 EVT tails (classifier condition CODE-ENFORCED OFF).
  A node is pseudo-positive if it clears ANY of:
    - hybrid_anomaly_score     >= hybrid EVT threshold       (general anomaly)
    - subspace_if_financial    >= financial EVT threshold    (income / fee fraud)
    - subspace_if_identity     >= identity EVT threshold     (name collision rings)
    - subspace_if_network      >= network EVT threshold      (IP concentration rings)
    - edge_pred_error          >= edge_pred_error EVT thresh (cross-channel fraud rings)

Round 1+: all EVT conditions still hold PLUS classifier agreement required.

HARD STOP: rounds do not advance automatically. Each round requires a Phase D
PR-AUC check by the project lead before its label set is used for training.

Writes: outputs/pseudo_labels_v3.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_v3 import RANDOM_SEED

np.random.seed(RANDOM_SEED)

HYBRID_CSV   = Path("outputs/hybrid_scores_v3.csv")
SUBSPACE_CSV = Path("outputs/subspace_if_scores_v3.csv")
RISK_CSV     = Path("outputs/risk_scores_v3.csv")
EVT_JSON     = Path("outputs/evt_thresholds_v3.json")
OUT_JSON     = Path("outputs/pseudo_labels_v3.json")

CLASSIFIER_CONFIDENCE_MIN = 0.6


def _extract_group_scores(subspace_df: pd.DataFrame, group: str) -> np.ndarray:
    return np.array([
        json.loads(row).get(group, 0.0)
        for row in subspace_df["group_scores_json"]
    ], dtype=np.float32)


def run_self_training(current_round: int = 0) -> None:
    print(f"[self_training] run_self_training() round={current_round} ...")

    hybrid_df   = pd.read_csv(HYBRID_CSV)
    subspace_df = pd.read_csv(SUBSPACE_CSV)
    thresholds  = json.loads(EVT_JSON.read_text())

    merged = hybrid_df.merge(subspace_df[["application_id", "subspace_if_score", "group_scores_json"]], on="application_id")

    # Expand group scores into columns
    for group in ("financial", "identity", "network"):
        merged[f"subspace_if_{group}"] = _extract_group_scores(subspace_df, group)

    # --- per-signal EVT flags ---
    signal_flags = {
        "EVT_HYBRID":     merged["hybrid_anomaly_score"] >= thresholds["hybrid"]["threshold"],
        "EVT_FINANCIAL":  merged["subspace_if_financial"] >= thresholds["subspace_if_financial"]["threshold"],
        "EVT_IDENTITY":   merged["subspace_if_identity"]  >= thresholds["subspace_if_identity"]["threshold"],
        "EVT_NETWORK":    merged["subspace_if_network"]   >= thresholds["subspace_if_network"]["threshold"],
        "EVT_EDGE_RING":  merged["edge_pred_error"]       >= thresholds["edge_pred_error"]["threshold"],
    }

    for name, flag in signal_flags.items():
        print(f"[self_training]   {name}: {flag.sum()} nodes above threshold")

    if current_round == 0:
        # Code-enforced: classifier agreement is OFF at round 0
        # Union of all EVT tails
        any_evt = signal_flags["EVT_HYBRID"]
        for flag in list(signal_flags.values())[1:]:
            any_evt = any_evt | flag
        promoted_mask = any_evt
        print(f"[self_training] Round 0: union of 5 EVT tails (classifier OFF by design)")
    else:
        if not RISK_CSV.exists():
            raise FileNotFoundError(
                f"Round {current_round} requires {RISK_CSV}. Run fusion_classifier_v3.py first."
            )
        risk_df = pd.read_csv(RISK_CSV)[["application_id", "risk_score_v3"]]
        merged  = merged.merge(risk_df, on="application_id")
        above_classifier = merged["risk_score_v3"] >= CLASSIFIER_CONFIDENCE_MIN

        any_evt = signal_flags["EVT_HYBRID"]
        for flag in list(signal_flags.values())[1:]:
            any_evt = any_evt | flag
        promoted_mask = any_evt & above_classifier
        print(f"[self_training] Round {current_round}: union EVT + classifier >= {CLASSIFIER_CONFIDENCE_MIN}")

    promoted_ids = merged.loc[promoted_mask, "application_id"].tolist()
    negative_ids = merged.loc[~promoted_mask, "application_id"].tolist()
    print(f"[self_training] Pseudo-positives: {len(promoted_ids)} | Negatives: {len(negative_ids)}")

    # Build positive records with per-signal trigger info
    positive_records = []
    for _, row in merged[promoted_mask].iterrows():
        triggers = [name for name, flag in signal_flags.items() if flag.loc[row.name]]
        rec = {
            "application_id":       row["application_id"],
            "round":                current_round,
            "trigger":              triggers,
            "hybrid_anomaly_score": float(round(row["hybrid_anomaly_score"], 6)),
            "subspace_if_score":    float(round(row["subspace_if_score"], 6)),
            "edge_pred_error":      float(round(row["edge_pred_error"], 6)),
            "subspace_if_financial": float(round(row["subspace_if_financial"], 6)),
            "subspace_if_identity":  float(round(row["subspace_if_identity"], 6)),
            "subspace_if_network":   float(round(row["subspace_if_network"], 6)),
        }
        if current_round > 0 and "risk_score_v3" in row:
            rec["classifier_confidence"] = float(round(row["risk_score_v3"], 6))
        positive_records.append(rec)

    # Breakdown by trigger type
    trigger_counts: dict[str, int] = {}
    for rec in positive_records:
        for t in rec["trigger"]:
            trigger_counts[t] = trigger_counts.get(t, 0) + 1
    print(f"[self_training] Trigger breakdown: {trigger_counts}")

    output = {
        "positive_set":   positive_records,
        "negative_ids":   negative_ids,
        "round":          current_round,
        "n_positive":     len(promoted_ids),
        "n_negative":     len(negative_ids),
        "trigger_counts": trigger_counts,
        "evt_thresholds": {
            k: thresholds[k]["threshold"] for k in thresholds
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"[self_training] Pseudo-labels saved -> {OUT_JSON}")

    if len(promoted_ids) < 10:
        print("[self_training] WARNING: fewer than 10 pseudo-positives -- EVT may be miscalibrated.")


if __name__ == "__main__":
    run_self_training(current_round=0)
