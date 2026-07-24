"""
ablation_lambda_edge.py — diagnostic: does dropping edge_pred_error from
hybrid_anomaly_score (LAMBDA_EDGE effectively 0) improve or hurt the MCM's
per-category detection, across ALL 7 held-out categories (not just the 3
relational ring types checked earlier)?

PROPOSED, PENDING — not adopted. LAMBDA_EDGE and the hybrid formula are
locked (AGENTS.md Sec 1); this script only measures, it does not change
config_v3.py or hybrid_graphmcm_v3.py. Any real change requires the Recipe 5
ablation process (MAINTAINER_PLAYBOOK.md) plus explicit lead sign-off
(AGENTS.md Sec 6 -- locked hyperparameter).

Uses the same held-out bundle as component_capability_test.py
(outputs/held_out/v3_44/). Live checkpoint is frozen (no retraining) --
this tests SCORE COMPOSITION only, not whether removing the edge-prediction
training objective would change embedding quality (a separate, untested
question).

  hybrid_anomaly_score (current) = minmax(feature_pred_error_raw + 0.3 * edge_pred_error_raw)
  feature_pred_error (candidate, LAMBDA_EDGE=0) = minmax(feature_pred_error_raw)
  -- these are mathematically identical to the "feature_pred_error" column
  compute_score_frame() already returns, since at LAMBDA_EDGE=0 the hybrid
  raw sum reduces to feature_pred_error_raw exactly.

Writes: outputs/ablation_lambda_edge.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score

LIVE_CKPT  = Path("models/hybrid_graphmcm_v3.pth")
MIN_POSITIVES = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path("outputs/held_out/v3_44"))
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    BUNDLE_DIR = args.bundle_dir
    OUT_JSON = args.out_json or Path(f"outputs/ablation_lambda_edge_{BUNDLE_DIR.name}.json")

    from src.hybrid_graphmcm_v3 import (
        HybridGraphMCM, _build_edge_index_and_types, _compute_isolated_mask, compute_score_frame,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_df = pd.read_csv(BUNDLE_DIR / "features.csv")
    schema  = json.loads((BUNDLE_DIR / "schema.json").read_text())
    graph   = torch.load(BUNDLE_DIR / "graph.pt", weights_only=False)
    gt      = pd.read_csv(BUNDLE_DIR / "ground_truth.csv")
    feat_cols = schema["features"]

    x_all = torch.tensor(feat_df[feat_cols].values, dtype=torch.float32).to(device)
    app_ids = feat_df["application_id"].values
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(graph, device)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], device)

    ckpt = torch.load(LIVE_CKPT, weights_only=False, map_location=device)
    model = HybridGraphMCM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.centroid = ckpt["centroid"].to(device)
    model.eval()

    with torch.no_grad():
        sdf = compute_score_frame(model, x_all, edge_index_list, edge_type_tensor,
                                   isolated_mask, app_ids, feat_cols).set_index("application_id")

    print(f"{'category':<25}{'n':>6}  {'current (feat+0.3*edge)':>26}  {'candidate (feat only)':>24}  {'delta':>10}")
    results = {}
    for category, ids in gt[gt["fraud_type"] != "NONE"].groupby("fraud_type")["application_id"]:
        pos_ids = set(ids)
        if len(pos_ids) < MIN_POSITIVES:
            continue
        labels = sdf.index.isin(pos_ids).astype(int)
        if labels.sum() < MIN_POSITIVES:
            continue
        pr_current   = average_precision_score(labels, sdf["hybrid_anomaly_score"].values)
        pr_candidate = average_precision_score(labels, sdf["feature_pred_error"].values)
        delta = pr_candidate - pr_current
        results[category] = {"n": int(labels.sum()), "current": pr_current, "candidate": pr_candidate, "delta": delta}
        print(f"{category:<25}{int(labels.sum()):>6}  {pr_current:>26.4f}  {pr_candidate:>24.4f}  {delta:>+10.4f}")

    mean_current   = sum(v["current"] for v in results.values()) / len(results)
    mean_candidate = sum(v["candidate"] for v in results.values()) / len(results)
    print(f"\n{'MEAN':<25}{'':>6}  {mean_current:>26.4f}  {mean_candidate:>24.4f}  {mean_candidate-mean_current:>+10.4f}")

    OUT_JSON.write_text(json.dumps({
        "status": "PROPOSED, PENDING -- not adopted, LAMBDA_EDGE is locked",
        "bundle": json.loads((BUNDLE_DIR / "manifest.json").read_text()),
        "per_category": results,
        "mean_current_hybrid_anomaly_score": mean_current,
        "mean_candidate_feature_pred_error_only": mean_candidate,
    }, indent=2))
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
