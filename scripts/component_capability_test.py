"""
component_capability_test.py — diagnostic: does each detector actually detect
what it was DESIGNED to detect, on data none of them trained on?

Not a gate, not a pipeline step. This is an investigation script (lives in
scripts/, matching generate_stress_test_dataset.py / stress_test_1_analysis.py
convention — one-off diagnostics, not maintained pipeline modules).

Uses the held-out bundle already built by src/build_held_out_set.py
(outputs/held_out/v3_44/ — the stress_testing_1 ground-truth cohort, merged
into the real population, features/graph rebuilt at the CURRENT schema).
None of the scored detectors trained on this bundle: subspace IF and
dense-block are refit/recomputed on it fresh (no persisted model), and the
hybrid checkpoint was trained only on the real 15k population.

Method: for every (signal, category) pair, compute PR-AUC treating that
category's ground-truth rows as positives and the rest of the 65k population
as negatives (same convention as evaluate_model_v3.evaluate_connected() and
deploy_gate.py). The FULL cross table (every signal x every category), not
just the "expected" pairing, is what actually demonstrates specialization —
a detector that scores well on ITS OWN category AND poorly on others is
evidence of genuine specialization, not a detector that just fires on
everything (which would score well everywhere and prove nothing).

Signals tested:
  dense_block_score_mobile, dense_block_score_ip   (deterministic, no seed)
  subspace_financial, subspace_identity,
  subspace_network, subspace_demographic            (IsolationForest, reseeded)
  hybrid_feature_pred_error, hybrid_edge_pred_error,
  hybrid_anomaly_score                              (frozen checkpoint, GPU)

3 seeds (42, 43, 44 — config_v3.COMPARE_SEEDS convention): subspace IF is
refit per seed (its only source of run-to-run variance — dense-block peeling
is deterministic, and the hybrid checkpoint's weights are frozen, so its
seed only re-seeds torch's RNG for hygiene and exposes the documented GPU
scatter-add noise floor, not training variance).

Writes: outputs/component_capability_test.json
Run: .venv/Scripts/python.exe scripts/component_capability_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

BUNDLE_DIR = Path("outputs/held_out/v3_44")
OUT_JSON   = Path("outputs/component_capability_test.json")
LIVE_CKPT  = Path("models/hybrid_graphmcm_v3.pth")

SEEDS = (42, 43, 44)
MIN_POSITIVES = 5

# Mirrors config_v3.SUBSPACE_GROUPS + evaluate_model_v3.EVAL_EXTRA_GROUPS
# (demographic is eval-only — not in the production pipeline's fusion input,
# but it's the correct specialist group to test AGE_VIOLATION against, same
# as evaluate_model_v3 already does for injected-anomaly evaluation).
SUBSPACE_GROUPS_TEST = {
    "financial": [
        "annual_family_income", "fee_income_ratio", "income_rank_in_district",
        "income_deviation_from_state_median", "admission_fee", "tution_fee", "misc_fee",
    ],
    "identity": [
        "name_similarity_score", "is_father_name_eq_mother",
        "is_applicant_name_eq_father", "is_applicant_name_eq_mother",
        "mobile_unique_names", "mobile_unique_fathers",
    ],
    "network": [
        "ip_application_count", "ip_to_mobile_ratio", "mobile_application_count",
        "institute_application_count", "degree_shares_ip", "degree_shares_mobile",
        "degree_shares_pincode",
    ],
    "demographic": [
        "age_at_registration", "competitive_exam_year", "admission_year", "c_course_year",
    ],
}

# What each signal was DESIGNED for (AGENTS.md §1 / config_v3 comments) — used
# only to mark the "expected" cell in the printed table, not to filter anything.
EXPECTED = {
    "dense_block_score_mobile": {"MOBILE_CLUSTER"},
    "dense_block_score_ip":     {"IP_CLUSTER"},
    "subspace_financial":       {"INCOME_VIOLATION", "FEE_INFLATION"},
    "subspace_identity":        {"MOTHER_NAME_COLLISION"},
    "subspace_network":         {"IP_CLUSTER", "MOBILE_CLUSTER", "PINCODE_CLUSTER"},
    "subspace_demographic":     {"AGE_VIOLATION"},
    "hybrid_feature_pred_error": set(),   # general anomaly, no single specialty
    "hybrid_edge_pred_error":   {"IP_CLUSTER", "MOBILE_CLUSTER", "PINCODE_CLUSTER"},
    "hybrid_anomaly_score":     set(),
}


def _load_bundle():
    feat_df = pd.read_csv(BUNDLE_DIR / "features.csv")
    schema  = json.loads((BUNDLE_DIR / "schema.json").read_text())
    graph   = torch.load(BUNDLE_DIR / "graph.pt", weights_only=False)
    gt      = pd.read_csv(BUNDLE_DIR / "ground_truth.csv")
    return feat_df, schema, graph, gt


def _subspace_group_score(feat_df: pd.DataFrame, cols: list[str], seed: int) -> np.ndarray:
    available = [c for c in cols if c in feat_df.columns]
    if not available:
        return np.zeros(len(feat_df), dtype=np.float32)
    X = feat_df[available].values
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed)
    clf.fit(X)
    raw = -clf.decision_function(X)
    lo, hi = raw.min(), raw.max()
    return ((raw - lo) / (hi - lo + 1e-9)).astype(np.float32)


def _dense_block(feat_df, graph) -> pd.DataFrame:
    from src.hybrid_graphmcm_v3 import _build_edge_index_and_types
    from src.interfaces.dense_block import dense_block_scores

    device = torch.device("cpu")  # dense-block peeling is CPU-only (no torch ops)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(graph, device)
    return dense_block_scores(edge_index_list, edge_type_tensor, len(feat_df), feat_df["application_id"].values)


def _hybrid_score(feat_df, schema, graph, seed: int) -> pd.DataFrame:
    from src.hybrid_graphmcm_v3 import (
        HybridGraphMCM, _build_edge_index_and_types, _compute_isolated_mask, compute_score_frame,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

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
        return compute_score_frame(model, x_all, edge_index_list, edge_type_tensor,
                                    isolated_mask, app_ids, feat_cols)


def _pr_auc_per_category(scores: pd.Series, gt: pd.DataFrame) -> dict[str, float]:
    results = {}
    for category, ids in gt[gt["fraud_type"] != "NONE"].groupby("fraud_type")["application_id"]:
        pos_ids = set(ids)
        if len(pos_ids) < MIN_POSITIVES:
            continue
        labels = scores.index.isin(pos_ids).astype(int)
        if labels.sum() < MIN_POSITIVES:
            continue
        results[category] = float(average_precision_score(labels, scores.values))
    return results


def main() -> None:
    if not (BUNDLE_DIR / "manifest.json").exists():
        raise FileNotFoundError(f"No held-out bundle at {BUNDLE_DIR} — run src.build_held_out_set first.")

    feat_df, schema, graph, gt = _load_bundle()
    print(f"[capability_test] Bundle: {len(feat_df)} rows, {gt['application_id'].nunique()} held-out with ground truth")
    print(f"[capability_test] Device for hybrid scoring: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Dense-block: deterministic, computed once (no seed dependence — Charikar
    # peeling has no randomness; documented in dense_block_detector_v3.py).
    dense_df = _dense_block(feat_df, graph)
    dense_df = dense_df.set_index("application_id")

    per_seed_results: dict[int, dict[str, dict[str, float]]] = {}

    for seed in SEEDS:
        print(f"\n[capability_test] === seed={seed} ===")
        signals: dict[str, pd.Series] = {}

        for group_name, cols in SUBSPACE_GROUPS_TEST.items():
            arr = _subspace_group_score(feat_df, cols, seed)
            signals[f"subspace_{group_name}"] = pd.Series(arr, index=feat_df["application_id"].values)

        signals["dense_block_score_mobile"] = dense_df["dense_block_score_mobile"]
        signals["dense_block_score_ip"]     = dense_df["dense_block_score_ip"]

        hybrid_df = _hybrid_score(feat_df, schema, graph, seed).set_index("application_id")
        signals["hybrid_feature_pred_error"] = hybrid_df["feature_pred_error"]
        signals["hybrid_edge_pred_error"]    = hybrid_df["edge_pred_error"]
        signals["hybrid_anomaly_score"]      = hybrid_df["hybrid_anomaly_score"]

        seed_results = {}
        for signal_name, series in signals.items():
            seed_results[signal_name] = _pr_auc_per_category(series, gt)
        per_seed_results[seed] = seed_results

    # Aggregate mean +- std across seeds
    all_signals = list(per_seed_results[SEEDS[0]].keys())
    all_categories = sorted({c for s in per_seed_results.values() for sig in s.values() for c in sig})

    summary: dict[str, dict[str, dict[str, float]]] = {}
    print(f"\n{'='*100}")
    print("SPECIALIZATION TABLE — PR-AUC mean (std) across 3 seeds. '*' = designed specialty for that category.")
    print(f"{'='*100}")
    header = f"{'signal':<28}" + "".join(f"{c:<22}" for c in all_categories)
    print(header)
    for signal_name in all_signals:
        summary[signal_name] = {}
        row = f"{signal_name:<28}"
        for category in all_categories:
            vals = [per_seed_results[s][signal_name].get(category) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if not vals:
                row += f"{'--':<22}"
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[signal_name][category] = {"mean": mean, "std": std}
            star = "*" if category in EXPECTED.get(signal_name, set()) else " "
            row += f"{mean:.3f}({std:.3f}){star:<8}"
        print(row)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "seeds": list(SEEDS),
        "bundle": json.loads((BUNDLE_DIR / "manifest.json").read_text()),
        "expected_specialties": {k: sorted(v) for k, v in EXPECTED.items()},
        "results_mean_std": summary,
        "per_seed_raw": per_seed_results,
    }, indent=2))
    print(f"\n[capability_test] Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
