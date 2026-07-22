"""
deploy_gate.py — manual pre-promotion quality gate for a candidate hybrid
checkpoint. Run this BEFORE src.checkpoint_manager.validate_and_hotswap().

checkpoint_manager already refuses a checkpoint with the wrong SHAPE
(N_FEATURES/GRAPH_EMB_DIM/N_EDGE_TYPES/ARCH_VERSION mismatch — hard stop 9).
That's structural, not a quality check: a shape-valid checkpoint that is
WORSE than what's live would sail through it. This script is that missing
quality check, run as a separate precondition so the atomic-swap code path
in checkpoint_manager stays untouched.

Fail-closed by design:
  - No held-out bundle for the current schema_version  -> FAIL (do not score
    blind against a stale or mismatched bundle; see MAINTAINER_PLAYBOOK.md
    Recipe 1/3 and src/build_held_out_set.py).
  - Candidate checkpoint fails structural validation     -> FAIL.
  - Candidate regresses beyond the documented noise floor
    (+-0.03-0.04, AGENTS.md/CLAUDE.md quantitative claims protocol) on ANY
    held-out fraud category vs. the CURRENTLY LIVE checkpoint, freshly
    re-measured in the same run (never a cited historical number) -> FAIL.

Every run (pass or fail) is logged to model_registry.json (run_type=
"deploy_gate") with the candidate/live checkpoint refs, schema_version,
held-out bundle version, git commit, per-category metrics for both, and the
verdict — so there's an audit trail of what was ever allowed to promote.

Exit code 0 only on PASS. On PASS, the operator still has to run
checkpoint_manager.validate_and_hotswap() themselves — this script does not
call it. Two separate, composable steps.

Run:
  .venv/Scripts/python.exe -m src.deploy_gate --candidate models/incoming_<ts>_<uuid>.pth --cycle 2026H2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.config_v3 import N_FEATURES, GRAPH_EMB_DIM, N_EDGE_TYPES, ARCH_VERSION

HELD_OUT_ROOT = Path("outputs/held_out")
LIVE_CKPT     = Path("models/hybrid_graphmcm_v3.pth")

# Documented GPU scatter-add noise floor (AGENTS.md §5 / CLAUDE.md quantitative
# claims protocol) — a regression smaller than this is not distinguishable
# from noise and does not fail the gate on its own.
NOISE_FLOOR = 0.04
MIN_POSITIVES_FOR_CATEGORY = 5

_REQUIRED_KEYS   = {"model_state_dict", "centroid", "config"}
_REQUIRED_CONFIG = {"N_FEATURES", "GRAPH_EMB_DIM", "N_EDGE_TYPES", "ARCH_VERSION"}


def _schema_version() -> str:
    return f"v3_{N_FEATURES}"


def _validate_shape(ckpt: dict, path: Path) -> None:
    """Same structural contract as checkpoint_manager._validate (hard stop 9),
    re-checked here so a shape-invalid candidate never even reaches scoring."""
    missing = _REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        raise ValueError(f"{path.name}: missing required keys {missing}")
    cfg = ckpt["config"]
    missing_cfg = _REQUIRED_CONFIG - set(cfg.keys())
    if missing_cfg:
        raise ValueError(f"{path.name}: config missing keys {missing_cfg}")
    for key, expected in (("N_FEATURES", N_FEATURES), ("GRAPH_EMB_DIM", GRAPH_EMB_DIM),
                           ("N_EDGE_TYPES", N_EDGE_TYPES), ("ARCH_VERSION", ARCH_VERSION)):
        if cfg[key] != expected:
            raise ValueError(f"{path.name}: {key} mismatch (checkpoint={cfg[key]}, expected={expected})")


def _load_bundle(bundle_dir: Path) -> dict:
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    feat_df  = pd.read_csv(bundle_dir / "features.csv")
    schema   = json.loads((bundle_dir / "schema.json").read_text())
    graph    = torch.load(bundle_dir / "graph.pt", weights_only=False)
    subspace = pd.read_csv(bundle_dir / "subspace_scores.csv")
    dense    = pd.read_csv(bundle_dir / "dense_scores.csv")
    gt       = pd.read_csv(bundle_dir / "ground_truth.csv")
    return {
        "manifest": manifest, "features": feat_df, "schema": schema,
        "graph": graph, "subspace": subspace, "dense": dense, "ground_truth": gt,
    }


def _score_hybrid(ckpt_path: Path, bundle: dict) -> pd.DataFrame:
    from src.hybrid_graphmcm_v3 import (
        HybridGraphMCM, _build_edge_index_and_types, _compute_isolated_mask,
        compute_score_frame,
    )

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    _validate_shape(ckpt, ckpt_path)

    feat_df   = bundle["features"]
    feat_cols = bundle["schema"]["features"]
    x_all     = torch.tensor(feat_df[feat_cols].values, dtype=torch.float32).to(device)
    app_ids   = feat_df["application_id"].values

    edge_index_list, edge_type_tensor = _build_edge_index_and_types(bundle["graph"], device)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], device)

    model = HybridGraphMCM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.centroid = ckpt["centroid"].to(device)
    model.eval()

    with torch.no_grad():
        return compute_score_frame(model, x_all, edge_index_list, edge_type_tensor,
                                    isolated_mask, app_ids, feat_cols)


def _fuse(hybrid_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    from src.interfaces.fusion import score_level_fusion

    merged = hybrid_df.merge(bundle["subspace"][["application_id", "subspace_if_score"]], on="application_id")
    merged = merged.merge(bundle["dense"][["application_id", "dense_block_score_relational"]], on="application_id", how="left")
    merged["dense_block_score_relational"] = merged["dense_block_score_relational"].fillna(0.0)

    merged["final_risk_score"] = score_level_fusion(
        merged["subspace_if_score"].values,
        merged["dense_block_score_relational"].values,
        merged["hybrid_anomaly_score"].values,
    )
    return merged


def _per_category_pr_auc(scored: pd.DataFrame, gt: pd.DataFrame) -> dict[str, float]:
    """PR-AUC per fraud_type: positives = held-out rows of that type, negatives =
    every other row in the merged population (real + other stress rows) — same
    convention evaluate_model_v3.evaluate_connected() uses (injected vs. whole
    real population)."""
    results = {}
    scored = scored.set_index("application_id")
    for fraud_type, ids in gt[gt["fraud_type"] != "NONE"].groupby("fraud_type")["application_id"]:
        pos_ids = set(ids)
        if len(pos_ids) < MIN_POSITIVES_FOR_CATEGORY:
            continue
        labels = scored.index.isin(pos_ids).astype(int)
        if labels.sum() < MIN_POSITIVES_FOR_CATEGORY:
            continue
        results[fraud_type] = float(average_precision_score(labels, scored["final_risk_score"].values))
    return results


def run_gate(candidate_path: Path, cycle: str = "unknown") -> bool:
    schema_version = _schema_version()
    bundle_dir = HELD_OUT_ROOT / schema_version

    print(f"[deploy_gate] schema_version={schema_version}")
    if not (bundle_dir / "manifest.json").exists():
        print(f"[deploy_gate] FAIL: no held-out bundle at {bundle_dir}. "
              f"Run: python -m src.build_held_out_set")
        _log_verdict(candidate_path, schema_version, None, {}, {}, "FAIL", "missing_held_out_bundle", cycle)
        return False

    bundle = _load_bundle(bundle_dir)
    print(f"[deploy_gate] Held-out bundle: {bundle['manifest']}")

    if not LIVE_CKPT.exists():
        print(f"[deploy_gate] FAIL: no live checkpoint at {LIVE_CKPT} to compare against.")
        _log_verdict(candidate_path, schema_version, bundle["manifest"], {}, {}, "FAIL", "no_live_checkpoint", cycle)
        return False

    try:
        candidate_hybrid = _score_hybrid(candidate_path, bundle)
    except Exception as e:
        print(f"[deploy_gate] FAIL: candidate checkpoint rejected: {e}")
        _log_verdict(candidate_path, schema_version, bundle["manifest"], {}, {}, "FAIL", f"candidate_invalid: {e}", cycle)
        return False

    live_hybrid = _score_hybrid(LIVE_CKPT, bundle)

    candidate_fused = _fuse(candidate_hybrid, bundle)
    live_fused      = _fuse(live_hybrid, bundle)

    gt = bundle["ground_truth"]
    candidate_metrics = _per_category_pr_auc(candidate_fused, gt)
    live_metrics      = _per_category_pr_auc(live_fused, gt)

    print("\n[deploy_gate] Per-category PR-AUC (candidate vs. live, same held-out bundle):")
    verdict = "PASS"
    reasons = []
    for category in sorted(set(candidate_metrics) | set(live_metrics)):
        c = candidate_metrics.get(category)
        l = live_metrics.get(category)
        if c is None or l is None:
            continue
        delta = c - l
        status = "OK" if delta >= -NOISE_FLOOR else "REGRESSION"
        if status == "REGRESSION":
            verdict = "FAIL"
            reasons.append(f"{category}: candidate={c:.4f} live={l:.4f} delta={delta:.4f}")
        print(f"  {category:<25} candidate={c:.4f}  live={l:.4f}  delta={delta:+.4f}  [{status}]")

    reason_str = "; ".join(reasons) if reasons else "no category regressed beyond noise floor"
    print(f"\n[deploy_gate] VERDICT: {verdict} ({reason_str})")

    _log_verdict(candidate_path, schema_version, bundle["manifest"], candidate_metrics, live_metrics, verdict, reason_str, cycle)
    return verdict == "PASS"


def _log_verdict(candidate_path, schema_version, bundle_manifest, candidate_metrics, live_metrics, verdict, reason, cycle) -> None:
    from src.model_registry import log_run
    log_run(
        "deploy_gate",
        cycle=cycle,
        schema_version=schema_version,
        status=verdict,
        params={
            "candidate_checkpoint": str(candidate_path),
            "live_checkpoint":      str(LIVE_CKPT),
            "held_out_bundle":      bundle_manifest,
            "reason":               reason,
        },
        metrics={
            **{f"candidate_pr_auc_{k}": v for k, v in candidate_metrics.items()},
            **{f"live_pr_auc_{k}": v for k, v in live_metrics.items()},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path, help="Path to the candidate checkpoint (temp path, not yet live)")
    parser.add_argument("--cycle", default="unknown")
    args = parser.parse_args()

    if not args.candidate.exists():
        print(f"[deploy_gate] FAIL: candidate checkpoint not found: {args.candidate}")
        sys.exit(1)

    passed = run_gate(args.candidate, cycle=args.cycle)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
