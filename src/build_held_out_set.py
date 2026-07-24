"""
build_held_out_set.py

Builds/refreshes the held-out evaluation bundle a candidate checkpoint must
clear before promotion (see src/deploy_gate.py). Ground truth: data/uploads/
stress_testing_1.csv + _ground_truth.csv (50k synthetic cohort, sampled +
perturbed from real rows per hard stop 7 — never GAN/CTGAN/TVAE), which
already exists and is used for the project's fusion/dense-block ablations
(see AGENTS.md §1, outputs/stress_testing_1_*.json). This script does not
generate new synthetic data — it re-runs feature engineering + graph
building over (real population + stress rows) so the bundle reflects
WHATEVER THE CURRENT FEATURE SCHEMA IS, and snapshots the result.

Why this must be re-run after a feature/schema change (MAINTAINER_PLAYBOOK.md
Recipe 1 / Recipe 3): the bundle is versioned by schema_version = "v3_<N_FEATURES>"
(matching the persist_version convention already used by
tabular_feature_engine_v3.build_base_pg). If N_FEATURES or the feature list
changes, the schema_version string changes, deploy_gate.py will look for a
bundle that doesn't exist yet, and refuse to run (fail closed) until this
script is re-run.

Uses src/api/dataset_ops.py's existing backup/merge/rebuild/restore helpers
unchanged (same ones the drift-simulation endpoints use) — canonical files
are always restored, even on error.

Writes: outputs/held_out/<schema_version>/
    features.csv       full merged population's engineered features (44-dim)
    graph.pt            full merged population's identity graph
    schema.json          feature list snapshot
    subspace_scores.csv  subspace IF scores over the merged population (checkpoint-independent)
    dense_scores.csv     dense-block scores over the merged population (checkpoint-independent)
    ground_truth.csv     application_id, is_fraud, fraud_type, ring_id (stress rows only)
    manifest.json         schema_version, git_commit, created_at, row counts

Run: .venv/Scripts/python.exe -m src.build_held_out_set
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import torch

from src.config_v3 import N_FEATURES
from src.api.dataset_ops import (
    backup_canonical_files,
    restore_canonical_files,
    merge_dataset_into_raw,
    rebuild_features_and_graph,
    RAW_CSV,
    FINAL_CSV,
    SCHEMA_JSON,
    GRAPH_PT,
)
from src.interfaces.subspace_if import compute_subspace_if_scores
from src.interfaces.dense_block import dense_block_scores

STRESS_CSV    = Path("data/uploads/stress_testing_1.csv")
STRESS_GT_CSV = Path("data/uploads/stress_testing_1_ground_truth.csv")

HELD_OUT_ROOT = Path("outputs/held_out")


def _schema_version() -> str:
    return f"v3_{N_FEATURES}"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def build_held_out_set(
    stress_csv: Path = STRESS_CSV,
    stress_gt_csv: Path = STRESS_GT_CSV,
    out_dir: Path | None = None,
) -> Path:
    """stress_csv/stress_gt_csv/out_dir let this build from an alternate stress
    population (e.g. a second, independently-seeded cohort for cross-validating
    a finding) without disturbing the canonical outputs/held_out/<schema_version>/
    bundle deploy_gate.py reads by default."""
    if not stress_csv.exists() or not stress_gt_csv.exists():
        raise FileNotFoundError(
            f"Missing stress-test source data: {stress_csv} / {stress_gt_csv}. "
            "Run scripts/generate_stress_test_dataset.py first."
        )

    schema_version = _schema_version()
    if out_dir is None:
        out_dir = HELD_OUT_ROOT / schema_version
    print(f"[build_held_out_set] schema_version={schema_version} -> {out_dir}")

    backup_dir = backup_canonical_files(label="held_out_build")
    print(f"[build_held_out_set] Canonical files backed up to {backup_dir}")

    try:
        n_appended = merge_dataset_into_raw(stress_csv)
        print(f"[build_held_out_set] Merged {n_appended} stress rows into raw population")

        rebuild_features_and_graph()
        print("[build_held_out_set] Rebuilt features + graph over merged population")

        feat_df = pd.read_csv(FINAL_CSV)
        schema  = json.loads(SCHEMA_JSON.read_text())
        graph   = torch.load(GRAPH_PT, weights_only=False)

        subspace_df = compute_subspace_if_scores(feat_df, features=set(schema["features"]))

        from src.hybrid_graphmcm_v3 import _build_edge_index_and_types
        edge_index_list, edge_type_tensor = _build_edge_index_and_types(graph, torch.device("cpu"))
        dense_df = dense_block_scores(
            edge_index_list, edge_type_tensor, len(feat_df), feat_df["application_id"].values
        )

        gt_df = pd.read_csv(stress_gt_csv)
        gt_df = gt_df[gt_df["application_id"].isin(set(feat_df["application_id"]))]

        out_dir.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(out_dir / "features.csv", index=False)
        shutil.copy2(GRAPH_PT, out_dir / "graph.pt")
        (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))
        subspace_df.to_csv(out_dir / "subspace_scores.csv", index=False)
        dense_df.to_csv(out_dir / "dense_scores.csv", index=False)
        gt_df.to_csv(out_dir / "ground_truth.csv", index=False)

        manifest = {
            "schema_version":      schema_version,
            "n_features":          N_FEATURES,
            "git_commit":          _git_commit(),
            "created_at":          pd.Timestamp.utcnow().isoformat(),
            "n_total_rows":        len(feat_df),
            "n_held_out_rows":     len(gt_df),
            "fraud_type_counts":   gt_df["fraud_type"].value_counts().to_dict() if "fraud_type" in gt_df.columns else {},
            "source":              stress_csv.stem,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[build_held_out_set] Wrote bundle: {manifest}")

    finally:
        restore_canonical_files(backup_dir)
        print(f"[build_held_out_set] Canonical files restored from {backup_dir}")

    return out_dir


if __name__ == "__main__":
    build_held_out_set()
