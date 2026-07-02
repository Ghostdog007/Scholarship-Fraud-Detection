"""
Dataset backup/merge/rebuild helpers for the drift-simulation endpoints.

These wrap the existing feature-engine and graph-builder modules unchanged —
no src/*.py pipeline module is modified (AGENTS.md Appendix F invariant).
Merges act on the canonical raw CSV path so build_base()/build_graph()/
add_degree_features() run exactly as they do in the normal pipeline.
"""
import shutil
import time
from pathlib import Path

import pandas as pd

RAW_CSV     = Path("data/raw/data_for_ml_model.csv")
NODEG_CSV   = Path("data/processed/engineered_features_v3_nodeg.csv")
FINAL_CSV   = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON = Path("data/processed/v3_feature_schema.json")
GRAPH_PT    = Path("data/processed/identity_graph_v3.pt")
DEGREE_CSV  = Path("data/processed/degree_features_v3.csv")

_CANONICAL_FILES = [RAW_CSV, NODEG_CSV, FINAL_CSV, SCHEMA_JSON, GRAPH_PT, DEGREE_CSV]

BACKUP_ROOT = Path("data/backups")


def backup_canonical_files(label: str = "snapshot") -> Path:
    """Copy the canonical raw/feature/graph files aside. Returns the backup dir."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"{ts}_{label}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in _CANONICAL_FILES:
        if f.exists():
            shutil.copy2(f, backup_dir / f.name)
    return backup_dir


def restore_canonical_files(backup_dir: Path) -> None:
    """Restore canonical files from a backup dir created by backup_canonical_files()."""
    for f in _CANONICAL_FILES:
        src = backup_dir / f.name
        if src.exists():
            shutil.copy2(src, f)


def merge_dataset_into_raw(new_dataset_path: Path) -> int:
    """
    Append a new dataset's rows onto the canonical raw CSV in place.
    Both files must share the raw schema (136 columns, incl. application_id).
    Returns the number of new rows appended.
    """
    base = pd.read_csv(RAW_CSV, low_memory=False)
    new  = pd.read_csv(new_dataset_path, low_memory=False)

    missing = set(base.columns) - set(new.columns)
    if missing:
        raise ValueError(f"New dataset missing columns required by raw schema: {sorted(missing)}")

    overlap = set(base["application_id"]) & set(new["application_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} application_id(s) in new dataset already exist in raw data: "
                          f"{sorted(overlap)[:5]}...")

    combined = pd.concat([base, new[base.columns]], ignore_index=True)
    combined.to_csv(RAW_CSV, index=False)
    return len(new)


def rebuild_features_and_graph() -> None:
    """Re-run the standard build order (AGENTS.md §8) on whatever is now at RAW_CSV."""
    from src.tabular_feature_engine_v3 import build_base, add_degree_features
    from src.graph_builder_v3 import build_graph

    build_base()
    build_graph()
    add_degree_features()
