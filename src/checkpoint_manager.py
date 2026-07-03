"""
checkpoint_manager.py — ADR-008

Atomic checkpoint validation and hot-swap.

Hard stop #14: this is the ONLY code path allowed to write
models/hybrid_graphmcm_v3.pth. Training code must save to a temp path
(models/incoming_<timestamp>_<uuid>.pth) and call validate_and_hotswap().

Validation contract (hard stop #15):
  - Required top-level keys: {model_state_dict, centroid, config}
  - config must contain N_FEATURES, GRAPH_EMB_DIM, N_EDGE_TYPES
  - Values must match src/config_v3.py constants exactly
"""
import shutil
import uuid
from pathlib import Path

import torch

from src.config_v3 import N_FEATURES, GRAPH_EMB_DIM, N_EDGE_TYPES, ARCH_VERSION

LIVE_PATH     = Path("models/hybrid_graphmcm_v3.pth")
BAK_PATH      = Path("models/hybrid_graphmcm_v3.pth.bak")
CKPT_DIR      = Path("models/checkpoints")
MAX_VERSIONED = 5

_REQUIRED_KEYS   = {"model_state_dict", "centroid", "config"}
_REQUIRED_CONFIG = {"N_FEATURES", "GRAPH_EMB_DIM", "N_EDGE_TYPES", "ARCH_VERSION"}


def _validate(d: dict, path: Path) -> None:
    missing = _REQUIRED_KEYS - set(d.keys())
    if missing:
        raise ValueError(f"Checkpoint {path.name} missing required keys: {missing}")

    cfg = d["config"]
    missing_cfg = _REQUIRED_CONFIG - set(cfg.keys())
    if missing_cfg:
        raise ValueError(f"Checkpoint config missing required keys: {missing_cfg}")

    if cfg["N_FEATURES"] != N_FEATURES:
        raise ValueError(f"N_FEATURES mismatch: checkpoint={cfg['N_FEATURES']}, expected={N_FEATURES}")
    if cfg["GRAPH_EMB_DIM"] != GRAPH_EMB_DIM:
        raise ValueError(f"GRAPH_EMB_DIM mismatch: checkpoint={cfg['GRAPH_EMB_DIM']}, expected={GRAPH_EMB_DIM}")
    if cfg["N_EDGE_TYPES"] != N_EDGE_TYPES:
        raise ValueError(f"N_EDGE_TYPES mismatch: checkpoint={cfg['N_EDGE_TYPES']}, expected={N_EDGE_TYPES}")
    if cfg["ARCH_VERSION"] != ARCH_VERSION:
        raise ValueError(f"ARCH_VERSION mismatch: checkpoint={cfg.get('ARCH_VERSION')}, expected={ARCH_VERSION}")


def _prune_old_checkpoints() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    versioned = sorted(CKPT_DIR.glob("hybrid_v3_*.pth"), key=lambda p: p.stat().st_mtime)
    while len(versioned) > MAX_VERSIONED:
        oldest = versioned.pop(0)
        oldest.unlink()
        print(f"[checkpoint_manager] Pruned: {oldest.name}")


def validate_and_hotswap(
    temp_path: Path,
    cycle: str = "unknown",
    mlflow_run_id: str = "none",
) -> dict:
    """
    Validate a checkpoint at temp_path then atomically swap it to LIVE_PATH.

    Steps:
      1. Load and schema-validate (keys + config dimension check)
      2. Save versioned copy to models/checkpoints/hybrid_v3_<cycle>_<tag>.pth
      3. Backup current live checkpoint to .bak
      4. Copy versioned → live (atomic on same volume)
      5. Prune versioned checkpoints older than MAX_VERSIONED
      6. Remove temp file (unless temp_path IS the live path, e.g. dvc pull)

    Returns metadata dict.
    Raises ValueError if validation fails — live checkpoint is NOT touched.
    """
    if not temp_path.exists():
        raise FileNotFoundError(f"Temp checkpoint not found: {temp_path}")

    print(f"[checkpoint_manager] Validating {temp_path.name} ...")
    d = torch.load(temp_path, weights_only=True, map_location="cpu")
    _validate(d, temp_path)
    cfg = d["config"]
    print(
        f"[checkpoint_manager] OK — N_FEATURES={cfg['N_FEATURES']} "
        f"GRAPH_EMB_DIM={cfg['GRAPH_EMB_DIM']} N_EDGE_TYPES={cfg['N_EDGE_TYPES']}"
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run_tag   = mlflow_run_id[:8] if mlflow_run_id != "none" else uuid.uuid4().hex[:8]
    cycle_tag = cycle.replace("/", "-").replace(" ", "_")
    versioned_path = CKPT_DIR / f"hybrid_v3_{cycle_tag}_{run_tag}.pth"
    shutil.copy2(temp_path, versioned_path)
    print(f"[checkpoint_manager] Versioned copy: {versioned_path.name}")

    if LIVE_PATH.exists():
        shutil.copy2(LIVE_PATH, BAK_PATH)
        print(f"[checkpoint_manager] Backed up live → {BAK_PATH.name}")

    LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(versioned_path, LIVE_PATH)
    print(f"[checkpoint_manager] Live checkpoint updated: {LIVE_PATH}")

    _prune_old_checkpoints()

    # Do not delete temp if it IS the live path (dvc pull writes directly)
    if temp_path.resolve() != LIVE_PATH.resolve() and temp_path.resolve() != versioned_path.resolve():
        temp_path.unlink(missing_ok=True)
        print(f"[checkpoint_manager] Cleaned temp: {temp_path.name}")

    return {
        "versioned_path": str(versioned_path),
        "size_mb": round(LIVE_PATH.stat().st_size / 1e6, 2),
        "cycle": cycle,
        "mlflow_run_id": mlflow_run_id,
    }


def get_checkpoint_info() -> dict:
    """Return metadata for the current live checkpoint."""
    if not LIVE_PATH.exists():
        return {"exists": False}

    size_mb = round(LIVE_PATH.stat().st_size / 1e6, 2)
    try:
        d   = torch.load(LIVE_PATH, weights_only=True, map_location="cpu")
        cfg = d.get("config", {})
    except Exception as e:
        return {"exists": True, "size_mb": size_mb, "load_error": str(e)}

    versioned = (
        sorted(p.name for p in CKPT_DIR.glob("hybrid_v3_*.pth"))
        if CKPT_DIR.exists() else []
    )
    return {
        "exists":                True,
        "size_mb":               size_mb,
        "n_features":            cfg.get("N_FEATURES"),
        "graph_emb_dim":         cfg.get("GRAPH_EMB_DIM"),
        "n_edge_types":          cfg.get("N_EDGE_TYPES"),
        "versioned_checkpoints": versioned,
    }
