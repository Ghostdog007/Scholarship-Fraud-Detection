"""
model_registry.py — local run/metric/checkpoint history (MLflow replacement).

Replaces the previous MLflow sqlite tracking that lived in
retraining_orchestrator.py + checkpoint_manager.py. Dependency-free: one JSON
file the pipeline writes and the API reads, so the supervisor sees running-model
state (last training run, its metrics, the live checkpoint, the decision trail)
directly in the console — no MLflow server, no mlruns/ dir, no mlflow.db.

Each entry is one "run":
  run_id, timestamp (UTC ISO), run_type, cycle, smoke_test, status,
  params{}, metrics{}, checkpoint{}

run_type is one of:
  incremental        — src.retraining_orchestrator incremental update
  full               — full pipeline (main_v3.py) — logged by the task wrapper
  checkpoint_swap    — validate_and_hotswap (upload / dvc pull / rollback)

Writes: outputs/model_registry.json   (atomic; worker concurrency is 1)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = Path("outputs/model_registry.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    """Best-effort current commit hash — traceability only, never fatal."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _load() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"runs": []}


def _atomic_write(data: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(REGISTRY.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, REGISTRY)   # atomic on the same volume
    finally:
        Path(tmp).unlink(missing_ok=True)


def log_run(
    run_type: str,
    *,
    cycle: str = "unknown",
    params: dict | None = None,
    metrics: dict | None = None,
    checkpoint: dict | None = None,
    smoke_test: bool = False,
    status: str = "complete",
    run_id: str | None = None,
    git_commit: str | None = None,
    schema_version: str | None = None,
) -> dict:
    """Append one run record and return it. Never raises on a bad numeric value —
    non-finite floats are dropped so the registry JSON stays valid.

    git_commit / schema_version (added 2026-07-23, deploy-gate traceability):
    auto-detected from `git rev-parse HEAD` when not passed explicitly. Lets any
    run — training, incremental, checkpoint_swap, deploy_gate — be traced back to
    the code state and feature schema it ran against. See MAINTAINER_PLAYBOOK.md
    Recipe 1/3 for why schema_version must change when the feature set changes."""
    def _clean(d: dict | None) -> dict:
        out = {}
        for k, v in (d or {}).items():
            if isinstance(v, float):
                if v != v or v in (float("inf"), float("-inf")):  # NaN/inf
                    continue
            out[k] = v
        return out

    data = _load()
    record = {
        "run_id":         run_id or uuid.uuid4().hex[:12],
        "timestamp":      _now_iso(),
        "run_type":       run_type,
        "cycle":          cycle,
        "smoke_test":     bool(smoke_test),
        "status":         status,
        "params":         _clean(params),
        "metrics":        _clean(metrics),
        "checkpoint":     checkpoint or {},
        "git_commit":     git_commit or _git_commit(),
        "schema_version": schema_version,
    }
    data["runs"].append(record)
    _atomic_write(data)
    # Postgres dual-write (migration step 1). JSON stays authoritative (hard
    # stop 13); failure is loud, never fatal.
    try:
        from src.db.stores import upsert_run
        upsert_run(record)
    except Exception as e:  # noqa: BLE001 — dual-write must not break callers
        print(f"[model_registry] WARNING: Postgres dual-write upsert_run FAILED: {e}")
    return record


def list_runs(limit: int | None = None, run_type: str | None = None) -> list[dict]:
    """Newest-first run history, optionally filtered by run_type."""
    runs = [r for r in reversed(_load().get("runs", []))
            if run_type is None or r.get("run_type") == run_type]
    return runs[:limit] if limit else runs


def latest_run(run_type: str | None = None) -> dict | None:
    for r in reversed(_load().get("runs", [])):
        if run_type is None or r.get("run_type") == run_type:
            return r
    return None
