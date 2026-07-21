"""
confirmed_fraud_store.py

Single source of truth for supervisor-confirmed fraud examples.

The supervisor calls add_confirmed() after reviewing an XAI card.
Everything downstream (LOE exposure, self-training labels, LightGBM
sample weights) reads from this file — no other file needs to be touched.

Writes: data/processed/confirmed_fraud.json
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

STORE_PATH   = Path("data/processed/confirmed_fraud.json")
FEATURES_CSV = Path("data/processed/engineered_features_v3.csv")

VALID_FRAUD_TYPES = {
    "IP_CLUSTER",
    "FEE_INFLATION",
    "INCOME_VIOLATION",
    "NAME_COLLISION",
    "CROSS_CHANNEL",
    "OTHER",
}


def _mirror(fn_name: str, *args) -> None:
    """Best-effort Postgres dual-write (migration step 1). JSON stays
    authoritative (hard stop 13); a PG failure never breaks the caller but is
    printed loudly so it can't go unnoticed."""
    try:
        from src.db import stores as db_stores
        getattr(db_stores, fn_name)(*args)
    except Exception as e:  # noqa: BLE001 — dual-write must not break the console
        print(f"[confirmed_fraud] WARNING: Postgres dual-write {fn_name} FAILED: {e}")


def _load_store() -> dict:
    if STORE_PATH.exists():
        return json.loads(STORE_PATH.read_text())
    return {"confirmed": [], "false_positives": []}


def _save_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2))


# ---------------------------------------------------------------------------
# Supervisor-facing API
# ---------------------------------------------------------------------------

def add_confirmed(
    app_id: str,
    fraud_type: str,
    confirmed_by: str,
    notes: str = "",
    cycle: str = "",
) -> None:
    """
    Supervisor marks an application as confirmed fraud.
    Feature vector is pulled automatically from the engineered features CSV.

    fraud_type must be one of VALID_FRAUD_TYPES.
    """
    if fraud_type not in VALID_FRAUD_TYPES:
        raise ValueError(f"fraud_type must be one of {VALID_FRAUD_TYPES}, got '{fraud_type}'")

    df = pd.read_csv(FEATURES_CSV)
    row = df[df["application_id"] == app_id]
    if row.empty:
        raise ValueError(f"application_id '{app_id}' not found in {FEATURES_CSV}")

    feat_cols   = [c for c in df.columns if c != "application_id"]
    feature_vec = row[feat_cols].values[0].tolist()

    store = _load_store()

    # Avoid duplicate entries
    existing_ids = {r["application_id"] for r in store["confirmed"]}
    if app_id in existing_ids:
        print(f"[confirmed_fraud] '{app_id}' already confirmed — skipping.")
        return

    record = {
        "application_id": app_id,
        "cycle":          cycle,
        "fraud_type":     fraud_type,
        "feature_vec":    feature_vec,
        "confirmed_by":   confirmed_by,
        "confirmed_at":   str(date.today()),
        "notes":          notes,
    }
    store["confirmed"].append(record)

    _save_store(store)
    _mirror("upsert_confirmed", record)
    total = len(store["confirmed"])
    print(f"[confirmed_fraud] Added '{app_id}' ({fraud_type}). Total confirmed: {total}")


def add_false_positive(app_id: str, confirmed_by: str, notes: str = "") -> None:
    """Supervisor marks a flag as a false positive — used as hard negative in LightGBM."""
    store = _load_store()
    existing_ids = {r["application_id"] for r in store["false_positives"]}
    if app_id in existing_ids:
        print(f"[confirmed_fraud] '{app_id}' already marked false positive — skipping.")
        return

    record = {
        "application_id": app_id,
        "confirmed_by":   confirmed_by,
        "confirmed_at":   str(date.today()),
        "notes":          notes,
    }
    store["false_positives"].append(record)
    _save_store(store)
    _mirror("upsert_false_positive", record)
    print(f"[confirmed_fraud] Marked '{app_id}' as false positive. Total FP: {len(store['false_positives'])}")


def remove_label(app_id: str) -> dict:
    """Undo a supervisor label — removes the application from BOTH the confirmed
    and false-positive lists. Used to reset a label (e.g. to re-demo the
    detection loop, or to correct a mis-click). Returns which lists it was
    removed from; both False means the application had no label."""
    store = _load_store()
    before_c = len(store["confirmed"])
    before_f = len(store["false_positives"])
    store["confirmed"]       = [r for r in store["confirmed"]       if r["application_id"] != app_id]
    store["false_positives"] = [r for r in store["false_positives"] if r["application_id"] != app_id]
    removed_c = len(store["confirmed"])       < before_c
    removed_f = len(store["false_positives"]) < before_f
    if removed_c or removed_f:
        _save_store(store)
        _mirror("delete_label", app_id)
        print(f"[confirmed_fraud] Cleared label for '{app_id}' "
              f"(confirmed={removed_c}, false_positive={removed_f}).")
    return {
        "removed_confirmed":       removed_c,
        "removed_false_positive":  removed_f,
        "n_confirmed":             len(store["confirmed"]),
        "n_false_positives":       len(store["false_positives"]),
    }


# ---------------------------------------------------------------------------
# Downstream consumers
# ---------------------------------------------------------------------------

def load_confirmed() -> list[dict]:
    """Returns list of confirmed fraud records."""
    return _load_store()["confirmed"]


def load_false_positive_ids() -> set[str]:
    """Returns set of application IDs confirmed as false positives."""
    return {r["application_id"] for r in _load_store()["false_positives"]}


def get_exposure_tensor(
    synthetic_fallback: torch.Tensor,
    min_real: int = 5,
) -> tuple[torch.Tensor, str]:
    """
    Returns the LOE exposure tensor and a label describing its source.

    If fewer than min_real confirmed fraud examples exist, returns synthetic_fallback.
    If enough real examples exist, returns real feature vectors (better geometry).
    If both exist, returns concatenation (real + synthetic as backup coverage).
    """
    records = load_confirmed()
    if len(records) < min_real:
        print(f"[confirmed_fraud] {len(records)} confirmed < {min_real} min — using synthetic exposure.")
        return synthetic_fallback, "synthetic"

    real_vecs = torch.tensor(
        np.array([r["feature_vec"] for r in records], dtype=np.float32)
    )

    # Concatenate: real examples anchor the LOE; synthetic covers archetypes
    # not yet seen in confirmed cases
    combined = torch.cat([real_vecs, synthetic_fallback], dim=0)
    print(f"[confirmed_fraud] Exposure set: {len(records)} real + {synthetic_fallback.shape[0]} synthetic = {combined.shape[0]} total")
    return combined, "real+synthetic"


def summary() -> None:
    store = _load_store()
    confirmed = store["confirmed"]
    fp        = store["false_positives"]
    print(f"[confirmed_fraud] Total confirmed fraud : {len(confirmed)}")
    print(f"[confirmed_fraud] Total false positives : {len(fp)}")
    if confirmed:
        from collections import Counter
        counts = Counter(r["fraud_type"] for r in confirmed)
        for ftype, n in counts.most_common():
            print(f"  {ftype}: {n}")


if __name__ == "__main__":
    summary()
