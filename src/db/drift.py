"""Postgres mirror of the yearly drift-baseline JSON files (cut-over prep).

outputs/prev_cycle_scores_ks.json and prev_cycle_features_ks.json hold FULL
score/feature arrays re-read whole on every drift check — at 3.5M rows that's
~3.5M floats (scores) or 68 x 3.5M floats (features) parsed from one JSON
blob per call. This mirrors them into a queryable table. JSON files remain
authoritative until cut-over (hard stop 13); dual-write, PG-first read with
file fallback, same pattern as the rest of this migration.
"""

import json

from src.db.connection import get_connection


def save_baseline(kind: str, payload: dict) -> None:
    assert kind in {"scores", "features"}
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO drift_baselines (baseline_kind, payload)"
            " VALUES (%s, %s)"
            " ON CONFLICT (baseline_kind) DO UPDATE SET"
            " payload = EXCLUDED.payload, saved_at = now()",
            (kind, json.dumps(payload)),
        )


def load_baseline(kind: str) -> dict | None:
    assert kind in {"scores", "features"}
    with get_connection() as conn:
        row = conn.execute(
            "SELECT payload FROM drift_baselines WHERE baseline_kind = %s", (kind,)
        ).fetchone()
    return row[0] if row else None


def has_baseline(kind: str) -> bool:
    return load_baseline(kind) is not None
