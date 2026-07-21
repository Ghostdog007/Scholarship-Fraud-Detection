"""Postgres mirrors of the three JSON stores (migration step 1, dual-write).

JSON remains AUTHORITATIVE until the Gate 1 parity check passes and the lead
approves cut-over (AGENTS.md hard stop 13). Reads stay on JSON; these
functions only mirror writes. All are best-effort from the caller's side —
the store modules wrap them so a Postgres outage never breaks the console —
but failures are printed loudly, never swallowed silently.

Mirrored shapes (field-for-field with the JSON):
  confirmed_fraud  <- data/processed/confirmed_fraud.json
  loe_patterns     <- data/processed/confirmed_fraud_graph_store.json
  training_runs    <- outputs/model_registry.json
"""

import json

from src.db.connection import get_connection


# ── confirmed_fraud ──────────────────────────────────────────────────────────

def upsert_confirmed(record: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO confirmed_fraud
                (application_id, label, fraud_type, confirmed_by, cycle,
                 feature_vec, confirmed_at, notes)
            VALUES (%s, 'confirmed', %s, %s, %s, %s, %s, %s)
            ON CONFLICT (application_id) DO UPDATE SET
                label = EXCLUDED.label,
                fraud_type = EXCLUDED.fraud_type,
                confirmed_by = EXCLUDED.confirmed_by,
                cycle = EXCLUDED.cycle,
                feature_vec = EXCLUDED.feature_vec,
                confirmed_at = EXCLUDED.confirmed_at,
                notes = EXCLUDED.notes
            """,
            (
                record["application_id"],
                record.get("fraud_type"),
                record.get("confirmed_by"),
                record.get("cycle"),
                record.get("feature_vec"),
                record.get("confirmed_at"),
                record.get("notes"),
            ),
        )


def upsert_false_positive(record: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO confirmed_fraud
                (application_id, label, confirmed_by, confirmed_at, notes)
            VALUES (%s, 'false_positive', %s, %s, %s)
            ON CONFLICT (application_id) DO UPDATE SET
                label = 'false_positive',
                fraud_type = NULL,
                cycle = NULL,
                feature_vec = NULL,
                confirmed_by = EXCLUDED.confirmed_by,
                confirmed_at = EXCLUDED.confirmed_at,
                notes = EXCLUDED.notes
            """,
            (
                record["application_id"],
                record.get("confirmed_by"),
                record.get("confirmed_at"),
                record.get("notes"),
            ),
        )


def delete_label(app_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM confirmed_fraud WHERE application_id = %s", (app_id,))


# ── loe_patterns ─────────────────────────────────────────────────────────────

def upsert_pattern(p: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO loe_patterns
                (pattern_id, center_app_id, fraud_type, state, subgraph,
                 exposure, confirmed_by, notes, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (pattern_id) DO UPDATE SET
                center_app_id = EXCLUDED.center_app_id,
                fraud_type = EXCLUDED.fraud_type,
                state = EXCLUDED.state,
                subgraph = EXCLUDED.subgraph,
                exposure = EXCLUDED.exposure,
                confirmed_by = EXCLUDED.confirmed_by,
                notes = EXCLUDED.notes,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at
            """,
            (
                p["pattern_id"],
                p.get("center_app_id"),
                p.get("fraud_type"),
                p.get("state"),
                json.dumps(p.get("subgraph")) if p.get("subgraph") is not None else None,
                json.dumps(p.get("exposure")) if p.get("exposure") is not None else None,
                p.get("confirmed_by"),
                p.get("notes"),
                p.get("created_at"),
                p.get("updated_at"),
            ),
        )


def delete_patterns(pattern_ids: list[str]) -> None:
    if not pattern_ids:
        return
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM loe_patterns WHERE pattern_id = ANY(%s)", (pattern_ids,)
        )


def mirror_patterns(store: dict) -> None:
    """Full sync: make loe_patterns match the JSON store exactly (upsert all,
    delete rows the store no longer has). Pattern counts are small (tens), so
    a full re-mirror per save is cheap and immune to missed transitions."""
    patterns = store.get("patterns", {})
    with get_connection() as conn:
        db_ids = {r[0] for r in conn.execute("SELECT pattern_id FROM loe_patterns").fetchall()}
        gone = db_ids - set(patterns.keys())
        if gone:
            conn.execute("DELETE FROM loe_patterns WHERE pattern_id = ANY(%s)", (list(gone),))
    for p in patterns.values():
        upsert_pattern(p)


# ── training_runs ────────────────────────────────────────────────────────────

def upsert_run(record: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO training_runs
                (run_id, ts, run_type, cycle, smoke_test, status,
                 params, metrics, checkpoint)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                ts = EXCLUDED.ts,
                run_type = EXCLUDED.run_type,
                cycle = EXCLUDED.cycle,
                smoke_test = EXCLUDED.smoke_test,
                status = EXCLUDED.status,
                params = EXCLUDED.params,
                metrics = EXCLUDED.metrics,
                checkpoint = EXCLUDED.checkpoint
            """,
            (
                record["run_id"],
                record.get("timestamp"),
                record.get("run_type"),
                record.get("cycle"),
                bool(record.get("smoke_test", False)),
                record.get("status"),
                json.dumps(record.get("params") or {}),
                json.dumps(record.get("metrics") or {}),
                json.dumps(record.get("checkpoint") or {}),
            ),
        )


# ── Replay (Gate 1): re-mirror the entire JSON stores into Postgres ──────────

def replay_all() -> dict:
    """Load each JSON store and upsert every record. Returns counts."""
    from pathlib import Path

    counts = {"confirmed": 0, "false_positives": 0, "patterns": 0, "runs": 0}

    cf_path = Path("data/processed/confirmed_fraud.json")
    if cf_path.exists():
        store = json.loads(cf_path.read_text())
        for r in store.get("confirmed", []):
            upsert_confirmed(r)
            counts["confirmed"] += 1
        for r in store.get("false_positives", []):
            upsert_false_positive(r)
            counts["false_positives"] += 1

    gp_path = Path("data/processed/confirmed_fraud_graph_store.json")
    if gp_path.exists():
        store = json.loads(gp_path.read_text())
        for p in store.get("patterns", {}).values():
            upsert_pattern(p)
            counts["patterns"] += 1

    reg_path = Path("outputs/model_registry.json")
    if reg_path.exists():
        data = json.loads(reg_path.read_text())
        for r in data.get("runs", []):
            upsert_run(r)
            counts["runs"] += 1

    return counts


if __name__ == "__main__":
    print(f"[db.stores] replay complete: {replay_all()}")
