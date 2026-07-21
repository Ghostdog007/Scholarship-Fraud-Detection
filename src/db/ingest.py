"""Populate Postgres from the file pipeline's outputs (migration step 2).

Until cut-over, files remain authoritative and this ingest re-mirrors them —
rerun after every pipeline run:

    .\\.venv\\Scripts\\python.exe -m src.db.ingest

Loads the primary 15k batch: applications (raw JSONB), identity_keys
(normalised EXACTLY like graph_builder_v3._build_edges: str().lower().strip(),
empty/'nan' -> NULL — Gate 2 requires identical ego-graph neighbourhoods),
features (44-dim), scores (risk + hybrid + subspace CSVs merged).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.connection import get_connection

RAW_CSV      = Path("data/raw/data_for_ml_model.csv")
FEATURES_CSV = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON  = Path("data/processed/v3_feature_schema.json")
RISK_CSV     = Path("outputs/risk_scores_v3.csv")
HYBRID_CSV   = Path("outputs/hybrid_scores_v3.csv")
SUBSPACE_CSV = Path("outputs/subspace_if_scores_v3.csv")

PRIMARY_BATCH_NAME = "primary_15k"
MODEL_VERSION      = "v3_current"

IDENTITY_RAW_COLS = {
    "mobile_no":        "mobile_no",
    "ip_address":       "ip_address",
    "father_name_norm": "father_name",
    "mother_name_norm": "mother_name",
    "pincode":          "permanent_pincode",
}


def _norm(v) -> str | None:
    """graph_builder_v3._build_edges normalisation, exactly."""
    s = str(v).lower().strip()
    if s == "" or s == "nan":
        return None
    return s


def _nan_to_none(v):
    if isinstance(v, float) and (v != v):
        return None
    return v


def _get_or_create_primary_batch(conn) -> int:
    row = conn.execute(
        "SELECT batch_id FROM batches WHERE name = %s AND kind = 'primary'",
        (PRIMARY_BATCH_NAME,),
    ).fetchone()
    if row:
        return row[0]
    return conn.execute(
        "INSERT INTO batches (name, kind, row_count, status)"
        " VALUES (%s, 'primary', 0, 'merged') RETURNING batch_id",
        (PRIMARY_BATCH_NAME,),
    ).fetchone()[0]


def ingest_primary() -> None:
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"[db.ingest] raw: {len(raw)} rows x {len(raw.columns)} cols")

    with get_connection() as conn:
        batch_id = _get_or_create_primary_batch(conn)

        # Re-mirror: wipe this batch's derived rows first (idempotent rerun).
        conn.execute("DELETE FROM scores WHERE batch_id = %s", (batch_id,))
        conn.execute(
            "DELETE FROM identity_keys WHERE application_id IN"
            " (SELECT application_id FROM applications WHERE batch_id = %s)", (batch_id,))
        conn.execute("DELETE FROM features WHERE batch_id = %s", (batch_id,))
        conn.execute("DELETE FROM applications WHERE batch_id = %s", (batch_id,))

        # -- applications (raw JSONB, lossless) --------------------------------
        with conn.cursor() as cur:
            with cur.copy(
                "COPY applications (application_id, batch_id, raw, source) FROM STDIN"
            ) as copy:
                for rec in raw.to_dict(orient="records"):
                    app_id = str(rec["application_id"])
                    payload = json.dumps(
                        {k: _nan_to_none(v) for k, v in rec.items()}, default=str
                    )
                    copy.write_row((app_id, batch_id, payload, "bulk_copy"))
        print("[db.ingest] applications loaded")

        # -- identity_keys (graph-builder normalisation) -----------------------
        with conn.cursor() as cur:
            with cur.copy(
                "COPY identity_keys (application_id, mobile_no, ip_address,"
                " father_name_norm, mother_name_norm, pincode) FROM STDIN"
            ) as copy:
                for rec in raw.to_dict(orient="records"):
                    copy.write_row((
                        str(rec["application_id"]),
                        _norm(rec.get("mobile_no")),
                        _norm(rec.get("ip_address")),
                        _norm(rec.get("father_name")),
                        _norm(rec.get("mother_name")),
                        _norm(rec.get("permanent_pincode")),
                    ))
        print("[db.ingest] identity_keys loaded")

        # -- features (44-dim, schema-ordered) ---------------------------------
        feats = pd.read_csv(FEATURES_CSV)
        schema_version = "v3_44"
        if SCHEMA_JSON.exists():
            meta = json.loads(SCHEMA_JSON.read_text())
            ordered = meta.get("features") or [c for c in feats.columns if c != "application_id"]
        else:
            ordered = [c for c in feats.columns if c != "application_id"]
        mat = feats[ordered].to_numpy(dtype=np.float32)
        ids = feats["application_id"].astype(str).tolist()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO features (application_id, batch_id, schema_version, vec)"
                " VALUES (%s, %s, %s, %s)",
                [(i, batch_id, schema_version, row.tolist()) for i, row in zip(ids, mat)],
            )
        print(f"[db.ingest] features loaded ({len(ids)} x {len(ordered)})")

        # -- scores (risk + hybrid + subspace merged) --------------------------
        risk = pd.read_csv(RISK_CSV)
        hyb  = pd.read_csv(HYBRID_CSV) if HYBRID_CSV.exists() else None
        sub  = pd.read_csv(SUBSPACE_CSV) if SUBSPACE_CSV.exists() else None
        merged = risk.copy()
        merged["application_id"] = merged["application_id"].astype(str)
        if hyb is not None:
            hyb["application_id"] = hyb["application_id"].astype(str)
            merged = merged.merge(hyb, on="application_id", how="left")
        if sub is not None:
            sub["application_id"] = sub["application_id"].astype(str)
            merged = merged.merge(sub, on="application_id", how="left")

        rows = []
        for rec in merged.to_dict(orient="records"):
            rows.append((
                rec["application_id"], batch_id, MODEL_VERSION,
                _nan_to_none(rec.get("hybrid_anomaly_score")),
                _nan_to_none(rec.get("feature_pred_error")),
                _nan_to_none(rec.get("edge_pred_error")),
                _nan_to_none(rec.get("subspace_if_score")),
                rec.get("group_scores_json") if isinstance(rec.get("group_scores_json"), str) else None,
                _nan_to_none(rec.get("dense_block_score_ip")),
                _nan_to_none(rec.get("risk_score_v3")),
                rec.get("label_source"),
                rec.get("per_feature_error_json") if isinstance(rec.get("per_feature_error_json"), str) else None,
                rec.get("per_feature_predicted_json") if isinstance(rec.get("per_feature_predicted_json"), str) else None,
            ))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO scores (application_id, batch_id, model_version,"
                " hybrid_anomaly_score, feature_pred_error, edge_pred_error,"
                " subspace_if_score, group_scores, dense_block_ip,"
                " final_risk_score, label_source, feature_errors, predicted_values)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
        print(f"[db.ingest] scores loaded ({len(rows)})")

        conn.execute(
            "UPDATE batches SET row_count = %s, status = 'merged' WHERE batch_id = %s",
            (len(raw), batch_id),
        )
    print("[db.ingest] primary batch ingest complete")


# ── Step 3: staged-batch lifecycle (admin-gated derivation contract) ─────────
# Senders deliver raw-schema rows only. stage_raw_csv() lands them in
# `applications` under a staged batch; evaluate_batch() populates the derived
# tables when the admin runs Evaluate; merge_batch() flips the human gate;
# delete_staged_batch() cleans up a removed cohort. Files remain authoritative
# until cut-over (hard stop 13) — these mirror the file flow, never replace it.

def stage_raw_csv(csv_path, name: str | None = None, kind: str = "cohort",
                  source: str = "csv_upload") -> dict:
    """Land a raw-schema CSV in `applications` under a staged batch.
    Re-staging the same name replaces the previous staged/evaluated batch
    (merged batches are permanent and never replaced)."""
    csv_path = Path(csv_path)
    name = name or csv_path.stem
    df = pd.read_csv(csv_path, low_memory=False)

    with get_connection() as conn:
        row = conn.execute(
            "SELECT batch_id, status FROM batches WHERE name = %s", (name,)
        ).fetchone()
        if row and row[1] == "merged":
            return {"batch_id": row[0], "name": name, "status": "merged",
                    "note": "batch already merged — not restaged"}
        if row:
            _wipe_batch(conn, row[0])
            conn.execute("DELETE FROM batches WHERE batch_id = %s", (row[0],))

        batch_id = conn.execute(
            "INSERT INTO batches (name, kind, row_count, status)"
            " VALUES (%s, %s, %s, 'staged') RETURNING batch_id",
            (name, kind, len(df)),
        ).fetchone()[0]

        n_conflict = 0
        with conn.cursor() as cur:
            for rec in df.to_dict(orient="records"):
                app_id = str(rec["application_id"])
                payload = json.dumps({k: _nan_to_none(v) for k, v in rec.items()}, default=str)
                cur.execute(
                    "INSERT INTO applications (application_id, batch_id, raw, source)"
                    " VALUES (%s, %s, %s, %s) ON CONFLICT (application_id) DO NOTHING",
                    (app_id, batch_id, payload, source),
                )
                if cur.rowcount == 0:
                    n_conflict += 1
    print(f"[db.ingest] staged batch '{name}' ({kind}): {len(df) - n_conflict} rows"
          + (f", {n_conflict} IDs already existed (skipped)" if n_conflict else ""))
    return {"batch_id": batch_id, "name": name, "status": "staged",
            "n_rows": len(df) - n_conflict, "n_id_conflicts": n_conflict}


def evaluate_batch(name: str, staged_scores_csv, staged_features_csv,
                   drift_p: float | None) -> dict:
    """Admin ran Evaluate: populate identity_keys + features + scores for the
    staged batch (pre-fusion preview scores, model_version 'staged_<name>').
    Idempotent — re-evaluating wipes and repopulates."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT batch_id FROM batches WHERE name = %s", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No staged batch named '{name}' — upload first")
        batch_id = row[0]

        conn.execute("DELETE FROM scores WHERE batch_id = %s", (batch_id,))
        conn.execute(
            "DELETE FROM identity_keys WHERE application_id IN"
            " (SELECT application_id FROM applications WHERE batch_id = %s)", (batch_id,))
        conn.execute("DELETE FROM features WHERE batch_id = %s", (batch_id,))

        # identity_keys from the staged raw JSONB (same normalisation as primary)
        apps = conn.execute(
            "SELECT application_id, raw FROM applications WHERE batch_id = %s", (batch_id,)
        ).fetchall()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO identity_keys (application_id, mobile_no, ip_address,"
                " father_name_norm, mother_name_norm, pincode)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [(
                    app_id,
                    _norm(raw.get("mobile_no")),
                    _norm(raw.get("ip_address")),
                    _norm(raw.get("father_name")),
                    _norm(raw.get("mother_name")),
                    _norm(raw.get("permanent_pincode")),
                ) for app_id, raw in apps],
            )

        # features from the staged features CSV (schema order)
        feats = pd.read_csv(staged_features_csv)
        ordered = ([c for c in json.loads(SCHEMA_JSON.read_text()).get("features", [])
                    if c in feats.columns]
                   if SCHEMA_JSON.exists()
                   else [c for c in feats.columns if c != "application_id"])
        mat = feats[ordered].to_numpy(dtype=np.float32)
        ids = feats["application_id"].astype(str).tolist()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO features (application_id, batch_id, schema_version, vec)"
                " VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (application_id) DO UPDATE SET"
                " batch_id = EXCLUDED.batch_id, vec = EXCLUDED.vec",
                [(i, batch_id, "v3_44", row.tolist()) for i, row in zip(ids, mat)],
            )

        # pre-fusion preview scores
        staged = pd.read_csv(staged_scores_csv)
        staged["application_id"] = staged["application_id"].astype(str)
        model_version = f"staged_{name}"
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO scores (application_id, batch_id, model_version,"
                " hybrid_anomaly_score, feature_pred_error, edge_pred_error)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [(
                    rec["application_id"], batch_id, model_version,
                    _nan_to_none(rec.get("hybrid_anomaly_score")),
                    _nan_to_none(rec.get("feature_pred_error")),
                    _nan_to_none(rec.get("edge_pred_error")),
                ) for rec in staged.to_dict(orient="records")],
            )

        conn.execute(
            "UPDATE batches SET status = 'evaluated', drift_p = %s WHERE batch_id = %s",
            (drift_p, batch_id),
        )
    print(f"[db.ingest] batch '{name}' evaluated: {len(apps)} identity_keys,"
          f" {len(ids)} features, {len(staged)} preview scores")
    return {"batch_id": batch_id, "n_scores": len(staged)}


def batch_exists(name: str) -> bool:
    with get_connection() as conn:
        return conn.execute(
            "SELECT 1 FROM batches WHERE name = %s", (name,)
        ).fetchone() is not None


def merge_batch(name: str) -> bool:
    """Human gate fired (Decide -> Merge / pattern ingest): mark permanent."""
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE batches SET status = 'merged' WHERE name = %s", (name,)
        )
        updated = cur.rowcount > 0
    if updated:
        print(f"[db.ingest] batch '{name}' merged (permanent)")
    return updated


def _wipe_batch(conn, batch_id: int) -> dict:
    n_s = conn.execute("DELETE FROM scores WHERE batch_id = %s", (batch_id,)).rowcount
    n_k = conn.execute(
        "DELETE FROM identity_keys WHERE application_id IN"
        " (SELECT application_id FROM applications WHERE batch_id = %s)", (batch_id,)).rowcount
    n_f = conn.execute("DELETE FROM features WHERE batch_id = %s", (batch_id,)).rowcount
    n_a = conn.execute("DELETE FROM applications WHERE batch_id = %s", (batch_id,)).rowcount
    return {"scores": n_s, "identity_keys": n_k, "features": n_f, "applications": n_a}


def delete_staged_batch(name: str) -> dict:
    """Console removed the cohort: drop its staged/evaluated Postgres rows.
    Merged batches are permanent — refuse."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT batch_id, status FROM batches WHERE name = %s", (name,)
        ).fetchone()
        if row is None:
            return {"deleted": False, "reason": "no such batch"}
        if row[1] == "merged":
            return {"deleted": False, "reason": "batch is merged (permanent)"}
        counts = _wipe_batch(conn, row[0])
        conn.execute("DELETE FROM batches WHERE batch_id = %s", (row[0],))
    print(f"[db.ingest] deleted staged batch '{name}': {counts}")
    return {"deleted": True, **counts}


if __name__ == "__main__":
    ingest_primary()
