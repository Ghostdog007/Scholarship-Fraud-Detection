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


if __name__ == "__main__":
    ingest_primary()
