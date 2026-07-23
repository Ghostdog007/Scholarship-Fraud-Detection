"""SQL side of the scale-path feature engine + graph builder (step 4).

Owns every query these modules need (hard stop 14). The semantics of each
query replicate the pandas operations in tabular_feature_engine_v3
EXACTLY — Gate 4 is a bit-for-bit comparison. Where a query cannot be
bit-identical (PERCENTILE_CONT's interpolation expression differs from
pandas' median by one float op), the consuming column is documented as
tolerance-compared in the gate.

Population scope: all applications in MERGED batches — identical to the
file pipeline's raw CSV (primary + permanently merged cohorts).
"""

import csv
import io
from pathlib import Path

import pandas as pd

from src.db.connection import get_connection

RAW_CSV = Path("data/raw/data_for_ml_model.csv")

_MERGED_ORDERED = (
    "SELECT a.application_id, a.raw FROM applications a JOIN batches b ON a.batch_id = b.batch_id"
    " WHERE b.status = 'merged' ORDER BY b.batch_id, a.application_id"
)


def fetch_raw_frame() -> pd.DataFrame:
    """Reconstruct the raw dataframe from Postgres with IDENTICAL dtype
    inference to pd.read_csv on the file: rows are re-serialised to CSV text
    and re-parsed. Column list comes from the raw CSV header (static 136-col
    raw schema, not row data — safe to read once). Row set and order come
    from Postgres itself (every MERGED batch, ordered by batch then id) —
    NOT from the raw CSV's id list, so any batch merged after the primary
    15k (Decide -> Merge, or Pattern-queue Promote) is included here too.
    This is what lets a Postgres-sourced full retrain (NIC_DATA_SOURCE=
    postgres, config_v3.DATA_SOURCE) actually see newly merged data without
    a CSV round-trip."""
    columns = list(pd.read_csv(RAW_CSV, nrows=0).columns)

    with get_connection() as conn:
        fetched = conn.execute(_MERGED_ORDERED).fetchall()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns)
    n_extra_cols = 0
    for _app_id, raw in fetched:
        row = ["" if raw.get(c) is None else raw.get(c) for c in columns]
        extra = set(raw.keys()) - set(columns)
        if extra:
            n_extra_cols += 1
        w.writerow(row)
    if n_extra_cols:
        print(f"[db.features] NOTE: {n_extra_cols} rows carried columns outside the "
              f"canonical raw schema (ignored, schema check should have blocked these)")
    print(f"[db.features] fetch_raw_frame(): {len(fetched)} rows from all merged batches")
    buf.seek(0)
    return pd.read_csv(buf, low_memory=False)


def aggregate_features() -> pd.DataFrame:
    """The cross-row aggregates, pushed down to SQL. Returns one row per
    application_id with columns matching the pandas groupby-transforms:

      mobile_application_count      COUNT(*) by raw mobile_no        (NULL key -> NULL)
      ip_application_count          COUNT(*) by raw ip_address
      mobile_unique_names           COUNT(DISTINCT applicant_name) by mobile
      mobile_unique_fathers         COUNT(DISTINCT father_name)    by mobile
      institute_application_count   COUNT(*) by c_institution_id
      income_rank_in_district       pandas rank(pct=True, method='average'):
                                    (RANK + (ties-1)/2) / district_size
      income_deviation_from_state_median   income - PERCENTILE_CONT(0.5)
                                    (tolerance column: interpolation float-op
                                     order differs from pandas median)

    Keys group on the JSONB values' text form — every row was serialised by
    the same ingest path, so equal values have equal text. Income is coerced
    exactly like the pandas path: NULL->0, negative->0.
    """
    sql = f"""
    WITH pop AS (
        SELECT a.application_id,
               a.raw->>'mobile_no'         AS mobile_key,
               a.raw->>'ip_address'        AS ip_key,
               a.raw->>'applicant_name'    AS name_val,
               a.raw->>'father_name'       AS father_val,
               a.raw->>'c_institution_id'  AS inst_key,
               a.raw->>'permanent_district_id' AS district_key,
               a.raw->>'domicile_state_id' AS state_key,
               GREATEST(COALESCE((a.raw->>'annual_family_income')::float8, 0), 0) AS income
        FROM applications a JOIN batches b ON a.batch_id = b.batch_id
        WHERE b.status = 'merged'
    ),
    mob AS (
        SELECT mobile_key,
               COUNT(*)                        AS mobile_application_count,
               COUNT(DISTINCT name_val)        AS mobile_unique_names,
               COUNT(DISTINCT father_val)      AS mobile_unique_fathers
        FROM pop WHERE mobile_key IS NOT NULL GROUP BY mobile_key
    ),
    ip AS (
        SELECT ip_key, COUNT(*) AS ip_application_count
        FROM pop WHERE ip_key IS NOT NULL GROUP BY ip_key
    ),
    inst AS (
        SELECT inst_key, COUNT(*) AS institute_application_count
        FROM pop WHERE inst_key IS NOT NULL GROUP BY inst_key
    ),
    med AS (
        SELECT state_key,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY income) AS state_median
        FROM pop WHERE state_key IS NOT NULL GROUP BY state_key
    ),
    ranked AS (
        SELECT application_id,
               CASE WHEN district_key IS NULL THEN NULL ELSE
                 (RANK() OVER (PARTITION BY district_key ORDER BY income)
                  + (COUNT(*) OVER (PARTITION BY district_key, income) - 1) / 2.0
                 )::float8
                 / (COUNT(*) OVER (PARTITION BY district_key))::float8
               END AS income_rank_in_district
        FROM pop WHERE district_key IS NOT NULL
    )
    SELECT p.application_id,
           mob.mobile_application_count::float8,
           mob.mobile_unique_names::float8,
           mob.mobile_unique_fathers::float8,
           ip.ip_application_count::float8,
           inst.institute_application_count::float8,
           r.income_rank_in_district,
           CASE WHEN med.state_key IS NULL THEN NULL
                ELSE p.income - med.state_median END AS income_deviation_from_state_median
    FROM pop p
    LEFT JOIN mob    ON mob.mobile_key = p.mobile_key
    LEFT JOIN ip     ON ip.ip_key = p.ip_key
    LEFT JOIN inst   ON inst.inst_key = p.inst_key
    LEFT JOIN med    ON med.state_key = p.state_key
    LEFT JOIN ranked r ON r.application_id = p.application_id
    """
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return pd.DataFrame(rows, columns=[
        "application_id", "mobile_application_count", "mobile_unique_names",
        "mobile_unique_fathers", "ip_application_count",
        "institute_application_count", "income_rank_in_district",
        "income_deviation_from_state_median",
    ])


# ── persisted scaling parameters (hard stop 11) ──────────────────────────────

def save_scaling_params(schema_version: str, params: list[dict]) -> None:
    """params: [{feature_name, col_min, col_max, scale_factor, offset, log1p}]"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO feature_scaling (schema_version, feature_name,"
                " col_min, col_max, scale_factor, offset_, log1p)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (schema_version, feature_name) DO UPDATE SET"
                " col_min = EXCLUDED.col_min, col_max = EXCLUDED.col_max,"
                " scale_factor = EXCLUDED.scale_factor, offset_ = EXCLUDED.offset_,"
                " log1p = EXCLUDED.log1p, fitted_at = now()",
                [(schema_version, p["feature_name"], p["col_min"], p["col_max"],
                  p["scale_factor"], p["offset"], p["log1p"]) for p in params],
            )
    print(f"[db.features] persisted {len(params)} scaling params ({schema_version})")


def load_scaling_params(schema_version: str) -> dict[str, dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT feature_name, col_min, col_max, scale_factor, offset_, log1p"
            " FROM feature_scaling WHERE schema_version = %s", (schema_version,)
        ).fetchall()
    return {r[0]: {"col_min": r[1], "col_max": r[2], "scale_factor": r[3],
                   "offset": r[4], "log1p": r[5]} for r in rows}


# ── edge groups for the graph builder ────────────────────────────────────────

def edge_groups(identity_col: str) -> list[tuple[str, list[str]]]:
    """Shared-value groups (>= 2 members) for one identity relation, sorted by
    value — the SQL replacement for pandas groupby in graph_builder. Values in
    identity_keys already carry the graph builder's exact normalisation
    (str().lower().strip(); ''/'nan' -> NULL), proven identical in Gate 2."""
    assert identity_col in {"mobile_no", "ip_address", "father_name_norm",
                            "mother_name_norm", "pincode"}
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT ik.{identity_col}, array_agg(ik.application_id)"
            f" FROM identity_keys ik"
            f" JOIN applications a ON a.application_id = ik.application_id"
            f" JOIN batches b ON b.batch_id = a.batch_id"
            f" WHERE b.status = 'merged' AND ik.{identity_col} IS NOT NULL"
            f" GROUP BY ik.{identity_col} HAVING COUNT(*) >= 2"
            f" ORDER BY ik.{identity_col}",
        ).fetchall()
    return [(r[0], r[1]) for r in rows]
