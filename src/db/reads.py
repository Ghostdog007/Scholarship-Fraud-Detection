"""Typed read queries for the API (migration step 2).

Each function returns payloads shaped EXACTLY like the file-based path it
mirrors — Gate 2 is an automated payload diff, so field names, ordering
rules, and float precision must match. Handlers switch between paths via the
NIC_READS_FROM_PG env flag (see src/api handlers); files remain written
regardless (hard stop 13).
"""

import os
from collections import Counter

from src.db.connection import get_connection

MODEL_VERSION = "v3_current"


def reads_from_pg() -> bool:
    """Step-2 switch: Postgres is the default read path (Gate 2 passed
    2026-07-21). Set NIC_READS_FROM_PG=0 to force the file path (escape
    hatch); handlers also fall back to files if a query raises."""
    return os.environ.get("NIC_READS_FROM_PG", "1") != "0"


def top_suspicious(n: int = 20) -> list[dict]:
    """Mirror of GET /v3/monitoring/top-suspicious reading risk_scores_v3.csv:
    records of {application_id, risk_score_v3, label_source}, sorted by risk
    desc. Ties broken by application_id for determinism (pandas' quicksort is
    unstable on ties; the Gate 2 harness compares ties as sets)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT application_id, final_risk_score, label_source FROM scores"
            " WHERE model_version = %s"
            " ORDER BY final_risk_score DESC NULLS LAST, application_id"
            " LIMIT %s",
            (MODEL_VERSION, n),
        ).fetchall()
    return [
        {"application_id": r[0], "risk_score_v3": r[1], "label_source": r[2]}
        for r in rows
    ]


def n_scored() -> int:
    """Mirror of the scored-population count (rows of risk_scores_v3.csv)."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM scores WHERE model_version = %s", (MODEL_VERSION,)
        ).fetchone()[0]


def fraud_store_summary() -> dict:
    """Mirror of GET /v3/monitoring/fraud-store-summary reading the JSON store."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT label, fraud_type FROM confirmed_fraud"
        ).fetchall()
    confirmed = [r for r in rows if r[0] == "confirmed"]
    fps = [r for r in rows if r[0] == "false_positive"]
    return {
        "n_confirmed": len(confirmed),
        "n_false_positives": len(fps),
        "by_fraud_type": dict(Counter(r[1] for r in confirmed)),
    }


def identity_row_exists(app_id: str) -> bool:
    with get_connection() as conn:
        return conn.execute(
            "SELECT 1 FROM identity_keys WHERE application_id = %s", (app_id,)
        ).fetchone() is not None


def induced_subgraph_edges(app_ids: list[str]) -> list[tuple[str, str, str]]:
    """All typed edges among the GIVEN node set (undirected, src<dst), from
    indexed identity_keys — powers ring/ego rendering without loading the
    .pt graph. Same shared-value semantics as the graph builder."""
    relations = {
        "shares_mobile":      "mobile_no",
        "shares_ip":          "ip_address",
        "shares_father_name": "father_name_norm",
        "shares_mother_name": "mother_name_norm",
        "shares_pincode":     "pincode",
    }
    out: list[tuple[str, str, str]] = []
    with get_connection() as conn:
        for rel, col in relations.items():
            rows = conn.execute(
                f"SELECT a.application_id, b.application_id FROM identity_keys a"
                f" JOIN identity_keys b ON b.{col} = a.{col}"
                f" AND a.application_id < b.application_id"
                f" WHERE a.application_id = ANY(%s) AND b.application_id = ANY(%s)"
                f" AND a.{col} IS NOT NULL",
                (app_ids, app_ids),
            ).fetchall()
            out.extend((r[0], r[1], rel) for r in rows)
    return out


def risk_scores_for(app_ids: list[str], score_col: str = "final_risk_score") -> dict[str, float]:
    """Score lookup for the given apps (primary batch), for ring/ego
    colouring. score_col is whitelisted — rings colour by final_risk_score,
    ego-graphs by hybrid_anomaly_score (matching their file-path sources)."""
    assert score_col in {"final_risk_score", "hybrid_anomaly_score"}
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT application_id, {score_col} FROM scores"
            f" WHERE model_version = %s AND application_id = ANY(%s)",
            (MODEL_VERSION, app_ids),
        ).fetchall()
    return {r[0]: r[1] for r in rows if r[1] is not None}


def ego_neighbors(app_id: str) -> dict[str, list[str]]:
    """1-hop typed neighbourhood from indexed identity_keys — the Postgres
    replacement for scanning the in-memory .pt graph. Returns
    {relation: sorted [neighbor application_ids]}; empty lists for relations
    where the app shares no value. Neighbourhoods must equal the .pt graph's
    per-relation adjacency exactly (Gate 2)."""
    relations = {
        "shares_mobile":      "mobile_no",
        "shares_ip":          "ip_address",
        "shares_father_name": "father_name_norm",
        "shares_mother_name": "mother_name_norm",
        "shares_pincode":     "pincode",
    }
    out: dict[str, list[str]] = {}
    with get_connection() as conn:
        for rel, col in relations.items():
            rows = conn.execute(
                f"SELECT b.application_id FROM identity_keys a"
                f" JOIN identity_keys b ON b.{col} = a.{col}"
                f" WHERE a.application_id = %s AND b.application_id <> %s"
                f" AND a.{col} IS NOT NULL",
                (app_id, app_id),
            ).fetchall()
            out[rel] = sorted(r[0] for r in rows)
    return out
