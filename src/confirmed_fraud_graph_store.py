"""
confirmed_fraud_graph_store.py

Supervisor-confirmed fraud pattern store (Track F).
Persists subgraphs and manages their lifecycle: 
FLAGGED -> CONFIRMED -> SELECTED -> PROMOTED / REJECTED

Writes: data/processed/confirmed_fraud_graph_store.json
WARNING: By project-lead decision, this store retains real application IDs
and therefore holds PII.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
import torch

STORE_PATH = Path("data/processed/confirmed_fraud_graph_store.json")
EXPOSURE_GRAPH_PT = Path("data/processed/synthetic_exposure_graph_v3.pt")

VALID_FRAUD_TYPES = {
    "IP_CLUSTER",
    "FEE_INFLATION",
    "INCOME_VIOLATION",
    "NAME_COLLISION",
    "CROSS_CHANNEL",
    "OTHER",
}

VALID_STATES = {
    "FLAGGED",
    "CONFIRMED",
    "SELECTED",
    "PROMOTED",
    "REJECTED"
}

def _load_store() -> dict:
    if STORE_PATH.exists():
        return json.loads(STORE_PATH.read_text())
    return {"patterns": {}}


def _save_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2))


def add_confirmed_pattern(
    app_id: str,
    fraud_type: str,
    subgraph: dict,
    confirmed_by: str,
    notes: str = ""
) -> str:
    """
    Adds a confirmed pattern to the store, initially in CONFIRMED state.
    Returns the generated pattern_id.
    """
    if fraud_type not in VALID_FRAUD_TYPES:
        raise ValueError(f"fraud_type must be one of {VALID_FRAUD_TYPES}, got '{fraud_type}'")

    store = _load_store()
    
    pattern_id = f"pat_{uuid.uuid4().hex[:8]}"
    
    store["patterns"][pattern_id] = {
        "pattern_id": pattern_id,
        "center_app_id": app_id,
        "fraud_type": fraud_type,
        "subgraph": subgraph,
        "confirmed_by": confirmed_by,
        "state": "CONFIRMED",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "notes": notes
    }
    
    _save_store(store)
    return pattern_id


def list_pending() -> list[dict]:
    """List patterns in CONFIRMED state waiting for review/selection."""
    store = _load_store()
    return [p for p in store["patterns"].values() if p["state"] == "CONFIRMED"]


def count_pending() -> int:
    return len(list_pending())


def select(pattern_ids: list[str]) -> None:
    """Transition specified patterns from CONFIRMED to SELECTED."""
    store = _load_store()
    for pid in pattern_ids:
        if pid in store["patterns"]:
            p = store["patterns"][pid]
            if p["state"] == "CONFIRMED":
                p["state"] = "SELECTED"
                p["updated_at"] = datetime.utcnow().isoformat()
    _save_store(store)


def promote(pattern_ids: list[str]) -> list[dict]:
    """
    Transition specified patterns from SELECTED to PROMOTED.
    Writes the selected subgraphs into the topology exposure set (T6d logic hook).
    Returns the promoted patterns for the retrain loop.
    """
    store = _load_store()
    promoted = []
    
    for pid in pattern_ids:
        if pid in store["patterns"]:
            p = store["patterns"][pid]
            if p["state"] == "SELECTED":
                p["state"] = "PROMOTED"
                p["updated_at"] = datetime.utcnow().isoformat()
                promoted.append(p)

    _save_store(store)

    # Append each promoted pattern's ring to the topology-exposure set so the
    # RGCN stream actually learns it (previously a no-op stub). The subgraph
    # carries the member application_ids (Flag-for-LOE stores {"nodes": [...],
    # "edges": [{"type": <relation>}]}); we extract their REAL intra-ring edges
    # from the identity graph, falling back to a clique on the reviewer-asserted
    # relation when the members share no typed attribute. Best-effort: a pattern
    # whose members aren't in the current graph is skipped, not fatal.
    from src.pattern_ingest_v3 import append_ring_to_topology_exposure
    for p in promoted:
        sg = p.get("subgraph") or {}
        member_ids = [str(m) for m in (sg.get("nodes") or sg.get("member_ids") or [])]
        if not member_ids:
            member_ids = [str(p.get("center_app_id"))] if p.get("center_app_id") else []
        asserted = [e.get("type") for e in (sg.get("edges") or []) if isinstance(e, dict) and e.get("type")]
        if len(member_ids) < 2:
            p["exposure"] = {"appended": False, "reason": "fewer than 2 members"}
            continue
        try:
            info = append_ring_to_topology_exposure(member_ids, fallback_edge_types=asserted or None)
            p["exposure"] = {"appended": True, **info}
        except Exception as e:  # noqa: BLE001 — never let a bad subgraph break promotion
            p["exposure"] = {"appended": False, "reason": str(e)}

    _save_store(store)
    return promoted


def reject(pattern_ids: list[str]) -> None:
    """Transition specified patterns to REJECTED."""
    store = _load_store()
    for pid in pattern_ids:
        if pid in store["patterns"]:
            p = store["patterns"][pid]
            p["state"] = "REJECTED"
            p["updated_at"] = datetime.utcnow().isoformat()
    _save_store(store)
