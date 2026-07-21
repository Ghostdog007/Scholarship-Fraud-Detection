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


def remove_patterns(pattern_ids: list[str]) -> dict:
    """Hard-delete pattern records from the store (flagged-history cleanup).

    Removes only the store entry. IMPORTANT: it does NOT unwind anything a
    PROMOTED pattern already did — its ring may already sit in the topology-
    exposure set (synthetic_exposure_graph_v3.pt) and be baked into the current
    checkpoint. Deleting the record just stops it showing in the history/queue;
    reversing its training effect needs a rebuild/retrain. The return value flags
    which deleted ids were PROMOTED so the caller can warn the reviewer."""
    store = _load_store()
    removed, not_found, removed_promoted = [], [], []
    for pid in pattern_ids:
        p = store["patterns"].pop(pid, None)
        if p is None:
            not_found.append(pid)
            continue
        removed.append(pid)
        if p.get("state") == "PROMOTED":
            removed_promoted.append(pid)
    if removed:
        _save_store(store)
    return {
        "removed": removed,
        "not_found": not_found,
        "removed_promoted": removed_promoted,
        "remaining": len(store["patterns"]),
    }


def list_all() -> list[dict]:
    """Every pattern ever flagged, all states, newest first — the persistent,
    cross-session 'flagged directory'. This store is written on disk
    (STORE_PATH) so it survives restarts and sessions; this just surfaces it."""
    store = _load_store()
    pats = list(store["patterns"].values())
    pats.sort(key=lambda p: p.get("updated_at") or p.get("created_at") or "", reverse=True)
    return pats


# ── IP-cluster coverage check (soft, read-only) ─────────────────────────────────
# Answers "has this application's IP cluster already been flagged for LOE?" so a
# reviewer doesn't re-add the same ring. HEURISTIC by design: it matches on the
# shares_ip identity edge only and NEVER mutates anything — the reviewer confirms
# via the 3D identity ring. Not a rule/threshold (hard stop #1): it's graph-edge
# membership, no numeric cutoff against a domain concept.
_GRAPH_PT   = Path("data/processed/identity_graph_v3.pt")
_NODEG_CSV  = Path("data/processed/engineered_features_v3_nodeg.csv")


# Cache the shares_ip adjacency so an interactive coverage check doesn't reload
# the ~9 MB graph on every card open. Keyed on the (graph, nodeg) file mtimes, so
# a rebuild (retrain / pattern ingest) transparently invalidates it — advisory
# data, but this keeps it honest after the graph changes.
_ip_cache: dict = {"key": None, "ids": None, "id_to_node": None, "adj": None}


def _load_ip_adjacency() -> dict | None:
    if not (_GRAPH_PT.exists() and _NODEG_CSV.exists()):
        return None
    key = (_GRAPH_PT.stat().st_mtime, _NODEG_CSV.stat().st_mtime)
    if _ip_cache["key"] == key:
        return _ip_cache
    import pandas as pd

    ids = pd.read_csv(_NODEG_CSV, usecols=["application_id"])["application_id"].astype(str).tolist()
    id_to_node = {a: i for i, a in enumerate(ids)}
    data = torch.load(_GRAPH_PT, weights_only=False)
    ei = data["application", "shares_ip", "application"].edge_index
    adj: dict[int, set[int]] = {}
    if ei.numel():
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for u, v in zip(src, dst):      # graph stores both directions
            if u == v:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    _ip_cache.update(key=key, ids=ids, id_to_node=id_to_node, adj=adj)
    return _ip_cache


def _shares_ip_neighbors(app_id: str) -> set[str] | None:
    """Application IDs sharing a shares_ip edge with app_id in the current
    identity graph (excludes app_id). Returns None when the graph/mapping files
    are absent (so callers can say 'graph unavailable' rather than 'no match')."""
    c = _load_ip_adjacency()
    if c is None:
        return None
    node = c["id_to_node"].get(str(app_id))
    if node is None:
        return set()  # scored population is known but this id isn't in it
    ids = c["ids"]
    return {ids[n] for n in c["adj"].get(node, ())}


def ip_coverage_for_app(app_id: str) -> dict:
    """Soft, IP-only 'is this cluster already flagged?' check across ALL prior
    sessions (reads the persistent store + the identity graph). Reports every
    non-REJECTED pattern that either lists app_id as a member OR shares an IP
    edge with one of its members. Purely informational — the caller surfaces it
    as a warning the reviewer verifies manually; nothing is mutated."""
    app_id = str(app_id)
    store  = _load_store()
    patterns  = [p for p in store["patterns"].values() if p.get("state") != "REJECTED"]
    neighbors = _shares_ip_neighbors(app_id)   # None => graph unavailable

    matches = []
    for p in patterns:
        sg = p.get("subgraph") or {}
        members = {str(m) for m in (sg.get("nodes") or sg.get("member_ids") or [])}
        if not members and p.get("center_app_id"):
            members = {str(p["center_app_id"])}
        direct = app_id in members
        shared = sorted(members & neighbors) if neighbors else []
        if not (direct or shared):
            continue
        exposure = p.get("exposure") or {}
        matches.append({
            "pattern_id":    p["pattern_id"],
            "fraud_type":    p.get("fraud_type"),
            "state":         p.get("state"),
            "in_exposure":   bool(exposure.get("appended")),
            "cluster_id":    exposure.get("cluster_id"),
            "confirmed_by":  p.get("confirmed_by"),
            "updated_at":    p.get("updated_at"),
            "is_member":     direct,
            "shared_ip_with": shared[:8],
            "n_shared_ip":   len(shared),
        })

    return {
        "application_id":  app_id,
        "covered":         bool(matches),
        "graph_available": neighbors is not None,
        "matches":         matches,
    }
