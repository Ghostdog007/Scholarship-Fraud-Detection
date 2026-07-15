"""
pattern_ingest_v3.py

Supervisor-supplied fraud-PATTERN intake (Track F extension).

Motivation: the LOE / confirmed-fraud loop only ever learns from patterns the
MODEL surfaced (flagged apps, or reviewer "Flag for LOE" rings). This module
lets a supervisor bring in a brand-new, relationally-complex fraud ring the model
has never seen — as a CSV of full raw-schema rows — and:

  1. TEST it read-only: merge → rebuild features+graph → score with the current
     checkpoint → restore. Answers "does the model already catch this ring?"
     (the relational context is real — the identity graph is rebuilt, so the
     members' shared IP/mobile/name/pincode edges exist).

  2. INGEST it as a relational pattern: permanently merge the ring, extract its
     real intra-ring subgraph (all 5 edge types), APPEND it as a new cluster to
     the topology-exposure set (data/processed/synthetic_exposure_graph_v3.pt),
     and record it in BOTH confirmed stores (graph store as a PROMOTED pattern +
     tabular store as confirmed rows). The next incremental fine-tune then learns
     the ring through the RGCN topology-exposure stream. Retrain dispatch is left
     to the caller (endpoint) — this module never trains and never auto-advances.

This is REAL supervisor-provided data flowing into the confirmed set — it does
NOT synthesize exposure and so does not touch hard stop #7 (no CTGAN/TVAE/GAN).

Reuses, unchanged: src.api.dataset_ops (merge/rebuild/backup/restore),
src.api.inference.score_dataset_only, src.graph_builder_v3.EDGE_RAW_COLS,
src.confirmed_fraud_store, src.confirmed_fraud_graph_store.

Run:
  python -m src.pattern_ingest_v3 --test    data/uploads/ring.csv
  python -m src.pattern_ingest_v3 --ingest  data/uploads/ring.csv \
        --fraud-type IP_CLUSTER --by lead_1 --notes "landlord-address ring"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config_v3 import EDGE_TYPES, N_FEATURES

NODEG_CSV   = Path("data/processed/engineered_features_v3_nodeg.csv")
FINAL_CSV   = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON = Path("data/processed/v3_feature_schema.json")
GRAPH_PT    = Path("data/processed/identity_graph_v3.pt")
TOPO_PT     = Path("data/processed/synthetic_exposure_graph_v3.pt")


# ── read-only test ────────────────────────────────────────────────────────────
def test_pattern(csv_path: str | Path) -> dict:
    """Score a supervisor ring against the CURRENT model without changing
    anything. Merges the rows, rebuilds features+graph, scores just the ring,
    then restores every canonical file. Returns per-member scores."""
    from src.api import dataset_ops
    from src.api.inference import score_dataset_only

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise ValueError(f"CSV not found: {csv_path}")
    ring_df = pd.read_csv(csv_path, low_memory=False)
    member_ids = set(ring_df["application_id"].astype(str))

    backup_dir = dataset_ops.backup_canonical_files(label="pattern_test")
    try:
        dataset_ops.merge_dataset_into_raw(csv_path)
        dataset_ops.rebuild_features_and_graph()
        scored = score_dataset_only(app_ids_to_return=member_ids)
    finally:
        dataset_ops.restore_canonical_files(backup_dir)
        import shutil
        shutil.rmtree(backup_dir, ignore_errors=True)

    scores = (scored.sort_values("hybrid_anomaly_score", ascending=False)
                    [["application_id", "hybrid_anomaly_score"]]
                    .to_dict(orient="records"))
    vals = [r["hybrid_anomaly_score"] for r in scores]
    return {
        "n_members": len(scores),
        "members":   scores,
        "mean_score": float(np.mean(vals)) if vals else None,
        "max_score":  float(np.max(vals))  if vals else None,
        "min_score":  float(np.min(vals))  if vals else None,
    }


# ── subgraph extraction + topology-exposure append ────────────────────────────
def _schema_feature_order() -> list[str]:
    return json.loads(SCHEMA_JSON.read_text())["features"]


def _extract_ring_subgraph(
    member_ids: list[str],
    fallback_edge_types: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """From the CURRENT canonical graph + features, pull the ring's real
    structure: 44-dim node features (schema order), intra-ring edges (both
    directions, as stored) across all 5 relations, and their edge-type ids.

    Node index = row order of NODEG_CSV (what the graph was built from). Returns
    (x[k,44], edge_index[2,E_local], edge_type[E_local], meta).

    fallback_edge_types: when the members share no real attribute edge (e.g. a
    reviewer-asserted ring whose link isn't one of the 5 typed identity fields),
    build a full clique on these relation names instead of erroring. Used by the
    Flag-for-LOE promote path, where the reviewer vouches for the relationship."""
    member_ids = [str(m) for m in member_ids]

    # id -> global node index, from the CSV the graph was built on.
    nodeg = pd.read_csv(NODEG_CSV, low_memory=False)
    id_to_node = {str(a): i for i, a in enumerate(nodeg["application_id"].values)}
    missing = [m for m in member_ids if m not in id_to_node]
    if missing:
        raise ValueError(f"{len(missing)} member id(s) not in the graph after rebuild: {missing[:5]}")

    # Stable local ordering of the ring's nodes.
    local_ids   = list(dict.fromkeys(member_ids))          # de-dupe, keep order
    global_idx  = [id_to_node[m] for m in local_ids]
    g2l         = {g: l for l, g in enumerate(global_idx)}
    member_set  = set(global_idx)

    # 44-dim features from FINAL_CSV, aligned to schema feature order.
    feats = _schema_feature_order()
    final = pd.read_csv(FINAL_CSV, low_memory=False).set_index("application_id")
    x = final.loc[local_ids, feats].to_numpy(dtype=np.float32)
    if x.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {x.shape[1]}")

    # Intra-ring edges from the identity graph, per relation.
    data = torch.load(GRAPH_PT, weights_only=False)
    src_l, dst_l, etype_l = [], [], []
    per_relation = {}
    for t, et in enumerate(EDGE_TYPES):
        ei = data["application", et, "application"].edge_index
        if ei.numel() == 0:
            per_relation[et] = 0
            continue
        ei = ei.numpy()
        keep = [(u, v) for u, v in zip(ei[0], ei[1]) if u in member_set and v in member_set]
        per_relation[et] = len(keep)
        for u, v in keep:
            src_l.append(g2l[u]); dst_l.append(g2l[v]); etype_l.append(t)

    if not src_l:
        if fallback_edge_types:
            # Reviewer-asserted ring: build a full clique on the named relation(s).
            k = len(local_ids)
            for et in fallback_edge_types:
                if et not in EDGE_TYPES:
                    continue
                t = EDGE_TYPES.index(et)
                for a in range(k):
                    for b in range(k):
                        if a != b:
                            src_l.append(a); dst_l.append(b); etype_l.append(t)
                per_relation[et] = per_relation.get(et, 0) + k * (k - 1)
        if not src_l:
            raise ValueError(
                "Members share no identity attribute (IP/mobile/name/pincode) and no "
                "asserted relation was given — no connected ring to learn.")

    edge_index = torch.tensor([src_l, dst_l], dtype=torch.long)
    edge_type  = torch.tensor(etype_l, dtype=torch.long)
    meta = {
        "n_nodes": len(local_ids),
        "n_edges_directed": edge_index.shape[1],
        "edges_per_relation": {k: v for k, v in per_relation.items() if v},
        "member_ids": local_ids,
    }
    return torch.tensor(x), edge_index, edge_type, meta


def append_ring_to_topology_exposure(
    member_ids: list[str],
    fallback_edge_types: list[str] | None = None,
) -> dict:
    """Extract the ring's subgraph and append it as ONE new cluster to
    synthetic_exposure_graph_v3.pt (creating the file if absent). This is the
    real implementation behind the previously-stubbed topology-LOE injection."""
    x, edge_index, edge_type, meta = _extract_ring_subgraph(member_ids, fallback_edge_types)

    if TOPO_PT.exists():
        pack = torch.load(TOPO_PT, weights_only=False)
    else:
        pack = {
            "x": torch.zeros((0, N_FEATURES), dtype=torch.float32),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_type": torch.zeros((0,), dtype=torch.long),
            "cluster_id": torch.zeros((0,), dtype=torch.long),
        }

    offset = pack["x"].shape[0]
    next_cluster = int(pack["cluster_id"].max().item()) + 1 if pack["cluster_id"].numel() else 0

    pack["x"]          = torch.cat([pack["x"], x], dim=0)
    pack["edge_index"] = torch.cat([pack["edge_index"], edge_index + offset], dim=1)
    pack["edge_type"]  = torch.cat([pack["edge_type"], edge_type], dim=0)
    pack["cluster_id"] = torch.cat(
        [pack["cluster_id"], torch.full((x.shape[0],), next_cluster, dtype=torch.long)], dim=0)

    TOPO_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, TOPO_PT)

    return {
        "cluster_id": next_cluster,
        "n_nodes": meta["n_nodes"],
        "n_edges_directed": meta["n_edges_directed"],
        "edges_per_relation": meta["edges_per_relation"],
        "topology_exposure_total_nodes": int(pack["x"].shape[0]),
    }


# ── full ingest ───────────────────────────────────────────────────────────────
def ingest_pattern(
    csv_path: str | Path,
    fraud_type: str,
    confirmed_by: str,
    notes: str = "",
    cycle: str = "",
) -> dict:
    """Permanently ingest a supervisor ring as a relational pattern:
    merge → rebuild → append to topology exposure → record in both confirmed
    stores. Does NOT retrain (the caller dispatches the human-gated fine-tune)."""
    from src.api import dataset_ops
    from src.confirmed_fraud_store import add_confirmed
    from src.confirmed_fraud_graph_store import (
        add_confirmed_pattern, select, promote, VALID_FRAUD_TYPES,
    )

    if fraud_type not in VALID_FRAUD_TYPES:
        raise ValueError(f"fraud_type must be one of {VALID_FRAUD_TYPES}, got '{fraud_type}'")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise ValueError(f"CSV not found: {csv_path}")
    ring_df = pd.read_csv(csv_path, low_memory=False)
    member_ids = [str(a) for a in ring_df["application_id"].astype(str).tolist()]

    # Permanent merge + rebuild (NOT restored — the ring joins the dataset).
    n_added = dataset_ops.merge_dataset_into_raw(csv_path)
    dataset_ops.rebuild_features_and_graph()

    # Record in the graph pattern store, then promote it — promote() is what
    # appends the ring to the topology-exposure set (single source of that
    # logic, shared with the Flag-for-LOE path). The members are now in the
    # rebuilt graph, so their real intra-ring edges are extracted.
    pattern_id = add_confirmed_pattern(
        app_id=member_ids[0], fraud_type=fraud_type,
        subgraph={"nodes": member_ids, "source": "supervisor_csv_ingest"},
        confirmed_by=confirmed_by, notes=notes,
    )
    select([pattern_id])
    promoted = promote([pattern_id])
    topo = (promoted[0].get("exposure") if promoted else None) or {"appended": False}
    if not topo.get("appended"):
        raise ValueError(f"Topology-exposure append failed: {topo.get('reason', 'unknown')}")

    # Record each member in the tabular confirmed store too ("add to confirmed
    # fraud"), so it anchors the tabular LOE and the store counts.
    recorded, skipped = [], []
    for m in member_ids:
        try:
            add_confirmed(app_id=m, fraud_type=fraud_type,
                          confirmed_by=confirmed_by, notes=notes, cycle=cycle)
            recorded.append(m)
        except ValueError as e:
            skipped.append({"application_id": m, "error": str(e)})

    return {
        "status": "ok",
        "pattern_id": pattern_id,
        "n_rows_added": n_added,
        "n_members": len(member_ids),
        "topology_exposure": topo,
        "n_confirmed_recorded": len(recorded),
        "confirm_errors": skipped,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test",   metavar="CSV", help="read-only: score the ring with the current model")
    g.add_argument("--ingest", metavar="CSV", help="permanent: ingest the ring as a relational pattern")
    ap.add_argument("--fraud-type", default="OTHER")
    ap.add_argument("--by", default="cli")
    ap.add_argument("--notes", default="")
    ap.add_argument("--cycle", default="")
    args = ap.parse_args()

    if args.test:
        print(json.dumps(test_pattern(args.test), indent=2))
    else:
        print(json.dumps(ingest_pattern(args.ingest, args.fraud_type, args.by,
                                        args.notes, args.cycle), indent=2))
