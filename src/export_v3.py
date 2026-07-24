"""
export_v3.py

Export flagged applications for offline review / audit / appeals.

Two products, both built from artefacts the pipeline already wrote — this module
computes NOTHING new and invents no thresholds:

  * single-application bundle  (build_single_export)
        <app_id>_export.zip
          ├─ <app_id>_scorecard.csv   flat one-row audit summary (opens in Excel)
          ├─ <app_id>_card.html       the interactive reviewer card (self-contained)
          └─ <app_id>_evidence.json   the machine-readable card object (appeals)

  * bulk bundle                (build_bulk_export)
        flagged_export_<ts>.zip
          ├─ manifest.csv             one scorecard row per flagged application
          ├─ cards/<app_id>.html      per-application reviewer cards
          └─ evidence/<app_id>.json   per-application evidence objects

The scorecard row carries the model-traceability summary (which model drove the
score, each detector's share, the fired triggers) so the CSV alone answers
"which model flagged this, and why".

Reads:  outputs/explanation_cards_v3.json, outputs/risk_scores_v3.csv
Writes: (API) zip bytes in-memory; (CLI) outputs/exports/<name>.zip

Run:
  python -m src.export_v3 --app-id APP00042      # single
  python -m src.export_v3 --bulk                 # all flagged
"""

import argparse
import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CARDS_JSON = Path("outputs/explanation_cards_v3.json")
RISK_CSV   = Path("outputs/risk_scores_v3.csv")
OUT_DIR    = Path("outputs/exports")

# Flat scorecard columns, in order. Kept explicit so the CSV schema is stable
# for downstream tooling / auditors.
SCORECARD_COLUMNS = [
    "application_id", "risk_score_v3", "risk_rank", "risk_percentile",
    "label_source", "review_status",
    # max fusion (changed 2026-07-22): exactly one detector drives the score, so
    # each detector's OWN normalised value is reported (not a "share of a blend")
    "driving_model", "driving_margin",
    "subspace_normalized", "dense_relational_normalized", "hybrid_normalized",
    # Deep SAD (V4.2, 2026-07-22): supplementary, NOT a fusion driver — exported
    # for traceability only, does not factor into driving_model/driving_margin above.
    "deepsad_percentile",
    "n_triggers", "triggers", "n_evt_crossings",
    "top_feat_1", "top_feat_1_value", "top_feat_1_expected", "top_feat_1_error_pct",
    "top_feat_2", "top_feat_2_value", "top_feat_2_expected", "top_feat_2_error_pct",
    "top_feat_3", "top_feat_3_value", "top_feat_3_expected", "top_feat_3_error_pct",
]


def _load_cards() -> list[dict]:
    if not CARDS_JSON.exists():
        return []
    return json.loads(CARDS_JSON.read_text())


def _find_card(cards: list[dict], app_id: str) -> dict | None:
    return next((c for c in cards if str(c["application_id"]) == str(app_id)), None)


def scorecard_row(card: dict) -> dict:
    """Flatten one card into the flat audit row. Pure projection of the card
    JSON — no recomputation."""
    ev    = card.get("evidence", {})
    fc    = ev.get("fusion_contributions", {})
    prov  = ev.get("provenance", [])
    row = {c: "" for c in SCORECARD_COLUMNS}
    row.update({
        "application_id":  card.get("application_id"),
        "risk_score_v3":   card.get("risk_score_v3"),
        "risk_rank":       ev.get("risk_rank"),
        "risk_percentile": ev.get("risk_percentile"),
        "label_source":    ev.get("label_source"),
        "review_status":   card.get("review_status"),
        # provenance is ordered by normalised value desc; first entry is the driver
        "driving_model":   prov[0] if prov else "",
        "driving_margin":  next((round(c["margin_over_next"], 4) for c in fc.values()
                                 if c.get("is_driver") and c.get("margin_over_next") is not None), ""),
        "subspace_normalized":           round(fc.get("subspace", {}).get("normalized", 0.0), 4),
        "dense_relational_normalized":   round(fc.get("dense_relational", {}).get("normalized", 0.0), 4),
        "hybrid_normalized":             round(fc.get("hybrid", {}).get("normalized", 0.0), 4),
        "deepsad_percentile": (ev.get("deepsad") or {}).get("percentile", ""),
        "n_triggers":      len(card.get("triggers", [])),
        "triggers":        ";".join(card.get("triggers", [])),
        "n_evt_crossings": len(ev.get("evt_crossings", [])),
    })
    for i, f in enumerate(card.get("top_feature_errors", [])[:3], start=1):
        row[f"top_feat_{i}"]           = f.get("feature_label", f.get("feature"))
        row[f"top_feat_{i}_value"]     = f.get("value")
        row[f"top_feat_{i}_expected"]  = f.get("expected")
        row[f"top_feat_{i}_error_pct"] = f.get("error_percentile")
    return row


def _scorecard_csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SCORECARD_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def _card_html_bytes(app_id: str) -> bytes:
    """Render the reviewer card HTML for the bundle. Falls back to a stub if the
    graph/risk context is unavailable so the export never hard-fails."""
    from src.xai_card_html_v3 import build_card_html
    html = build_card_html(app_id)
    if html is None:
        html = (f"<!doctype html><meta charset='utf-8'>"
                f"<p style='font-family:sans-serif'>No reviewer card available for "
                f"{app_id} (not flagged, or cards not generated).</p>")
    return html.encode("utf-8")


def build_single_export(app_id: str) -> tuple[str, bytes] | None:
    """Zip bytes for one application: scorecard.csv + card.html + evidence.json.
    Returns (filename, data) or None if the application has no card."""
    cards = _load_cards()
    card  = _find_card(cards, app_id)
    if card is None:
        return None

    safe = str(app_id).replace("/", "_")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{safe}_scorecard.csv", _scorecard_csv_bytes([scorecard_row(card)]))
        z.writestr(f"{safe}_card.html",     _card_html_bytes(app_id))
        z.writestr(f"{safe}_evidence.json", json.dumps(card, indent=2).encode("utf-8"))
    return f"{safe}_export.zip", buf.getvalue()


def build_bulk_export() -> tuple[str, bytes] | None:
    """Zip bytes for every flagged application: manifest.csv + cards/ + evidence/.
    Returns (filename, data) or None if there are no cards at all."""
    cards = _load_cards()
    if not cards:
        return None

    rows = [scorecard_row(c) for c in cards]
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    buf  = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.csv", _scorecard_csv_bytes(rows))
        for c in cards:
            safe = str(c["application_id"]).replace("/", "_")
            z.writestr(f"cards/{safe}.html",     _card_html_bytes(c["application_id"]))
            z.writestr(f"evidence/{safe}.json",  json.dumps(c, indent=2).encode("utf-8"))
    return f"flagged_export_{ts}.zip", buf.getvalue()


def _ring_html_bytes(app_id: str, risk_map: dict | None) -> bytes | None:
    """Render the interactive 3D identity-ring HTML for one application, or None
    if the application has no typed edges (not in the identity graph). This is
    the 'relevant SVG' for the selected bundle — a self-contained rotatable
    Plotly view, not a static image (no kaleido dependency)."""
    from src.xai_card_html_v3 import build_ring_html
    html = build_ring_html(app_id, risk_map=risk_map)
    if html is None:
        return None
    return html.encode("utf-8")


def build_selected_export(app_ids: list[str]) -> tuple[str, bytes] | None:
    """Zip bytes for a reviewer-chosen subset of flagged applications.

    Per app, bundles: scorecard row (combined manifest.csv) + reviewer-card HTML
    (cards/) + interactive identity-ring HTML (rings/) + evidence JSON
    (evidence/). IDs with no card are skipped and listed in _skipped.txt; IDs
    with a card but no graph edges get a card+evidence but no ring (noted in the
    manifest is not needed — absence of the rings/ file is the signal).

    Like the other builders, this recomputes NOTHING except the lazy Plotly ring
    (which is a pure render of existing graph + risk artefacts). Returns
    (filename, data), or None if none of the requested IDs have a card."""
    cards = _load_cards()
    if not cards:
        return None

    # Preserve caller order; de-dupe while keeping first occurrence.
    seen: set[str] = set()
    ordered_ids = [x for x in (str(a) for a in app_ids)
                   if not (x in seen or seen.add(x))]

    picked  = [(aid, _find_card(cards, aid)) for aid in ordered_ids]
    found   = [(aid, c) for aid, c in picked if c is not None]
    skipped = [aid for aid, c in picked if c is None]
    if not found:
        return None

    # Load risk_map once and reuse for every ring render (build_ring_html would
    # otherwise re-read risk_scores_v3.csv per application).
    risk_map: dict = {}
    if RISK_CSV.exists():
        import pandas as pd
        rdf = pd.read_csv(RISK_CSV)
        risk_map = dict(zip(rdf["application_id"], rdf["risk_score_v3"]))

    rows = [scorecard_row(c) for _, c in found]
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    buf  = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.csv", _scorecard_csv_bytes(rows))
        for aid, c in found:
            safe = str(aid).replace("/", "_")
            z.writestr(f"cards/{safe}.html",    _card_html_bytes(aid))
            z.writestr(f"evidence/{safe}.json", json.dumps(c, indent=2).encode("utf-8"))
            ring = _ring_html_bytes(aid, risk_map)
            if ring is not None:
                z.writestr(f"rings/{safe}.html", ring)
        if skipped:
            z.writestr("_skipped.txt",
                       ("These requested IDs had no reviewer card and were skipped "
                        "(not flagged, or cards not generated):\n"
                        + "\n".join(skipped)).encode("utf-8"))
    return f"selected_export_{len(found)}apps_{ts}.zip", buf.getvalue()


# ── cohort (staged, read-only preview) export ──────────────────────────────────
# Mirrors the committed builders but sourced from a cohort's staged files
# (outputs/staged_scores_<name>.csv + the staged graph bundle). Scores are
# PRE-FUSION — see the README.txt written into every bundle.
STAGED_SCORECARD_COLUMNS = [
    "application_id", "hybrid_anomaly_score", "feature_pred_error", "edge_pred_error",
    "score_type",
    "top_feat_1", "top_feat_1_error",
    "top_feat_2", "top_feat_2_error",
    "top_feat_3", "top_feat_3_error",
]


def _staged_paths(name: str):
    return (Path(f"outputs/staged_scores_{name}.csv"),
            Path(f"outputs/staged_graph_{name}.pt"),
            Path(f"outputs/staged_nodeorder_{name}.csv"))


def _staged_scorecard_row(row) -> dict:
    d = {c: "" for c in STAGED_SCORECARD_COLUMNS}
    d.update({
        "application_id":      row["application_id"],
        "hybrid_anomaly_score": row["hybrid_anomaly_score"],
        "feature_pred_error":  row.get("feature_pred_error", ""),
        "edge_pred_error":     row.get("edge_pred_error", ""),
        "score_type":          "hybrid_anomaly_score (pre-fusion preview)",
    })
    per = row.get("per_feature_error_json")
    if isinstance(per, str):
        top = sorted(json.loads(per).items(), key=lambda kv: kv[1], reverse=True)[:3]
        for i, (f, e) in enumerate(top, start=1):
            d[f"top_feat_{i}"] = f
            d[f"top_feat_{i}_error"] = round(float(e), 6)
    return d


def _staged_evidence_bytes(row) -> bytes:
    per = row.get("per_feature_error_json")
    ev = {
        "application_id":       str(row["application_id"]),
        "hybrid_anomaly_score": float(row["hybrid_anomaly_score"]),
        "feature_pred_error":   float(row.get("feature_pred_error", 0) or 0),
        "edge_pred_error":      float(row.get("edge_pred_error", 0) or 0),
        "per_feature_error":    json.loads(per) if isinstance(per, str) else {},
        "note": "PREVIEW — pre-fusion staged cohort score; NOT the committed risk_score_v3",
    }
    return json.dumps(ev, indent=2).encode("utf-8")


def _cohort_bundle(name: str, app_ids: list[str] | None,
                   include_rings: bool = False) -> tuple[str, bytes] | None:
    """Zip a cohort's staged evidence — manifest + per-app card/evidence (+ rings
    when include_rings). app_ids=None → all cohort rows (bulk). None if the cohort
    has no staged scores. Ring inclusion mirrors the committed builders: only the
    'selected' export embeds the heavy Plotly rings; bulk/single stay light."""
    import pandas as pd
    from src.xai_card_html_v3 import build_staged_card_html, build_ring_html

    scores_path, graph_pt, nodeorder = _staged_paths(name)
    if not scores_path.exists():
        return None
    sdf = pd.read_csv(scores_path)
    if app_ids is not None:
        wanted = {str(a) for a in app_ids}
        sdf = sdf[sdf["application_id"].astype(str).isin(wanted)]
    if sdf.empty:
        return None

    risk_map = dict(zip(sdf["application_id"], sdf["hybrid_anomaly_score"]))
    ring_ok  = include_rings and graph_pt.exists() and nodeorder.exists()
    rows     = [_staged_scorecard_row(r) for _, r in sdf.iterrows()]
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        buf2 = io.StringIO()
        w = csv.DictWriter(buf2, fieldnames=STAGED_SCORECARD_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        z.writestr("manifest.csv", buf2.getvalue().encode("utf-8"))
        z.writestr("README.txt", (
            f"Cohort preview export — {name}\n\n"
            "Scores here are the PRE-FUSION hybrid_anomaly_score (higher = more\n"
            "anomalous), NOT the committed fused risk_score_v3. This is a read-only\n"
            "preview of an evaluated (not-yet-ingested) cohort.\n").encode("utf-8"))
        for _, r in sdf.iterrows():
            aid  = str(r["application_id"])
            safe = aid.replace("/", "_")
            card = build_staged_card_html(name, aid)
            if card:
                z.writestr(f"cards/{safe}.html", card.encode("utf-8"))
            z.writestr(f"evidence/{safe}.json", _staged_evidence_bytes(r))
            if ring_ok:
                ring = build_ring_html(aid, risk_map=risk_map, graph_pt=graph_pt, nodeorder_csv=nodeorder)
                if ring:
                    z.writestr(f"rings/{safe}.html", ring.encode("utf-8"))
    label = "all" if app_ids is None else f"{len(sdf)}apps"
    return f"cohort_{name}_{label}_{ts}.zip", buf.getvalue()


def build_cohort_single_export(name: str, app_id: str):
    return _cohort_bundle(name, [app_id], include_rings=False)


def build_cohort_bulk_export(name: str):
    return _cohort_bundle(name, None, include_rings=False)


def build_cohort_selected_export(name: str, app_ids: list[str]):
    return _cohort_bundle(name, app_ids, include_rings=True)


# ── confirmed-pattern (flagged-history) export ─────────────────────────────────
# Mirrors the application exports above, sourced from confirmed_fraud_graph_store
# instead of the explanation cards. One manifest row + one full JSON record per
# pattern — the record already carries the subgraph, state, and (if promoted)
# the exposure cluster it landed in, so nothing is recomputed here either.
PATTERN_MANIFEST_COLUMNS = [
    "pattern_id", "fraud_type", "state", "center_app_id", "n_members",
    "confirmed_by", "created_at", "updated_at", "in_exposure", "exposure_cluster_id", "notes",
]


def _pattern_manifest_row(p: dict) -> dict:
    sg = p.get("subgraph") or {}
    members = sg.get("nodes") or sg.get("member_ids") or []
    exposure = p.get("exposure") or {}
    return {
        "pattern_id":          p.get("pattern_id"),
        "fraud_type":          p.get("fraud_type"),
        "state":               p.get("state"),
        "center_app_id":       p.get("center_app_id"),
        "n_members":           len(members),
        "confirmed_by":        p.get("confirmed_by"),
        "created_at":          p.get("created_at"),
        "updated_at":          p.get("updated_at"),
        "in_exposure":         bool(exposure.get("appended")),
        "exposure_cluster_id": exposure.get("cluster_id", ""),
        "notes":               p.get("notes", ""),
    }


def _pattern_manifest_csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=PATTERN_MANIFEST_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def build_pattern_bulk_export() -> tuple[str, bytes] | None:
    """Zip bytes for every pattern in the flagged-history store (all states,
    all sessions): manifest.csv + patterns/<pattern_id>.json (full record).
    Returns None if the store is empty."""
    from src.confirmed_fraud_graph_store import list_all
    patterns = list_all()
    if not patterns:
        return None

    rows = [_pattern_manifest_row(p) for p in patterns]
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    buf  = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.csv", _pattern_manifest_csv_bytes(rows))
        for p in patterns:
            z.writestr(f"patterns/{p['pattern_id']}.json", json.dumps(p, indent=2).encode("utf-8"))
    return f"patterns_export_{ts}.zip", buf.getvalue()


def build_pattern_selected_export(pattern_ids: list[str]) -> tuple[str, bytes] | None:
    """Zip bytes for a reviewer-chosen subset of the flagged-history store.
    Unknown ids are skipped and listed in _skipped.txt. Returns None if none
    of the requested ids exist."""
    from src.confirmed_fraud_graph_store import list_all
    patterns = list_all()
    if not patterns:
        return None

    seen: set[str] = set()
    ordered_ids = [x for x in (str(a) for a in pattern_ids)
                   if not (x in seen or seen.add(x))]

    by_id   = {p["pattern_id"]: p for p in patterns}
    found   = [by_id[pid] for pid in ordered_ids if pid in by_id]
    skipped = [pid for pid in ordered_ids if pid not in by_id]
    if not found:
        return None

    rows = [_pattern_manifest_row(p) for p in found]
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    buf  = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.csv", _pattern_manifest_csv_bytes(rows))
        for p in found:
            z.writestr(f"patterns/{p['pattern_id']}.json", json.dumps(p, indent=2).encode("utf-8"))
        if skipped:
            z.writestr("_skipped.txt",
                       "These requested pattern_ids were not found in the store:\n"
                       + "\n".join(skipped))
    return f"patterns_selected_export_{len(found)}_{ts}.zip", buf.getvalue()


def _write(result: tuple[str, bytes] | None, label: str) -> None:
    if result is None:
        print(f"[export] nothing to export for {label} — is outputs/explanation_cards_v3.json present?")
        return
    filename, data = result
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_bytes(data)
    print(f"[export] wrote {label}: {path}  ({len(data):,} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--app-id", help="export a single flagged application")
    g.add_argument("--bulk", action="store_true", help="export all flagged applications")
    g.add_argument("--ids", help="export a chosen subset: comma-separated application IDs")
    args = ap.parse_args()
    if args.bulk:
        _write(build_bulk_export(), "bulk")
    elif args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        _write(build_selected_export(ids), f"selected ({len(ids)} ids)")
    else:
        _write(build_single_export(args.app_id), f"application {args.app_id}")
