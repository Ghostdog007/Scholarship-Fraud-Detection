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
OUT_DIR    = Path("outputs/exports")

# Flat scorecard columns, in order. Kept explicit so the CSV schema is stable
# for downstream tooling / auditors.
SCORECARD_COLUMNS = [
    "application_id", "risk_score_v3", "risk_rank", "risk_percentile",
    "label_source", "review_status",
    "driving_model", "subspace_share", "dense_ip_share", "hybrid_share",
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
        # provenance is already ordered by share desc; first entry is the driver
        "driving_model":   prov[0] if prov else "",
        "subspace_share":  round(fc.get("subspace", {}).get("share", 0.0), 4),
        "dense_ip_share":  round(fc.get("dense_ip", {}).get("share", 0.0), 4),
        "hybrid_share":    round(fc.get("hybrid", {}).get("share", 0.0), 4),
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
    args = ap.parse_args()
    if args.bulk:
        _write(build_bulk_export(), "bulk")
    else:
        _write(build_single_export(args.app_id), f"application {args.app_id}")
