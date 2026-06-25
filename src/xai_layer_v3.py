"""
xai_layer_v3.py

Produces per-application explanation cards from hybrid model outputs.
Reads per_feature_error_json (68-feature breakdown) and risk_scores_v3.csv.
No training code. No raw embeddings.
Writes: outputs/explanation_cards_v3.json
"""

import json
from pathlib import Path

import pandas as pd

HYBRID_CSV     = Path("outputs/hybrid_scores_v3.csv")
RISK_CSV       = Path("outputs/risk_scores_v3.csv")
SCHEMA_JSON    = Path("data/processed/v3_feature_schema.json")
GRAPH_PT       = Path("data/processed/identity_graph_v3.pt")
FEATURES_CSV   = Path("data/processed/engineered_features_v3.csv")
PSEUDO_LABELS  = Path("outputs/pseudo_labels_v3.json")
OUT_JSON       = Path("outputs/explanation_cards_v3.json")

TOP_K_FEATURES  = 5
TOP_K_NEIGHBORS = 3

TRIGGER_DESCRIPTIONS = {
    "EVT_HYBRID":    "general anomaly (hybrid model)",
    "EVT_FINANCIAL": "income/fee fraud signal",
    "EVT_IDENTITY":  "identity collision signal",
    "EVT_NETWORK":   "IP concentration signal",
    "EVT_EDGE_RING": "cross-channel fraud ring",
}


def _magnitude(value: float) -> str:
    if value < 0.05:
        return "near-zero"
    elif value < 0.15:
        return "very low"
    elif value < 0.35:
        return "low"
    elif value < 0.65:
        return "moderate"
    elif value < 0.85:
        return "high"
    elif value < 0.95:
        return "very high"
    else:
        return "at maximum"


def _top_features(per_feat: dict, actual_vals: dict, k: int) -> list[dict]:
    sorted_feats = sorted(per_feat.items(), key=lambda kv: kv[1], reverse=True)
    result = []
    for f, err in sorted_feats[:k]:
        val = actual_vals.get(f)
        entry: dict = {"feature": f, "error": round(err, 6)}
        if val is not None:
            entry["value"] = round(float(val), 6)
            entry["magnitude"] = _magnitude(float(val))
        result.append(entry)
    return result


def _build_neighbor_index(graph_pt_path: Path) -> dict[int, list[dict]]:
    import torch
    data = torch.load(graph_pt_path, weights_only=False)
    neighbor_index: dict[int, list[dict]] = {}

    for edge_type_tuple in data.edge_types:
        et = edge_type_tuple[1]
        ei = data[edge_type_tuple].edge_index
        if ei.shape[1] == 0:
            continue
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for s, d in zip(src, dst):
            neighbor_index.setdefault(s, []).append({"neighbor_idx": d, "edge_type": et})

    return neighbor_index


def _narrative(card: dict) -> str:
    parts = []
    score = card["risk_score_v3"]

    if score >= 0.7:
        parts.append("High risk application.")
    elif score >= 0.4:
        parts.append("Moderate risk application.")
    else:
        parts.append("Low risk application.")

    triggers = card.get("triggers", [])
    if triggers:
        descs = [TRIGGER_DESCRIPTIONS.get(t, t) for t in triggers]
        parts.append(f"Triggered by: {', '.join(descs)}.")

    top_feats = card["top_feature_errors"]
    if top_feats:
        feat_strs = []
        for f in top_feats[:3]:
            if "value" in f and "magnitude" in f:
                feat_strs.append(f"{f['feature']}={f['value']:.3f} ({f['magnitude']})")
            else:
                feat_strs.append(f["feature"])
        parts.append(f"Largest prediction errors: {', '.join(feat_strs)}.")

    neighbors = card["top_graph_neighbors"]
    if neighbors:
        et_counts: dict[str, int] = {}
        for n in neighbors:
            et_counts[n["edge_type"]] = et_counts.get(n["edge_type"], 0) + 1
        conn_str = "; ".join(f"{v} via {k}" for k, v in et_counts.items())
        parts.append(f"Connected to other applications: {conn_str}.")
    else:
        parts.append("No graph connections (isolated node).")

    return " ".join(parts)


def run_xai(top_n: int = 500) -> None:
    print("[xai] run_xai() starting ...")

    schema     = json.loads(SCHEMA_JSON.read_text())
    features   = schema["features"]
    hybrid_df  = pd.read_csv(HYBRID_CSV)
    risk_df    = pd.read_csv(RISK_CSV)

    # Load actual (scaled) feature values for enriching narratives
    feat_df = pd.read_csv(FEATURES_CSV)
    feat_value_map: dict[str, dict] = {}
    for _, row in feat_df.iterrows():
        feat_value_map[row["application_id"]] = {
            col: row[col] for col in feat_df.columns if col != "application_id"
        }

    # Load trigger info from pseudo-labels
    trigger_map: dict[str, list[str]] = {}
    if PSEUDO_LABELS.exists():
        pl = json.loads(PSEUDO_LABELS.read_text())
        for rec in pl.get("positive_set", []):
            trigger_map[rec["application_id"]] = rec.get("trigger", [])

    merged = hybrid_df.merge(risk_df, on="application_id")
    merged = merged.sort_values("risk_score_v3", ascending=False)
    merged_top = merged.head(top_n)

    print(f"[xai] Building neighbor index ...")
    neighbor_index = _build_neighbor_index(GRAPH_PT)

    all_ids   = hybrid_df["application_id"].tolist()
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}

    cards = []
    for _, row in merged_top.iterrows():
        app_id   = row["application_id"]
        node_idx = id_to_idx.get(app_id, -1)

        per_feat    = json.loads(row["per_feature_error_json"])
        actual_vals = feat_value_map.get(app_id, {})

        # Validate key set against schema
        if set(per_feat.keys()) != set(features):
            extra   = set(per_feat.keys()) - set(features)
            missing = set(features) - set(per_feat.keys())
            raise ValueError(
                f"per_feature_error_json key mismatch for {app_id}. "
                f"Extra: {extra}. Missing: {missing}."
            )

        top_feats = _top_features(per_feat, actual_vals, TOP_K_FEATURES)

        neighbors = neighbor_index.get(node_idx, [])
        top_neighbors = neighbors[:TOP_K_NEIGHBORS]
        neighbor_records = [
            {"edge_type": n["edge_type"], "neighbor_idx": n["neighbor_idx"]}
            for n in top_neighbors
        ]

        triggers = trigger_map.get(app_id, [])

        card = {
            "application_id":       app_id,
            "risk_score_v3":        float(round(row["risk_score_v3"], 6)),
            "hybrid_anomaly_score": float(round(row["hybrid_anomaly_score"], 6)),
            "feature_pred_error":   float(round(row["feature_pred_error"], 6)),
            "edge_pred_error":      float(round(row["edge_pred_error"], 6)),
            "label_source":         row["label_source"],
            "triggers":             triggers,
            "top_feature_errors":   top_feats,
            "top_graph_neighbors":  neighbor_records,
        }
        card["narrative"] = _narrative(card)
        cards.append(card)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(cards, indent=2))
    print(f"[xai] Saved {len(cards)} explanation cards -> {OUT_JSON}")
    if cards:
        print(f"[xai] Top application: {cards[0]['application_id']} | risk={cards[0]['risk_score_v3']:.4f}")
        print(f"[xai] Narrative sample: {cards[0]['narrative']}")


if __name__ == "__main__":
    run_xai()
