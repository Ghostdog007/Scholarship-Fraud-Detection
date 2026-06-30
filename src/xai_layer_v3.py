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

# Human-readable labels for every engineered feature column
FEATURE_LABELS: dict[str, str] = {
    # Identity
    "application_id":            "Application ID",
    "c_institution_id":          "Institution ID (college-reported)",
    "c_course_id":               "Course ID (college-reported)",
    "c_student_id":              "Student ID (college-reported)",
    "inst_verify_by":            "Institution verifier code",
    "state_verify_by":           "State verifier code",
    "district_verify_by":        "District verifier code",
    # Temporal
    "admission_year":            "Year of admission",
    "x_course_year":             "Course year (external record)",
    "p_course_year":             "Course year (portal record)",
    "x_passing_year":            "Passing year (external record)",
    "p_passing_year":            "Passing year (portal record)",
    # Financial
    "annual_income":             "Declared annual family income",
    "total_fee":                 "Total tuition fee declared",
    "fee_to_income_ratio":       "Fee-to-income ratio",
    "income_percentile":         "Income percentile (state-normalised)",
    "fee_percentile":            "Fee percentile (course-normalised)",
    "fee_deviation_from_median": "Fee deviation from course median",
    # Network / IP
    "ip_application_count":      "No. of applications from same IP",
    "ip_to_mobile_ratio":        "IP-to-mobile concentration ratio",
    "mobile_application_count":  "No. of applications from same mobile",
    "shared_ip_flag":            "Shared IP address flag",
    "shared_mobile_flag":        "Shared mobile number flag",
    # Name-sharing
    "shared_father_name_count":  "No. of applications sharing father's name",
    "shared_mother_name_count":  "No. of applications sharing mother's name",
    "father_name_entropy":       "Father name entropy (uniqueness)",
    "mother_name_entropy":       "Mother name entropy (uniqueness)",
    # Geography
    "is_urban":                  "Urban applicant flag",
    "pincode_application_count": "No. of applications from same pincode",
    "state_code":                "State code",
    "district_code":             "District code",
    # Degree features (from graph)
    "degree_shares_ip":          "Graph degree — shared IP edges",
    "degree_shares_mobile":      "Graph degree — shared mobile edges",
    "degree_shares_father_name": "Graph degree — shared father-name edges",
    "degree_shares_mother_name": "Graph degree — shared mother-name edges",
    "degree_shares_pincode":     "Graph degree — shared pincode edges",
}

TRIGGER_DESCRIPTIONS = {
    "EVT_HYBRID":    "overall anomaly detected by hybrid model (EVT threshold crossed)",
    "EVT_FINANCIAL": "income or fee amount is statistically extreme for this cohort",
    "EVT_IDENTITY":  "identity fields (name / verifier codes) form an unusual cluster",
    "EVT_NETWORK":   "IP address shared with an unusually large number of other applications",
    "EVT_EDGE_RING": "part of a cross-channel fraud ring (IP + mobile + name overlap)",
}

LABEL_SOURCE_DESCRIPTIONS = {
    "negative":       "Pending Review — no confirmed fraud label assigned yet",
    "evt_positive":   "EVT Flagged — statistical tail signal crossed threshold",
    "pseudo_positive":"Model Flagged — promoted by self-training (round 0)",
    "confirmed":      "Confirmed Fraud — manually verified by reviewer",
}


def _readable(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


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
        entry: dict = {
            "feature":       f,
            "feature_label": _readable(f),
            "error":         round(err, 6),
        }
        if val is not None:
            entry["value"]     = round(float(val), 6)
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
    score    = card["risk_score_v3"]
    triggers = card.get("triggers", [])
    neighbors = card["top_graph_neighbors"]
    top_feats = card["top_feature_errors"]

    # Risk level opener
    if score >= 0.7:
        parts.append("HIGH RISK — recommend manual review before processing.")
    elif score >= 0.4:
        parts.append("MODERATE RISK — flag for secondary verification.")
    else:
        parts.append("LOW RISK — no immediate action required.")

    # What drove the score
    if triggers:
        descs = [TRIGGER_DESCRIPTIONS.get(t, t) for t in triggers]
        parts.append("Flagged because: " + "; ".join(descs) + ".")
    elif score >= 0.7:
        parts.append(
            "No single EVT threshold was crossed, but the hybrid model's "
            "overall anomaly score is in the top risk tier — the combination "
            "of feature inconsistencies is collectively suspicious."
        )

    # Feature explanation — WHY these fields are suspicious
    if top_feats:
        feat_lines = []
        for f in top_feats[:3]:
            label = f.get("feature_label", _readable(f["feature"]))
            mag   = f.get("magnitude", "unknown")
            val   = f.get("value")
            val_str = f" (value={val:.3f})" if val is not None else ""
            feat_lines.append(
                f"{label}{val_str} is {mag} — the model could not predict "
                f"this from the rest of the application, suggesting it is "
                f"inconsistent with legitimate application patterns."
            )
        parts.append("Key anomalous fields: " + " | ".join(feat_lines) + ".")

    # Graph connections
    if neighbors:
        et_groups: dict[str, list[str]] = {}
        for n in neighbors:
            et_groups.setdefault(n["edge_type"], []).append(
                n.get("application_id", f'idx:{n.get("neighbor_idx","?")}')
            )
        conn_parts = []
        for et, ids in et_groups.items():
            edge_label = et.replace("shares_", "shares same ").replace("_", " ")
            id_list = ", ".join(ids[:3])
            conn_parts.append(f"{len(ids)} application(s) via {edge_label} ({id_list})")
        parts.append(
            "Graph alert: this application is linked to "
            + "; ".join(conn_parts)
            + ". Linked applications should be reviewed together."
        )
    else:
        parts.append(
            "Isolated node — no shared IP, mobile, name, or pincode links found. "
            "Suspicion is driven entirely by feature-level inconsistencies."
        )

    # Recommended action
    if score >= 0.7:
        parts.append(
            "Recommended action: hold disbursement and request supporting documents "
            "(fee receipt, admission letter, income certificate) before approval."
        )

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
    idx_to_id = {i: aid for aid, i in id_to_idx.items()}   # resolve idx → application_id

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

        # Resolve neighbor indices to application IDs
        neighbors = neighbor_index.get(node_idx, [])
        top_neighbors = neighbors[:TOP_K_NEIGHBORS]
        neighbor_records = [
            {
                "edge_type":      n["edge_type"],
                "application_id": idx_to_id.get(n["neighbor_idx"], f'idx:{n["neighbor_idx"]}'),
            }
            for n in top_neighbors
        ]

        triggers = trigger_map.get(app_id, [])
        label_src = row["label_source"]

        card = {
            "application_id":       app_id,
            "risk_score_v3":        float(round(row["risk_score_v3"], 6)),
            "hybrid_anomaly_score": float(round(row["hybrid_anomaly_score"], 6)),
            "feature_pred_error":   float(round(row["feature_pred_error"], 6)),
            "edge_pred_error":      float(round(row["edge_pred_error"], 6)),
            # Human-readable status — "negative" was confusing for reviewers
            "review_status":        LABEL_SOURCE_DESCRIPTIONS.get(label_src, label_src),
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
