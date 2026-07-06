"""
xai_layer_v3.py

Produces per-application explanation cards from hybrid model outputs.

Evidence-first design (2026-07-03): every claim in the narrative is a measured
deviation from the scored population rather than a canned sentence —
  * value percentiles vs the population distribution of each feature,
  * model-expected vs declared values (per_feature_predicted_json), with
    direction ("the model expected =0.29, the declared value is 0.002"),
  * per-feature prediction-error percentiles (how rare this miss is),
  * graph-degree percentiles per edge type,
  * subspace IF group scores vs their EVT thresholds and percentiles.
No hand-set narrative thresholds: the only numeric gates quoted are
EVT-derived (evt_thresholds_v3.json); everything else is stated as a
computed percentile. Same evidence always renders the same prose —
deterministic and auditable for appeals.

Reads:  outputs/hybrid_scores_v3.csv, outputs/risk_scores_v3.csv,
        outputs/pseudo_labels_v3.json, outputs/subspace_if_scores_v3.csv,
        outputs/evt_thresholds_v3.json, data/processed/engineered_features_v3.csv,
        data/processed/identity_graph_v3.pt
No training code. No raw embeddings.
Writes: outputs/explanation_cards_v3.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_v3 import (
    MIN_SIGNALS_FOR_PROMOTION, EDGE_TYPES, ENCODER_ARCH,
    FUSION_W_SUBSPACE, FUSION_W_DENSE_IP, FUSION_W_HYBRID,
)

HYBRID_CSV     = Path("outputs/hybrid_scores_v3.csv")
RISK_CSV       = Path("outputs/risk_scores_v3.csv")
SUBSPACE_CSV   = Path("outputs/subspace_if_scores_v3.csv")
EVT_JSON       = Path("outputs/evt_thresholds_v3.json")
SCHEMA_JSON    = Path("data/processed/v3_feature_schema.json")
GRAPH_PT       = Path("data/processed/identity_graph_v3.pt")
FEATURES_CSV   = Path("data/processed/engineered_features_v3.csv")
PSEUDO_LABELS  = Path("outputs/pseudo_labels_v3.json")
OUT_JSON       = Path("outputs/explanation_cards_v3.json")
DENSE_CSV      = Path("outputs/dense_block_scores_v3.csv")

TOP_K_FEATURES  = 5
TOP_K_NEIGHBORS = 3
NARRATIVE_FEATURES = 3   # how many top features the prose cites

# Human-readable labels for engineered feature columns
FEATURE_LABELS: dict[str, str] = {
    # Identity
    "application_id":              "Application ID",
    "c_institution_id":            "Institution ID (college-reported)",
    "c_course_id":                 "Course ID (college-reported)",
    "c_student_id":                "Student ID (college-reported)",
    "inst_verify_by":              "Institution verifier code",
    "state_verify_by":             "State verifier code",
    "district_verify_by":          "District verifier code",
    "name_similarity_score":       "Applicant/parent name similarity",
    "is_father_name_eq_mother":    "Father's name identical to mother's",
    "is_applicant_name_eq_father": "Applicant's name identical to father's",
    "is_applicant_name_eq_mother": "Applicant's name identical to mother's",
    "mobile_unique_names":         "Distinct applicant names on same mobile",
    "mobile_unique_fathers":       "Distinct father names on same mobile",
    # Temporal
    "admission_year":              "Year of admission",
    "age_at_registration":         "Age at registration",
    "competitive_exam_year":       "Competitive exam year",
    "c_course_year":               "Course year (college-reported)",
    "x_course_year":               "Course year (external record)",
    "p_course_year":               "Course year (portal record)",
    "x_passing_year":              "Passing year (external record)",
    "p_passing_year":              "Passing year (portal record)",
    # Financial
    "annual_family_income":        "Declared annual family income",
    "annual_income":               "Declared annual family income",
    "admission_fee":               "Admission fee declared",
    "tution_fee":                  "Tuition fee declared",
    "misc_fee":                    "Miscellaneous fee declared",
    "total_fee":                   "Total tuition fee declared",
    "fee_income_ratio":            "Fee-to-income ratio",
    "fee_to_income_ratio":         "Fee-to-income ratio",
    "income_rank_in_district":     "Income rank within district",
    "income_deviation_from_state_median": "Income deviation from state median",
    "income_percentile":           "Income percentile (state-normalised)",
    "fee_percentile":              "Fee percentile (course-normalised)",
    "fee_deviation_from_median":   "Fee deviation from course median",
    # Network / IP
    "ip_application_count":        "No. of applications from same IP",
    "ip_to_mobile_ratio":          "IP-to-mobile concentration ratio",
    "mobile_application_count":    "No. of applications from same mobile",
    "institute_application_count": "No. of applications from same institute",
    "shared_ip_flag":              "Shared IP address flag",
    "shared_mobile_flag":          "Shared mobile number flag",
    # Name-sharing
    "shared_father_name_count":    "No. of applications sharing father's name",
    "shared_mother_name_count":    "No. of applications sharing mother's name",
    "father_name_entropy":         "Father name entropy (uniqueness)",
    "mother_name_entropy":         "Mother name entropy (uniqueness)",
    # Geography
    "is_urban":                    "Urban applicant flag",
    "pincode_application_count":   "No. of applications from same pincode",
    "state_code":                  "State code",
    "district_code":               "District code",
    # Degree features (from graph)
    "degree_shares_ip":            "Graph degree — shared IP edges",
    "degree_shares_mobile":        "Graph degree — shared mobile edges",
    "degree_shares_father_name":   "Graph degree — shared father-name edges",
    "degree_shares_mother_name":   "Graph degree — shared mother-name edges",
    "degree_shares_pincode":       "Graph degree — shared pincode edges",
}

TRIGGER_DESCRIPTIONS = {
    "EVT_HYBRID":    "overall anomaly detected by hybrid model",
    "EVT_FINANCIAL": "income or fee profile statistically extreme for this cohort",
    "EVT_IDENTITY":  "identity fields (name / verifier codes) form an unusual cluster",
    "EVT_NETWORK":   "IP/mobile sharing pattern statistically extreme",
    "EVT_EDGE_RING": "graph connectivity inconsistent with the declared feature profile",
}

# Maps each self-training trigger to its evt_thresholds_v3.json signal key
TRIGGER_SIGNAL_KEY = {
    "EVT_HYBRID":    "hybrid",
    "EVT_FINANCIAL": "subspace_if_financial",
    "EVT_IDENTITY":  "subspace_if_identity",
    "EVT_NETWORK":   "subspace_if_network",
    "EVT_EDGE_RING": "edge_pred_error",
}

LABEL_SOURCE_DESCRIPTIONS = {
    "negative":       "Pending Review — no confirmed fraud label assigned yet",
    "evt_positive":   "EVT Flagged — statistical tail signal crossed threshold",
    "pseudo_positive":"Model Flagged — promoted by self-training (round 0)",
    "confirmed":      "Confirmed Fraud — manually verified by reviewer",
}

GROUP_LABELS = {
    "financial": "financial-features detector",
    "identity":  "identity-features detector",
    "network":   "network-features detector",
}

EDGE_TYPE_LABELS = {
    "shares_ip":          "IP address",
    "shares_mobile":      "mobile number",
    "shares_father_name": "father's name",
    "shares_mother_name": "mother's name",
    "shares_pincode":     "pincode",
}


DETECTOR_LABELS = {
    "subspace": "tabular subspace detector",
    "dense_ip": "shared-IP dense-block",
    "hybrid":   "relational RGCN model",
}


def _readable(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _minmax(x: np.ndarray) -> np.ndarray:
    """Population min-max to [0,1] — identical to fusion_classifier_v3._minmax
    so the reported contributions reproduce the actual fused score exactly."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def build_fusion_contributions(
    hybrid_df: pd.DataFrame,
    subspace_df: pd.DataFrame,
    dense_map: dict,
) -> dict[str, dict]:
    """Closed-form per-detector contribution to the score-level fusion.

    Replaces the removed LightGBM TreeSHAP path. Because the fusion is
      risk = minmax( W_S*minmax(subspace) + W_D*minmax(dense_ip) + W_H*minmax(hybrid) )
    each detector's contribution to the pre-normalisation sum is exact — no
    approximation. Returns app_id -> {detector: {weighted, share, normalized}}.
    """
    fus = hybrid_df[["application_id", "hybrid_anomaly_score"]].merge(
        subspace_df[["application_id", "subspace_if_score"]], on="application_id"
    )
    fus["dense_block_score_ip"] = fus["application_id"].map(
        lambda a: dense_map.get(a, {}).get("dense_block_score_ip", 0.0)
    ).astype(float)

    s = _minmax(fus["subspace_if_score"].values)
    d = _minmax(fus["dense_block_score_ip"].values)
    h = _minmax(fus["hybrid_anomaly_score"].values)

    c_s = FUSION_W_SUBSPACE * s
    c_d = FUSION_W_DENSE_IP * d
    c_h = FUSION_W_HYBRID   * h
    total = c_s + c_d + c_h + 1e-12

    out: dict[str, dict] = {}
    for i, aid in enumerate(fus["application_id"].values):
        out[aid] = {
            "subspace": {"weighted": round(float(c_s[i]), 6),
                         "share": round(float(c_s[i] / total[i]), 4),
                         "normalized": round(float(s[i]), 6)},
            "dense_ip": {"weighted": round(float(c_d[i]), 6),
                         "share": round(float(c_d[i] / total[i]), 4),
                         "normalized": round(float(d[i]), 6)},
            "hybrid":   {"weighted": round(float(c_h[i]), 6),
                         "share": round(float(c_h[i] / total[i]), 4),
                         "normalized": round(float(h[i]), 6)},
        }
    return out


# ---------------------------------------------------------------------------
# Population statistics — the "normalcy" every narrative claim is measured against
# ---------------------------------------------------------------------------

def _pct_rank(sorted_arr: np.ndarray, v: float) -> float:
    """Percentage of the population strictly below v, counting ties at half
    weight (midrank). 99.7 means: higher than 99.7% of applicants."""
    if sorted_arr.size == 0:
        return 50.0
    left  = np.searchsorted(sorted_arr, v, side="left")
    right = np.searchsorted(sorted_arr, v, side="right")
    return float(100.0 * (left + 0.5 * (right - left)) / sorted_arr.size)


def _fmt_pct(p: float) -> str:
    return "99.9+%" if p >= 99.95 else f"{p:.1f}%"


def build_population_stats(
    hybrid_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    risk_df: pd.DataFrame | None = None,
    subspace_df: pd.DataFrame | None = None,
    degree_counts: dict[str, np.ndarray] | None = None,
) -> dict:
    """Precompute the population reference distributions used for every card."""
    stats: dict = {"n": len(feat_df)}

    feat_cols = [c for c in feat_df.columns if c != "application_id"]
    stats["feat_sorted"] = {c: np.sort(feat_df[c].values.astype(float)) for c in feat_cols}
    stats["feat_median"] = {c: float(np.median(feat_df[c].values)) for c in feat_cols}

    # Per-feature prediction-error distribution across the whole population
    err_records = [json.loads(s) for s in hybrid_df["per_feature_error_json"]]
    err_matrix  = pd.DataFrame(err_records)
    stats["err_sorted"] = {c: np.sort(err_matrix[c].values.astype(float)) for c in err_matrix.columns}

    if risk_df is not None:
        stats["risk_sorted"] = np.sort(risk_df["risk_score_v3"].values.astype(float))

    if subspace_df is not None:
        group_records = [json.loads(s) for s in subspace_df["group_scores_json"]]
        group_matrix  = pd.DataFrame(group_records)
        stats["group_sorted"] = {g: np.sort(group_matrix[g].values.astype(float)) for g in group_matrix.columns}

    if degree_counts is not None:
        stats["degree_sorted"] = {et: np.sort(counts) for et, counts in degree_counts.items()}
        total = np.zeros_like(next(iter(degree_counts.values())))
        for counts in degree_counts.values():
            total = total + counts
        stats["isolated_pct"] = float(100.0 * (total == 0).mean())

    return stats


# ---------------------------------------------------------------------------
# Per-card evidence
# ---------------------------------------------------------------------------

def _top_features(
    per_feat: dict,
    actual_vals: dict,
    k: int,
    predicted: dict | None = None,
    stats: dict | None = None,
) -> list[dict]:
    """Top-k features by prediction error, enriched with all available
    evidence: declared value, model-expected value, and population
    percentiles (when stats are supplied)."""
    sorted_feats = sorted(per_feat.items(), key=lambda kv: kv[1], reverse=True)
    result = []
    for f, err in sorted_feats[:k]:
        entry: dict = {
            "feature":       f,
            "feature_label": _readable(f),
            "error":         round(err, 6),
        }
        val = actual_vals.get(f)
        if val is not None:
            entry["value"] = round(float(val), 6)
        if predicted is not None and f in predicted:
            entry["expected"] = round(float(predicted[f]), 6)
        if stats is not None:
            if val is not None and f in stats["feat_sorted"]:
                entry["value_percentile"]  = round(_pct_rank(stats["feat_sorted"][f], float(val)), 2)
                entry["population_median"] = round(stats["feat_median"][f], 6)
            if f in stats.get("err_sorted", {}):
                entry["error_percentile"] = round(_pct_rank(stats["err_sorted"][f], float(err)), 2)
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


def _degree_counts_from_index(
    neighbor_index: dict[int, list[dict]],
    n_nodes: int,
    edge_types: list[str],
) -> dict[str, np.ndarray]:
    counts = {et: np.zeros(n_nodes, dtype=np.int64) for et in edge_types}
    for node, entries in neighbor_index.items():
        for e in entries:
            counts[e["edge_type"]][node] += 1
    return counts


# ---------------------------------------------------------------------------
# Narrative rendering — deterministic prose composed from the evidence
# ---------------------------------------------------------------------------

def _feature_sentence(entry: dict) -> str:
    label = entry.get("feature_label", _readable(entry["feature"]))
    val   = entry.get("value")
    parts = [label]

    vp = entry.get("value_percentile")
    if val is not None and vp is not None:
        med = entry.get("population_median")
        med_str = f", population median {med:.3f}" if med is not None else ""
        if med is not None and np.isclose(val, med):
            # Midrank percentiles are tie-inflated when the value sits on the
            # median — the value itself is ordinary; the model miss is the story.
            parts.append(f"matches the population median (scaled value {val:.3f})")
        elif vp >= 50.0:
            parts.append(f"is higher than {_fmt_pct(vp)} of applicants (scaled value {val:.3f}{med_str})")
        else:
            parts.append(f"is lower than {_fmt_pct(100.0 - vp)} of applicants (scaled value {val:.3f}{med_str})")
    elif val is not None:
        parts.append(f"has scaled value {val:.3f}")
    else:
        parts.append("could not be predicted from the rest of the application")

    exp = entry.get("expected")
    if exp is not None and val is not None:
        direction = "above" if val > exp else "below"
        parts.append(
            f"; given the other declared fields and the application's network context, "
            f"the model expected about {exp:.3f} — the declared value is {direction} expectation"
        )

    ep = entry.get("error_percentile")
    if ep is not None:
        if ep >= 50.0:
            parts.append(f"; this prediction miss is larger than {_fmt_pct(ep)} of all applications' misses on this field")
        else:
            parts.append(f"; the prediction miss on this field is unremarkable ({_fmt_pct(ep)} percentile)")

    return " ".join(parts[:2]) + "".join(parts[2:])


def _narrative(card: dict) -> str:
    parts: list[str] = []
    ev        = card.get("evidence") or {}
    score     = card.get("risk_score_v3", 0.0)
    triggers  = card.get("triggers", [])
    neighbors = card.get("top_graph_neighbors", [])
    top_feats = card.get("top_feature_errors", [])

    # 1. Where this application sits in the scored population
    risk_pct  = ev.get("risk_percentile")
    risk_rank = ev.get("risk_rank")
    n_pop     = ev.get("population_size")
    if risk_pct is not None and risk_rank is not None and n_pop:
        parts.append(
            f"Risk score {score:.4f} — higher than {_fmt_pct(risk_pct)} of the "
            f"{n_pop:,} scored applications (rank {risk_rank})."
        )
    else:
        parts.append(f"Anomaly score {score:.4f} (higher = more anomalous).")

    # 2. EVT flags — observed value vs fitted threshold, per fired signal.
    #    "Crossed a threshold" and "promoted to pseudo-positive" are distinct:
    #    promotion requires MIN_SIGNALS_FOR_PROMOTION signals to agree.
    evt_signals = ev.get("evt_signals", {})
    if triggers:
        flag_lines = []
        for t in triggers:
            desc = TRIGGER_DESCRIPTIONS.get(t, t)
            sig  = evt_signals.get(t)
            if sig and sig.get("observed") is not None and sig.get("threshold") is not None:
                flag_lines.append(f"{desc} (observed {sig['observed']:.3f} vs threshold {sig['threshold']:.3f})")
            else:
                flag_lines.append(desc)
        q = next((s.get("q") for s in evt_signals.values() if s.get("q")), None)
        q_str = f", fitted to the score tails at a {q * 100:.1f}% target false-positive rate" if q else ""
        plural = "s" if len(triggers) != 1 else ""
        parts.append(
            f"Crossed {len(triggers)} independent extreme-value threshold{plural}{q_str}: "
            + "; ".join(flag_lines) + "."
        )
    elif ev:
        crossings = ev.get("evt_crossings", [])
        if crossings:
            lines = [
                f"{c['label']} (observed {c['observed']:.3f} vs threshold {c['threshold']:.3f})"
                for c in crossings
            ]
            qualifier = (
                f" — below the {MIN_SIGNALS_FOR_PROMOTION}-signal agreement required for automatic flag promotion"
                if len(crossings) < MIN_SIGNALS_FOR_PROMOTION else ""
            )
            plural = "s" if len(crossings) != 1 else ""
            parts.append(
                f"Crossed {len(crossings)} extreme-value threshold{plural}: "
                + "; ".join(lines) + qualifier + "."
            )
        else:
            parts.append("No extreme-value threshold was crossed.")

    # 3. Feature-level evidence — measured deviations, not boilerplate
    if top_feats:
        feat_lines = [_feature_sentence(f) for f in top_feats[:NARRATIVE_FEATURES]]
        parts.append("Key evidence: " + " | ".join(feat_lines) + ".")

    # 3b. Fusion breakdown — which detector drove the fused score (closed-form,
    #     replaces the removed LightGBM TreeSHAP attributions).
    fc = ev.get("fusion_contributions", {})
    if fc:
        order = sorted(fc.items(), key=lambda kv: kv[1].get("share", 0.0), reverse=True)
        parts_fc = [
            f"{DETECTOR_LABELS.get(k, k)} {c['share'] * 100:.0f}%"
            for k, c in order if c.get("share", 0.0) > 0.005
        ]
        if parts_fc:
            parts.append("Score composition: " + ", ".join(parts_fc) + ".")

    # 3c. Dense-block-IP: the IP specialist. Only surfaced when it actually fires.
    dib = ev.get("dense_block_ip")
    if dib and dib.get("score", 0.0) > 0.0:
        pct = dib.get("percentile")
        pct_str = f", more concentrated than {_fmt_pct(pct)} of applicants" if pct is not None else ""
        parts.append(
            f"Shared-IP dense-block: this application sits inside a dense cluster of "
            f"co-applications on the same IP address{pct_str} — the signal the IP specialist adds "
            f"on top of the tabular detectors."
        )

    # 4. Subspace detector context. Crossed groups are already reported in
    #    section 2 (triggers or evt_crossings) — here we only add the strongest
    #    detector's percentile when nothing crossed, as ranking context.
    groups = ev.get("subspace_groups", {})
    if groups and not triggers and not any(d.get("crossed") for d in groups.values()):
        top_g, top_d = max(groups.items(), key=lambda kv: kv[1].get("percentile", 0.0))
        if top_d.get("percentile") is not None:
            parts.append(
                f"Strongest independent detector: {GROUP_LABELS.get(top_g, top_g)} "
                f"places this application above {_fmt_pct(top_d['percentile'])} of applicants "
                f"(below its EVT threshold)."
            )

    # 5. Graph evidence — connectivity measured against the population
    connections = ev.get("graph_connections", [])
    if connections:
        conn_lines = []
        for c in connections:
            et_label = EDGE_TYPE_LABELS.get(c["edge_type"], c["edge_type"].replace("_", " "))
            pct_str  = f", more connected than {_fmt_pct(c['percentile'])} of applicants" if c.get("percentile") is not None else ""
            ids      = ", ".join(c.get("sample_ids", [])[:TOP_K_NEIGHBORS])
            id_str   = f" (e.g. {ids})" if ids else ""
            conn_lines.append(f"shares the same {et_label} with {c['count']} other application(s){pct_str}{id_str}")
        parts.append(
            "Network links: " + "; ".join(conn_lines)
            + ". Linked applications should be reviewed together."
        )
    elif ev:
        iso_pct = ev.get("isolated_population_pct")
        iso_str = f" — {iso_pct:.1f}% of applicants are similarly isolated" if iso_pct is not None else ""
        parts.append(
            f"No shared IP, mobile, parent-name, or pincode links found{iso_str}. "
            "Suspicion rests on feature-level evidence and the subspace detectors."
        )
    elif not neighbors:
        parts.append("No shared IP, mobile, parent-name, or pincode links found.")

    attention = card.get("attention")
    if attention:
        beta_r = attention.get("beta_r", {})
        if beta_r:
            top_rel, top_w = max(beta_r.items(), key=lambda kv: kv[1])
            top_rel_label = EDGE_TYPE_LABELS.get(top_rel, top_rel.replace("_", " "))
            top_edges = attention.get("top_edges", [])
            n_edges = len(top_edges)
            parts.append(
                f"The model weighted this application's {top_rel_label} links most heavily (β={top_w:.2f}), "
                f"focusing on {n_edges} co-application(s)."
            )

    # 6. Recommended action — driven by label state and EVT crossings,
    #    not a hand-set score cut
    label_source = ev.get("label_source")
    if label_source == "confirmed":
        parts.append("Recommended action: confirmed fraud — keep disbursement on hold and coordinate with the investigation team.")
    elif triggers:
        parts.append(
            "Recommended action: hold disbursement and request supporting documents "
            "(fee receipt, admission letter, income certificate); review linked applications together before approval."
        )
    elif ev.get("evt_crossings"):
        parts.append(
            "Recommended action: not auto-flagged (signal agreement below the promotion requirement), "
            "but the crossed threshold above warrants secondary verification before disbursement."
        )
    elif ev:
        parts.append("No statistical flag crossed — no hold required; card provided for ranking context.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_xai(top_n: int = 500) -> None:
    print("[xai] run_xai() starting ...")

    schema     = json.loads(SCHEMA_JSON.read_text())
    features   = schema["features"]
    hybrid_df  = pd.read_csv(HYBRID_CSV)
    risk_df    = pd.read_csv(RISK_CSV)

    has_predicted = "per_feature_predicted_json" in hybrid_df.columns
    if not has_predicted:
        print("[xai] WARNING: per_feature_predicted_json not in hybrid scores — "
              "expected-vs-actual evidence unavailable; re-run scoring for full narratives.")
              
    # Dense-block-IP scores (V4 IP specialist). Consumed for the closed-form
    # fusion breakdown and as narrative evidence.
    dense_map: dict = {}
    dense_ip_sorted = np.array([])
    if DENSE_CSV.exists():
        dense_df = pd.read_csv(DENSE_CSV)
        dense_cols = [c for c in dense_df.columns if c.startswith("dense_block_score_")]
        for _, r in dense_df.iterrows():
            dense_map[r["application_id"]] = {c: float(r[c]) for c in dense_cols}
        if "dense_block_score_ip" in dense_df.columns:
            dense_ip_sorted = np.sort(dense_df["dense_block_score_ip"].values.astype(float))
    else:
        print(f"[xai] WARNING: {DENSE_CSV} not found — dense-block-IP evidence omitted.")

    # Actual (scaled) feature values per application
    feat_df = pd.read_csv(FEATURES_CSV)
    feat_value_map = feat_df.set_index("application_id").to_dict("index")

    # Subspace IF group scores (§7 lists xai as consumer)
    subspace_df, subspace_map = None, {}
    if SUBSPACE_CSV.exists():
        subspace_df = pd.read_csv(SUBSPACE_CSV)
        for _, r in subspace_df.iterrows():
            subspace_map[r["application_id"]] = json.loads(r["group_scores_json"])
    else:
        print(f"[xai] WARNING: {SUBSPACE_CSV} not found — subspace corroboration omitted.")

    # Closed-form fusion breakdown (replaces the removed LightGBM TreeSHAP path)
    fusion_contrib: dict = {}
    if subspace_df is not None:
        fusion_contrib = build_fusion_contributions(hybrid_df, subspace_df, dense_map)
        print(f"[xai] Computed closed-form fusion contributions for {len(fusion_contrib)} applications.")

    # EVT thresholds — the only numeric gates the narrative quotes
    evt = json.loads(EVT_JSON.read_text()) if EVT_JSON.exists() else {}
    if not evt:
        print(f"[xai] WARNING: {EVT_JSON} not found — thresholds omitted from narratives.")

    # Trigger info from pseudo-labels
    trigger_map: dict[str, list[str]] = {}
    if PSEUDO_LABELS.exists():
        pl = json.loads(PSEUDO_LABELS.read_text())
        for rec in pl.get("positive_set", []):
            trigger_map[rec["application_id"]] = rec.get("trigger", [])

    merged = hybrid_df.merge(risk_df, on="application_id")
    merged = merged.sort_values("risk_score_v3", ascending=False)
    merged_top = merged.head(top_n)

    print("[xai] Building neighbor index ...")
    neighbor_index = _build_neighbor_index(GRAPH_PT)

    all_ids   = hybrid_df["application_id"].tolist()
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    idx_to_id = {i: aid for aid, i in id_to_idx.items()}

    edge_types    = sorted({e["edge_type"] for entries in neighbor_index.values() for e in entries})
    degree_counts = _degree_counts_from_index(neighbor_index, len(all_ids), edge_types)

    print("[xai] Computing population statistics (values, errors, degrees, groups) ...")
    stats = build_population_stats(hybrid_df, feat_df, risk_df, subspace_df, degree_counts)

    from src.hybrid_graphmcm_v3 import load_model_and_inputs
    import torch
    print("[xai] Loading hybrid model for attention extraction ...")
    model, x_all, edge_index_list, edge_type_tensor, isolated_mask, app_ids_arr, _ = load_model_and_inputs()
    with torch.no_grad():
        model(x_all, edge_index_list, edge_type_tensor, isolated_mask)
    beta_r_tensor = model.last_beta_r.cpu().numpy() if model.last_beta_r is not None else np.ones(len(EDGE_TYPES)) / len(EDGE_TYPES)

    risk_sorted = stats["risk_sorted"]
    n_pop       = stats["n"]

    cards = []
    for _, row in merged_top.iterrows():
        app_id   = row["application_id"]
        node_idx = id_to_idx.get(app_id, -1)

        per_feat    = json.loads(row["per_feature_error_json"])
        predicted   = json.loads(row["per_feature_predicted_json"]) if has_predicted else None
        actual_vals = feat_value_map.get(app_id, {})

        # Validate key sets against schema
        if set(per_feat.keys()) != set(features):
            extra   = set(per_feat.keys()) - set(features)
            missing = set(features) - set(per_feat.keys())
            raise ValueError(
                f"per_feature_error_json key mismatch for {app_id}. "
                f"Extra: {extra}. Missing: {missing}."
            )
        if predicted is not None and set(predicted.keys()) != set(features):
            raise ValueError(
                f"per_feature_predicted_json key mismatch for {app_id}: "
                f"key set does not match v3_feature_schema.json features."
            )

        top_feats = _top_features(per_feat, actual_vals, TOP_K_FEATURES, predicted, stats)

        # Graph connections: per-edge-type counts + population percentile
        neighbors = neighbor_index.get(node_idx, [])
        by_type: dict[str, list[int]] = {}
        for n in neighbors:
            by_type.setdefault(n["edge_type"], []).append(n["neighbor_idx"])
        connections = []
        for et, idxs in by_type.items():
            count = len(idxs)
            conn = {
                "edge_type":  et,
                "count":      count,
                "sample_ids": [str(idx_to_id.get(i, f"idx:{i}")) for i in idxs[:TOP_K_NEIGHBORS]],
            }
            if et in stats.get("degree_sorted", {}):
                conn["percentile"] = round(_pct_rank(stats["degree_sorted"][et], count), 2)
            connections.append(conn)
        connections.sort(key=lambda c: c["count"], reverse=True)

        neighbor_records = [
            {"edge_type": n["edge_type"],
             "application_id": str(idx_to_id.get(n["neighbor_idx"], f'idx:{n["neighbor_idx"]}'))}
            for n in neighbors[:TOP_K_NEIGHBORS]
        ]

        # Subspace group evidence: score, percentile, EVT threshold, crossed
        groups_ev: dict[str, dict] = {}
        app_groups = subspace_map.get(app_id, {})
        for g, g_score in app_groups.items():
            g_ev: dict = {"score": round(float(g_score), 6)}
            if g in stats.get("group_sorted", {}):
                g_ev["percentile"] = round(_pct_rank(stats["group_sorted"][g], float(g_score)), 2)
            thr = evt.get(f"subspace_if_{g}", {}).get("threshold")
            if thr is not None:
                g_ev["threshold"] = round(float(thr), 6)
                g_ev["crossed"]   = bool(float(g_score) >= float(thr))
            groups_ev[g] = g_ev

        # EVT signal detail for fired triggers (observed vs fitted threshold)
        triggers = trigger_map.get(app_id, [])
        evt_signals: dict[str, dict] = {}
        for t in triggers:
            key = TRIGGER_SIGNAL_KEY.get(t)
            sig = evt.get(key, {}) if key else {}
            observed = None
            if t == "EVT_HYBRID":
                observed = float(row["hybrid_anomaly_score"])
            elif t == "EVT_EDGE_RING":
                observed = float(row["edge_pred_error"])
            elif t == "EVT_FINANCIAL":
                observed = app_groups.get("financial")
            elif t == "EVT_IDENTITY":
                observed = app_groups.get("identity")
            elif t == "EVT_NETWORK":
                observed = app_groups.get("network")
            evt_signals[t] = {
                "observed":  round(float(observed), 6) if observed is not None else None,
                "threshold": round(float(sig["threshold"]), 6) if "threshold" in sig else None,
                "q":         sig.get("q"),
            }

        # EVT crossings independent of promotion — a single crossed signal is
        # real evidence even when it does not meet the multi-signal promotion
        # requirement, and the narrative must not contradict itself about it.
        evt_crossings = []
        hyb_thr = evt.get("hybrid", {}).get("threshold")
        if hyb_thr is not None and float(row["hybrid_anomaly_score"]) >= float(hyb_thr):
            evt_crossings.append({
                "signal":    "hybrid",
                "label":     "hybrid-model overall anomaly score",
                "observed":  round(float(row["hybrid_anomaly_score"]), 6),
                "threshold": round(float(hyb_thr), 6),
            })
        edge_thr = evt.get("edge_pred_error", {}).get("threshold")
        if edge_thr is not None and float(row["edge_pred_error"]) >= float(edge_thr):
            evt_crossings.append({
                "signal":    "edge_pred_error",
                "label":     "graph-connectivity signal",
                "observed":  round(float(row["edge_pred_error"]), 6),
                "threshold": round(float(edge_thr), 6),
            })
        for g, g_ev in groups_ev.items():
            if g_ev.get("crossed"):
                evt_crossings.append({
                    "signal":    f"subspace_if_{g}",
                    "label":     GROUP_LABELS.get(g, g),
                    "observed":  g_ev["score"],
                    "threshold": g_ev["threshold"],
                })

        score     = float(row["risk_score_v3"])
        label_src = row["label_source"]
        risk_rank = int(n_pop - np.searchsorted(risk_sorted, score, side="right")) + 1

        evidence = {
            "population_size":         n_pop,
            "risk_rank":               risk_rank,
            "risk_percentile":         round(_pct_rank(risk_sorted, score), 2),
            "label_source":            label_src,
            "evt_signals":             evt_signals,
            "evt_crossings":           evt_crossings,
            "subspace_groups":         groups_ev,
            "graph_connections":       connections,
            "isolated_population_pct": round(stats.get("isolated_pct", 0.0), 2),
        }

        if ENCODER_ARCH == "han" and node_idx != -1:
            top_edges = model.top_alpha(node_idx, TOP_K_NEIGHBORS)
            for edge in top_edges:
                edge["neighbor_id"] = str(idx_to_id.get(edge["neighbor_idx"], f'idx:{edge["neighbor_idx"]}'))
            attention_dict = {
                "beta_r": {EDGE_TYPES[r]: float(beta_r_tensor[r]) for r in range(len(beta_r_tensor))},
                "top_edges": top_edges
            }
        else:
            attention_dict = None
            
        # Closed-form fusion breakdown — exact per-detector contribution to the
        # score-level fusion (subspace / dense-IP / hybrid). Provenance is the
        # ordered list of detectors that actually contributed (>0.5% share).
        contrib = fusion_contrib.get(app_id, {})
        evidence["fusion_contributions"] = contrib
        evidence["provenance"] = [
            DETECTOR_LABELS.get(k, k)
            for k, c in sorted(contrib.items(), key=lambda kv: kv[1].get("share", 0.0), reverse=True)
            if c.get("share", 0.0) > 0.005
        ]

        # Dense-block-IP membership evidence (the IP specialist)
        dense_ip_score = dense_map.get(app_id, {}).get("dense_block_score_ip", 0.0)
        dense_ip_ev = {"score": round(float(dense_ip_score), 6)}
        if dense_ip_sorted.size and dense_ip_score > 0.0:
            dense_ip_ev["percentile"] = round(_pct_rank(dense_ip_sorted, float(dense_ip_score)), 2)
        evidence["dense_block_ip"] = dense_ip_ev

        card = {
            "application_id":       app_id,
            "risk_score_v3":        float(round(score, 6)),
            "hybrid_anomaly_score": float(round(row["hybrid_anomaly_score"], 6)),
            "feature_pred_error":   float(round(row["feature_pred_error"], 6)),
            "attention":            attention_dict,
            "edge_pred_error":      float(round(row["edge_pred_error"], 6)),
            "review_status":        LABEL_SOURCE_DESCRIPTIONS.get(label_src, label_src),
            "triggers":             triggers,
            "top_feature_errors":   top_feats,
            "top_graph_neighbors":  neighbor_records,
            "evidence":             evidence,
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
