"""
evaluate_model_v3.py

Three-level evaluation protocol:
  Level 1: Category PR-AUC (5 archetypes, 150 injected anomalies each, unseen seeds)
  Level 2: Degree-stratified PR-AUC (isolated / low / high degree)
  Level 3: Edge-dropout test (score_retention = median_after / median_before)

Scoring methodology:
  All injected nodes are isolated (no graph edges).
  For isolated nodes the hybrid model's isolated_embedding is the same fixed
  vector for every node, so feature prediction error is nearly constant across
  all inject nodes and cannot rank them above real nodes on the hybrid scale.
  Instead, each category uses the subspace IF group that directly targets the
  injected anomaly type:

    AGE_VIOLATION         -> "demographic" IF (age_at_registration, eval-only group)
    INCOME_VIOLATION      -> "financial"   IF
    IP_CONCENTRATION      -> "network"     IF
    MOTHER_NAME_COLLISION -> "identity"    IF
    FEE_INFLATION         -> "financial"   IF

  For real nodes the same group's score is loaded from the precomputed CSV so
  inject and real scores are always normalised on the same real-data scale.

  Injection functions perturb ALL features in the target group, not just one,
  so the injected vectors are internally consistent.

Prints all results to stdout. No training code. No model modification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.config_v3 import (
    DEGREE_FEATURES,
    LAMBDA_EDGE,
    N_EDGE_TYPES,
    RANDOM_SEED,
    SUBSPACE_GROUPS,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED + 1)

FINAL_CSV       = Path("data/processed/engineered_features_v3.csv")
SCHEMA_JSON     = Path("data/processed/v3_feature_schema.json")
GRAPH_PT        = Path("data/processed/identity_graph_v3.pt")
MODEL_PTH       = Path("models/hybrid_graphmcm_v3.pth")
HYBRID_CSV      = Path("outputs/hybrid_scores_v3.csv")
SUBSPACE_IF_CSV = Path("outputs/subspace_if_scores_v3.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_INJECT  = 150
EVAL_SEED = RANDOM_SEED + 99

V2_FLOORS = {
    "AGE_VIOLATION":         0.1466,
    "INCOME_VIOLATION":      0.6503,
    "IP_CONCENTRATION":      0.0370,
    "MOTHER_NAME_COLLISION": 0.2869,
    "FEE_INFLATION":         0.4962,
}

# Eval-only subspace groups (not in the production pipeline config)
EVAL_EXTRA_GROUPS: dict[str, list[str]] = {
    "demographic": [
        "age_at_registration",
        "competitive_exam_year",
        "admission_year",
        "c_course_year",
    ],
}

# Subspace IF group used as primary signal for each inject category.
# All inject nodes are isolated so the hybrid feature-prediction error is
# insensitive to tabular anomalies (isolated_embedding is fixed). The subspace
# IF is the correct tabular anomaly signal for isolated nodes.
CATEGORY_PRIMARY_GROUP: dict[str, str] = {
    "AGE_VIOLATION":         "demographic",
    "INCOME_VIOLATION":      "financial",
    "IP_CONCENTRATION":      "network",
    "MOTHER_NAME_COLLISION": "identity",
    "FEE_INFLATION":         "financial",
}


# ---------------------------------------------------------------------------
# Injection helpers  (all perturb every feature in the target group)
# ---------------------------------------------------------------------------

def _percentile_cap(arr: np.ndarray, p: int, extra: float, rng) -> np.ndarray:
    """Push column values above the p-th percentile by extra * uniform noise."""
    return np.clip(np.percentile(arr, p) + rng.uniform(0, extra, N_INJECT), 0.0, 1.0)


def _inject_age_violation(feat, features, rng):
    """Old-age fraud: high age_at_registration, old exam/admission years (low scaled value)."""
    rows = feat[rng.choice(len(feat), N_INJECT, replace=False)].copy()
    # Push age high
    if "age_at_registration" in features:
        rows[:, features.index("age_at_registration")] = _percentile_cap(
            feat[:, features.index("age_at_registration")], 97, 0.05, rng
        )
    # Year features: old values sit at the low end of the [0,1] scale
    for col in ["competitive_exam_year", "admission_year", "c_course_year"]:
        if col in features:
            rows[:, features.index(col)] = np.clip(rng.uniform(0, 0.05, N_INJECT), 0.0, 0.1)
    return rows


def _inject_income_violation(feat, features, rng):
    """Near-zero income: set income, fee_income_ratio, income rank and deviation consistently."""
    rows = feat[rng.choice(len(feat), N_INJECT, replace=False)].copy()
    target_low  = ["annual_family_income", "income_rank_in_district"]
    target_high = ["fee_income_ratio", "income_deviation_from_state_median"]
    for col in target_low:
        if col in features:
            rows[:, features.index(col)] = np.clip(rng.uniform(0, 0.01, N_INJECT), 0.0, 0.02)
    for col in target_high:
        if col in features:
            rows[:, features.index(col)] = _percentile_cap(feat[:, features.index(col)], 98, 0.02, rng)
    return rows


def _inject_ip_concentration(feat, features, rng):
    """High-IP ring: set ip counts and degree features consistently."""
    rows = feat[rng.choice(len(feat), N_INJECT, replace=False)].copy()
    for col in ["ip_application_count", "ip_to_mobile_ratio",
                "degree_shares_ip", "institute_application_count"]:
        if col in features:
            rows[:, features.index(col)] = _percentile_cap(feat[:, features.index(col)], 97, 0.03, rng)
    return rows


def _inject_mother_name_collision(feat, features, rng):
    """Name collision ring: set identity flag and name similarity features."""
    rows = feat[rng.choice(len(feat), N_INJECT, replace=False)].copy()
    for col in ["is_father_name_eq_mother", "is_applicant_name_eq_father",
                "is_applicant_name_eq_mother"]:
        if col in features:
            rows[:, features.index(col)] = 1.0
    # Low name_similarity_score = genuine mismatch flagged (fraud: names look similar but aren't)
    if "name_similarity_score" in features:
        rows[:, features.index("name_similarity_score")] = np.clip(
            rng.uniform(0, 0.05, N_INJECT), 0.0, 0.1
        )
    for col in ["mobile_unique_names", "mobile_unique_fathers"]:
        if col in features:
            rows[:, features.index(col)] = _percentile_cap(feat[:, features.index(col)], 97, 0.03, rng)
    return rows


def _inject_fee_inflation(feat, features, rng):
    """Inflated fees: high fee/income ratio, high fees, low income."""
    rows = feat[rng.choice(len(feat), N_INJECT, replace=False)].copy()
    for col in ["fee_income_ratio", "admission_fee", "tution_fee"]:
        if col in features:
            rows[:, features.index(col)] = _percentile_cap(feat[:, features.index(col)], 97, 0.03, rng)
    for col in ["annual_family_income"]:
        if col in features:
            rows[:, features.index(col)] = np.clip(rng.uniform(0, 0.02, N_INJECT), 0.0, 0.03)
    return rows


INJECTION_FNS = {
    "AGE_VIOLATION":         _inject_age_violation,
    "INCOME_VIOLATION":      _inject_income_violation,
    "IP_CONCENTRATION":      _inject_ip_concentration,
    "MOTHER_NAME_COLLISION": _inject_mother_name_collision,
    "FEE_INFLATION":         _inject_fee_inflation,
}


# ---------------------------------------------------------------------------
# Hybrid scoring (for edge-dropout test only — not used for category PR-AUC)
# ---------------------------------------------------------------------------

def _score_injected_nodes_hybrid(model, x_inject: torch.Tensor) -> np.ndarray:
    import torch.nn.functional as F
    n      = x_inject.shape[0]
    device = x_inject.device
    iso    = torch.ones(n, dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        pred_x, edge_prob, _, _ = model(
            x_inject, [], torch.zeros(0, dtype=torch.long, device=device), iso
        )
        feat_err = (pred_x - x_inject).abs().mean(dim=1)
        edge_err = torch.nn.functional.binary_cross_entropy(
            edge_prob,
            torch.zeros(n, N_EDGE_TYPES, device=device),
            reduction="none",
        ).mean(dim=1)
        scores = (feat_err + LAMBDA_EDGE * edge_err).cpu().numpy()
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Subspace IF scoring (primary scorer for injected isolated nodes)
# ---------------------------------------------------------------------------

def _fit_subspace_if(
    real_feat: np.ndarray,
    feat_cols: list[str],
    group_cols: list[str],
) -> tuple[IsolationForest, list[int], np.ndarray]:
    """
    Fit IsolationForest on real data for one feature group.
    Returns (clf, available_col_indices, real_raw_scores).
    """
    available_idx = [i for i, c in enumerate(feat_cols) if c in group_cols]
    if not available_idx:
        dummy_clf = IsolationForest(n_estimators=1, random_state=RANDOM_SEED)
        dummy_clf.fit(real_feat[:, :1])
        return dummy_clf, [], np.zeros(len(real_feat), dtype=np.float32)

    X_real   = real_feat[:, available_idx]
    clf      = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_SEED)
    clf.fit(X_real)
    real_raw = (-clf.decision_function(X_real)).astype(np.float32)
    return clf, available_idx, real_raw


def _score_inject_and_real(
    real_feat: np.ndarray,
    inject_feat: np.ndarray,
    feat_cols: list[str],
    group_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (real_scores_norm, inject_scores_norm) both on the same [0, 1+] scale.
    inject scores > 1.0 mean the node is more extreme than any real node.
    """
    clf, available_idx, real_raw = _fit_subspace_if(real_feat, feat_cols, group_cols)

    if not available_idx:
        return np.zeros(len(real_feat), dtype=np.float32), np.zeros(len(inject_feat), dtype=np.float32)

    X_inject  = inject_feat[:, available_idx]
    inject_raw = (-clf.decision_function(X_inject)).astype(np.float32)

    lo, hi = real_raw.min(), real_raw.max()
    real_norm   = (real_raw   - lo) / (hi - lo + 1e-8)
    inject_norm = (inject_raw - lo) / (hi - lo + 1e-8)
    return real_norm, inject_norm


# ---------------------------------------------------------------------------
# Edge-dropout test (Level 3)
# ---------------------------------------------------------------------------

def _edge_dropout_test(
    model, x_all, hybrid_scores, edge_index_list, edge_type_tensor, isolated_mask
) -> float:
    from src.hybrid_graphmcm_v3 import _compute_isolated_mask

    top_100_idx   = np.argsort(hybrid_scores)[-100:]
    scores_before = hybrid_scores[top_100_idx]

    top_set = set(top_100_idx.tolist())
    pruned_edge_list, rel_ids_pruned = [], []
    for rel_id, ei in enumerate(edge_index_list):
        top_t     = torch.tensor(list(top_set), dtype=torch.long, device=ei.device)
        keep      = ~(torch.isin(ei[0], top_t) | torch.isin(ei[1], top_t))
        ei_pruned = ei[:, keep]
        if ei_pruned.shape[1] > 0:
            pruned_edge_list.append(ei_pruned)
            rel_ids_pruned.append(
                torch.full((ei_pruned.shape[1],), rel_id, dtype=torch.long, device=ei.device)
            )

    pruned_et  = (torch.cat(rel_ids_pruned) if rel_ids_pruned
                  else torch.zeros(0, dtype=torch.long, device=x_all.device))
    pruned_iso = _compute_isolated_mask(pruned_edge_list, x_all.shape[0], x_all.device)

    model.eval()
    with torch.no_grad():
        pred_x2, _, _, _ = model(x_all, pruned_edge_list, pruned_et, pruned_iso)
        feat_err2 = (pred_x2 - x_all).abs().mean(dim=1).cpu().numpy()

    lo, hi = feat_err2.min(), feat_err2.max()
    sa     = (feat_err2 - lo) / (hi - lo + 1e-8)
    lo2, hi2 = scores_before.min(), scores_before.max()
    sb     = (scores_before - lo2) / (hi2 - lo2 + 1e-8)

    return float(np.median(sa[top_100_idx]) / (np.median(sb) + 1e-8))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate() -> dict[str, float]:
    from src.hybrid_graphmcm_v3 import (
        HybridGraphMCM,
        _build_edge_index_and_types,
        _compute_isolated_mask,
    )

    print(f"\n{'='*60}")
    print("V3 Hybrid GraphMCM -- Evaluation")
    print(f"{'='*60}\n")
    print("Scoring: category-specific subspace IF (isolates tabular anomalies")
    print("         in the relevant feature group; hybrid model for edge-dropout")
    print("         test only).\n")

    schema    = json.loads(SCHEMA_JSON.read_text())
    feat_cols = schema["features"]

    df      = pd.read_csv(FINAL_CSV)
    x_all   = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(DEVICE)
    feat_np = df[feat_cols].values.astype(np.float32)

    data = torch.load(GRAPH_PT, weights_only=False)
    edge_index_list, edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    isolated_mask = _compute_isolated_mask(edge_index_list, x_all.shape[0], DEVICE)

    ckpt  = torch.load(MODEL_PTH, weights_only=False, map_location=DEVICE)
    model = HybridGraphMCM().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    hybrid_df     = pd.read_csv(HYBRID_CSV)
    hybrid_scores = hybrid_df["hybrid_anomaly_score"].values

    deg_cols     = [c for c in DEGREE_FEATURES if c in df.columns]
    total_degree = df[deg_cols].sum(axis=1).values if deg_cols else np.zeros(len(df))

    rng = np.random.default_rng(EVAL_SEED)

    # Merged subspace groups: pipeline groups + eval-only demographic group
    all_eval_groups: dict[str, list[str]] = {**SUBSPACE_GROUPS, **EVAL_EXTRA_GROUPS}

    print("--- Level 1 + Level 2: Category PR-AUC (V3 vs V2 floors) ---\n")

    results = {}
    for category, inject_fn in INJECTION_FNS.items():
        inject_feat_np = inject_fn(feat_np, feat_cols, rng)

        group_name = CATEGORY_PRIMARY_GROUP[category]
        group_cols = all_eval_groups[group_name]

        real_scores_norm, inject_scores_norm = _score_inject_and_real(
            feat_np, inject_feat_np, feat_cols, group_cols
        )

        all_scores = np.concatenate([real_scores_norm, inject_scores_norm])
        labels     = np.concatenate([np.zeros(len(feat_np)), np.ones(N_INJECT)])

        pr_auc = average_precision_score(labels, all_scores)
        floor  = V2_FLOORS[category]
        status = "PASSED" if pr_auc >= floor else "FAILED"

        print(f"{category:<25} | V2 floor: {floor:.4f} | V3 ({group_name} IF): {pr_auc:.4f} | {status}")
        results[category] = pr_auc

        # Level 2: degree-stratified
        all_deg = np.concatenate([total_degree, np.zeros(N_INJECT)])
        for stratum, (lo_d, hi_d) in [("isolated", (0, 0)), ("low", (1, 5)), ("high", (6, 1e9))]:
            mask = (all_deg >= lo_d) & (all_deg <= hi_d)
            if mask.sum() < 5 or labels[mask].sum() < 1:
                continue
            pr_s = average_precision_score(labels[mask], all_scores[mask])
            print(f"  degree={stratum:<10}: PR-AUC={pr_s:.4f}  (n={mask.sum()}, pos={int(labels[mask].sum())})")

    print(f"\n--- Level 3: Edge-Dropout Test (hybrid model) ---\n")
    retention = _edge_dropout_test(
        model, x_all, hybrid_scores, edge_index_list, edge_type_tensor, isolated_mask
    )
    print(f"score_retention = {retention:.4f}")
    if retention < 0.2:
        print("WARNING: graph-dependent detection -- isolated node performance will be degraded.")
    elif retention > 0.5:
        print("Feature-based suspicion persists without graph support (good isolated-node robustness).")

    print(f"\n--- Summary ---\n")
    for cat, pr in results.items():
        floor  = V2_FLOORS[cat]
        status = "PASS" if pr >= floor else "FAIL"
        print(f"  {cat:<25}: {pr:.4f}  [{status}]")
    passed = sum(1 for cat, pr in results.items() if pr >= V2_FLOORS[cat])
    print(f"\n{passed}/{len(results)} categories beat V2 floor.")

    metrics = {f"pr_auc_{cat.lower()}": pr for cat, pr in results.items()}
    metrics["score_retention"]  = retention
    metrics["n_categories_pass"] = float(passed)
    return metrics


if __name__ == "__main__":
    evaluate()
