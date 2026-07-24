"""
compare_architectures_v3.py

LOCKED-fusion validation harness (score-level, LightGBM removed).

Validates the settled V4.1 architecture — the max SCORE-LEVEL fusion
(`max(minmax(subspace), minmax(dense_relational), minmax(hybrid))`, see
fusion_classifier_v3.score_level_fusion; changed 2026-07-22 from the additive
weighted-sum after the stress_testing_1 ablation showed the sum diluting
whichever detector actually found the fraud — see README changelog) — on the
connected-cluster harness (T1) and the held-out star/bipartite harness (T9b),
using the PRETRAINED per-seed detectors (models/hybrid_v3_seed{seed}.pth), frozen and
never retrained. The 14-positive LightGBM combiner was removed (it destroyed the raw
signals, docs/AGENTS.md H.8); there is no label fit set here — the fusion is
label-independent.

Reports per seed / per category PR-AUC for:
  - locked_fusion        : the locked architecture (primary)
  - subspace_only        : raw subspace IF              (tabular backbone)
  - hybrid_only          : raw hybrid_anomaly            (RGCN relational)
  - dense_relational_only: raw dense-block, mobile/IP IP-priority-weighted max

so the fusion can be read against each of its parts (mirrors H.8).

Writes: outputs/ablation/locked_fusion_validation.json
        (does NOT overwrite the canonical tier_comparison.json / H.7 record)
"""

import json
import warnings
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score

# _score_inject_and_real calls sklearn with bare numpy arrays -> feature-name
# UserWarnings. Silence so the run log stays readable.
warnings.filterwarnings("ignore", category=UserWarning)

from src.config_v3 import (
    COMPARE_SEEDS, EVAL_HELDOUT_SIZE_RANGE, EVAL_CONNECTED_N_CLUSTERS,
    EVAL_CONNECTED_SIZE_RANGE, SUBSPACE_GROUPS, DENSE_BLOCK_RELATIONS, EDGE_TYPES,
    DENSE_BLOCK_RELATION_WEIGHTS,
)
from src.evaluate_model_v3 import (
    INJECTION_FNS, CATEGORY_PRIMARY_GROUP, EVAL_EXTRA_GROUPS,
)
import src.evaluate_model_v3 as eval_m
from src.hybrid_graphmcm_v3 import (
    HybridGraphMCM, _build_edge_index_and_types, _compute_isolated_mask,
    compute_score_frame, DEVICE,
)
from src.dense_block_detector_v3 import dense_block_scores
from src.fusion_classifier_v3 import score_level_fusion

# RELATION_MAP is defined *inside* evaluate_connected() (not module-level), so we
# replicate it here. Keep in sync with evaluate_model_v3.
RELATION_MAP = {
    "IP_CONCENTRATION": 1,
    "MOTHER_NAME_COLLISION": 3,
    "FEE_INFLATION": 4,
    "AGE_VIOLATION": 4,
    "INCOME_VIOLATION": 4,
}
# all_eval_groups is a local inside evaluate(); rebuild the identical merge here.
ALL_EVAL_GROUPS = {**SUBSPACE_GROUPS, **EVAL_EXTRA_GROUPS}

# Dense-block relational column: IP-priority-weighted max across mobile/IP
# (DENSE_BLOCK_RELATIONS=[0,1], DENSE_BLOCK_RELATION_WEIGHTS) — see config_v3.
# Pincode dropped 2026-07-22 -- not a valid fraud signal on its own.
DENSE_RELATIONAL_COL = "dense_block_score_relational"

MODES = ["locked_fusion", "subspace_only", "hybrid_only", "dense_relational_only"]

ABLATION_DIR = Path("outputs/ablation")
JSON_OUT = ABLATION_DIR / "locked_fusion_validation.json"


def _build_augmented_graph(
    x_real: torch.Tensor, edge_index_list: list, edge_type_tensor: torch.Tensor,
    feat_np: np.ndarray, feat_cols: list, category: str, rng: np.random.Generator,
    mode_is_heldout: bool = False,
):
    """
    Builds the augmented graph containing either the standard T1 cliques
    or the T9b held-out shapes (star/bipartite).
    """
    import src.evaluate_model_v3 as eval_mod
    all_x_inject = []
    cluster_edges = []

    n_real = x_real.shape[0]
    node_offset = n_real
    rel_idx = RELATION_MAP[category]

    n_clusters = EVAL_CONNECTED_N_CLUSTERS
    size_range = EVAL_HELDOUT_SIZE_RANGE if mode_is_heldout else EVAL_CONNECTED_SIZE_RANGE

    old_n_inject = eval_mod.N_INJECT
    inject_fn = INJECTION_FNS[category]

    for c in range(n_clusters):
        c_size = rng.integers(size_range[0], size_range[1] + 1)
        eval_mod.N_INJECT = c_size

        x_c_np = inject_fn(feat_np, feat_cols, rng)
        all_x_inject.append(x_c_np)

        nodes = np.arange(node_offset, node_offset + c_size)
        if mode_is_heldout:
            if c % 2 == 0:
                # Star shape
                hub = nodes[0]
                leaves = nodes[1:]
                ei = np.vstack([np.repeat(hub, len(leaves)), leaves])
                ei = np.hstack([ei, ei[::-1]])  # undirected
            else:
                # Bipartite
                group1 = nodes[:c_size // 2]
                group2 = nodes[c_size // 2:]
                idx_i, idx_j = np.meshgrid(group1, group2)
                ei = np.vstack([idx_i.ravel(), idx_j.ravel()])
                ei = np.hstack([ei, ei[::-1]])
            cluster_edges.append(ei)
        else:
            # Clique
            idx_i, idx_j = np.meshgrid(nodes, nodes)
            mask = idx_i != idx_j
            ei = np.vstack([idx_i[mask], idx_j[mask]])
            cluster_edges.append(ei)

        node_offset += c_size

    eval_mod.N_INJECT = old_n_inject

    x_inject_np = np.vstack(all_x_inject) if all_x_inject else np.zeros((0, x_real.shape[1]), dtype=np.float32)
    x_inject = torch.tensor(x_inject_np, dtype=torch.float32).to(DEVICE)
    x_all = torch.cat([x_real, x_inject], dim=0)

    eval_edge_index_list = [ei.clone() for ei in edge_index_list]

    if cluster_edges:
        new_edges = np.hstack(cluster_edges)
        new_edges_t = torch.tensor(new_edges, dtype=torch.long, device=DEVICE)
        if eval_edge_index_list[rel_idx].shape[1] > 0:
            eval_edge_index_list[rel_idx] = torch.cat([eval_edge_index_list[rel_idx], new_edges_t], dim=1)
        else:
            eval_edge_index_list[rel_idx] = new_edges_t

    eval_edge_type_list = []
    for r_id, ei in enumerate(eval_edge_index_list):
        if ei.shape[1] > 0:
            eval_edge_type_list.append(torch.full((ei.shape[1],), r_id, dtype=torch.long, device=DEVICE))

    eval_edge_type_tensor = torch.cat(eval_edge_type_list) if eval_edge_type_list else torch.zeros(0, dtype=torch.long, device=DEVICE)
    eval_isolated_mask = _compute_isolated_mask(eval_edge_index_list, x_all.shape[0], DEVICE)

    return x_all, eval_edge_index_list, eval_edge_type_tensor, eval_isolated_mask, x_inject.shape[0]


def _score_category(model, x_real, base_edge_index_list, base_edge_type_tensor,
                    feat_np, feat_cols, real_app_ids, category, rng_eval, mode_is_heldout):
    """
    Score one category on the augmented graph with the frozen detector and return
    {mode: preds_in_node_order} plus the binary labels (injected = 1).

    The three raw components are aligned by node order:
      - hybrid_anomaly_score            : compute_score_frame over the whole augmented graph
      - subspace_if_score               : _score_inject_and_real (real + injected, same scale)
      - dense_block_score_relational    : dense_block_scores, mobile/IP IP-priority-weighted max
    then fused via the LOCKED score_level_fusion.
    """
    n_real = x_real.shape[0]
    x_all, eval_ei, eval_et, eval_iso, n_inject = _build_augmented_graph(
        x_real, base_edge_index_list, base_edge_type_tensor, feat_np, feat_cols,
        category, rng_eval, mode_is_heldout=mode_is_heldout,
    )
    aug_app_ids = np.concatenate([real_app_ids, np.array([f"inj_{i}" for i in range(n_inject)])])

    with torch.no_grad():
        x_all_d = x_all.to(DEVICE)
        eval_ei_d = [ei.to(DEVICE) for ei in eval_ei]
        eval_et_d = eval_et.to(DEVICE)
        eval_iso_d = _compute_isolated_mask(eval_ei_d, x_all_d.shape[0], DEVICE)
        aug_score_df = compute_score_frame(model, x_all_d, eval_ei_d, eval_et_d, eval_iso_d, aug_app_ids, feat_cols)

    # Subspace IF: fit on real, score real + injected on the same [0,1+] scale.
    real_norm, inject_norm = eval_m._score_inject_and_real(
        feat_np, x_all[n_real:].cpu().numpy(), feat_cols, ALL_EVAL_GROUPS[CATEGORY_PRIMARY_GROUP[category]]
    )
    aug_score_df["subspace_if_score"] = np.concatenate([real_norm, inject_norm])

    # Dense-block, relational (mobile/IP, IP-priority-weighted max).
    # Merge to align by application_id.
    aug_dense_df = dense_block_scores(eval_ei_d, eval_et_d, x_all.shape[0], aug_app_ids)
    merged = aug_score_df.merge(aug_dense_df, on="application_id", how="left")
    if DENSE_RELATIONAL_COL not in merged.columns:
        merged[DENSE_RELATIONAL_COL] = 0.0
    merged[DENSE_RELATIONAL_COL] = merged[DENSE_RELATIONAL_COL].fillna(0.0)

    subspace = merged["subspace_if_score"].values
    hybrid = merged["hybrid_anomaly_score"].values
    dense_relational = merged[DENSE_RELATIONAL_COL].values

    preds = {
        "locked_fusion": score_level_fusion(subspace, dense_relational, hybrid),
        "subspace_only": subspace,
        "hybrid_only": hybrid,
        "dense_relational_only": dense_relational,
    }

    labels = np.zeros(x_all.shape[0])
    labels[n_real:] = 1.0

    # merge preserved node order (real 0..n_real-1 then injected); assert alignment
    assert len(labels) == len(preds["locked_fusion"]) == x_all.shape[0]
    return preds, labels


def run_comparison():
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    print("[validate] LOCKED max score-level fusion "
          "(risk = minmax(max(minmax(subspace), minmax(dense_relational), minmax(hybrid))))")
    print(f"[validate] Dense-block relations -> {DENSE_BLOCK_RELATIONS} "
          f"weights={DENSE_BLOCK_RELATION_WEIGHTS} -> '{DENSE_RELATIONAL_COL}'")
    print(f"[validate] seeds: {list(COMPARE_SEEDS)}  (frozen pretrained detectors, no retraining)")

    # Load real data once
    df = pd.read_csv(Path("data/processed/engineered_features_v3.csv"))
    schema = json.loads(Path("data/processed/v3_feature_schema.json").read_text())
    feat_cols = schema["features"]

    x_real = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(DEVICE)
    feat_np = df[feat_cols].values.astype(np.float32)
    real_app_ids = df["application_id"].values
    cat_order = list(INJECTION_FNS)

    data = torch.load(Path("data/processed/identity_graph_v3.pt"), weights_only=False)
    base_edge_index_list, base_edge_type_tensor = _build_edge_index_and_types(data, DEVICE)

    out_json = {}

    for seed in COMPARE_SEEDS:
        print(f"\n{'='*46}\nLOCKED-fusion validation | SEED {seed}\n{'='*46}")
        ckpt_path = Path(f"models/hybrid_v3_seed{seed}.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Pretrained detector {ckpt_path} not found. This harness reuses the "
                f"frozen per-seed checkpoints and does not train. Available seeds must "
                f"cover COMPARE_SEEDS={list(COMPARE_SEEDS)}."
            )

        model = HybridGraphMCM().to(DEVICE)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        out_json[str(seed)] = {m: {} for m in MODES}

        for cat in INJECTION_FNS:
            rng_eval = np.random.default_rng(seed * 100 + cat_order.index(cat))
            preds, labels = _score_category(
                model, x_real, base_edge_index_list, base_edge_type_tensor,
                feat_np, feat_cols, real_app_ids, cat, rng_eval, mode_is_heldout=False,
            )
            for m in MODES:
                pr = average_precision_score(labels, preds[m])
                out_json[str(seed)][m][f"conn_pr_auc_{cat}"] = float(pr)
            print(f"  {cat:<24} | locked={out_json[str(seed)]['locked_fusion'][f'conn_pr_auc_{cat}']:.4f}"
                  f"  sub={out_json[str(seed)]['subspace_only'][f'conn_pr_auc_{cat}']:.4f}"
                  f"  hyb={out_json[str(seed)]['hybrid_only'][f'conn_pr_auc_{cat}']:.4f}"
                  f"  rel={out_json[str(seed)]['dense_relational_only'][f'conn_pr_auc_{cat}']:.4f}")

        for m in MODES:
            vals = list(out_json[str(seed)][m].values())
            out_json[str(seed)][m]["mean_conn_pr_auc"] = float(np.mean(vals))

    # Aggregate mean / std across seeds
    out_json["aggregate"] = {}
    for m in MODES:
        out_json["aggregate"][m] = {}
        for cat in INJECTION_FNS:
            k = f"conn_pr_auc_{cat}"
            vals = [out_json[str(s)][m][k] for s in COMPARE_SEEDS]
            out_json["aggregate"][m][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        mean_vals = [out_json[str(s)][m]["mean_conn_pr_auc"] for s in COMPARE_SEEDS]
        out_json["aggregate"][m]["mean_conn_pr_auc"] = {"mean": float(np.mean(mean_vals)), "std": float(np.std(mean_vals))}

    # Held-out T9b (novel star/bipartite topology), seed 42 — reload frozen detector
    print(f"\n{'='*46}\nHeld-out T9b (star/bipartite) | SEED {COMPARE_SEEDS[0]}\n{'='*46}")
    seed = COMPARE_SEEDS[0]
    model = HybridGraphMCM().to(DEVICE)
    ckpt = torch.load(Path(f"models/hybrid_v3_seed{seed}.pth"), weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_json["heldout"] = {}
    for cat in INJECTION_FNS:
        rng_eval = np.random.default_rng(seed * 100 + 777 + cat_order.index(cat))
        preds, labels = _score_category(
            model, x_real, base_edge_index_list, base_edge_type_tensor,
            feat_np, feat_cols, real_app_ids, cat, rng_eval, mode_is_heldout=True,
        )
        for m in MODES:
            out_json["heldout"][f"{cat}_{m}"] = float(average_precision_score(labels, preds[m]))
        print(f"  {cat:<24} | locked={out_json['heldout'][f'{cat}_locked_fusion']:.4f}"
              f"  sub={out_json['heldout'][f'{cat}_subspace_only']:.4f}"
              f"  hyb={out_json['heldout'][f'{cat}_hybrid_only']:.4f}"
              f"  rel={out_json['heldout'][f'{cat}_dense_relational_only']:.4f}")

    out_json["_meta"] = {
        "harness": "locked max score-level fusion (LightGBM removed)",
        "combine": "max(minmax(subspace), minmax(dense_relational), minmax(hybrid))",
        "dense_block_relations": DENSE_BLOCK_RELATIONS,
        "dense_block_relation_weights": DENSE_BLOCK_RELATION_WEIGHTS,
        "dense_block_column": DENSE_RELATIONAL_COL,
        "seeds": list(COMPARE_SEEDS),
        "detectors": "frozen pretrained models/hybrid_v3_seed{seed}.pth",
        "note": f"{len(COMPARE_SEEDS)} seeds; H.6 formal gate wants >3 — treat as proposed, pending a 4th seed.",
    }

    with open(JSON_OUT, "w") as f:
        json.dump(out_json, f, indent=2)

    # Console summary
    print(f"\n{'='*46}\nAGGREGATE (mean over seeds {list(COMPARE_SEEDS)})\n{'='*46}")
    print(f"{'Category':<24} {'locked':>8} {'subspace':>9} {'hybrid':>8} {'dense_ip':>9}")
    for cat in INJECTION_FNS:
        k = f"conn_pr_auc_{cat}"
        row = [out_json["aggregate"][m][k]["mean"] for m in MODES]
        print(f"{cat:<24} {row[0]:>8.4f} {row[1]:>9.4f} {row[2]:>8.4f} {row[3]:>9.4f}")
    means = [out_json["aggregate"][m]["mean_conn_pr_auc"]["mean"] for m in MODES]
    stds = [out_json["aggregate"][m]["mean_conn_pr_auc"]["std"] for m in MODES]
    print(f"{'MEAN':<24} {means[0]:>8.4f} {means[1]:>9.4f} {means[2]:>8.4f} {means[3]:>9.4f}")
    print(f"{'(std of mean)':<24} {stds[0]:>8.4f} {stds[1]:>9.4f} {stds[2]:>8.4f} {stds[3]:>9.4f}")
    print(f"\nSaved -> {JSON_OUT}")


if __name__ == "__main__":
    run_comparison()
