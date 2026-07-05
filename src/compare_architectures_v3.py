"""
compare_architectures_v3.py

Track N: T9 / T9b Head-to-head comparison
Runs the baseline fusion, Tier-1 attention features, and Ring-classifier modes
on the same augmented graph and outputs per-category PR-AUC to JSON.
"""

import os
import json
import warnings
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# project_to_nodes calls predict_proba per candidate with a numpy array, which
# triggers sklearn's "X does not have valid feature names" UserWarning thousands
# of times. Silence it here so the run log stays readable (no env-var prefix
# needed at the shell — keeps the harness invocation simple).
warnings.filterwarnings("ignore", category=UserWarning)

from src.config_v3 import (
    COMPARE_SEEDS, EVAL_HELDOUT_SIZE_RANGE, EVAL_CONNECTED_N_CLUSTERS,
    EVAL_CONNECTED_SIZE_RANGE, RANDOM_SEED, CONFIRMED_WEIGHT, SUBSPACE_GROUPS,
)
from src.evaluate_model_v3 import INJECTION_FNS, CATEGORY_PRIMARY_GROUP, EVAL_EXTRA_GROUPS
from src.hybrid_graphmcm_v3 import (
    HybridGraphMCM, load_model_and_inputs, _build_edge_index_and_types,
    _compute_isolated_mask, train, compute_score_frame, run_attention_summary, DEVICE
)
from src.ring_candidate_v3 import generate_candidates
from src.ring_classifier_v3 import RingClassifier, project_to_nodes, LGB_PARAMS
from src.subspace_if_v3 import run_subspace_if
from src.dense_block_detector_v3 import dense_block_scores
from src.deviation_layer_v3 import train_and_score_oof, build_dev_features, score_devnet
from src.config_v3 import DEGREE_FEATURES, DENSE_BLOCK_RELATIONS, EDGE_TYPES
import src.evaluate_model_v3 as eval_m

# RELATION_MAP is defined *inside* evaluate_connected() (not module-level), so we
# replicate it here rather than import it. Keep in sync with evaluate_model_v3.
RELATION_MAP = {
    "IP_CONCENTRATION": 1,
    "MOTHER_NAME_COLLISION": 3,
    "FEE_INFLATION": 4,
    "AGE_VIOLATION": 4,
    "INCOME_VIOLATION": 4,
}
# all_eval_groups is a local inside evaluate(); rebuild the identical merge here.
ALL_EVAL_GROUPS = {**SUBSPACE_GROUPS, **EVAL_EXTRA_GROUPS}

ABLATION_DIR = Path("outputs/ablation")
JSON_OUT = ABLATION_DIR / "tier_comparison.json"


def _build_augmented_graph(
    x_real: torch.Tensor, edge_index_list: list, edge_type_tensor: torch.Tensor,
    feat_np: np.ndarray, feat_cols: list, category: str, rng: np.random.Generator,
    mode_is_heldout: bool = False
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
                ei = np.hstack([ei, ei[::-1]]) # undirected
            else:
                # Bipartite
                group1 = nodes[:c_size//2]
                group2 = nodes[c_size//2:]
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


def run_comparison():
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # 0. Load real data once
    final_csv = Path("data/processed/engineered_features_v3.csv")
    df = pd.read_csv(final_csv)
    schema = json.loads(Path("data/processed/v3_feature_schema.json").read_text())
    feat_cols = schema["features"]
    
    x_real = torch.tensor(df[feat_cols].values, dtype=torch.float32).to(DEVICE)
    feat_np = df[feat_cols].values.astype(np.float32)
    real_app_ids = df["application_id"].values
    n_real = x_real.shape[0]
    cat_order = list(INJECTION_FNS)  # stable, deterministic index per category
    
    data = torch.load(Path("data/processed/identity_graph_v3.pt"), weights_only=False)
    base_edge_index_list, base_edge_type_tensor = _build_edge_index_and_types(data, DEVICE)
    
    # We will build a single LightGBM baseline fit set per seed for real data + pseudo-labels.
    labels_data = json.loads(Path("outputs/pseudo_labels_v3.json").read_text())
    positive_ids = {r["application_id"] for r in labels_data["positive_set"]}
    source_map = {r["application_id"]: r.get("source", "evt_pseudo") for r in labels_data["positive_set"]}
    
    y_real = np.array([1 if a in positive_ids else 0 for a in real_app_ids])
    src_real = np.array([source_map.get(a, "negative") for a in real_app_ids])
    weight_map = {"confirmed": CONFIRMED_WEIGHT, "evt_pseudo": 1.0, "negative": 1.0}
    w_real = np.array([weight_map[s] for s in src_real])
    
    confirmed_patterns = {}
    for r in labels_data["positive_set"]:
        if r.get("source") == "confirmed":
            cat = r.get("category", "UNKNOWN")
            confirmed_patterns.setdefault(cat, []).append(r["application_id"])
    
    real_deg_df = df[["application_id"] + [c for c in DEGREE_FEATURES if c in df.columns]]
    
    out_json = {}
    
    for seed in COMPARE_SEEDS:
        print(f"\n{'='*40}\nRunning comparison for SEED {seed}\n{'='*40}")
        out_json[str(seed)] = {
            "baseline": {}, "tier1": {}, "ring": {}, "max_fusion": {},
            "dense_block_fusion": {}, "dense_block_only": {}
        }
        
        # 1. Setup per-seed detector checkpoint
        os.environ["V4_ENCODER_ARCH"] = "rgcn"
        os.environ["V4_TOPO_EXPOSURE"] = "1"
        os.environ["V4_SEED"] = str(seed)

        # config_v3.RANDOM_SEED / torch.manual_seed are frozen at import time, so
        # the env var above cannot re-seed the detector. Re-seed the global RNGs
        # here so each seed trains a genuinely different detector (seed-everything).
        import random as _random
        torch.manual_seed(seed)
        np.random.seed(seed)
        _random.seed(seed)

        ckpt_path = Path(f"models/hybrid_v3_seed{seed}.pth")
        if not ckpt_path.exists():
            print(f"Training detector for seed {seed}...")
            train(smoke_test=False)
            import shutil
            shutil.copy2("models/hybrid_graphmcm_v3.pth", ckpt_path)
            run_subspace_if()
            run_attention_summary()
        
        # Load the per-seed model (on CPU for deterministic scoring)
        # Score on GPU (CPU-determinism dropped). Detectors are FROZEN — reused
        # from models/hybrid_v3_seed{seed}.pth, never retrained per run — which
        # removes the detector-instance variance that swamped the architecture
        # signal. GPU scatter-add adds only ~±0.04 scoring noise.
        score_device = DEVICE
        model = HybridGraphMCM().to(score_device)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=score_device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        # Compute baseline and tier1 features on REAL data for fit
        with torch.no_grad():
            x_real_cpu = x_real.to(score_device)
            base_ei_cpu = [ei.to(score_device) for ei in base_edge_index_list]
            base_et_cpu = base_edge_type_tensor.to(score_device)
            base_iso_cpu = _compute_isolated_mask(base_ei_cpu, x_real_cpu.shape[0], score_device)
            
            real_score_df = compute_score_frame(model, x_real_cpu, base_ei_cpu, base_et_cpu, base_iso_cpu, real_app_ids, feat_cols)
            real_attn_df = model.attention_summary(x_real_cpu, base_ei_cpu, base_et_cpu, base_iso_cpu, real_app_ids)
            
        real_subspace_df = pd.read_csv("outputs/subspace_if_scores_v3.csv")
        
        # Build features for fit
        merged_real = real_score_df.merge(real_subspace_df[["application_id", "subspace_if_score"]], on="application_id")
        merged_real_t1 = merged_real.merge(real_attn_df, on="application_id")
        
        baseline_cols = ["hybrid_anomaly_score", "feature_pred_error", "edge_pred_error", "subspace_if_score"]
        tier1_cols = baseline_cols + [f"beta_{r}" for r in range(5)] + [f"alpha_entropy_{r}" for r in range(5)] + ["alpha_top1"]
        
        # Dense Block & Deviation logic
        real_dense_df = dense_block_scores(base_ei_cpu, base_et_cpu, n_real, real_app_ids)
        
        rng_syn = np.random.default_rng(seed)
        synthetic_exposure = {}
        for cat, fn in INJECTION_FNS.items():
            synthetic_exposure[cat] = fn(feat_np, feat_cols, rng_syn)
            
        real_dev_df, dev_model = train_and_score_oof(
            x_real.cpu().numpy(), real_dense_df, real_deg_df,
            synthetic_exposure, confirmed_patterns, real_app_ids
        )
        
        merged_real_dense = merged_real_t1.merge(real_dense_df, on="application_id").merge(real_dev_df[["application_id", "deviation_score"]], on="application_id")
        
        dense_cols = [f"dense_block_score_{EDGE_TYPES[r].replace('shares_', '')}" for r in DENSE_BLOCK_RELATIONS]
        db_fusion_cols = baseline_cols + dense_cols + ["deviation_score"]
        db_only_cols = ["subspace_if_score"] + dense_cols + ["deviation_score"]
        
        X_base_real = merged_real_dense[baseline_cols].values
        X_tier1_real = merged_real_dense[tier1_cols].values
        X_db_fusion_real = merged_real_dense[db_fusion_cols].values
        X_db_only_real = merged_real_dense[db_only_cols].values
        
        # Fit models on REAL data
        clf_base = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
        clf_base.fit(X_base_real, y_real, sample_weight=w_real)
        
        clf_tier1 = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
        clf_tier1.fit(X_tier1_real, y_real, sample_weight=w_real)
        
        clf_db_fusion = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
        clf_db_fusion.fit(X_db_fusion_real, y_real, sample_weight=w_real)
        
        clf_db_only = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
        clf_db_only.fit(X_db_only_real, y_real, sample_weight=w_real)
        
        # 2. Train Ring Classifier on REAL data candidate rings
        # First generate candidate rings from the real graph
        base_hybrid_scores = torch.tensor(merged_real["hybrid_anomaly_score"].values, dtype=torch.float32, device=DEVICE)
        real_candidates = generate_candidates(base_edge_index_list, base_edge_type_tensor, x_real, base_hybrid_scores)
        
        # Split candidates into pos/neg based on overlap with confirmed nodes
        confirmed_nodes = set(np.where(y_real == 1)[0])
        pos_sg = []
        neg_sg = []
        for sg in real_candidates:
            if any(u in confirmed_nodes for u in sg["node_ids"]):
                pos_sg.append(sg)
            else:
                neg_sg.append(sg)
                
        # Also add synthetic clusters to pos_sg
        syn_exp = torch.load("data/processed/synthetic_exposure_graph_v3.pt", weights_only=False)
        for cid in torch.unique(syn_exp["cluster_id"]):
            mask = syn_exp["cluster_id"] == cid
            c_nodes = torch.where(mask)[0].tolist()
            if len(c_nodes) >= 4:
                # We need edges among them. Since they are cliques or stars, we just add dummy subgraph
                sg = {"node_ids": c_nodes, "x": syn_exp["x"][mask].cpu().numpy(), "edges": [], "scores": np.zeros(len(c_nodes))}
                # Rebuild edges from syn_exp["edge_index"]
                e_src, e_dst = syn_exp["edge_index"]
                e_rel = syn_exp["edge_type"]
                for i in range(len(e_src)):
                    if e_src[i].item() in c_nodes and e_dst[i].item() in c_nodes:
                        sg["edges"].append((c_nodes.index(e_src[i].item()), c_nodes.index(e_dst[i].item()), e_rel[i].item()))
                pos_sg.append(sg)
                
        # Subsample negatives
        rng_neg = np.random.default_rng(seed)
        if len(neg_sg) > 400:
            neg_sg = rng_neg.choice(neg_sg, 400, replace=False).tolist()
            
        ring_clf = RingClassifier()
        ring_clf.clf.set_params(random_state=seed)
        ring_clf.fit(pos_sg, neg_sg)
        
        # 3. Evaluate each category
        for cat in INJECTION_FNS:
            rng_eval = np.random.default_rng(seed * 100 + cat_order.index(cat))
            
            x_all, eval_ei, eval_et, eval_iso, n_inject = _build_augmented_graph(
                x_real, base_edge_index_list, base_edge_type_tensor, feat_np, feat_cols, cat, rng_eval, mode_is_heldout=False
            )
            
            aug_app_ids = np.concatenate([real_app_ids, np.array([f"inj_{i}" for i in range(n_inject)])])
            
            with torch.no_grad():
                x_all_cpu = x_all.to(score_device)
                eval_ei_cpu = [ei.to(score_device) for ei in eval_ei]
                eval_et_cpu = eval_et.to(score_device)
                eval_iso_cpu = _compute_isolated_mask(eval_ei_cpu, x_all_cpu.shape[0], score_device)
                
                aug_score_df = compute_score_frame(model, x_all_cpu, eval_ei_cpu, eval_et_cpu, eval_iso_cpu, aug_app_ids, feat_cols)
                aug_attn_df = model.attention_summary(x_all_cpu, eval_ei_cpu, eval_et_cpu, eval_iso_cpu, aug_app_ids)
                
            # Assume injected nodes get subspace IF score from distribution ?
            # Evaluate_connected uses score_inject_and_real. The hybrid features + subspace IF.
            # But here we are using the LIGHTGBM model to score them. We must supply the subspace IF score.
            # Let's mock the subspace IF score for injected nodes as we did in evaluate_connected
            real_scores_norm, inject_scores_norm = eval_m._score_inject_and_real(
                feat_np, x_all[n_real:].cpu().numpy(), feat_cols, ALL_EVAL_GROUPS[CATEGORY_PRIMARY_GROUP[cat]]
            )
            subspace_scores = np.concatenate([real_scores_norm, inject_scores_norm])
            
            aug_score_df["subspace_if_score"] = subspace_scores
            # Augmented dense-block scores (per relation, deterministic)
            aug_dense_df = dense_block_scores(eval_ei_cpu, eval_et_cpu, x_all.shape[0], aug_app_ids)
            # Injected nodes inherit degree 0 (isolated); real nodes keep their degree features
            aug_deg_df = pd.DataFrame({"application_id": aug_app_ids})
            for col in DEGREE_FEATURES:
                if col in df.columns:
                    deg_vals = np.zeros(x_all.shape[0], dtype=np.float32)
                    deg_vals[:n_real] = df[col].values
                    aug_deg_df[col] = deg_vals

            # Deviation: score with the per-seed model trained ONLY on real data.
            # Real nodes keep their leakage-safe OOF score (matches the fit set);
            # injected nodes are scored by that model (never in its training set).
            X_aug_dev = build_dev_features(x_all_cpu.cpu().numpy(), aug_dense_df, aug_deg_df)
            aug_dev_scores = score_devnet(dev_model, X_aug_dev)
            aug_dev_scores[:n_real] = real_dev_df["deviation_score"].values
            aug_dev_df = pd.DataFrame({"application_id": aug_app_ids, "deviation_score": aug_dev_scores})

            aug_merged = aug_score_df.merge(aug_attn_df, on="application_id").merge(
                aug_dense_df, on="application_id").merge(aug_deg_df, on="application_id").merge(
                aug_dev_df, on="application_id")
            
            X_base_aug = aug_merged[baseline_cols].values
            X_tier1_aug = aug_merged[tier1_cols].values
            X_db_fusion_aug = aug_merged[db_fusion_cols].values
            X_db_only_aug = aug_merged[db_only_cols].values
            
            base_preds = clf_base.predict_proba(X_base_aug)[:, 1]
            tier1_preds = clf_tier1.predict_proba(X_tier1_aug)[:, 1]
            db_fusion_preds = clf_db_fusion.predict_proba(X_db_fusion_aug)[:, 1]
            db_only_preds = clf_db_only.predict_proba(X_db_only_aug)[:, 1]
            
            # Ring classifier preds
            aug_hybrid_scores = torch.tensor(aug_merged["hybrid_anomaly_score"].values, dtype=torch.float32, device=DEVICE)
            aug_candidates = generate_candidates(eval_ei_cpu, eval_et_cpu, x_all_cpu, aug_hybrid_scores)
            ring_preds = project_to_nodes(aug_candidates, ring_clf, x_all.shape[0])
            
            maxf_preds = np.maximum(base_preds, tier1_preds)
            
            labels = np.zeros(x_all.shape[0])
            labels[n_real:] = 1.0
            
            out_json[str(seed)]["baseline"][f"conn_pr_auc_{cat}"] = average_precision_score(labels, base_preds)
            out_json[str(seed)]["tier1"][f"conn_pr_auc_{cat}"] = average_precision_score(labels, tier1_preds)
            out_json[str(seed)]["ring"][f"conn_pr_auc_{cat}"] = average_precision_score(labels, ring_preds)
            out_json[str(seed)]["max_fusion"][f"conn_pr_auc_{cat}"] = average_precision_score(labels, maxf_preds)
            out_json[str(seed)]["dense_block_fusion"][f"conn_pr_auc_{cat}"] = average_precision_score(labels, db_fusion_preds)
            out_json[str(seed)]["dense_block_only"][f"conn_pr_auc_{cat}"] = average_precision_score(labels, db_only_preds)
            
            # Assert no leakage
            assert not np.any(np.isnan(base_preds))
            assert len(base_preds) == x_all.shape[0]

        # Calculate means
        for mode in ["baseline", "tier1", "ring", "max_fusion", "dense_block_fusion", "dense_block_only"]:
            vals = list(out_json[str(seed)][mode].values())
            out_json[str(seed)][mode]["mean_conn_pr_auc"] = np.mean(vals)

    # Aggregate means and stds
    out_json["aggregate"] = {}
    for mode in ["baseline", "tier1", "ring", "max_fusion", "dense_block_fusion", "dense_block_only"]:
        out_json["aggregate"][mode] = {}
        for cat in INJECTION_FNS:
            cat_key = f"conn_pr_auc_{cat}"
            vals = [out_json[str(s)][mode][cat_key] for s in COMPARE_SEEDS]
            out_json["aggregate"][mode][cat_key] = {"mean": np.mean(vals), "std": np.std(vals)}
        mean_vals = [out_json[str(s)][mode]["mean_conn_pr_auc"] for s in COMPARE_SEEDS]
        out_json["aggregate"][mode]["mean_conn_pr_auc"] = {"mean": np.mean(mean_vals), "std": np.std(mean_vals)}
        
    # T9b: Held-out evaluation
    out_json["heldout"] = {}
    seed = COMPARE_SEEDS[0]
    for cat in INJECTION_FNS:
        rng_eval = np.random.default_rng(seed * 100 + 777 + cat_order.index(cat))
        x_all, eval_ei, eval_et, eval_iso, n_inject = _build_augmented_graph(
            x_real, base_edge_index_list, base_edge_type_tensor, feat_np, feat_cols, cat, rng_eval, mode_is_heldout=True
        )
        aug_app_ids = np.concatenate([real_app_ids, np.array([f"inj_{i}" for i in range(n_inject)])])
        with torch.no_grad():
            x_all_cpu = x_all.to(score_device)
            eval_ei_cpu = [ei.to(score_device) for ei in eval_ei]
            eval_et_cpu = eval_et.to(score_device)
            eval_iso_cpu = _compute_isolated_mask(eval_ei_cpu, x_all_cpu.shape[0], score_device)
            
            aug_score_df = compute_score_frame(model, x_all_cpu, eval_ei_cpu, eval_et_cpu, eval_iso_cpu, aug_app_ids, feat_cols)
            aug_attn_df = model.attention_summary(x_all_cpu, eval_ei_cpu, eval_et_cpu, eval_iso_cpu, aug_app_ids)
            
        real_scores_norm, inject_scores_norm = eval_m._score_inject_and_real(
            feat_np, x_all[n_real:].cpu().numpy(), feat_cols, ALL_EVAL_GROUPS[CATEGORY_PRIMARY_GROUP[cat]]
        )
        subspace_scores = np.concatenate([real_scores_norm, inject_scores_norm])
        aug_score_df["subspace_if_score"] = subspace_scores
        aug_dense_df = dense_block_scores(eval_ei_cpu, eval_et_cpu, x_all.shape[0], aug_app_ids)
        aug_deg_df = pd.DataFrame({"application_id": aug_app_ids})
        for col in DEGREE_FEATURES:
            if col in df.columns:
                deg_vals = np.zeros(x_all.shape[0], dtype=np.float32)
                deg_vals[:n_real] = df[col].values
                aug_deg_df[col] = deg_vals

        X_aug_dev = build_dev_features(x_all_cpu.cpu().numpy(), aug_dense_df, aug_deg_df)
        aug_dev_scores = score_devnet(dev_model, X_aug_dev)
        aug_dev_scores[:n_real] = real_dev_df["deviation_score"].values
        aug_dev_df = pd.DataFrame({"application_id": aug_app_ids, "deviation_score": aug_dev_scores})

        aug_merged = aug_score_df.merge(aug_attn_df, on="application_id").merge(
            aug_dense_df, on="application_id").merge(aug_dev_df, on="application_id")
        
        X_base_aug = aug_merged[baseline_cols].values
        X_tier1_aug = aug_merged[tier1_cols].values
        X_db_fusion_aug = aug_merged[db_fusion_cols].values
        X_db_only_aug = aug_merged[db_only_cols].values
        
        base_preds = clf_base.predict_proba(X_base_aug)[:, 1]
        tier1_preds = clf_tier1.predict_proba(X_tier1_aug)[:, 1]
        maxf_preds = np.maximum(base_preds, tier1_preds)
        db_fusion_preds = clf_db_fusion.predict_proba(X_db_fusion_aug)[:, 1]
        db_only_preds = clf_db_only.predict_proba(X_db_only_aug)[:, 1]
        
        aug_hybrid_scores = torch.tensor(aug_merged["hybrid_anomaly_score"].values, dtype=torch.float32, device=DEVICE)
        aug_candidates = generate_candidates(eval_ei_cpu, eval_et_cpu, x_all_cpu, aug_hybrid_scores)
        ring_preds = project_to_nodes(aug_candidates, ring_clf, x_all.shape[0])
        
        labels = np.zeros(x_all.shape[0])
        labels[n_real:] = 1.0
        
        out_json["heldout"][f"{cat}_baseline"] = average_precision_score(labels, base_preds)
        out_json["heldout"][f"{cat}_tier1"] = average_precision_score(labels, tier1_preds)
        out_json["heldout"][f"{cat}_ring"] = average_precision_score(labels, ring_preds)
        out_json["heldout"][f"{cat}_max_fusion"] = average_precision_score(labels, maxf_preds)
        out_json["heldout"][f"{cat}_dense_block_fusion"] = average_precision_score(labels, db_fusion_preds)
        out_json["heldout"][f"{cat}_dense_block_only"] = average_precision_score(labels, db_only_preds)
        
    with open(JSON_OUT, "w") as f:
        json.dump(out_json, f, indent=2, default=float)  # cast any stray numpy scalars
        
    print(f"\nSaved results to {JSON_OUT}")

if __name__ == "__main__":
    run_comparison()
