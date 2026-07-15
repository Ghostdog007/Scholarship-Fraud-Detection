"""
ablation_noid_v3.py  —  Noisy-feature ablation (Wave 3, PROPOSED / PENDING)

Question: do the nominal identifier/code features (mobile_no, aadhaar token,
course/district/institution IDs, verifier codes, low-cardinality nominal
categoricals — the set the XAI narration policy already refuses to speak) carry
any DETECTION signal for the Hybrid GraphMCM, or are they noise the reconstruction
can drop with no cost?

Method (fully isolated — never touches the locked production pipeline):
  * a reduced feature matrix is built by DROPPING xai_layer_v3.IDENTIFIER_FEATURES;
  * two arms are retrained from scratch — "full" (all 68) and "noid" (reduced) —
    at seeds 42/43/44, EACH IN ITS OWN PROCESS (clean N_FEATURES), to separate
    checkpoints under models/ablation/;
  * both arms run with topology-exposure OFF (the canonical topo tensor is 68-dim
    and its path is hard-coded in train(); disabling it for BOTH arms keeps the
    comparison a fair internal control — this is a RELATIVE effect measurement,
    so absolute PR-AUC differs from the production topo-ON baseline);
  * each arm is scored on the connected-cluster harness (the AGENTS.md Appendix H
    relational test), mean PR-AUC over the 5 categories.

Reproducibility caveat (from CLAUDE.md): RGCNConv(aggr="add") CUDA scatter-add is
not seed-controlled → ±0.03–0.04 run-to-run noise floor. Interpret deltas smaller
than that as null. Per the Quantitative Claims Protocol, results are logged as
PROPOSED / PENDING — adoption (which would change the LOCKED N_FEATURES) is a
separate, explicit decision, not made here.

Usage (orchestrated by a shell loop; each --run is its own process):
  python -m src.ablation_noid_v3 --prepare
  python -m src.ablation_noid_v3 --run --arm full --seed 42
  python -m src.ablation_noid_v3 --run --arm noid --seed 42
  python -m src.ablation_noid_v3 --summarize
Set ABLATION_SMOKE=1 to run 2+2 epochs for a fast end-to-end validation.
"""

import os
import sys
import json
from pathlib import Path

ABL_DIR   = Path("data/processed/ablation")
CKPT_DIR  = Path("models/ablation")
RES_DIR   = Path("outputs/ablation/noid")
CANON_CSV      = Path("data/processed/engineered_features_v3.csv")
CANON_SCHEMA   = Path("data/processed/v3_feature_schema.json")
CANON_EXPOSURE = Path("data/processed/synthetic_exposure_set_v3.pt")
SEEDS = (42, 43, 44)


# --------------------------------------------------------------------------
# Step 0 — build the reduced (identifier-dropped) artefacts, once
# --------------------------------------------------------------------------
def prepare() -> None:
    import pandas as pd
    import torch
    from src.xai_layer_v3 import IDENTIFIER_FEATURES

    schema = json.loads(CANON_SCHEMA.read_text())
    feats  = schema["features"]
    keep   = [f for f in feats if f not in IDENTIFIER_FEATURES]
    drop   = [f for f in feats if f in IDENTIFIER_FEATURES]
    keep_idx = [i for i, f in enumerate(feats) if f in set(keep)]

    ABL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CANON_CSV)
    df[["application_id"] + keep].to_csv(ABL_DIR / "engineered_features_noid.csv", index=False)

    rs = dict(schema)
    rs["features"] = keep
    rs["n_features"] = len(keep)
    (ABL_DIR / "schema_noid.json").write_text(json.dumps(rs, indent=2))

    x = torch.load(CANON_EXPOSURE, weights_only=True)
    torch.save(x[:, keep_idx].contiguous(), ABL_DIR / "exposure_noid.pt")

    (ABL_DIR / "drop_manifest.json").write_text(json.dumps(
        {"n_full": len(feats), "n_noid": len(keep), "dropped": drop}, indent=2))
    print(f"[ablation] prepared reduced artefacts: {len(feats)} -> {len(keep)} features "
          f"(dropped {len(drop)}): {', '.join(drop)}")


# --------------------------------------------------------------------------
# Step 1 — retrain + connected-cluster eval for ONE arm (own process)
# --------------------------------------------------------------------------
def _eval_connected(H, features, seed):
    """Connected-cluster PR-AUC, mirroring evaluate_model_v3.evaluate_connected
    but parametrised on the arm's feature list + freshly-trained checkpoint."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score
    import src.evaluate_model_v3 as EV
    from src.config_v3 import EVAL_CONNECTED_N_CLUSTERS, EVAL_CONNECTED_SIZE_RANGE

    RELATION_MAP = {"IP_CONCENTRATION": 1, "MOTHER_NAME_COLLISION": 3,
                    "FEE_INFLATION": 4, "AGE_VIOLATION": 4, "INCOME_VIOLATION": 4}

    df       = pd.read_csv(H.FINAL_CSV)
    x_real   = torch.tensor(df[features].values, dtype=torch.float32).to(H.DEVICE)
    feat_np  = df[features].values.astype(np.float32)
    real_ids = df["application_id"].values
    n_real   = x_real.shape[0]

    data = torch.load(H.GRAPH_PT, weights_only=False)
    edge_index_list, _ = H._build_edge_index_and_types(data, H.DEVICE)

    ckpt  = torch.load(H.MODEL_PTH, weights_only=False, map_location=H.DEVICE)
    model = H.HybridGraphMCM().to(H.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    inj_seed = seed + 99          # depends only on seed → identical clusters both arms
    results = {}
    old_n = EV.N_INJECT
    for cat_idx, (category, inject_fn) in enumerate(EV.INJECTION_FNS.items()):
        rng = np.random.default_rng(inj_seed + cat_idx)
        all_x, cluster_edges, node_offset = [], [], n_real
        for _ in range(EVAL_CONNECTED_N_CLUSTERS):
            c_size = int(rng.integers(EVAL_CONNECTED_SIZE_RANGE[0], EVAL_CONNECTED_SIZE_RANGE[1] + 1))
            EV.N_INJECT = c_size
            all_x.append(inject_fn(feat_np, features, rng))
            nodes = np.arange(node_offset, node_offset + c_size)
            ii, jj = np.meshgrid(nodes, nodes)
            m = ii != jj
            cluster_edges.append(np.vstack([ii[m], jj[m]]))
            node_offset += c_size

        x_inject = torch.tensor(np.vstack(all_x), dtype=torch.float32).to(H.DEVICE)
        x_all = torch.cat([x_real, x_inject], dim=0)
        rel_idx = RELATION_MAP[category]

        eev = [ei.clone() for ei in edge_index_list]
        new_edges = torch.tensor(np.hstack(cluster_edges), dtype=torch.long, device=H.DEVICE)
        if eev[rel_idx].shape[1] > 0:
            eev[rel_idx] = torch.cat([eev[rel_idx], new_edges], dim=1)
        else:
            eev[rel_idx] = new_edges

        et = []
        for r_id, ei in enumerate(eev):
            if ei.shape[1] > 0:
                et.append(torch.full((ei.shape[1],), r_id, dtype=torch.long, device=H.DEVICE))
        et_tensor = torch.cat(et) if et else torch.zeros(0, dtype=torch.long, device=H.DEVICE)
        iso = H._compute_isolated_mask(eev, x_all.shape[0], H.DEVICE)
        ids = np.concatenate([real_ids, np.array([f"inject_{i}" for i in range(x_inject.shape[0])])])

        with torch.no_grad():
            sdf = H.compute_score_frame(model, x_all, eev, et_tensor, iso, ids, features)
        labels = np.zeros(x_all.shape[0]); labels[n_real:] = 1.0
        results[category] = float(average_precision_score(labels, sdf["hybrid_anomaly_score"].values))
        print(f"    {category:<24} PR-AUC={results[category]:.4f}")
    EV.N_INJECT = old_n
    results["mean"] = float(np.mean([v for k, v in results.items()]))
    return results


def run_arm(arm: str, seed: int) -> None:
    import numpy as np
    import torch
    smoke = os.environ.get("ABLATION_SMOKE") == "1"

    # env MUST be set before config/H import so N_FEATURES resolves per arm
    if arm == "noid":
        features = json.loads((ABL_DIR / "schema_noid.json").read_text())["features"]
    else:
        features = json.loads(CANON_SCHEMA.read_text())["features"]
    os.environ["V4_N_FEATURES"] = str(len(features))
    os.environ["V4_SEED"]       = str(seed)
    os.environ["V4_TOPO_EXPOSURE"] = "0"

    import src.hybrid_graphmcm_v3 as H
    torch.manual_seed(seed); np.random.seed(seed)

    if arm == "noid":
        H.FINAL_CSV   = ABL_DIR / "engineered_features_noid.csv"
        H.SCHEMA_JSON = ABL_DIR / "schema_noid.json"
        H.EXPOSURE_PT = ABL_DIR / "exposure_noid.pt"
    H.TOPO_EXPOSURE_ENABLED = False          # fair internal control (both arms)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    H.MODEL_PTH = CKPT_DIR / f"{arm}_seed{seed}.pth"
    H.OUT_CSV   = ABL_DIR / f"scores_{arm}_seed{seed}.csv"

    print(f"[ablation] === arm={arm} seed={seed} n_features={len(features)} "
          f"smoke={smoke} device={H.DEVICE} ===")
    H.train(smoke_test=smoke)

    print(f"[ablation] connected-cluster eval (arm={arm} seed={seed}) ...")
    connected = _eval_connected(H, features, seed)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = {"arm": arm, "seed": seed, "n_features": len(features),
           "topo_exposure": False, "smoke": smoke, "connected": connected}
    (RES_DIR / f"{arm}_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(f"[ablation] arm={arm} seed={seed} mean_connected_pr_auc={connected['mean']:.4f}")


# --------------------------------------------------------------------------
# Step 2 — aggregate across seeds and both arms
# --------------------------------------------------------------------------
def summarize() -> None:
    import numpy as np
    files = sorted(RES_DIR.glob("*_seed*.json"))
    if not files:
        print("[ablation] no per-arm results found — run the arms first."); return
    runs = [json.loads(f.read_text()) for f in files]
    cats = [c for c in runs[0]["connected"] if c != "mean"]

    def arm_matrix(arm):
        rs = [r for r in runs if r["arm"] == arm]
        return rs, {c: np.mean([r["connected"][c] for r in rs]) for c in cats + ["mean"]}

    full_rs, full_m = arm_matrix("full")
    noid_rs, noid_m = arm_matrix("noid")

    lines = []
    header = f"{'category':<26}{'full(68)':>12}{'noid':>12}{'delta':>12}"
    lines.append(header); lines.append("-" * len(header))
    for c in cats + ["mean"]:
        d = noid_m[c] - full_m[c]
        lines.append(f"{c:<26}{full_m[c]:>12.4f}{noid_m[c]:>12.4f}{d:>+12.4f}")
    table = "\n".join(lines)
    print("\n" + table)

    summary = {
        "_meta": {
            "question": "does dropping nominal identifier features change connected-cluster PR-AUC?",
            "status": "PROPOSED / PENDING — not adopted; N_FEATURES stays LOCKED at 68",
            "topo_exposure": "OFF for both arms (fair internal control; not the production topo-ON baseline)",
            "noise_floor": "±0.03–0.04 (RGCN CUDA scatter-add); deltas within this are null",
            "seeds_full": sorted(r["seed"] for r in full_rs),
            "seeds_noid": sorted(r["seed"] for r in noid_rs),
            "n_features_noid": noid_rs[0]["n_features"] if noid_rs else None,
            "dropped": json.loads((ABL_DIR / "drop_manifest.json").read_text())["dropped"]
                        if (ABL_DIR / "drop_manifest.json").exists() else [],
        },
        "full_mean": full_m, "noid_mean": noid_m,
        "delta": {c: noid_m[c] - full_m[c] for c in cats + ["mean"]},
        "runs": runs,
    }
    outp = Path("outputs/ablation/noid_ablation.json")
    outp.write_text(json.dumps(summary, indent=2))
    print(f"\n[ablation] wrote {outp}")
    print(f"[ablation] mean delta (noid - full) = {summary['delta']['mean']:+.4f}  "
          f"(noise floor ±0.03–0.04; PROPOSED/PENDING)")


# --------------------------------------------------------------------------
# Wave 3b — topology-ON CONFIRMATION, scored through the LOCKED fusion
# --------------------------------------------------------------------------
CANON_TOPO = Path("data/processed/synthetic_exposure_graph_v3.pt")


def prepare_topo() -> None:
    """Reduced topology-exposure pack (identifier columns dropped from node x)."""
    import torch
    from src.xai_layer_v3 import IDENTIFIER_FEATURES
    feats = json.loads(CANON_SCHEMA.read_text())["features"]
    keep_idx = [i for i, f in enumerate(feats) if f not in IDENTIFIER_FEATURES]
    pack = torch.load(CANON_TOPO, weights_only=False)
    pack["x"] = pack["x"][:, keep_idx].contiguous()
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(pack, ABL_DIR / "exposure_graph_noid.pt")
    print(f"[ablation] reduced topo pack: x -> {tuple(pack['x'].shape)}  (topology-ON confirmation)")


def retrain_noid_topoon(seed: int) -> None:
    """Retrain the noid detector WITH topology exposure ON (production setting)."""
    import numpy as np, torch
    smoke = os.environ.get("ABLATION_SMOKE") == "1"
    features = json.loads((ABL_DIR / "schema_noid.json").read_text())["features"]
    os.environ["V4_N_FEATURES"] = str(len(features))
    os.environ["V4_SEED"] = str(seed)
    os.environ["V4_TOPO_EXPOSURE"] = "1"

    import src.hybrid_graphmcm_v3 as H
    torch.manual_seed(seed); np.random.seed(seed)
    H.FINAL_CSV   = ABL_DIR / "engineered_features_noid.csv"
    H.SCHEMA_JSON = ABL_DIR / "schema_noid.json"
    H.EXPOSURE_PT = ABL_DIR / "exposure_noid.pt"
    H.TOPO_PT     = ABL_DIR / "exposure_graph_noid.pt"
    H.TOPO_EXPOSURE_ENABLED = True
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    H.MODEL_PTH = CKPT_DIR / f"noid_topoon_seed{seed}.pth"
    H.OUT_CSV   = ABL_DIR / f"scores_noid_topoon_seed{seed}.csv"
    print(f"[ablation] === retrain noid TOPO-ON seed={seed} n_features={len(features)} smoke={smoke} ===")
    H.train(smoke_test=smoke)


def fused_eval(arm: str, seed: int) -> None:
    """Score one arm through the LOCKED score-level fusion on the connected-cluster
    harness, reusing compare_architectures_v3._score_category verbatim. full arm =
    the frozen backbone models/hybrid_v3_seed{seed}.pth; noid arm = the topo-ON
    reduced checkpoint just trained."""
    import numpy as np, pandas as pd, torch
    from sklearn.metrics import average_precision_score

    if arm == "noid":
        features = json.loads((ABL_DIR / "schema_noid.json").read_text())["features"]
        csv, ckpt = ABL_DIR / "engineered_features_noid.csv", CKPT_DIR / f"noid_topoon_seed{seed}.pth"
    else:
        features = json.loads(CANON_SCHEMA.read_text())["features"]
        csv, ckpt = CANON_CSV, Path(f"models/hybrid_v3_seed{seed}.pth")
    os.environ["V4_N_FEATURES"] = str(len(features))

    import src.compare_architectures_v3 as C
    import src.hybrid_graphmcm_v3 as H
    from src.evaluate_model_v3 import INJECTION_FNS

    df = pd.read_csv(csv)
    x_real  = torch.tensor(df[features].values, dtype=torch.float32).to(H.DEVICE)
    feat_np = df[features].values.astype(np.float32)
    real_ids = df["application_id"].values
    data = torch.load(H.GRAPH_PT, weights_only=False)
    base_ei, base_et = H._build_edge_index_and_types(data, H.DEVICE)

    model = H.HybridGraphMCM().to(H.DEVICE)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location=H.DEVICE)["model_state_dict"])
    model.eval()

    cat_order = list(INJECTION_FNS)
    res = {m: {} for m in C.MODES}
    for cat in cat_order:
        rng = np.random.default_rng(seed * 100 + cat_order.index(cat))   # matches locked harness
        preds, labels = C._score_category(model, x_real, base_ei, base_et,
                                          feat_np, features, real_ids, cat, rng, mode_is_heldout=False)
        for m in C.MODES:
            res[m][cat] = float(average_precision_score(labels, preds[m]))
        print(f"    {cat:<24} locked={res['locked_fusion'][cat]:.4f} hyb={res['hybrid_only'][cat]:.4f}")
    for m in C.MODES:
        res[m]["mean"] = float(np.mean([res[m][c] for c in cat_order]))
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"fused_{arm}_seed{seed}.json").write_text(
        json.dumps({"arm": arm, "seed": seed, "topo_exposure": True,
                    "n_features": len(features), "modes": res}, indent=2))
    print(f"[ablation] fused {arm} seed={seed} locked_mean={res['locked_fusion']['mean']:.4f}")


def summarize_fused() -> None:
    import numpy as np
    files = sorted(RES_DIR.glob("fused_*_seed*.json"))
    if not files:
        print("[ablation] no fused results yet."); return
    runs = [json.loads(f.read_text()) for f in files]
    cats = [c for c in runs[0]["modes"]["locked_fusion"] if c != "mean"]

    def arm_mean(arm, mode):
        rs = [r for r in runs if r["arm"] == arm]
        return {c: float(np.mean([r["modes"][mode][c] for r in rs])) for c in cats + ["mean"]}, sorted(r["seed"] for r in rs)

    lf_full, seeds_full = arm_mean("full", "locked_fusion")
    lf_noid, seeds_noid = arm_mean("noid", "locked_fusion")
    hy_full, _ = arm_mean("full", "hybrid_only")
    hy_noid, _ = arm_mean("noid", "hybrid_only")

    print(f"\nLOCKED FUSION (topo-ON) — mean over seeds full={seeds_full} noid={seeds_noid}")
    hdr = f"{'category':<24}{'full':>10}{'noid':>10}{'delta':>10}  |{'hyb full':>10}{'hyb noid':>10}"
    print(hdr); print('-'*len(hdr))
    for c in cats + ["mean"]:
        print(f"{c:<24}{lf_full[c]:>10.4f}{lf_noid[c]:>10.4f}{lf_noid[c]-lf_full[c]:>+10.4f}  |"
              f"{hy_full[c]:>10.4f}{hy_noid[c]:>10.4f}")
    out = {"_meta": {"status": "PROPOSED / PENDING — N_FEATURES stays LOCKED at 68",
                     "harness": "LOCKED score-level fusion, topology-exposure ON (production setting)",
                     "full_arm": "frozen backbone models/hybrid_v3_seed{seed}.pth",
                     "noid_arm": "retrained topo-ON reduced (44-feat) models/ablation/noid_topoon_seed{seed}.pth",
                     "noise_floor": "±0.03–0.04 (RGCN CUDA scatter-add)"},
           "locked_fusion": {"full": lf_full, "noid": lf_noid,
                             "delta": {c: lf_noid[c]-lf_full[c] for c in cats+["mean"]}},
           "hybrid_only": {"full": hy_full, "noid": hy_noid},
           "runs": runs}
    Path("outputs/ablation/noid_fused_confirmation.json").write_text(json.dumps(out, indent=2))
    print(f"\n[ablation] wrote outputs/ablation/noid_fused_confirmation.json  "
          f"(fused mean delta {lf_noid['mean']-lf_full['mean']:+.4f}; PROPOSED/PENDING)")


if __name__ == "__main__":
    if "--prepare-topo" in sys.argv:
        prepare_topo()
    elif "--retrain-noid-topoon" in sys.argv:
        retrain_noid_topoon(int(sys.argv[sys.argv.index("--seed") + 1]))
    elif "--fused-eval" in sys.argv:
        fused_eval(sys.argv[sys.argv.index("--arm") + 1], int(sys.argv[sys.argv.index("--seed") + 1]))
    elif "--summarize-fused" in sys.argv:
        summarize_fused()
    elif "--prepare" in sys.argv:
        prepare()
    elif "--summarize" in sys.argv:
        summarize()
    elif "--run" in sys.argv:
        arm  = sys.argv[sys.argv.index("--arm") + 1]
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        assert arm in ("full", "noid")
        run_arm(arm, seed)
    else:
        print(__doc__)
