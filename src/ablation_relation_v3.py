"""
ablation_relation_v3.py  —  Family-name edge ablation (PROPOSED / PENDING)

Question: do shares_father_name / shares_mother_name graph edges carry any
DETECTION signal for the Hybrid GraphMCM, or do they behave like the already-
rejected shares_pincode relation (dense, demographic, not collusion)?

Motivating evidence (pre-retrain, real 15k population, see conversation log):
  * structural — shares_father_name: 9,151 edges / 23.2% of nodes touched /
    max degree 35; shares_mother_name: 38,146 edges / 35.7% of nodes / max
    degree 109, mean degree 5.09 — same diffuse-hub shape as shares_pincode
    (104,081 edges / 82.8%), not the sparse shares_mobile (83 edges / 0.9%)
    or shares_ip (7,202 edges / 27.6%) shape that IS trusted (dense-block gate).
  * behavioural — compute_relation_ablation() per-node deltas (locked
    checkpoint) do NOT separate top-5%-risk nodes from the rest for father/
    mother name (ratio ~1.0x, mother_name reversed 0.76x) the way mobile
    (19x) and ip (2.2x) do.
  * stress_testing_1_stats.json (locked production checkpoint):
    MOTHER_NAME_COLLISION hybrid_only_pr_auc=0.045 vs subspace_only_pr_auc
    (no graph at all)=0.289 — the graph model does WORSE than the graph-free
    tabular IF at catching the fraud archetype this edge exists to catch.

Method (fully isolated — never touches the locked production pipeline, no
LOCKED-hyperparameter change made here):
  * "full" arm = canonical 5-relation graph, retrained from scratch;
  * "norelfm" arm = identical graph with shares_father_name (rel 2) and
    shares_mother_name (rel 3) edge_index emptied before training — N_EDGE_TYPES
    stays 5 (architecture unchanged, relations 2/3 just never populated,
    matching the precedent set by compute_relation_ablation's masking);
  * both arms: topology-exposure OFF (fair internal control, same convention
    as ablation_noid_v3.py), 44 features (current N_FEATURES), same seed(s);
  * scored on the connected-cluster harness (5 injected fraud archetypes,
    incl. MOTHER_NAME_COLLISION), reusing ablation_noid_v3._eval_connected
    verbatim (generic over H.GRAPH_PT / H.FINAL_CSV / H.MODEL_PTH).

Reproducibility caveat (CLAUDE.md): RGCNConv(aggr="add") CUDA scatter-add is
not seed-controlled -> +/-0.03-0.04 run-to-run noise floor (CPU-only in this
environment, so this run is deterministic modulo library nondeterminism).
Per the Quantitative Claims Protocol, results are PROPOSED / PENDING —
adopting them (dropping N_EDGE_TYPES or the two relations from graph_builder)
is a separate, explicit lead decision, not made here.

Usage:
  python -m src.ablation_relation_v3 --prepare
  python -m src.ablation_relation_v3 --run --arm full --seed 42
  python -m src.ablation_relation_v3 --run --arm norelfm --seed 42
  python -m src.ablation_relation_v3 --summarize
Set ABLATION_SMOKE=1 to run 2+2 epochs for a fast end-to-end validation.
"""

import os
import sys
import json
from pathlib import Path

ABL_DIR  = Path("data/processed/ablation")
CKPT_DIR = Path("models/ablation")
RES_DIR  = Path("outputs/ablation/relation")
CANON_GRAPH_PT = Path("data/processed/identity_graph_v3.pt")
DROPPED_RELATIONS = ["shares_father_name", "shares_mother_name"]
SEEDS = (42, 43, 44)


def prepare() -> None:
    import torch
    data = torch.load(CANON_GRAPH_PT, weights_only=False)
    n_before = {et: data["application", et, "application"].edge_index.shape[1] for et in
                [t[1] for t in data.edge_types]}
    for et in DROPPED_RELATIONS:
        data["application", et, "application"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    outp = ABL_DIR / "identity_graph_norelfm.pt"
    torch.save(data, outp)
    print(f"[ablation-rel] prepared {outp}: dropped {DROPPED_RELATIONS}")
    print(f"[ablation-rel] edge counts before drop: {n_before}")


def _eval_connected_fixed(H, features, seed):
    """Connected-cluster PR-AUC, like ablation_noid_v3._eval_connected but building
    a FIXED 5-slot edge_index_list (empty tensor for a dropped relation, not
    compacted) so RELATION_MAP's positional rel_idx still addresses the correct
    relation when shares_father_name/shares_mother_name are absent from the
    trained graph -- H._build_edge_index_and_types() drops empty relations from
    the list entirely, which silently misaligns positional indexing here."""
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
    edge_index_list = [data["application", et, "application"].edge_index.to(H.DEVICE)
                        for et in [t[1] for t in data.edge_types]]

    ckpt  = torch.load(H.MODEL_PTH, weights_only=False, map_location=H.DEVICE)
    model = H.HybridGraphMCM().to(H.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    inj_seed = seed + 99
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
        eev_nonempty = [ei for ei in eev if ei.shape[1] > 0]
        iso = H._compute_isolated_mask(eev_nonempty, x_all.shape[0], H.DEVICE)
        ids = np.concatenate([real_ids, np.array([f"inject_{i}" for i in range(x_inject.shape[0])])])

        with torch.no_grad():
            sdf = H.compute_score_frame(model, x_all, eev_nonempty, et_tensor, iso, ids, features)
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

    os.environ["V4_SEED"] = str(seed)
    os.environ["V4_TOPO_EXPOSURE"] = "0"

    import src.hybrid_graphmcm_v3 as H
    torch.manual_seed(seed); np.random.seed(seed)

    features = json.loads(H.SCHEMA_JSON.read_text())["features"]
    H.TOPO_EXPOSURE_ENABLED = False
    if arm == "norelfm":
        H.GRAPH_PT = ABL_DIR / "identity_graph_norelfm.pt"

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    H.MODEL_PTH = CKPT_DIR / f"{arm}_seed{seed}.pth"
    H.OUT_CSV   = ABL_DIR / f"scores_{arm}_seed{seed}.csv"

    print(f"[ablation-rel] === arm={arm} seed={seed} graph={H.GRAPH_PT} "
          f"smoke={smoke} device={H.DEVICE} ===")
    H.train(smoke_test=smoke)

    print(f"[ablation-rel] connected-cluster eval (arm={arm} seed={seed}) ...")
    connected = _eval_connected_fixed(H, features, seed)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = {"arm": arm, "seed": seed, "dropped_relations": DROPPED_RELATIONS if arm == "norelfm" else [],
           "topo_exposure": False, "smoke": smoke, "connected": connected}
    (RES_DIR / f"{arm}_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(f"[ablation-rel] arm={arm} seed={seed} mean_connected_pr_auc={connected['mean']:.4f}")


def summarize() -> None:
    import numpy as np
    files = sorted(RES_DIR.glob("*_seed*.json"))
    if not files:
        print("[ablation-rel] no per-arm results found — run the arms first."); return
    runs = [json.loads(f.read_text()) for f in files]
    cats = [c for c in runs[0]["connected"] if c != "mean"]

    def arm_matrix(arm):
        rs = [r for r in runs if r["arm"] == arm]
        return rs, {c: np.mean([r["connected"][c] for r in rs]) for c in cats + ["mean"]}

    full_rs, full_m = arm_matrix("full")
    norelfm_rs, norelfm_m = arm_matrix("norelfm")

    lines = []
    header = f"{'category':<26}{'full(5-rel)':>14}{'norelfm(3-rel)':>16}{'delta':>12}"
    lines.append(header); lines.append("-" * len(header))
    for c in cats + ["mean"]:
        d = norelfm_m[c] - full_m[c]
        lines.append(f"{c:<26}{full_m[c]:>14.4f}{norelfm_m[c]:>16.4f}{d:>+12.4f}")
    table = "\n".join(lines)
    print("\n" + table)

    summary = {
        "_meta": {
            "question": "do shares_father_name/shares_mother_name edges change connected-cluster PR-AUC?",
            "status": "PROPOSED / PENDING — architecture (5 edge types) stays LOCKED until lead decision",
            "topo_exposure": "OFF for both arms (fair internal control; not the production topo-ON baseline)",
            "noise_floor": "±0.03–0.04 (RGCN CUDA scatter-add); deltas within this are null (CPU-only here)",
            "dropped_relations": DROPPED_RELATIONS,
            "seeds_full": sorted(r["seed"] for r in full_rs),
            "seeds_norelfm": sorted(r["seed"] for r in norelfm_rs),
        },
        "full_mean": full_m, "norelfm_mean": norelfm_m,
        "delta": {c: norelfm_m[c] - full_m[c] for c in cats + ["mean"]},
        "runs": runs,
    }
    outp = Path("outputs/ablation/relation_ablation_result.json")
    outp.write_text(json.dumps(summary, indent=2))
    print(f"\n[ablation-rel] wrote {outp}")
    print(f"[ablation-rel] mean delta (norelfm - full) = {summary['delta']['mean']:+.4f}  "
          f"(noise floor ±0.03–0.04; PROPOSED/PENDING)")
    if "MOTHER_NAME_COLLISION" in summary["delta"]:
        print(f"[ablation-rel] MOTHER_NAME_COLLISION delta = "
              f"{summary['delta']['MOTHER_NAME_COLLISION']:+.4f}")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    elif "--summarize" in sys.argv:
        summarize()
    elif "--run" in sys.argv:
        arm  = sys.argv[sys.argv.index("--arm") + 1]
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        assert arm in ("full", "norelfm")
        run_arm(arm, seed)
    else:
        print(__doc__)
