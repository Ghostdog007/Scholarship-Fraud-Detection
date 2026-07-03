"""
smoke_v4.py — fast pre-flight for the V4 ablation (run before any GPU run).

Runs, for each of the 3 ablation configs (env-selected):
  UNIT  : tiny-graph forward, isolated-node parity, score-frame schema,
          ARCH_VERSION rejection gate  (seconds, no training)
  TRAIN : train(smoke_test=True) on the real graph, 2+2 epochs
  EVAL  : evaluate_connected() (shrunk to a few clusters)
          + evaluate() edge-dropout for the HAN config (exercises the
            empty-relation path that B4 fixed)

The live checkpoint + scores are backed up and restored, so a smoke run never
clobbers a real trained model.

Usage:  .\.venv\Scripts\python.exe scripts/smoke_v4.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = [
    ("config1_rgcn_feature", {"V4_ENCODER_ARCH": "rgcn", "V4_TOPO_EXPOSURE": "0"}),
    ("config2_rgcn_topo",    {"V4_ENCODER_ARCH": "rgcn", "V4_TOPO_EXPOSURE": "1"}),
    ("config3_han_topo",     {"V4_ENCODER_ARCH": "han",  "V4_TOPO_EXPOSURE": "1"}),
]
# Shrink everything so the smoke is fast; the real runs use the config defaults.
SMOKE_ENV = {"V4_TOPO_CLUSTERS": "5", "V4_EVAL_CLUSTERS": "3"}

BACKUP = [
    "models/hybrid_graphmcm_v3.pth",
    "models/hybrid_graphmcm_v3.pth.bak",
    "outputs/hybrid_scores_v3.csv",
]


def run_one() -> None:
    import numpy as np
    import torch

    from src.config_v3 import (
        ENCODER_ARCH, TOPO_EXPOSURE_ENABLED,
        N_FEATURES, N_EDGE_TYPES, GRAPH_EMB_DIM,
    )
    from src.hybrid_graphmcm_v3 import (
        HybridGraphMCM, _compute_isolated_mask, compute_score_frame,
    )

    print(f"[smoke] arch={ENCODER_ARCH}  topo_exposure={TOPO_EXPOSURE_ENABLED}")

    # ---- UNIT: tiny graph ----
    torch.manual_seed(0)
    dev = torch.device("cpu")
    N = 20
    x = torch.rand(N, N_FEATURES)
    eil, ets = [], []
    for r in range(N_EDGE_TYPES):
        src = torch.randint(0, N, (6,)); dst = torch.randint(0, N, (6,))
        eil.append(torch.stack([src, dst])); ets.append(torch.full((6,), r))
    et = torch.cat(ets)
    iso = _compute_isolated_mask(eil, N, dev)

    m = HybridGraphMCM().to(dev)
    m.init_centroid(x, eil, et, iso)
    pred, edge, h_n, concat = m(x, eil, et, iso)
    assert h_n.shape == (N, GRAPH_EMB_DIM), f"h_N shape {h_n.shape}"

    iso2 = torch.zeros(N, dtype=torch.bool); iso2[0] = True
    h2 = m.encode_graph(x, eil, et, iso2)
    assert torch.allclose(h2[0], m.isolated_embedding), "isolated-node parity FAILED"

    ids = np.array([f"a{i}" for i in range(N)])
    feats = [f"f{i}" for i in range(N_FEATURES)]
    frame = compute_score_frame(m, x, eil, et, iso, ids, feats)
    expected = {"application_id", "hybrid_anomaly_score", "feature_pred_error",
                "edge_pred_error", "per_feature_error_json", "per_feature_predicted_json"}
    assert set(frame.columns) == expected, f"schema drift: {list(frame.columns)}"

    from src.checkpoint_manager import _validate
    bad = {"model_state_dict": {}, "centroid": torch.zeros(GRAPH_EMB_DIM),
           "config": {"N_FEATURES": N_FEATURES, "GRAPH_EMB_DIM": GRAPH_EMB_DIM,
                      "N_EDGE_TYPES": N_EDGE_TYPES, "ARCH_VERSION": "WRONG_v0"}}
    try:
        _validate(bad, Path("x")); raise AssertionError("ARCH_VERSION gate did NOT raise")
    except ValueError:
        pass
    print("[smoke] UNIT ok: h_N shape, isolated parity, score schema, ARCH_VERSION gate")

    # ---- INTEGRATION: build topo pack (if enabled) + smoke train + eval ----
    if TOPO_EXPOSURE_ENABLED:
        from src.synthetic_exposure_builder_v3 import build_topology_exposure
        build_topology_exposure()

    from src.hybrid_graphmcm_v3 import train
    train(smoke_test=True)

    from src.evaluate_model_v3 import evaluate_connected
    mets = evaluate_connected()
    print("[smoke] connected PR-AUC:", {k: round(v, 4) for k, v in mets.items()
                                        if k.startswith("conn_")})

    if ENCODER_ARCH == "han":
        # exercises the edge-dropout path (pruned edges -> possibly empty relation),
        # which is exactly what B4 fixed for HAN.
        from src.evaluate_model_v3 import evaluate
        evaluate()

    print("[smoke] PASS")


def main() -> int:
    if "--one" in sys.argv:
        run_one()
        return 0

    tmp = ROOT / ".smoke_backup"
    tmp.mkdir(exist_ok=True)
    saved = []
    for rel in BACKUP:
        p = ROOT / rel
        if p.exists():
            shutil.copy2(p, tmp / p.name)
            saved.append(rel)
    print(f"[smoke] backed up {len(saved)} live file(s) -> {tmp}")

    try:
        for name, env in CONFIGS:
            print("\n" + "=" * 72)
            print(f"SMOKE :: {name}")
            print("=" * 72)
            child_env = dict(os.environ)
            child_env.update(SMOKE_ENV)
            child_env.update(env)
            r = subprocess.run([sys.executable, __file__, "--one"],
                               cwd=str(ROOT), env=child_env)
            if r.returncode != 0:
                print(f"\n[smoke] {name} FAILED (rc={r.returncode}) — stopping.")
                return 1
        print("\n" + "=" * 72)
        print("ALL 3 SMOKE CONFIGS PASSED — safe to launch the full GPU runs.")
        print("=" * 72)
        return 0
    finally:
        for rel in saved:
            shutil.copy2(tmp / Path(rel).name, ROOT / rel)
        shutil.rmtree(tmp, ignore_errors=True)
        print("[smoke] restored live checkpoint + scores")


if __name__ == "__main__":
    raise SystemExit(main())
