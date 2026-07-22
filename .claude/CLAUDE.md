# CLAUDE.md — NIC Scholarship Fraud Detection (V4-Scale phase)

## What This Project Is

ML-based, **rule-free** fraud detection for the NIC scholarship portal. The
detection architecture is **fixed and validated** (see `docs/HISTORY.md` for
how we got here — v1 rules → v2 rule-free stack → v3 Hybrid GraphMCM → V4
locked fusion). The active branch is **`V4-Scale`**: migrating I/O to
**PostgreSQL as the system of record** and detector training to mini-batch
sampling so the system runs at **30–40 lakh (3–4 million) applications** on
the production server (16 vCPU, 64 GB RAM, Ubuntu 22.04, no GPU, k3s).

You are almost always working the migration plan in `docs/IMPLEMENTATION.md`.
You are **not** redesigning detection.

> **Naming:** "V4" / "V4-Scale" are capability/branch labels. Source files
> stay `_v3`. Do NOT rename `_v3`→`_v4` or `/v3`→`/v4`.

---

## Read These Files First — Every Session

1. **`docs/AGENTS.md`** — the working contract: fixed architecture, module
   ownership, all 14 hard stops, open decisions. Lead-owned (hard stop 8) —
   never edit autonomously; flag and propose a redline.
2. **`docs/TECHNICAL_REFERENCE_AND_SCALING.md`** — Part I: how every
   component works; Part II: the target Postgres/k8s architecture. The
   deepest reference; cite it rather than re-deriving.
3. **`docs/IMPLEMENTATION.md`** — the 5-step migration plan with 15k
   acceptance gates. Find the current step before writing code.
4. **`docs/HISTORY.md`** — read-only metric record. Cite it (or the ablation
   JSONs in `outputs/ablation/`) for any past number; never extend it.

Pre-V4-Scale docs (incl. the 2,377-line AGENTS.md with Appendix H result
tables) live in git history: `git show 9b772de:docs/AGENTS.md` on `main`.

If a referenced file is not in the working directory, stop and say so before
writing any code.

---

## Documentation Freshness — Standing Instruction

**After completing any instruction that changes behaviour, interfaces, files,
or how the system is run/deployed, update the relevant docs in the same turn.**

- Code/endpoint/CLI/config change → `README.md`; API change →
  `docs/API_TESTING_GUIDE.md`.
- Serving/deploy/container change → `deploy/README.md`; operator-flow change
  → `docs/OPERATIONS_RUNBOOK.md`.
- Migration-step progress (gate passed / step started) → status markers in
  `docs/IMPLEMENTATION.md`.
- `docs/AGENTS.md` is lead-owned — flag staleness, propose a redline, never
  self-serve edits. `docs/HISTORY.md` is a closed record — never extend.

Treat a task as unfinished until its docs match reality. State in your
summary which docs you touched (or that none needed changes).

---

## Project Directory Layout

```
NIC fraud Detection Project/
├── main_v3.py                          # Pipeline orchestrator (entry point)
├── README.md / requirements.txt / requirements-dev.txt
├── docker-compose.yml                  # redis + nic-api + nic-worker + nginx (+ postgres, step 0)
│
├── src/                                # Python source (_v3 names)
│   ├── config_v3.py                    # ALL hyperparameters — single source of truth
│   ├── tabular_feature_engine_v3.py    # 44-feature engineering (SQL-pushdown in step 4)
│   ├── graph_builder_v3.py             # 5-relation identity graph (hub-capped in step 4)
│   ├── synthetic_exposure_builder_v3.py# programmatic LOE exposure
│   ├── hybrid_graphmcm_v3.py           # Hybrid GraphMCM detector (NeighborLoader in step 5)
│   ├── subspace_if_v3.py               # per-group Isolation Forest (tabular backbone)
│   ├── dense_block_detector_v3.py      # FRAUDAR-style peeling, mobile+ip+pincode (IP-weighted max)
│   ├── deepsad_detector_v3.py          # Deep SAD center-distance, XAI-only (not in fusion)
│   ├── evt_scorer_v3.py                # EVT/GPD thresholds
│   ├── self_training_loop_v3.py        # human-gated pseudo-labels
│   ├── fusion_classifier_v3.py         # LOCKED score-level fusion
│   ├── xai_layer_v3.py / xai_card_html_v3.py  # evidence cards (JSON / HTML)
│   ├── evaluate_model_v3.py            # synthetic harness (v2 floors = pass bar)
│   ├── checkpoint_manager.py           # atomic checkpoint hot-swap
│   ├── retraining_orchestrator.py      # drift check + retrain paths
│   ├── confirmed_fraud_store.py / confirmed_fraud_graph_store.py / model_registry.py
│   │                                   # JSON stores (dual-write → Postgres, steps 1–3)
│   ├── db/                             # (step 0) ALL SQL lives here — nowhere else
│   └── api/                            # FastAPI app, handlers, Celery tasks
│
├── frontend/                           # vanilla-JS console (UNCHANGED this phase)
├── deploy/                             # nginx, k8s manifests, (step 0) postgres/schema.sql
├── data/                               # raw/ 15k dataset + processed/ artifacts
├── models/                             # checkpoints (checkpoint_manager-managed)
├── outputs/                            # scores, cards, ablation JSON (incl.
│                                       #   ablation/locked_fusion_validation.json)
└── docs/
    ├── AGENTS.md                       # working contract (lead-owned)
    ├── TECHNICAL_REFERENCE_AND_SCALING.md
    ├── IMPLEMENTATION.md               # 5-step migration plan + gates
    ├── HISTORY.md                      # read-only metric record
    ├── OPERATIONS_RUNBOOK.md           # console/operator guide
    └── API_TESTING_GUIDE.md
```

Source in `src/`, data in `data/`, checkpoints in `models/`, outputs in
`outputs/`. Never write outputs to the project root.

---

## Architecture in Brief (fixed — do not relitigate)

```
applications (Postgres system of record; console CSV intake preserved)
  → 44 engineered features (MinMax, persisted params)
  → 5-relation identity graph (shares_mobile/ip/father_name/mother_name/pincode)
  → three detectors (ALL higher = more anomalous):
      hybrid_anomaly_score = feature_pred_error + 0.3·edge_pred_error   (RGCN GraphMCM)
      subspace_if_score                                                  (backbone)
      dense_block_ip                                                     (IP specialist)
  → EVT thresholds (the only thresholds allowed)
  → LOCKED fusion: final_risk = minmax(1.0·subspace + 0.5·dense_ip + 0.3·hybrid)
  → human-gated self-training | XAI evidence cards
```

**LightGBM is NOT the fusion layer** — it was removed (14 positives destroyed
calibrated components; `docs/HISTORY.md`). Any record saying otherwise is
stale.

## Key Hyperparameters (locked; source of truth `src/config_v3.py`)

| Parameter | Value |
|---|---|
| Model features | 44 (`N_FEATURES`; 24 nominal identifiers dropped, noid ablation 2026-07-15) |
| Graph edge types / masks | 5 / 8 |
| Graph hidden / emb dim | 128 / 64 · MLP hidden / Z dim 256 / 64 |
| LOE margin / λ_edge / λ_exposure | data-derived (`_derive_loe_margin`, was fixed 2.0 — found ~3x too small for this embedding scale, changed 2026-07-22) / 0.3 / 1.0 · Stage 2 persistent LOE weight 0.15 (`LOE_STAGE2_WEIGHT`) |
| Stage 1 / Stage 2 epochs | 80 / 120 · LR 1e-3 · batch 256 · seed 42 (`V4_SEED`) |
| Encoder | `rgcn`, `root_weight=False` since 2026-07-22 (HAN available; drop-in regresses −0.091, 3-seed) |
| Incremental fine-tune | 10 epochs @ 1e-4, RGCN frozen |
| Fusion (LOCKED) | max, not weighted-sum, since 2026-07-22: `minmax(max(minmax(subspace), minmax(dense_relational), minmax(hybrid)))` — no per-component weight (`FUSION_W_*` retired); Deep SAD `center_dist_score` is NOT a fusion input (XAI-only, see below) |
| Dense-block gate | `shares_mobile`+`shares_ip`+`shares_pincode` since 2026-07-22 (`DENSE_BLOCK_RELATIONS=[0,1,4]`), IP-priority-weighted max (`DENSE_BLOCK_RELATION_WEIGHTS={0:0.3,1:1.0,4:0.2}`) — was `shares_ip` only |
| RGCN root weight | `root_weight=False` since 2026-07-22 (`hybrid_graphmcm_v3.RGCNEncoder`) — default `True` let each node's own unmasked features leak into `h_n` via the self-transform, independent of MCM masking; disabling it made `h_n` pure neighbor aggregation. Validated on stress_testing_1 (0.153→0.201 overall, 0.029→0.078 mobile-ring) and on the real 15k set (5/5 V2 floors still pass, edge-dropout retention 2.34) |
| Deep SAD (XAI-only) | Separate encoder/checkpoint (`deepsad_detector_v3.py`, `models/deepsad_v3.pth`), center-pull/exposure-push objective, no reconstruction loss. `center_dist_score` surfaced on XAI cards as a supplementary signal (>75th pct) — deliberately NOT in `FUSION_COMPONENTS`. Validated on stress_testing_1: 0.201 overall / 0.093 mobile-ring / 0.050 IP-ring, strongest single relational signal found this session. Fusion inclusion TESTED AND REJECTED 2026-07-22: candidate 4-way max fusion scored 0.4181 vs locked 3-way's 0.4182 (noise-level; Deep SAD won the argmax in <1% of nodes — the existing trio already covers its specialties too well for a 4th input to matter) (`DEEPSAD_*` in config_v3.py) |
| Drift alert | KS p < 0.01 (`DRIFT_KS_THRESHOLD`) |
| Confirmed-fraud weight | 3.0 · promotion needs ≥2 EVT signals |
| EVT GPD shape valid range | [-0.5, 1.0] · centroid clean percentile 95 |

> Reproducibility: `RGCNConv(aggr="add")` CUDA scatter-add gives a
> ±0.03–0.04 noise floor on detector scores. Use deterministic algorithms or
> CPU scoring before comparing smaller effects.

## How to Run

```bash
python main_v3.py                                  # full pipeline (project root)
.\.venv\Scripts\python.exe -m src.<module_name>    # individual module
docker compose up --build                          # serving stack (console at :8080)
```

Modules assume the working directory is the project root and run as modules
(`-m src.<name>`).

---

## Hard Stops — Never Proceed Past These

(Mirrors `docs/AGENTS.md` §4; AGENTS.md wording governs on any divergence.)

1. **No rules.** No numeric threshold against a domain concept, no rule
   codes, no policy-boundary features. Only EVT-derived or learned
   thresholds. The scale-phase hub-cap / group-size ceiling must be derived
   from the observed group-size distribution, never hand-picked.
2. **No raw GNN embeddings leave `hybrid_graphmcm_v3.py`** — scalar scores
   and attention weights only. No embedding columns in Postgres, ever.
3. **Higher = more anomalous.** Document any inversion at the point of
   inversion.
4. **`sanity` column is never used** (feature, label, or evaluation).
5. **Self-training rounds are human-gated**; Round 0 classifier-agreement is
   code-enforced OFF.
6. **No v1/v2 model outputs anywhere** (`lgbm_risk_score`,
   `vae_anomaly_score`, `graph_anomaly_score`, old checkpoints).
7. **Synthetic exposure is programmatic** — never CTGAN/TVAE/copula.
8. **`docs/AGENTS.md` is lead-owned** — redlines only, applied under explicit
   lead direction for the current branch.
9. **Checkpoints go through `checkpoint_manager`** (temp path → validate
   `{model_state_dict, centroid, config}` with `N_FEATURES`, `GRAPH_EMB_DIM`,
   `N_EDGE_TYPES` → atomic rename). Never `torch.save` onto the live path.
10. **`nic-worker` replicas = 1**, enforced in the k8s manifest with a
    comment.
11. **(scale) Scaler params are persisted, never refit per batch** — fit once
    per `schema_version`, store, re-apply. Refitting on a scoring batch is a
    batch-statistics leak.
12. **(scale) Every migration step passes its 15k parity gate** before the
    next starts (`docs/IMPLEMENTATION.md`).
13. **(scale) Dual-write before cut-over** — a Postgres table becomes
    authoritative only after demonstrated parity with its file predecessor.
14. **(scale) `src/db/` owns all SQL** — no inline SQL in handlers or model
    modules; schema changes via versioned migrations.

---

## Known Structural Weaknesses (MAR critique — still true, watch during migration)

| Component | What breaks |
|---|---|
| DeepSVDD centroid | If fraud dominates the population, hypersphere silently inflates |
| EVT | Discontinuous score distributions explode/collapse thresholds (mitigated at 3.5M — larger tails; `TECHNICAL_REFERENCE_AND_SCALING.md` §13.1) |
| Self-training | If the EVT tail is data-entry errors, the classifier anchors on typos — this is what breaks first in production |
| Isolated nodes | Zero typed edges → no structural signal; subspace IF carries them |
| Dense rings vs reconstruction | Dense cliques reconstruct easily — that's why dense-block exists, gated to `shares_ip` |

---

## Quantitative Claims Protocol

1. **Raw stdout only** — no number enters a doc without a traceable print
   line or artifact file.
2. **Name the baseline explicitly** (file, timestamp, or commit).
3. **Seed everything** before comparing (and mind the GPU noise floor).
4. **Row-level counting only** — unique flagged rows, isolated masking.
5. **No same-turn resolution** — log as "proposed, pending."
6. **Conflicting numbers halt.** Surface, re-derive; never reconcile by
   narrative.

---

## When to Stop and Ask

- A task requires modifying two modules simultaneously.
- A new dependency not in `requirements.txt`.
- A result conflicts with `docs/HISTORY.md` or an ablation JSON.
- A migration-step parity gate fails.
- A Postgres schema change (`deploy/postgres/` migrations).
- A self-training round is ready to advance.
- Anything would touch the fixed architecture or a locked hyperparameter.
- A referenced file is missing from the working directory.

## Open Decisions — Lead-Owned, Do Not Resolve Autonomously

Tracked in `docs/TECHNICAL_REFERENCE_AND_SCALING.md` §15 and
`docs/AGENTS.md` §7: K_CAP + group-size ceiling (needs 3.5M profiling);
NeighborLoader fan-out magnitude **and shape**; `pg_trgm` vs `difflib`
equivalence; Postgres HA policy; batch cadence; 4th-seed confirmation of the
locked fusion (3 seeds recorded as "proposed, pending").
