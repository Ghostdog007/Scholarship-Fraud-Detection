# Project History — How We Arrived at V4-Scale

<!-- VERSION: 1.0 | DATE: 2026-07-21 | Read-only record. Do not extend with new results;
     new work is documented in AGENTS.md / IMPLEMENTATION.md. Full pre-V4-Scale docs
     live in git history on `main` (see AGENTS.md and ARCHITECTURE_EVOLUTION_v4.md
     at commit 9b772de and earlier). -->

This file is the condensed, metric-backed record of every architectural
generation and the decisions that ended in the locked architecture now being
scaled on this branch. Every number here traces to a named artifact; none are
from memory.

---

## Generation 1 — v1: rules + supervised LightGBM (superseded)

99 hand-written NIC rules + 8 bridge features feeding a supervised LightGBM.
Canonical run archived. Abandoned because rule-based labels cap the system at
"fraud we already knew how to describe" and every threshold was a policy
liability. **Legacy: the "no rules" hard stop.**

## Generation 2 — v2: first rule-free stack (superseded)

Tabular VAE + Graph AE (DOMINANT + DeepSVDD) + full-space Isolation Forest +
MCM. Its synthetic-harness PR-AUC results became the **v3 pass floors** (still
enforced in `src/evaluate_model_v3.py::evaluate()`):

```
AGE_VIOLATION 0.1466 | INCOME_VIOLATION 0.6503 | IP_CONCENTRATION 0.0370
MOTHER_NAME_COLLISION 0.2869 | FEE_INFLATION 0.4962
```

Superseded because three separate detectors with three separate training
objectives were harder to reason about than one hybrid, and the full-space IF
diluted strong few-feature signals.

## Generation 3 — v3: Hybrid GraphMCM (the current detector core)

One two-stream detector (8 learned masks over the tabular features + RGCN over
5 identity relations → joint MLP predictor), a **per-group subspace IF**
(financial / identity / network), EVT thresholds, human-gated self-training,
LOE synthetic exposure (programmatic — CTGAN-family generators tested and
rejected for 24× composite degradation of fraud behavioral signal).

Key v3-era locked decisions, each ablation-backed:

| Decision | Evidence |
|---|---|
| RGCN over HAN encoder | HAN drop-in regressed **−0.091 mean over 3 seeds** (2026-07-04 ablation; `config_v3.py` comment at `ENCODER_ARCH`) |
| 68 → 44 features (drop 24 nominal identifier/code columns) | noid ablation 2026-07-15: no regression at detector or fused level; sharing signal preserved via raw-column graph edges + degree/count features (`src/ablation_noid_v3.py`) |
| GPU scatter-add noise floor | `RGCNConv(aggr="add")` CUDA atomics give ±0.03–0.04 run-to-run variance; effects smaller than that require deterministic/CPU scoring |

## Generation 4 — V4 capability experiments (2026-07, the comparison that fixed the architecture)

Six modes were compared head-to-head on connected-cluster injection
(baseline / tier1 attention read-out / ring classifier / max_fusion /
dense_block_fusion / dense_block_only), 3 seeds. Source tables: AGENTS.md
Appendix H at commit `9b772de` on `main`; raw JSON in `outputs/ablation/`.

**Finding 1 — dense-block is an IP specialist.** Largest single-category gain
in the project: IP_CONCENTRATION **0.155 (baseline) → 0.673 (dense_block_fusion)**
— but it regressed MOTHER_NAME to 0.098 because mother-name/pincode relations
are *legitimately* dense (families, geography). Hence: **dense-block gated to
`shares_ip` only** (`DENSE_BLOCK_RELATIONS=[1]`).

**Finding 2 — "GNN harmful" was a fusion artifact, not a detector fact**
(H.7 correction, verified 2026-07-05 on a frozen detector set). The raw
detector scored IP **0.511** and MOTHER **0.452**; the 14-positive LightGBM
fusion degraded those to 0.169 and 0.197. The RGCN raw score is the strongest
relational detector in the system; the weak link was the fusion.

**Finding 3 — LightGBM fusion removed.** With only 14 pseudo-label positives
the meta-learner destroyed calibrated components (subspace 0.966 → 0.315 on
INCOME through the fusion; H.8). Score-level weighted fusion of raw components
replaced it:

```
final_risk = minmax( 1.0·subspace + 0.5·dense_ip + 0.3·hybrid )
```

**Finding 4 — RGCN retirement disproven.** dense_block_only (GNN dropped)
was worst overall (mean 0.132, H.2); on held-out novel topologies the RGCN
generalises where dense-block evaporates (H.8: held-out rgcn IP 0.367 >
dense_ip 0.282).

## The locked-fusion validation (the numbers that justify this branch)

Source: `outputs/ablation/locked_fusion_validation.json` (3 seeds 42/43/44,
frozen pretrained detectors, connected-cluster harness). Aggregate mean
connected PR-AUC:

| Component | Mean over 5 categories (±std) |
|---|---|
| **locked_fusion (shipped)** | **0.659 ± 0.015** |
| subspace_only | 0.743 ± 0.002 |
| hybrid_only (raw RGCN) | 0.227 ± 0.028 |
| dense_ip_only | 0.197 ± 0.004 |

Why ship the fusion when subspace-only has a higher mean: the mean hides the
IP blind spot. Per-category (same file, aggregate block):

| Category | subspace_only | locked_fusion |
|---|---|---|
| AGE_VIOLATION | 0.673 | 0.597 |
| INCOME_VIOLATION | 0.970 | 0.705 |
| **IP_CONCENTRATION** | **0.357** | **0.581** |
| MOTHER_NAME_COLLISION | 0.791 | 0.752 |
| FEE_INFLATION | 0.922 | 0.662 |

The fusion trades tabular headroom for **+0.22 on the one category subspace
cannot see** (coordinated IP rings — the fraud type that matters most
operationally), while every category stays comfortably above the v2 floors.
Held-out novel-topology check (same file, `heldout` block): locked_fusion
0.569 / 0.718 / 0.315 / 0.741 / 0.692 across the five categories.

**Standing caveat, recorded honestly:** the validation `_meta` notes "3 seeds;
H.6 formal gate wants >3 — treat as proposed, pending a 4th seed." The
architecture is adopted; a 4th-seed confirmation remains open.

## Serving & operations layer (2026-07-10 → 2026-07-17)

Built around the frozen detection core, all file-contract based: FastAPI +
Celery + Redis + nginx console (review queue with 50/page triage, LOE pattern
queue with promote-to-exposure, admin deploy loop), supervisor CSV
fraud-pattern intake, cohort preview scoring, checkpoint manager with atomic
hot-swap, drift monitoring (KS, threshold p < 0.01), export/traceability.
MLflow was removed in favor of `model_registry.json`. Operator guide:
`OPERATIONS_RUNBOOK.md`.

## 2026-07-21 — V4-Scale branch created

Target: **30–40 lakh (3–4M) applications** on the production server (16 vCPU /
64 GB, Ubuntu 22.04, no GPU, k3s) with **PostgreSQL as the system of record**.
The detection architecture above is **fixed** — this branch changes I/O
boundaries, not model math. The four things that break at 233× scale and their
fixes are specified in `TECHNICAL_REFERENCE_AND_SCALING.md` Part II (with
externally-reviewed, in-session-verified citations); the working contract for
agents is `AGENTS.md`; the step plan is `IMPLEMENTATION.md`.
