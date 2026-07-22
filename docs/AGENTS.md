# AGENTS.md — V4-Scale Working Contract

<!-- VERSION: 3.0 (V4-Scale rewrite, 2026-07-21, applied under explicit project-lead
     direction for branch V4-Scale — see .claude/CLAUDE.md hard stop #8) -->
<!-- OWNER: Project Lead. Agents do not modify this file autonomously; flag and
     propose a redline instead. -->
<!-- Pre-rewrite AGENTS.md (2,377 lines incl. Appendix H result tables) is preserved
     in git history: `git show 9b772de:docs/AGENTS.md` on main. -->

You are working on the **scale phase** of the NIC scholarship fraud detection
system. The detection architecture is **fixed and validated** — you are
migrating its I/O boundaries to PostgreSQL and its training loop to
mini-batch sampling so it runs at 3–4 million applications on a CPU-only
Kubernetes server. You are **not** redesigning detection.

Read order for a cold start:
1. This file — the contract.
2. `TECHNICAL_REFERENCE_AND_SCALING.md` — Part I: how every component works
   today; Part II: the target architecture (Postgres schema, hub-capped edges,
   NeighborLoader, pod sizing). The single deepest reference.
3. `IMPLEMENTATION.md` — the 5-step migration plan with acceptance gates.
4. `HISTORY.md` — how we got here, with the metrics. Read-only; never extend.
5. `.claude/CLAUDE.md` — session instructions and hard stops. Binding.

---

## 1. The fixed detection architecture (do not relitigate)

```
raw applications (PostgreSQL `applications` / today: data/raw CSV)
   → tabular_feature_engine_v3  : 44 numeric features, MinMax-scaled
   → graph_builder_v3           : 5-relation identity graph
       (shares_mobile, shares_ip, shares_father_name,
        shares_mother_name, shares_pincode)
   → three FUSED detectors, all higher = more anomalous:
       hybrid_graphmcm_v3       : 8-mask feature stream + RGCN graph stream
                                  (root_weight=False since 2026-07-22 — pure
                                  neighbor aggregation, no self-leak around
                                  the MCM mask)
                                  → hybrid_anomaly_score
                                    = feature_pred_error + 0.3·edge_pred_error
       subspace_if_v3           : per-group IF (financial/identity/network)
       dense_block_detector_v3  : FRAUDAR-style peeling over shares_mobile +
                                  shares_ip + shares_pincode (independently
                                  peeled, IP-priority-weighted max-combined —
                                  since 2026-07-22; was shares_ip only)
   → evt_scorer_v3              : GPD tail thresholds (the only thresholds allowed)
   → LOCKED fusion              : final_risk = minmax(max(minmax(subspace),
                                     minmax(dense_relational), minmax(hybrid)))
                                   (since 2026-07-22; was a weighted sum)
   → self_training_loop_v3      : EVT-tail pseudo-labels, HUMAN-GATED
   → xai_layer_v3 / xai_card_html_v3 : evidence cards (JSON + HTML)

A 4th detector, deepsad_detector_v3 (Deep SAD center-distance: separate
RGCN encoder, center-pull/exposure-push, no reconstruction loss), runs
alongside but is NOT fused — tested directly as a 4th max-fusion input and
rejected (2026-07-22: candidate 4-way scored 0.4181 vs the locked 3-way's
0.4182 on stress_testing_1, noise-level). It reads into xai_layer_v3 only,
as a supplementary XAI-card signal, never into final_risk_score.
```

Why each piece is locked (metrics in `HISTORY.md`, raw JSON in
`outputs/ablation/locked_fusion_validation.json`):
- **Subspace IF is the backbone** (3-seed mean connected PR-AUC 0.743).
- **Dense-block extended beyond IP-only 2026-07-22** (shares_ip-only scored
  mobile-sharing rings near zero, PR-AUC 0.030; extending to mobile+pincode
  with IP-priority weighting — not equal weighting, which let ordinary
  non-fraud density outrank true IP rings — fixed that while holding IP-ring
  detection ~unchanged).
- **RGCN stays, and its root_weight was fixed 2026-07-22** (retirement
  disproven; best generalisation to novel topology; the root_weight=False
  fix recovered signal the LOE-margin fix below had cost it).
- **LOE margin was fixed 2026-07-22** (`LOE_MARGIN=2.0` was ~3x too small
  for this embedding scale — the exposure-push term contributed
  effectively zero gradient throughout training; replaced with a
  data-derived margin + a small persistent Stage-2 term).
- **LightGBM fusion is gone, and so is the weighted-sum fusion that
  replaced it** (LightGBM: 14 positives destroyed calibrated components.
  Weighted sum, superseded 2026-07-22: it diluted strong single-detector
  signals — e.g. mobile-ring, subspace alone 0.674 vs summed fusion 0.349 —
  replaced with an unweighted max). If you see LightGBM OR a `+` between
  detector terms referenced as the fusion layer anywhere, that record is stale.
- **HAN encoder available but off** (−0.091 vs RGCN, 3 seeds).
- **Deep SAD exists but is XAI-only, not fused** (validated strongest single
  relational signal this session on stress_testing_1, but tested directly
  as a 4th fusion input and found to not improve the fused score — see
  pipeline diagram above).

Hyperparameter source of truth: `src/config_v3.py`. Naming rule: **source
files keep `_v3` names** — "V4"/"V4-Scale" are capability/branch labels only.
Never rename `_v3` → `_v4`.

## 2. The scale target and what changes

**Target: 30–40 lakh = 3.0–4.0 million applications** (not 30–40 million) on
one server: 16 vCPU, 64 GB RAM, Ubuntu 22.04, **no GPU**, k3s Kubernetes.
**PostgreSQL is the system of record** — training data, LOE patterns,
confirmed fraud, scores, batches all live in it; every ingestion path (console
CSV upload, portal sync, bulk COPY) writes to it. The console's CSV-intake
UX is preserved exactly — only handler internals change.

Exactly four things change (design in `TECHNICAL_REFERENCE_AND_SCALING.md`
Part II — cite it, don't re-derive):

| Change | Replaces | Why |
|---|---|---|
| SQL-pushdown feature engineering, chunked, **persisted scaler params** | whole-frame pandas, fit-on-population MinMax | 20–30 GB peak → <1 GB; also fixes a batch-statistics leak |
| Hub-capped edge topology (K_CAP cliques, star above, statistical group-size ceiling) | all-pairs edges per shared value | O(k²) blowup: one 5,000-member pincode = 12.5M edges |
| NeighborLoader mini-batch training/scoring for the hybrid detector | full-graph forward passes | full graph at 3.5M nodes cannot fit 64 GB |
| Postgres tables behind existing interfaces | CSV/JSON file stores | indexed queries, concurrency, one source of truth |

Everything else — model classes, losses, hyperparameters, score semantics,
subspace IF / EVT / fusion / XAI logic, the frontend — **does not change**.

## 3. Module ownership

One module per response. If a task spans two rows, stop and confirm scope.

| Module | File(s) | Reads | Writes |
|---|---|---|---|
| Feature engineering | `src/tabular_feature_engine_v3.py` | `applications`/raw CSV | `features` table / `engineered_features_v3.csv`, `v3_feature_schema.json` |
| Graph construction | `src/graph_builder_v3.py` | features + `identity_keys` | graph artifact, degree features |
| Synthetic exposure | `src/synthetic_exposure_builder_v3.py` | features, promoted `loe_patterns` | exposure `.pt` artifacts |
| Hybrid detector | `src/hybrid_graphmcm_v3.py` | features, graph, exposure | `hybrid_scores`, `models/hybrid_graphmcm_v3.pth` |
| Subspace IF | `src/subspace_if_v3.py` | features | subspace scores |
| Dense-block | `src/dense_block_detector_v3.py` | `shares_mobile`/`shares_ip`/`shares_pincode` edges | per-relation scores + `dense_block_score_relational` |
| Deep SAD (XAI-only, not fused) | `src/deepsad_detector_v3.py` | features, graph, exposure | `center_dist_score`, `models/deepsad_v3.pth` |
| EVT | `src/evt_scorer_v3.py` | score vectors | `evt_thresholds` |
| Self-training | `src/self_training_loop_v3.py` | scores, EVT, `confirmed_fraud` | `pseudo_labels` |
| Fusion | `src/fusion_classifier_v3.py` | three score vectors | `final_risk_score` |
| XAI | `src/xai_layer_v3.py`, `src/xai_card_html_v3.py` | scores, per-feature errors | explanation cards |
| Evaluation | `src/evaluate_model_v3.py` | processed data, checkpoints | console / ablation JSON |
| Checkpoint manager | `src/checkpoint_manager.py` | incoming `.pth` (temp path) | live checkpoint (atomic rename), `models/checkpoints/` |
| Orchestrator | `src/retraining_orchestrator.py` | scores, drift state | training runs, drift JSON |
| **(new)** DB layer | `src/db/` | — | the only module that owns SQL; everything else goes through it |
| API | `src/api/` | via `src/db/` + stores | HTTP responses, Celery jobs |
| Frontend | `frontend/` | API | — (unchanged in this phase) |

**Migration rule:** during steps 1–4 of `IMPLEMENTATION.md`, `src/` model
modules are untouched; new code lives in `src/db/` and handler internals.
Model-module edits (feature engine internals, graph builder, NeighborLoader
loop) happen only in their designated steps, one module at a time.

## 4. Hard stops (all binding; carried forward + scale-phase additions)

1. **No rules.** No numeric threshold against a domain concept, no named rule
   codes, no policy-boundary features. Only EVT-derived or learned thresholds.
   The hub-cap / group-size ceiling must be derived from the observed
   group-size distribution (a statistical cutoff), never a hand-picked domain
   number.
2. **No raw GNN embeddings leave `hybrid_graphmcm_v3.py`.** Scalar scores and
   attention weights only. This includes the Postgres schema: **no embedding
   columns, ever.**
3. **Higher = more anomalous.** Any inversion must be documented at the point
   of inversion.
4. **`sanity` column is never used** — not as feature, label, or evaluation.
5. **Self-training rounds are human-gated.** No loop advances rounds
   automatically; Round 0 classifier-agreement is code-enforced OFF.
6. **No v1/v2 model outputs anywhere.**
7. **Synthetic exposure is programmatic.** Never CTGAN/TVAE/copula generators.
8. **This file is lead-owned.** Flag staleness; propose redlines; edit only
   under explicit lead direction for the current branch.
9. **Checkpoints go through `checkpoint_manager`** — temp path, validate
   `{model_state_dict, centroid, config}` (config must contain `N_FEATURES`,
   `GRAPH_EMB_DIM`, `N_EDGE_TYPES`), atomic rename. Never
   `torch.save(...)` directly onto the live path.
10. **`nic-worker` replicas = 1**, enforced in the k8s manifest with a
    comment. Training jobs write fixed paths; two workers corrupt each other.
11. **(scale) Scaler parameters are persisted, never refit per batch.** Fit
    on the training population once per `schema_version`, store (Postgres
    `feature_scaling` / artifact), apply stored params to every subsequent
    batch. Refitting on a scoring batch is a correctness bug (batch-statistics
    leak), not a style issue.
12. **(scale) Every migration step passes its 15k parity gate before the next
    starts.** Gates are defined in `IMPLEMENTATION.md`. No skipping ahead
    because a step "looks done."
13. **(scale) Dual-write before cut-over.** A Postgres table becomes
    authoritative only after parity with its file predecessor is demonstrated;
    until then the file store remains the source of truth.
14. **(scale) `src/db/` owns all SQL.** No inline SQL in handlers or model
    modules. Schema changes go through versioned migration files.

## 5. Quantitative claims protocol (unchanged, binding)

1. Raw stdout only — no number enters a doc without a traceable print line or
   artifact file.
2. Name the baseline explicitly (file, timestamp, or commit).
3. Seed everything before comparing; remember the ±0.03–0.04 GPU scatter-add
   noise floor on detector scores (use deterministic algorithms or CPU scoring
   for smaller effects).
4. Row-level counting only.
5. No same-turn resolution — a number generated this turn cannot settle an
   open question this turn; log as "proposed, pending."
6. Conflicting numbers halt the task. Surface; re-derive; never reconcile by
   narrative.

## 6. When to stop and ask the project lead

- A task requires modifying two modules at once.
- A new dependency not in `requirements.txt`.
- A result conflicts with a number recorded in `HISTORY.md` or an ablation
  JSON.
- A migration-step parity gate fails.
- You are about to change the Postgres schema (`deploy/postgres/schema.sql` /
  migrations).
- A self-training round is ready to advance.
- Anything would touch the fixed detection architecture (§1) or a locked
  hyperparameter.
- A referenced file is missing from the working directory.

## 7. Open decisions (lead-owned; do not resolve autonomously)

Tracked in `TECHNICAL_REFERENCE_AND_SCALING.md` §15:
1. K_CAP and the group-size ceiling — needs a profiling query on real 3.5M
   ingest. **Concrete supporting evidence (2026-07-22):** a `stress_testing_1`
   artifact (382-node shares_ip structure, from sampling 50k rows with
   replacement out of only 15k real applicants — see the generator's own
   docstring) demonstrated the exact failure mode this decision guards
   against: one oversized structure sets the max-anchor for
   `dense_block_detector_v3`'s per-relation min-max normalization, compressing
   every genuine IP-fraud ring's score (true rings topped out at 0.52, not
   1.0) and materially hurting fused IP-cluster detection (PR-AUC 0.095→0.055
   in that ablation). Confirmed absent from real 15k production data today
   (max shares_ip degree 38, vs the artifact's 382) — not an active bug — but
   directly motivates capping the normalization anchor (not just raw edge
   count) once hub-capping is designed, not just capping edge fan-out.
2. ~~NeighborLoader fan-out magnitude and shape~~ — **CLOSED (lead,
   2026-07-21): exact-neighborhood batching adopted** (fanout (-1,-1); ablation
   in IMPLEMENTATION.md step 5 — truncating fan-outs deviate up to 0.44,
   exact mode is bit-equal to full-graph for the 2-layer RGCN; memory bounded
   by the hub cap).
3. `pg_trgm` vs `difflib` name similarity — equivalence check required.
4. Postgres HA / warm standby — NIC ops policy call.
5. Batch cadence (yearly 3.5M vs rolling cohorts) — drives the retrain
   calendar.
6. (carried from detection phase) Multi-seed confirmation of the locked
   fusion — `locked_fusion_validation.json`'s 3-seed record predates the
   2026-07-22 weighted-sum→max fusion change and the root_weight/LOE-margin
   fixes to `hybrid_graphmcm_v3`; it validates a formula and an encoder that
   no longer exist in production. Needs re-running against the CURRENT max
   fusion + fixed encoder before it can be cited as confirming today's system.
