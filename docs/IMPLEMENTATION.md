# IMPLEMENTATION.md — V4 Final Detection Architecture (LOCKED)

**Status:** Locked and implemented (2026-07-05). Every component choice below is
backed by measured results in `docs/AGENTS.md` Appendix H. Sits under
`.claude/CLAUDE.md` and `docs/AGENTS.md`.

**Design principle:** several *specialised raw detectors*, each catching what the
others structurally cannot, combined by a **weighted score-level fusion** — NOT a
learned tree. The 14-positive LightGBM fusion was dropped because it *destroyed* the
raw signals (subspace INCOME 0.966→0.315, RGCN IP 0.51→0.169; AGENTS.md H.8).

---

## The locked architecture

### Layer 1 — Two backbones (raw scores)
- **RGCN Hybrid GraphMCM + topology exposure** → `hybrid_anomaly_score`. The
  relational backbone: raw IP **0.51**, MOTHER **0.45**, and the best generalisation
  to novel topology. Its topology-exposure layer is validated (LOE +0.148, H.9).
- **Subspace Isolation Forest** → `subspace_if_score`. The tabular backbone and the
  single strongest component (raw mean **0.727**; INCOME 0.966, FEE 0.916).

### Layer 2 — IP specialist: dense-block, gated to `shares_ip`
FRAUDAR-style camouflage-resistant greedy peeling, run on **`shares_ip` only**
(`DENSE_BLOCK_RELATIONS=[1]`). Raw IP **0.713** — fills subspace's one blind spot.
Not run on `shares_mother_name`/`shares_pincode` (legitimately dense → false
positives). Deterministic, self-terminating, unthresholded.
`src/dense_block_detector_v3.py` → `outputs/dense_block_scores_v3.csv`.

### Layer 3 — Deviation layer (wired, DORMANT)
DevNet/PReNet weak-supervision. Cold-start synthetic-only today → dormant. Exposure
layer validated (LOE +0.092, H.9). Activates per-category once confirmed labels
cross `DEV_MIN_CONFIRMED_PER_CATEGORY`. `src/deviation_layer_v3.py`.

### Layer 4 — Weighted SCORE-LEVEL fusion (LOCKED)
```
risk = minmax( 1.0*minmax(subspace_if_score)
             + 0.5*minmax(dense_block_score_ip)
             + 0.3*minmax(hybrid_anomaly_score) )
```
Weights in `config_v3.py` (`FUSION_W_SUBSPACE/DENSE_IP/HYBRID`). Label-independent —
no learned gate to bury a strong signal. Subspace dominant (wins 4/5 categories),
dense-block-IP boosts the IP blind spot, RGCN adds relational/topology signal.
`src/fusion_classifier_v3.py`.

### Layer 5 — EVT/SPOT thresholding (unchanged)
The single authoritative threshold on the fused score. `src/evt_scorer_v3.py`.

### Layer 6 — AAD feedback loop (unchanged)
Supervisor confirms/rejects top flags → feeds Layer 3's labels and, if novel,
topology exposure via `confirmed_fraud_graph_store` (FLAGGED→CONFIRMED→SELECTED→
PROMOTED). Batched, human-gated (hard stop #5).

### Standing, not fused: ring classifier
Independent audit signal, not a fusion input. Most stable, best held-out MOTHER.
`src/ring_*.py`.

---

## Measured performance (locked fusion, frozen detector, GPU, one run)

| | AGE | INCOME | IP | MOTHER | FEE | MEAN |
|---|---|---|---|---|---|---|
| Connected | 0.576 | 0.700 | **0.538** | 0.737 | 0.643 | **0.639** |
| Held-out | 0.603 | 0.744 | 0.409 | 0.752 | 0.689 | 0.640 |

vs the old LightGBM fusion baseline (~0.22 mean) and vs subspace-only (0.727 mean but
IP stuck at 0.327). The locked fusion trades a little tabular for a real **+0.21 on
IP** — the relational capability the architecture exists to provide.

---

## DROPPED (measured out)

- **Tier-1 attention read-out** — modest; redundant once the raw RGCN score is used.
- **HAN encoder** — regressed −0.091 (3-seed).
- **`max_fusion` / equal-weight score fusion** — dilutes strong per-category signal.
- **LightGBM as the primary combiner** — destroyed the raw signals on 14 labels.
  Parked; revisit (with monotonic constraints) only once labels grow.
- **RGCN retirement (`dense_block_only`)** — disproven; RGCN raw is the best
  relational detector.

---

## Config (locked in `src/config_v3.py`)

`ENCODER_ARCH="rgcn"`, `DENSE_BLOCK_ENABLED=1` (default ON), `DENSE_BLOCK_RELATIONS=[1]`,
`FUSION_W_SUBSPACE=1.0`, `FUSION_W_DENSE_IP=0.5`, `FUSION_W_HYBRID=0.3`,
`DEVIATION_LAYER_ENABLED=0` (dormant). `TIER1_ATTN_FEATURES`/`RING_CLASSIFIER_ENABLED`
remain OFF (deprecated as fusion inputs).

Pipeline order (`main_v3.py`): … subspace_if → **dense_block** → evt → self_training →
**fusion (score-level)** → xai → evaluate.

## Frozen boundaries (unchanged)

FastAPI routes, MLflow schema, `checkpoint_manager.py`, deployment spec, hard stops
#1–16. This architecture is entirely inside the detection pipeline.
