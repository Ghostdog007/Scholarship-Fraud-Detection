# V4 Architecture Evolution — What We Tested, Why We Pivoted, and What We Locked

**Branch:** `v4-han-graphmcm`
**Status of numbers:** every figure below traces to a file in the repo. Provenance is
named at each table. Where two runs disagree, both are shown with their source — not
reconciled by narrative (per the Quantitative Claims Protocol in `.claude/CLAUDE.md`).

**Primary sources**
- `outputs/ablation/tier_comparison.json` — the six-way comparison (its on-disk
  `aggregate`/`heldout` blocks are the **H.7 frozen-detector run**).
- `outputs/ablation/config{1,2}_rgcn_{feature,topo}_seed{42,43,44}.json` — the
  variance-controlled topology-exposure ablation.
- `docs/AGENTS.md` Appendix H (H.1–H.9) — canonical write-up of all runs.
- `docs/IMPLEMENTATION.md` — the locked architecture.

> **A note on "V4".** "V4" is a *capability label*, not a rename. Source files stay
> `_v3`. Everything here is an opt-in layer on top of the V3 Hybrid GraphMCM backbone.

---

## 0. TL;DR — the pivot chain in one line each

1. **Baseline (RGCN Hybrid GraphMCM + subspace IF → LightGBM fusion)** — balanced and
   tabular-strong, but **structurally blind to dense IP cliques** (they reconstruct
   *easily*). → motivated a relational add-on.
2. **HAN encoder swap** — two-level attention instead of RGCN. **Regressed −0.091
   (3-seed).** → rejected; RGCN kept.
3. **Tier-1 attention read-out** — export attention geometry as fusion columns.
   Recovered relational signal (IP, MOTHER) but **cratered tabular** (11 columns
   overfit 14 positives). → real signal, wrong plumbing; dropped as a fusion input.
4. **Ring classifier** — subgraph fingerprint. Most *stable*, best held-out MOTHER,
   but never wins a category. → kept as an **independent audit signal**, not fused.
5. **FRAUDAR-style dense-block detector** — explicit density peeling. **IP +0.52, the
   single largest category gain in the project**, but wrecks the legitimately-dense
   relations (MOTHER/PINCODE). → adopt, but **gated to `shares_ip` only**.
6. **`dense_block_only` (retire the RGCN)** — worst overall; **RGCN retirement
   disproven.** The raw RGCN score is our strongest relational detector.
7. **The real culprit was the combiner.** The 14-positive **LightGBM fusion destroys
   raw signal** (subspace INCOME 0.966→0.315; RGCN IP 0.51→0.169). → **replace
   LightGBM with a weighted score-level fusion.**

**Locked result:** two backbones (RGCN + subspace IF) + one IP-gated FRAUDAR
specialist, combined by a **label-independent weighted score-level fusion**, with a
dormant deviation layer and a standing (non-fused) ring auditor.

---

## 1. The problem the whole branch exists to solve

The V3 baseline is a reconstruction detector. Its core assumption — *anomalies are
hard to reconstruct* — is **exactly false for dense fraud cliques**. When 30
applications share one IP and look near-identical, an RGCN smooths them into a tight,
low-error cluster: the fraud **reconstructs beautifully** and scores as *normal*.
That is the MAR "dense rings" failure condition (folded into `.claude/CLAUDE.md`), and
it shows up numerically as the baseline's IP blind spot (IP PR-AUC 0.155, H.2).

Everything on this branch is an attempt to buy back that one relational blind spot
**without paying for it in tabular performance** (AGE/INCOME/FEE), which the baseline
is already very good at.

---

## 2. Architecture block diagrams — every model we tested

Score direction throughout: **higher = more anomalous** (hard stop #3).

### 2.1 `baseline` — RGCN Hybrid GraphMCM + subspace IF → LightGBM

```
                 68 numeric features            5-edge identity graph
                        │                       (mobile/ip/father/mother/pincode)
                        ▼                                │
        ┌───────── FEATURE STREAM ─────────┐            ▼
        │  K=8 random masks over 68 dims    │     ┌── GRAPH STREAM (RGCN) ──┐
        │  predict masked values            │     │  128 hidden → 64 emb h_N │
        └───────────────┬───────────────────┘     └─────────┬───────────────┘
                        └──────────► concat ◄────────────────┘
                                       │
                                   MLP (256→64)
                                       │
                        ┌──────────────┴───────────────┐
                        ▼                               ▼
               predicted x  ─► feature_pred_error   edge probs ─► edge_pred_error
                        │                               │
                        └── hybrid_anomaly_score = feat_err + 0.3·edge_err ──┐
                                                                             │
   68 features ─► SUBSPACE ISOLATION FOREST (financial | identity | network)│
                                    │  subspace_if_score ─────────────────┐  │
                                    ▼                                      ▼  ▼
                                                          LightGBM FUSION (14 positives)
                                                                    │
                                                              risk score
```

- **Good at:** AGE / INCOME / FEE (tabular), robustness, simplicity. The balanced
  incumbent and the backbone of everything downstream.
- **Blind to:** dense IP cliques (reconstruction smoothing) → IP 0.155.

### 2.2 `HAN` encoder swap — two-level attention instead of RGCN

```
        GRAPH STREAM (HAN)                      ← drop-in replacement for the RGCN box
   ┌───────────────────────────────┐
   │ node-level attention  (α)      │  per-relation neighbor weighting
   │        ▼                       │
   │ semantic-level attention (β_r) │  weight across the 5 relations
   └───────────────┬────────────────┘
                   ▼  h_N (64)   → (rest of the pipeline identical to baseline)
```

- **Result: regressed −0.091 (3-seed).** More attention parameters, same 5 relations,
  no extra supervision — the attention just added variance without new signal.
- **Verdict:** rejected. `ENCODER_ARCH="rgcn"` stays default; `han` remains available
  but unused. (`docs/IMPLEMENTATION.md` "DROPPED"; `.claude/CLAUDE.md` hyperparam table.)

### 2.3 `tier1` — attention read-out as fusion columns

```
   baseline (RGCN)  ──► 11 attention read-out columns
                        (per-relation β_r weights, α entropy, top-1 mass)
                                     │
   subspace_if_score ────────────────┤
   hybrid_anomaly_score ─────────────┼──► LightGBM FUSION (still 14 positives)
                                     ▼
                                 risk score
```

Only **scalar attention weights** leave the detector (hard stop #2 — raw `h_N` never
exported). Recovered relational signal but the 11 extra columns overfit 14 positives.

### 2.4 `ring` — subgraph fingerprint classifier (standing, not fused)

```
   identity graph ─► ring candidate generation (public structure only, never h_N)
                          │
                     structural fingerprint (density, degree profile, motif counts)
                          │
                     ring classifier + open-set novelty  ─► per-ring score
                          │
                     project ring score down to member nodes
```

- **Good at:** stability (std 0.004, lowest of any mode) and **held-out MOTHER (0.398,
  best of any mode, H.3).** Reads structure the clique-only harness under-measures.
- **Weak at:** never *wins* a category on aggregate (mean 0.238). → kept as an
  independent **audit** signal, not a fusion column.

### 2.5 `max_fusion` — per-node max(baseline, tier1)

```
   baseline score ─┐
                   ├─► elementwise max ─► risk score
   tier1 score ────┘
```

Highest mean in the H.2 run (0.445) but only +0.02 over baseline — inside the noise
floor. A "safe lateral," not a real gain.

### 2.6 `dense_block_fusion` — FRAUDAR dense-block (per relation) + DevNet, into LightGBM

```
   baseline ─────────────┐
   FRAUDAR dense-block ───┤  (run on ALL relations here)
   DevNet deviation ──────┼──► LightGBM FUSION
                          ▼
                      risk score
```

- **IP 0.673 vs baseline 0.155 → +0.52, the single largest category gain in the
  project (H.4).**
- **But** regresses everything and *craters MOTHER (0.098)*: the real graph has
  **legitimately** dense mother-name / pincode blocks (siblings, shared addresses), so
  an all-relations density score fires on benign structure. → the reason we later
  **gate it to `shares_ip` only.**

### 2.7 `dense_block_only` — GNN columns dropped (RGCN-retirement test)

```
   subspace IF ──────────┐
   FRAUDAR dense-block ───┼──► fusion   (RGCN / hybrid columns REMOVED)
   DevNet deviation ──────┘
```

- Tests "do we even need the GNN?" Answer: **no — worst overall (mean 0.132, H.2).**
  Loses tabular entirely. **RGCN retirement disproven.**

### 2.8 LOCKED — weighted score-level fusion (the current system)

```
     RGCN Hybrid GraphMCM + topology exposure ─► hybrid_anomaly_score ─┐  ×0.3
                                                                       │
     Subspace Isolation Forest ───────────────► subspace_if_score ─────┤  ×1.0
                                                                       │
     FRAUDAR dense-block, gated to shares_ip ─► dense_block_score_ip ──┤  ×0.5
                                                                       ▼
   risk = minmax( 1.0·minmax(subspace_if) + 0.5·minmax(dense_ip) + 0.3·minmax(hybrid) )
                                       │
                            EVT / SPOT threshold  ─► flags
                                       │
                    AAD supervisor loop (human-gated) ─► confirmed labels
                                       │
             (DORMANT) DevNet deviation layer — activates per-category on confirmed labels
             (STANDING) ring classifier — independent audit, not fused
```

No learned gate can bury a strong raw signal here — the weights are fixed in
`config_v3.py` (`FUSION_W_SUBSPACE=1.0`, `FUSION_W_DENSE_IP=0.5`, `FUSION_W_HYBRID=0.3`).

---

## 3. The ablation scoreboard

### 3.1 Six-way comparison — H.2 run (3-seed, CPU-deterministic, **LightGBM fusion**)

Injected-vs-real PR-AUC per category (higher = better). Source: `docs/AGENTS.md` H.2.
**Bold = best in row.**

| Category | baseline | tier1 | ring | max_fusion | dense_block_fusion | dense_block_only |
|---|---|---|---|---|---|---|
| AGE_VIOLATION | **0.402** | 0.141 | 0.182 | 0.286 | 0.163 | 0.091 |
| INCOME_VIOLATION | **0.638** | 0.180 | 0.212 | 0.586 | 0.262 | 0.052 |
| IP_CONCENTRATION | 0.155 | 0.413 | 0.186 | 0.404 | **0.673** | 0.453 |
| MOTHER_NAME_COLLISION | 0.341 | 0.391 | 0.390 | **0.411** | 0.098 | 0.032 |
| FEE_INFLATION | **0.587** | 0.165 | 0.220 | 0.537 | 0.235 | 0.033 |
| **MEAN** | 0.425 | 0.258 | 0.238 | **0.445** | 0.286 | 0.132 |
| (std of mean) | 0.040 | 0.043 | 0.004 | 0.061 | 0.077 | 0.021 |

### 3.2 Six-way comparison — H.7 frozen-detector run (**this is what `tier_comparison.json` on disk holds now**)

Detectors frozen (reused checkpoints, never retrained per run), IP-gated dense-block.
Source: `outputs/ablation/tier_comparison.json` `aggregate` block = `docs/AGENTS.md` H.7.

| Category | baseline | tier1 | ring | max_fusion | dense_block_fusion | dense_block_only |
|---|---|---|---|---|---|---|
| AGE_VIOLATION | 0.147 | 0.082 | 0.188 | 0.150 | 0.124 | **0.469** |
| INCOME_VIOLATION | 0.315 | 0.399 | 0.179 | 0.408 | 0.147 | **0.556** |
| IP_CONCENTRATION | 0.169 | 0.356 | 0.195 | **0.364** | 0.224 | 0.278 |
| MOTHER_NAME_COLLISION | 0.197 | 0.242 | **0.356** | 0.275 | 0.238 | 0.237 |
| FEE_INFLATION | 0.269 | 0.312 | 0.197 | 0.347 | 0.128 | **0.523** |
| **MEAN** | 0.220 | 0.278 | 0.223 | 0.309 | 0.172 | **0.413** |

> **⚠ Provenance conflict, surfaced not hidden.** H.2 gives baseline mean **0.425**;
> H.7 (current JSON) gives baseline mean **0.220**. The docs reconcile this explicitly:
> the "0.42 baseline" was **a lucky LightGBM fusion draw** above the typical V3 range
> (~0.25), not the norm (H.7 "Implications" #2). Both are real runs; the frozen-detector
> H.7 numbers are the trustworthy comparison because they remove per-run detector
> variance. **Do not average the two tables together.**

### 3.3 Held-out T9b — novel star / bipartite topologies (seed 42, H.3)

Tests generalisation to fraud *shapes never seen in training* (stars/bipartite, not
cliques).

| Category | baseline | tier1 | ring | max_fusion | dense_block_fusion | dense_block_only |
|---|---|---|---|---|---|---|
| AGE_VIOLATION | **0.444** | 0.105 | 0.149 | 0.304 | 0.226 | 0.058 |
| INCOME_VIOLATION | **0.675** | 0.101 | 0.210 | 0.545 | 0.293 | 0.064 |
| IP_CONCENTRATION | 0.271 | 0.424 | 0.157 | 0.387 | 0.285 | 0.230 |
| MOTHER_NAME_COLLISION | 0.167 | 0.300 | **0.398** | 0.328 | 0.027 | 0.045 |
| FEE_INFLATION | **0.669** | 0.084 | 0.230 | 0.539 | 0.300 | 0.057 |

Key reading: on **novel topology the dense-block IP edge evaporates** (a star is not
dense) — dense-block is a **clique specialist**, whereas RGCN+topology-exposure
generalises (see §5).

### 3.4 Raw-score analysis — bypassing LightGBM entirely (H.8, frozen detector, GPU, 1 run)

Each raw component rank-normalised, then compared. This is the run that exposed the
combiner as the real problem.

| Category | rgcn_topo | subspace | dense_ip | sl_sum | sl_max |
|---|---|---|---|---|---|
| AGE | 0.099 | **0.632** | 0.038 | 0.183 | 0.232 |
| INCOME | 0.082 | **0.966** | 0.038 | 0.165 | 0.340 |
| IP | 0.414 | 0.327 | **0.713** | 0.714 | 0.385 |
| MOTHER | 0.396 | **0.796** | 0.043 | 0.379 | 0.361 |
| FEE | 0.071 | **0.916** | 0.038 | 0.145 | 0.323 |
| **MEAN** | 0.212 | **0.727** | 0.174 | 0.317 | 0.328 |

Held-out (star/bipartite): subspace 0.757, rgcn 0.207, dense_ip 0.095, sl_sum 0.272,
sl_max 0.325. Note **rgcn IP 0.367 > dense_ip 0.282 out-of-distribution** — the RGCN
generalises, the dense-block does not.

### 3.5 Locked-fusion performance (`docs/IMPLEMENTATION.md`, frozen detector, GPU, 1 run)

| | AGE | INCOME | IP | MOTHER | FEE | MEAN |
|---|---|---|---|---|---|---|
| Connected | 0.576 | 0.700 | **0.538** | 0.737 | 0.643 | **0.639** |
| Held-out | 0.603 | 0.744 | 0.409 | 0.752 | 0.689 | 0.640 |

vs old LightGBM fusion (~0.22 mean) and vs subspace-only (0.727 mean but IP stuck at
0.327). The locked fusion trades a little tabular for **+0.21 on IP** — the relational
capability the whole architecture exists to provide.

> Reproducibility caveat (`.claude/CLAUDE.md`): `RGCNConv(aggr="add")` uses CUDA
> scatter-add atomics that are not seed-controlled → ±0.03–0.04 run-to-run noise on
> detector-derived scores. Differences smaller than that are not real.

---

## 4. The LightGBM weakness — why we dropped the learned combiner

This is the load-bearing finding of the branch.

**Symptom.** The LightGBM fusion is fit on the real 15k rows but only **14 EVT
pseudo-label positives**. Scoring the *same frozen detector* raw vs through the fusion:

| Signal | raw | after LightGBM fusion | damage |
|---|---|---|---|
| subspace INCOME | 0.966 | 0.315 | −0.651 |
| RGCN IP | 0.511 | 0.169 | −0.342 |
| RGCN MOTHER | 0.452 | 0.197 | −0.255 |

(Source: H.7 correction + H.8 "LightGBM weakness".)

**Mechanism.** With only 14 idiosyncratic positives, the tree learns *"fraud = these
14 specific points"* and discounts any feature that didn't happen to separate those
14 — **including the two strongest detectors we have.** The injected-fraud structure
in the harness doesn't match the 14 real positives, so the fitted tree actively
suppresses the raw graph score.

**Why this was initially misread.** `dense_block_only` "won" the fusion comparison
(H.2/H.7). It looked like *"drop the GNN."* The correction (H.7, verified 2026-07-05)
showed that was a **fusion artifact, not a detector fact**: `db_only` wins only
because that metric is tabular-dominated (subspace carries AGE/INCOME/FEE) and dropping
the GNN removes *fusion-fit noise*, not because the GNN signal is bad. Scored raw, the
RGCN is IP **0.511** / MOTHER **0.452** — 3× what the fusion reports.

**Decision.** Replace LightGBM with a **label-independent weighted score-level
fusion** (§2.8). No learned gate → no small-label overfitting → strong raw signals
survive to the risk score. LightGBM is **parked**, revisited only with monotonic
constraints once confirmed labels grow (`docs/IMPLEMENTATION.md` DROPPED).

---

## 5. LOE / outlier-exposure — how each component "learns a fraud shape"

**LOE = Latent Outlier Exposure.** The idea: don't just train on normal data and hope
anomalies stick out — *show the model synthetic examples of the fraud geometry* and
explicitly push them away from the normal region. Two components on this branch carry
an exposure layer, and **both were validated to genuinely teach the detector new
patterns** (H.9).

### 5.1 RGCN Hybrid GraphMCM — topology exposure

The LOE loss (`src/hybrid_graphmcm_v3.py:431`):

```python
def _loe_loss(h_synth, centroid, lam):
    dist = torch.norm(h_synth - centroid.unsqueeze(0), dim=1)
    exposure = torch.exp(-torch.sqrt(dist + 1e-8))   # ↑ when synth is NEAR centroid
    return lam * exposure.mean()                     # penalise fraud sitting in normal region
```

Training is two-stage (`hybrid_graphmcm_v3.py:649`):
- **Stage 1 — graph LOE warm-start (80 ep):** DeepSVDD pulls real nodes toward the
  centroid *while* the LOE term pushes **synthetic exposure nodes away**. The exposure
  weight is annealed linearly: `lam_t = LAMBDA_EXPOSURE·(1 − epoch/epochs_s1)`.
- **Stage 2 — free joint reconstruction (120 ep):** feature-pred + `0.3·`edge-pred
  loss, LOE off, so the geometry set up in Stage 1 relaxes into a usable detector.

The key upgrade on this branch: **topology exposure**. Instead of exposing only
*tabular* synthetic frauds, the exposure set includes injected **graph structures** —
dense clusters + edges (`synthetic_exposure_graph_v3.pt`), fed via
`_get_synth_h_topology`. This is what lets the graph stream learn *relational* fraud
shape, not just feature-space outliers.

**Validation (H.9 + config1→config2 ablation):**

| Test | BEFORE | AFTER | Δ |
|---|---|---|---|
| Star/bipartite IP, exposure had only cliques (LOE add-and-retrain) | 0.309 | **0.457** | **+0.148** |
| config1 (feature exposure) → config2 (topology exposure), IP, 3-seed mean | ~0.303 | ~0.464 | **+0.16** |

The config1→config2 mean comes from the seed JSONs:
`conn_pr_auc_ip_concentration` config1 = mean(0.327, 0.317, 0.266) = **0.303**;
config2 = mean(0.309, 0.419, 0.664) = **0.464**. Two independent tests, same ~+0.15
lift. **Topology exposure genuinely teaches the RGCN to catch unseen topologies** —
"show it a fraud shape → it detects that shape" is validated. (MOTHER also lifted:
config1 ~0.484 → config2 ~0.652.)

### 5.2 Deviation layer (DevNet/PReNet) — synthetic archetype exposure

The deviation layer (`src/deviation_layer_v3.py`) is weak-supervision: a small MLP
trained on real normals + a handful of anomalies (synthetic archetypes at cold-start,
real confirmed patterns once `DEV_MIN_CONFIRMED_PER_CATEGORY` is crossed). It sees
**tabular features only** (78-dim: 68 node + 5 dense-block + 5 degree), OOF-stacked to
stay leakage-safe.

**Validation (H.9, leave-one-archetype-out):** remove the IP archetype from the
exposure, test on IP, then add it back and retest → **0.034 → 0.126 (+0.092).** The
absolute numbers are low precisely *because it has no graph structure* — but the delta
is clearly positive, confirming the same exposure mechanism works here too.

**Why dormant:** cold-start synthetic-only today; it activates **per-category** only
once confirmed labels accrue via the AAD loop. Wired, validated, off.

### 5.3 What LOE does NOT fix

LOE teaches *shapes you expose it to*. It cannot invent the dense-clique blind spot
away — a clique that reconstructs easily still reconstructs easily. That residual gap
is exactly what the FRAUDAR arm exists to cover (§6). The MAR caveat stands:
too-narrow archetypes bias Stage 2 toward obvious fraud; the exposure set must stay
programmatically diverse (hard stop #7 — never CTGAN/TVAE-generated).

---

## 6. FRAUDAR dense-block — why it owns dense IP blocks (and only IP)

`src/dense_block_detector_v3.py`. A camouflage-resistant greedy peeling detector in
the FRAUDAR family — the right tool for the one thing reconstruction cannot see.

### 6.1 The algorithm

```
per relation r in DENSE_BLOCK_RELATIONS:
    build adjacency on relation r
    k-core prune (k=2)                         ← throw away tree/periphery nodes
    node_w[u] = 1 / log(global_deg[u] + C)     ← CAMOUFLAGE WEIGHT (the FRAUDAR trick)
    Charikar greedy peeling:
        repeatedly remove the min-weighted-degree node
        record density = weighted_edges / |active| at every step
    score[u] = max density of any subgraph u survived in
```

- **Camouflage resistance.** The weight `1/log(deg+C)` **down-weights high-degree
  nodes**. Fraud rings pad themselves with links to popular/legitimate nodes to look
  normal ("camouflage"); by discounting those popular columns, the peeling can't be
  fooled into diluting a dense core. This is the property plain density or plain
  k-core lacks.
- **Deterministic, self-terminating, unthresholded.** Stable tie-break by node id; no
  EVT gate inside it; peeling terminates when the graph empties. No rule, no domain
  threshold (hard stops #1).

### 6.2 Why it *wins* IP and *loses* MOTHER — the mechanism (H.5)

Dense-block flags **density-as-suspicious**. Whether that's correct depends entirely
on whether density is *natural* for the relation:

| Relation | Natural density in real data | Dense-block verdict |
|---|---|---|
| `shares_ip` | **sparse** — most people have their own IP | an injected IP clique stands out → **IP raw 0.713** |
| `shares_mother_name` | **dense** — siblings legitimately share a mother | fires on benign families → **MOTHER 0.098** |
| `shares_pincode` | **dense** — neighbours legitimately share an address area | fires on benign geography |

So the *same score* that gives the largest win in the project (IP +0.52) gives one of
the worst regressions (MOTHER −0.24) when applied to the wrong relation.

### 6.3 The gating decision

**Run dense-block on `shares_ip` only** — `DENSE_BLOCK_RELATIONS=[1]`. This keeps the
+0.52 IP win and gates the detector *away* from the legitimately-dense relations where
it produces false positives. It is a **specialist, not a general relational
detector** — and on held-out star/bipartite topology it fades (IP edge evaporates),
which is why the RGCN (not FRAUDAR) remains the general relational backbone.

### 6.4 The two detectors are complementary, not redundant

```
   RGCN graph stream   →  captures fraud TOPOLOGY in general
                          (rings, stars, bipartite; generalises to novel shapes;
                           topology-exposure teaches new geometries — §5.1)
                          BUT smooths dense cliques into "normal"

   FRAUDAR dense-block →  captures DENSE IP BLOCKS specifically
   (shares_ip only)       (camouflage-resistant; exactly the clique the RGCN misses)
                          BUT clique-only; fires on benign density; no OOD generalisation
```

The RGCN sees the *shape* of relational fraud across the whole graph; FRAUDAR sees the
*density* of the one sparse relation where density is itself the signal. Fusing them
(weights 0.3 and 0.5 respectively, subspace at 1.0) is why the locked system reaches
IP **0.538** connected — neither component gets there alone (RGCN raw 0.51, dense_ip
raw 0.713, but each with its own failure mode the other covers).

---

## 7. Component scorecard — what each piece is good at

| Component | Owns (strong) | Structurally blind to | Role in locked system |
|---|---|---|---|
| **Subspace Isolation Forest** | AGE/INCOME/FEE tabular (raw mean **0.727**; INCOME 0.966, FEE 0.916) | relational / structural fraud | **backbone, weight 1.0** |
| **RGCN Hybrid GraphMCM + topology exposure** | general relational shape (raw IP 0.51, MOTHER 0.45); **best OOD generalisation** | dense cliques (smoothing) | relational arm, weight 0.3 |
| **FRAUDAR dense-block (IP-gated)** | dense IP cliques (raw IP **0.713**) | anything sparse; benign density; novel topology | IP specialist, weight 0.5 |
| **Ring classifier** | stability (std 0.004); held-out MOTHER (0.398) | never wins a category | standing **audit**, not fused |
| **Deviation layer (DevNet)** | (validated exposure Δ +0.092) | graph structure (tabular-only) | wired, **dormant** |
| **LightGBM fusion** | — | destroys raw signal at 14 positives | **removed** (parked) |
| **HAN encoder** | — | added variance, no new signal | **rejected** (−0.091) |
| **Weighted score-level fusion** | preserves every raw signal; label-independent | (no learning → no adaptation) | **the locked combiner** |

---

## 8. Why we pivoted, step by step (the reasoning chain)

1. **Baseline → needed a relational fix.** Reconstruction is blind to dense cliques
   (IP 0.155). Not a bug to patch — a structural property. Needed a *different kind* of
   detector, not a better RGCN.
2. **Tried HAN first (cheapest fix: swap the encoder).** More attention, same 5
   relations, no new supervision → −0.091. Rejected. The problem wasn't encoder
   capacity.
3. **Tried exporting attention (tier1).** Real relational signal recovered (IP,
   MOTHER) — but 11 columns × 14 positives = tabular collapse. Right signal, wrong
   plumbing.
4. **Tried an explicit density detector (dense_block).** IP +0.52 — the win we were
   chasing — but MOTHER/PINCODE regressions from benign density. Answer: **gate to IP.**
5. **Tried retiring the RGCN (dense_block_only).** Worst overall. RGCN retirement
   **disproven** — but it *looked* right, which forced the raw-score audit.
6. **Audited raw vs fused (H.8).** Found the true culprit: **LightGBM at 14 positives
   destroys raw signal.** The GNN was never the problem; the combiner was.
7. **Replaced the combiner.** Label-independent weighted score-level fusion → locked
   architecture (§2.8), connected mean **0.639**, IP **0.538**.

---

## 9. What is locked, dropped, and pending

**Locked (`docs/IMPLEMENTATION.md`, `config_v3.py`):**
`ENCODER_ARCH="rgcn"`, `DENSE_BLOCK_ENABLED=1`, `DENSE_BLOCK_RELATIONS=[1]`,
`FUSION_W_SUBSPACE=1.0`, `FUSION_W_DENSE_IP=0.5`, `FUSION_W_HYBRID=0.3`,
`DEVIATION_LAYER_ENABLED=0`. Tier1 / ring OFF as fusion inputs.

**Dropped (measured out):** HAN (−0.091), tier1 as fusion, max/equal-weight fusion,
LightGBM as primary combiner, RGCN retirement.

**Pending (the one open gate):** the **IP-gated dense-block validation run across >3
seeds** on the frozen detector set. H.6 is explicit — *do not* flip any additional
flag ON until that combination is confirmed, because "small-seed comparisons are where
false improvements hide." A **3-seed** run of the locked fusion is now recorded (§10);
the >3-seed extension (a 4th detector) is the remaining step.

---

## 10. Locked-fusion validation — score-level fusion on frozen detectors (2026-07-10)

**What changed in the harness.** The 14-positive **LightGBM combiner was removed** from
the comparison harness (`src/compare_architectures_v3.py`) and replaced by the locked
**score-level fusion** (`src/fusion_classifier_v3.score_level_fusion`, the single source
of truth shared with the production `run_fusion`). The harness now does **no fitting at
all** — the fusion is label-independent — and reuses the **frozen pretrained detectors**
`models/hybrid_v3_seed{42,43,44}.pth` (never retrained). It reports the locked fusion
against each of its raw parts.

**Provenance:** `outputs/ablation/locked_fusion_validation.json`, one run, 3 seeds
(42/43/44), connected-cluster harness + T9b held-out. **Does not** overwrite the H.7
`tier_comparison.json`. Raw stdout below.

### 10.1 Aggregate connected-cluster PR-AUC — mean over seeds 42/43/44

| Category | **locked_fusion** | subspace_only | hybrid_only (RGCN) | dense_ip_only |
|---|---|---|---|---|
| AGE_VIOLATION | 0.597 | **0.673** | 0.086 | 0.046 |
| INCOME_VIOLATION | 0.705 | **0.970** | 0.058 | 0.042 |
| IP_CONCENTRATION | 0.581 | 0.357 | 0.363 | **0.816** |
| MOTHER_NAME_COLLISION | 0.752 | **0.791** | 0.564 | 0.042 |
| FEE_INFLATION | 0.662 | **0.923** | 0.062 | 0.042 |
| **MEAN** | **0.659** | 0.743 | 0.227 | 0.198 |
| (std of mean) | 0.0145 | 0.0023 | 0.0276 | 0.0038 |

### 10.2 Held-out T9b — novel star/bipartite topology (seed 42)

| Category | locked_fusion | subspace_only | hybrid_only | dense_ip_only |
|---|---|---|---|---|
| AGE_VIOLATION | 0.569 | 0.668 | 0.046 | 0.045 |
| INCOME_VIOLATION | 0.718 | 0.973 | 0.044 | 0.044 |
| IP_CONCENTRATION | 0.315 | 0.330 | 0.140 | 0.188 |
| MOTHER_NAME_COLLISION | 0.741 | 0.800 | 0.536 | 0.045 |
| FEE_INFLATION | 0.692 | 0.938 | 0.050 | 0.046 |

### 10.3 Reading (logged **proposed, pending** — not a gate resolution)

- **The locked fusion does what it was designed to.** It trades ~0.08 mean tabular
  (0.659 vs subspace-only 0.743) to lift the **IP blind spot 0.357 → 0.581 (+0.22)** —
  matching the IMPLEMENTATION.md single-run prediction ("+0.21 on IP"), now confirmed
  across **3 seeds** rather than one run.
- **Stable:** locked std of mean **0.0145**, inside the ±0.03–0.04 scatter-add noise
  floor.
- **Components behave as architected:** subspace = tabular backbone; `dense_ip` = pure
  IP specialist (0.816 on IP, ~0.04 everywhere else → correctly gated); `hybrid` = the
  relational carrier (MOTHER 0.564, IP 0.363). On held-out topology `dense_ip`
  evaporates (0.188) — the clique specialist can't see stars — and the fusion leans on
  subspace + hybrid.

**Caveats.** (1) **3 seeds, not >3** — per the Quantitative Claims Protocol #5 this is a
strong positive read, **not** the closed H.6 gate; that needs a 4th detector (seed 45).
(2) These numbers are recorded here and in the JSON only; `docs/AGENTS.md` (lead-owned)
is untouched.

---

*Generated 2026-07-10. All figures traced to `outputs/ablation/*.json`,
`docs/AGENTS.md` Appendix H, and `docs/IMPLEMENTATION.md`. Where H.2 and H.7 disagree,
both are shown with provenance and not reconciled by narrative. §10 traces to
`outputs/ablation/locked_fusion_validation.json`.*
