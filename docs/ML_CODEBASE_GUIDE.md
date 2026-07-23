# ML Codebase Guide — NIC Scholarship Fraud Detection (V3/V4)

<!-- OWNER: not lead-owned — explanatory reference, safe to extend. -->
<!-- Scope: the detection/ML core only (src/*_v3.py + main_v3.py). API, -->
<!-- serving, console, and Postgres-migration code are covered in -->
<!-- docs/TECHNICAL_REFERENCE_AND_SCALING.md and docs/OPERATIONS_RUNBOOK.md. -->

## Purpose of this document

You've been asked to explain this codebase. This doc walks the **ML-relevant
file structure** — one section per file, in pipeline order — with the
functions/classes that matter, real code snippets, and (most importantly)
**why** each one is built the way it is. The "why" is usually the least
obvious part of an ML codebase and the part a reviewer most needs — a lot of
what looks like an arbitrary constant or a strange comment here is the fossil
record of a specific ablation that failed a different way first.

For architecture background (what the system does, in prose) start with
`docs/AGENTS.md` and `docs/TECHNICAL_REFERENCE_AND_SCALING.md` Part I. This
doc assumes that context and goes file-by-file into the code itself.

**How to read this**: the pipeline runs in the order these sections appear
(mirrors `main_v3.py`'s `PIPELINE_STEPS`). Each detector section stands
mostly alone; the fusion/XAI sections at the end depend on all three.

---

## 0. Orchestration and configuration

### `main_v3.py` (project root, ~344 lines)

The pipeline entry point. `run_pipeline(steps=None, smoke_test=False,
cycle="manual")` (line 165) runs all 14 steps in order — feature engineering
→ graph → synthetic exposure → Hybrid GraphMCM training → relation ablation
→ Subspace IF → dense-block → Deep SAD → EVT → self-training → fusion → XAI
→ evaluate — wrapped in an MLflow experiment run (SQLite-backed, so it
survives being launched from any working directory). `should_run(name)`
gates each step so `--steps=build_base,train_hybrid` can run a subset.

**Why it looks like this**: line 17 carries a hard stop directly in the
orchestrator, not just in a policy doc —

```python
HARD STOP: self-training round advancement requires Phase D PR-AUC check by project lead.
           This orchestrator always runs round=0. Do not modify to auto-advance.
```

That's deliberate: a policy document can be missed by someone editing code;
a docstring at the top of the function that would need to change is much
harder to miss.

### `src/config_v3.py` (~300 lines)

The single source of truth for every hyperparameter, feature list, and
architecture switch. Almost nothing elsewhere hard-codes a number — everyone
imports from here. This is the file to read first when asking "what does the
system currently do" because it's where every locked decision and every
still-open ablation switch (`V4_ENCODER_ARCH`, `V4_TOPO_EXPOSURE`,
`V4_RING`, ...) actually lives, each usually with a one-line comment citing
the ablation that decided it.

Two constants worth understanding immediately because they show up
everywhere downstream:

```python
LAMBDA_EDGE = 0.3          # training-loss weight — edge-prediction head still trains
LAMBDA_EDGE_SCORE = 0.0    # inference SCORE weight — decoupled from the above
```

**Why two separate constants for what sounds like one thing**: `edge_pred_error`
(how well the model predicts which relations a node is connected by) showed
no usable ranking signal on any of 3 designed relational fraud categories
(held-out PR-AUC 0.011–0.023 — at or below what a random ranking would
score) and *actively diluted* the other detector term's real signal on one
category (0.268 standalone vs 0.017 once combined). Rather than rip the
edge-prediction head out of training entirely (which would be a bigger,
riskier change and lose it as an XAI signal), the fix was to keep it
training but zero its score contribution — hence two separate constants
instead of one.

`FUSION_COMPONENTS = ("subspace", "dense_relational", "hybrid")` (line ~286)
documents the three — and only three — inputs to the locked fusion; see §5.

---

## 1. Feature engineering

### `src/tabular_feature_engine_v3.py` (~398 lines)

Two-stage build: `build_base()` (line 263) produces a 63-feature CSV (before
graph-degree features exist yet); `add_degree_features()` (line 305) merges
in the 5 relation-degree columns from the graph builder to produce the final
44-feature model input (after the 24 nominal-identifier columns are dropped
here) plus `data/processed/v3_feature_schema.json` — the schema file every
other module reads feature order from, so it never has to be hard-coded
twice.

A parallel Postgres-backed path (`build_base_pg()`, line 282;
`apply_stored_scaling()`, line 233) exists for the V4-Scale migration —
same output, sourced from Postgres instead of the file, verified bit-exact
(see `docs/IMPLEMENTATION.md` Gate 4). `apply_stored_scaling()` **raises**
rather than silently refitting if scaler params for a schema version are
missing — this is hard stop #11 (scaler params persisted once, never
refit on a scoring batch, since refitting on a batch's own statistics is a
data leak).

**A precision detail worth knowing about** (lines ~251–253), because it's
the kind of thing that looks like paranoia until you've hit it:

```python
# numpy, not pandas Series arithmetic: pandas routes >10k-element ops
# through numexpr, which rounds the mul/add chain 1 ULP differently --
# numpy x*scale_+min_ is bit-identical to sklearn's transform.
```

This is why the Postgres-path parity gate (`docs/IMPLEMENTATION.md` Step 4)
could claim **bit-exact**, not just "close enough" — the codebase cares
about floating-point reproducibility to the ULP level in places where a
silent divergence between two supposedly-identical code paths would
otherwise be undetectable until scores quietly drifted.

### `src/graph_builder_v3.py` (~228 lines)

Builds the 5-relation identity graph as a PyG `HeteroData` object: every pair
of applications sharing a raw value (mobile, IP, father's name, mother's
name, pincode) gets an edge. `build_graph()` (line 58) is the file path;
`_build_edges()` (line 38) does the actual pairing, currently as a full
clique per shared value (O(n²) per group). `build_graph_pg()` (line 178) is
the SQL/scale-path equivalent, with **hub-capping** support:
`_edges_from_groups()` (line 139) turns oversized groups into a star instead
of a clique above `k_cap`, and skips groups above a `ceiling` entirely.
`derive_group_ceiling()` (line 170) computes that ceiling from the *observed*
group-size distribution (default: 99.9th percentile), never a hand-picked
number — that's hard stop #1 (no rule-like thresholds against a domain
concept).

**Why both caps default to `None`** (off): this is explicitly one of the
project's open decisions, not an oversight —

```python
# BOTH CAPS DEFAULT OFF ... K_CAP is open decision #1 (lead-owned,
# needs 3.5M profiling)
```

The machinery is built and validated, but choosing the actual cap value
needs real 3.5M-scale group-size data that doesn't exist yet on the 15k
dataset — picking a number now would be exactly the kind of hand-picked
threshold hard stop #1 forbids.

### `src/synthetic_exposure_builder_v3.py` (~308 lines)

Builds the **programmatic** LOE (exposure) set the detector is pushed away
from during training — 750 rows across 5 archetypes (IP concentration,
mother-name collision, fee inflation, age violation, income violation), each
a real row's features perturbed on the archetype's target columns.
`build_exposure_set()` (line 194); `_add_context_noise()` (line 47)
additionally perturbs a random 25% of *non-target* columns so exposure rows
aren't suspiciously identical everywhere except the one signal — otherwise
the model could learn "identical-except-one-field" as the tell, instead of
the actual fraud shape. `build_topology_exposure()` (line 224) is the
graph-structured sibling: synthetic connected cliques for the RGCN's
relational LOE term, feeding `synthetic_exposure_graph_v3.pt`.

**Why it's never a GAN**: hard stop #7. CTGAN/TVAE/copula generators were
tested and explicitly rejected — with only 15k real applicants, a learned
generative model risks memorizing/leaking real individuals' data into
synthetic fraud examples, and a rule-free system that's simultaneously
learning its "what fraud looks like" signal *from a generative model fit on
the same population it's supposed to be finding fraud in* is circular in an
uncomfortable way. Perturbation of real rows is auditable and bounded;
CTGAN-family output is neither.

---

## 2. Detector 1 — Hybrid GraphMCM (RGCN)

### `src/hybrid_graphmcm_v3.py` (~1,220 lines — the largest, most important file)

This is the core learned detector. It reconstructs each application's own
(masked) features conditioned on its graph neighborhood — the model that
can't reconstruct an application well, or whose neighborhood looks unlike
its own declared attributes, scores as anomalous. Two-stage training: Stage
1 warms up a DeepSVDD-style centroid + graph-side LOE push-away; Stage 2 is
free joint feature/edge reconstruction with a lighter, *persistent* LOE term.

**`RGCNEncoder`** (class, line 90) — two `RGCNConv` layers:

```python
self.conv1 = RGCNConv(N_FEATURES, GRAPH_HIDDEN, num_relations=N_EDGE_TYPES, aggr="add", root_weight=False)
self.conv2 = RGCNConv(GRAPH_HIDDEN, GRAPH_EMB_DIM, num_relations=N_EDGE_TYPES, aggr="add", root_weight=False)
```
`forward()` (line 101): `h = tanh(conv1(...)); h = tanh(conv2(...))`.

**Why `root_weight=False`** — this is the single most important non-obvious
line in the whole codebase, and it's commented in place (lines 87–99):
`RGCNConv` defaults to `root_weight=True`, which adds a learned
self-transform of each node's *own unmasked* features straight into its
output embedding — completely bypassing the masked-channel (MCM) mechanism
`_apply_masks()` is supposed to enforce. In other words, with the default on,
the "graph-conditioned" embedding could just be smuggling in the answer the
model is supposed to be predicting from context. Turning it off measurably
mattered: overall PR-AUC 0.153→0.201 on a stress test, mobile-ring
0.029→0.078, IP-ring 0.032→0.055, with no regression at low degree.

**`HybridGraphMCM`** (class, line 219) — the masked-channel mechanism:

```python
self.mask_logits = nn.Parameter(torch.randn(MASK_NUM, N_FEATURES))   # K=8 learned masks
...
def _apply_masks(self, x: torch.Tensor) -> torch.Tensor:
    """Average over K soft-masked versions of x."""
    masks = torch.softmax(self.mask_logits, dim=1)  # (K, N_FEATURES)
    masked = x.unsqueeze(0) * masks.unsqueeze(1)     # (K, B, N_FEATURES)
    return masked.mean(dim=0)                         # (B, N_FEATURES)
```

`K=8` learned, softmax-normalized masks are averaged — the model learns
*which* features to hide from itself before reconstructing them from
context, rather than a hand-picked masking scheme.

`encode_graph()` (line 260) is the isolated-node handling — a node with zero
edges gets a **trainable** `isolated_embedding` parameter substituted for
whatever the (meaningless, edgeless) RGCN output would be:

```python
def encode_graph(self, x, edge_index_list, edge_type_tensor, isolated_mask) -> torch.Tensor:
    h = self.encoder(x, edge_index_list, edge_type_tensor)
    iso_emb = self.isolated_embedding.unsqueeze(0).expand(h.shape[0], -1)
    mask_exp = isolated_mask.unsqueeze(1).expand_as(h)
    return torch.where(mask_exp, iso_emb, h)
```
This is *why* isolated nodes aren't simply broken by this detector — they
get a real, trained fallback embedding instead of a zero vector or garbage
from an encoder with no neighbors to aggregate.

**`compute_score_frame()`** (line 493) — the single source of truth for the
score formula, reused by training, incremental fine-tune, and the API's
read-only scoring path so the three can never silently diverge:

```python
per_feat_err       = (pred_x - x_all).abs()          # (N, N_FEATURES)
feature_pred_error = per_feat_err.mean(dim=1)        # (N,)
...
edge_pred_error = F.binary_cross_entropy(edge_prob, target, reduction="none").mean(dim=1)

# LAMBDA_EDGE_SCORE (0.0), not LAMBDA_EDGE (0.3, still the training-loss weight)
hybrid_anomaly_score = feature_pred_error + LAMBDA_EDGE_SCORE * edge_pred_error
```

**Why `feature_pred_error` is a per-feature mean absolute error, not
something fancier**: it's directly interpretable — the XAI layer pairs each
feature's predicted value against its actual declared value and states
"expected X, declared Y" with a direction, which only works cleanly because
the error metric *is* a literal difference in the same scaled feature space,
not a compressed embedding distance.

**`train()`** (line 737) — the two-stage loop:
- **Stage 1** (`EPOCHS_STAGE1`): `loss = svdd_loss + loe`, where
  `svdd_loss = torch.norm(h_n - centroid).mean()` pulls real embeddings
  toward a DeepSVDD centroid, and `loe` (`_loe_loss()`, line 467) pushes
  synthetic exposure embeddings away with a decaying weight
  `lam_t = LAMBDA_EXPOSURE * (1 - epoch/epochs_s1)`.
- **Stage 2** (`EPOCHS_STAGE2`): `loss = feat_loss + LAMBDA_EDGE * edge_loss
  + loe_s2` — free joint reconstruction, but now with a *persistent*
  (non-decaying) LOE term, `LOE_STAGE2_WEIGHT`.

**Why Stage 2 needed its own persistent LOE term** (this is the kind of
regression you only catch by testing, not by reasoning about the math):
Stage 2 originally had *zero* exposure term — pure reconstruction. But 120
epochs of unconstrained reconstruction can slowly re-absorb whatever
separation Stage 1 built between real and synthetic-fraud embeddings,
because dense synthetic cliques are, mechanically, *easy* to reconstruct
(the same reason `dense_block_detector_v3.py` exists as a separate
specialist — see §3). Adding a light, non-decaying LOE weight in Stage 2
keeps that separation from silently eroding over the longer training stage.

**`_loe_loss()`** (line 467) — a hinge loss on distance-to-centroid:
```python
clamp(margin - dist, min=0).mean() * lam
```
with `margin` **data-derived**
(`_derive_loe_margin()`, line 447) rather than the earlier fixed constant
`LOE_MARGIN=2.0` — that fixed value turned out to be roughly 3x smaller than
the real population's own median embedding-to-centroid distance at
`GRAPH_EMB_DIM=64` (median ≈ 5.9), meaning exposure examples were already
"past the margin" and contributing zero loss before training even started.
An earlier `exp(-sqrt(dist))` loss formulation was tried first and rejected
— it saturated to ~0 within the first few epochs at *both* 30 and 150 epoch
budgets, i.e. it wasn't a usable training signal at any reasonable schedule.

**`compute_relation_ablation()`** (line 618) — re-scores the *already
trained, locked* checkpoint 5 times, each time with one relation's edges
masked out of `edge_type_tensor`, to attribute how much each relation's
presence changes the reconstruction. This is XAI-only — it never feeds a
threshold or the fusion score, it only narrates "removing this relation
would have changed the neighborhood-expectation this much" on a reviewer
card.

**`train_incremental()`** (line 1074) — the yearly CPU fine-tune path:
freezes the RGCN encoder by default (`freeze_rgcn=True`) and only updates
the MLP predictor heads, unless there are ≥50 confirmed-fraud examples (see
`retraining_orchestrator.py`, §7) — below that, there isn't enough signal to
safely fine-tune the graph encoder itself without risking overfitting to a
handful of labels.

---

## 3. Detector 2 — Dense-block (FRAUDAR-style)

### `src/dense_block_detector_v3.py` (~220 lines)

A structural specialist for exactly the case Hybrid GraphMCM is weak at:
dense fraud rings reconstruct *too easily* (a tight clique's members all
look like each other, so a graph-conditioned reconstruction model finds
nothing surprising about them). This module runs Charikar greedy peeling —
repeatedly remove the minimum-weighted-degree node, track density at each
step — separately per relation (mobile, IP), then combines the two via a
priority-weighted max.

`_charikar_peeling()` (line 40) assigns each node the **prefix max** density
over the peeling steps it was present for:

> verified on a triangle+pendant toy case (comment at lines 91–95) — prefix,
> not suffix, max is the version that correctly scores a node still attached
> when the graph was at its densest, even if later peeling steps (after that
> node was already removed) reached an even higher density among the
> remaining core.

`dense_block_scores()` (line 107) produces one score per relation plus the
combined `dense_block_score_relational` using
`DENSE_BLOCK_RELATION_WEIGHTS = {0: 0.3, 1: 1.0}` (mobile : IP,
`config_v3.py`).

**Why IP is weighted more than 3x mobile, and why this isn't equal
weighting**: equal weighting was tried first and produced a *higher*
overall PR-AUC (0.268) — but it let ordinary, non-fraud mobile/pincode
density outrank true IP-ring members, which collapsed IP-specific PR-AUC to
0.067. Since IP sharing is the dominant real fraud vector in this domain,
that regression was judged unacceptable even though the aggregate number
looked better — a reminder that the single best-looking number isn't always
the right one to optimize when it's hiding a regression on the case that
matters most.

---

## 4. Detector 3 — Subspace Isolation Forest

### `src/subspace_if_v3.py` (~96 lines)

The backbone detector — unsupervised, no graph dependency, so it's the one
that still works for isolated nodes with zero edges. Fits one independent
`IsolationForest` per feature subgroup (`financial`, `identity`, `network` —
`config_v3.SUBSPACE_GROUPS`), then aggregates by per-group max.

```python
clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_SEED)
clf.fit(X)

# decision_function returns higher = more normal; negate so higher = more anomalous
raw_scores = clf.decision_function(X)
anomaly_scores = -raw_scores
```

That comment is doing real work: sklearn's `IsolationForest.decision_function`
convention (higher = more normal) is the **opposite** of this whole
codebase's convention (hard stop #3: higher = more anomalous, everywhere).
Missing that negation would silently invert every subspace score without
raising any error — the model would still "work," just backwards, which is
exactly the kind of bug that's invisible until someone notices the
supposedly-safest applications are flagged as most suspicious.

`compute_subspace_if_scores()` (line 24) is a **pure function** (no file
I/O) — reused unchanged by both the committed pipeline
(`run_subspace_if()`) and the read-only cohort-preview API path, so a
preview score is computed by the literal same code as a committed one, not
a re-implementation that could drift.

---

## 5. Fusion (LOCKED)

### `src/fusion_classifier_v3.py` (~122 lines)

The single combination point for the three production detectors. This file
is short because the decision is simple — the complexity is entirely in
*why* it's this simple:

```python
def score_level_fusion(subspace_if_score, dense_block_score_relational, hybrid_anomaly_score) -> np.ndarray:
    """
    risk = minmax( max( minmax(subspace), minmax(dense_relational), minmax(hybrid) ) )
    ...
    Label-independent: no learned gate can bury a strong raw signal — max is
    stricter about this than the old sum was, since it preserves the single
    strongest detector's value exactly regardless of the other two.
    """
    s = _minmax(subspace_if_score)
    d = _minmax(dense_block_score_relational)
    h = _minmax(hybrid_anomaly_score)
    combined = np.maximum.reduce([s, d, h])
    return _minmax(combined)
```

**Why max, not a weighted sum, and definitely not a learned combiner**: an
earlier additive weighted-sum diluted whichever single detector actually
found a given fraud shape — e.g. on a mobile-ring stress test, the Subspace
IF alone scored 0.674 PR-AUC, but the weighted-sum *fused* score dropped to
0.349, because averaging in two detectors that had no signal on that
specific ring pulled the one detector that did have signal back down.
Switching the combination function to max raised overall PR-AUC 0.403→0.447.
This is a separate, earlier decision from the LightGBM-learned-combiner
rejection (`docs/HISTORY.md` — LightGBM as a *learned* fusion layer
destroyed 14 real positive labels by learning the wrong decision boundary
from too few labels; that's a different failure mode than the additive-sum
dilution problem this file's docstring is about).

`FUSION_COMPONENTS` in `config_v3.py` documents that these are the **only**
three inputs — Deep SAD (`deepsad_detector_v3.py`, §6) is deliberately
absent, tested in a candidate 4-way fusion, and rejected on evidence (see
next section).

---

## 6. Supplementary / XAI-only signal — Deep SAD

### `src/deepsad_detector_v3.py` (~173 lines)

A second, architecturally independent detector (own 2-layer RGCN encoder,
own checkpoint `models/deepsad_v3.pth`) implementing Deep SAD (Ruff et al.,
ICLR 2020) — no reconstruction loss at all, just a center-pull/exposure-push
objective:

```python
loss_normal  = dist_real.mean()
loss_anomaly = (DEEPSAD_ETA / (dist_synth + 1e-6)).mean()
loss = loss_normal + loss_anomaly
```

**Why it exists at all if it's not in fusion**: it's genuinely the strongest
*single* relational signal found in this codebase's stress testing — 0.201
overall / 0.093 mobile-ring / 0.050 IP-ring, each individually higher than
any of the three fusion detectors alone on those categories. It was tested
in a candidate 4-way max fusion and **rejected on evidence, not by default**:
it only won the fusion argmax (i.e. was the single highest of the 4 scores)
in fewer than 1% of nodes — the existing 3-detector trio already covers its
specialties too well for a 4th input to move the aggregate. So it stays
XAI-card-only: reviewers see it as a "second opinion (supplementary)" signal
on the card, but it never gates a threshold or a fusion decision. This is a
good example of "a component being individually strong doesn't automatically
mean it belongs in the ensemble" — the actual test is marginal contribution,
not standalone performance.

---

## 7. Thresholds, self-training, retraining

### `src/evt_scorer_v3.py` (~152 lines)

Fits a Generalized Pareto Distribution to the extreme right tail of each
score distribution (Peaks-Over-Threshold), giving statistically-derived
flagging thresholds instead of hand-picked cutoffs — this is the *only*
thresholding mechanism the project allows (hard stop #1).

```python
Q            = 0.002   # false-positive rate (only human-set value in this module)
```
Everything else — the percentile the tail starts at, the GPD shape/scale,
the final threshold — is fit from data. `_fit_evt()` (line 43) jitters
scores by tiny noise before fitting (breaks exact ties that would otherwise
distort the tail fit), and falls back to an empirical quantile if the fitted
shape parameter is outside `[EVT_SHAPE_MIN, EVT_SHAPE_MAX] = [-0.5, 1.0]` or
there are fewer than 10 exceedances — a GPD fit on too few points or with an
invalid shape isn't trustworthy, so the code has an explicit fallback rather
than silently trusting a bad fit.

### `src/self_training_loop_v3.py` (~198 lines)

Round 0 promotes an application to pseudo-positive only if it clears at
least `MIN_SIGNALS_FOR_PROMOTION = 2` of 5 independent EVT tail thresholds:

```python
signal_count = sum(...)   # 5 independent EVT signal flags
promoted_mask = signal_count >= MIN_SIGNALS_FOR_PROMOTION
```

**Why require 2 signals, not any 1**: reduces confirmation bias from
single-signal noise — the code comment gives a concrete example: an income
of ₹5 (a data-entry error, not fraud) would trigger `EVT_FINANCIAL` alone
but look completely normal on every other dimension. Requiring agreement
across independent signals is cheap insurance against exactly the kind of
mistake the project's own "Known Structural Weaknesses" table (`CLAUDE.md`)
warns about — "if the EVT tail is data-entry errors, the classifier anchors
on typos."

The module's own hard stop (lines 16–17), mirrored in `main_v3.py`:
> rounds do not advance automatically. Each round requires a Phase D PR-AUC
> check by the project lead before its label set is used for training.

### `src/retraining_orchestrator.py` (~311 lines)

The yearly-cycle drift/retrain decision point. `_check_drift()` (line 74)
runs a KS test between this cycle's and the previous cycle's
`hybrid_anomaly_score` distributions; `p < DRIFT_KS_THRESHOLD` (0.01)
recommends a full retrain instead of an incremental one.

```python
freeze_rgcn = n_confirmed < 50
```
**Why 50**: below 50 confirmed-fraud examples, there isn't enough real
signal to safely fine-tune the graph encoder's weights without risking it
overfitting to a small, possibly-unrepresentative handful of labels — so
only the MLP predictor heads update. Above 50, the encoder itself
unfreezes. This is the same caution as the self-training 2-signal rule:
don't let a model update on too little evidence.

### `src/checkpoint_manager.py` (~164 lines)

`validate_and_hotswap()` (line 61) is the **only** authorized path to
replace the live checkpoint (hard stop #9 — never `torch.save` onto the
live path). It validates the checkpoint dict has exactly
`{model_state_dict, centroid, config}` with `config`'s `N_FEATURES` /
`GRAPH_EMB_DIM` / `N_EDGE_TYPES` matching the running `config_v3.py` before
touching anything — a dimension-mismatched checkpoint (e.g. from a stale
feature schema after a features change) can never overwrite the live model;
validation failure leaves the live checkpoint completely untouched, backed
up, and versioned.

---

## 8. Explainability (XAI)

### `src/xai_layer_v3.py` (~1,555 lines) + `src/xai_card_html_v3.py` (~1,358 lines)

`xai_layer_v3.py` computes every number a reviewer card shows; `xai_card_html_v3.py`
renders those numbers into HTML and computes nothing new — a clean
separation that matters for auditability: if a claim on a card is wrong, the
bug is in exactly one of these two files depending on whether it's a wrong
*number* or a wrong *rendering*.

The project's XAI design rule (module docstring, `xai_layer_v3.py` lines
6–17):
> No hand-set narrative thresholds: the only numeric gates quoted are
> EVT-derived; everything else is stated as a computed percentile.

`build_fusion_contributions()` (`xai_layer_v3.py` line 387) is the direct
consequence of max-fusion having no natural "percentage contribution" (a sum
does, a max doesn't): attribution is exact — whichever detector's own
minmax value equals the fused max **is** the driver, full stop, plus a
`margin_over_next` showing how decisively it won. This is why the fusion
section of a reviewer card shows a "WON DRIVER" badge on exactly one
detector rather than a stacked-percentage bar chart — the underlying math
doesn't support a percentage breakdown, so the UI doesn't pretend it does.

`DETECTOR_PILL["deepsad"]` (`xai_card_html_v3.py` line 71) is deliberately
excluded from `DETECTOR_ORDER` (the 3 locked fusion inputs) so the
fusion-composition footer always shows exactly the 3 real fusion inputs,
while Deep SAD still gets its own visual identity elsewhere on the card as
a clearly-labeled non-fusion "second opinion."

### `src/evaluate_model_v3.py` (~547 lines)

The synthetic evaluation harness — independent of training, injects 5
fresh (unseen-seed) archetype anomalies and checks PR-AUC against hard
regression floors (`V2_FLOORS`). One subtlety worth knowing (module
docstring, lines 9–20): every injected eval node is **isolated** (zero
edges), and Hybrid GraphMCM's `isolated_embedding` is one fixed learned
vector shared by every isolated node — so feature-reconstruction error is
nearly *constant* across different injected anomalies and can't rank them
against each other. The harness works around this by scoring each category
with the one Subspace IF group that targets its archetype (e.g. IP
concentration → the "network" group), never the hybrid score directly, for
this specific evaluation. This is a good illustration of why understanding
a detector's blind spot (isolated nodes) matters even for writing its own
test harness correctly.

---

## 9. Stores (JSON, dual-written to Postgres)

Brief — these hold state, not ML logic, but every detector above reads
labels from them:

- **`src/confirmed_fraud_store.py`** (~209 lines) — supervisor-confirmed
  fraud/false-positive labels; `get_exposure_tensor()` (line 168) is what
  feeds real confirmed-fraud examples into the LOE push-away term alongside
  synthetic exposure once ≥5 real examples exist.
- **`src/confirmed_fraud_graph_store.py`** (~306 lines) — confirmed fraud
  *ring* patterns, `FLAGGED → CONFIRMED → SELECTED → PROMOTED`; `promote()`
  (line 114) is what appends a real ring's edges into the topology exposure
  set so the RGCN's relational LOE term actually learns it.
- **`src/model_registry.py`** (~136 lines) — the MLflow-free run-tracking
  log (`log_run()`, line 69) behind the console's Run History panel.

### `src/db/` — Postgres-side SQL modules (V4-Scale migration, brief)

- `bootstrap.py` — one-shot container startup: schema + primary ingest.
- `connection.py` — pooled connections, `.env` config.
- `features.py` — SQL replicas of the pandas feature/graph logic, built to
  be bit-exact with `tabular_feature_engine_v3.py`/`graph_builder_v3.py`
  (see `docs/MAINTAINER_PLAYBOOK.md` Recipe 6 for the dual-path discipline
  this requires).
- `ingest.py` — stages raw batches, populates derived tables on
  Evaluate/Merge (the admin-gated contract — see `docs/IMPLEMENTATION.md`
  Step 3).
- `migrate.py` — schema migration runner.
- `reads.py` — Postgres-vs-file read toggle for the API.
- `stores.py` — Postgres mirrors of the JSON stores above.

This layer is infrastructure for the *migration*, not detection logic — see
`docs/TECHNICAL_REFERENCE_AND_SCALING.md` if you need to go deeper here.

---

## Quick map: "I want to understand X, start where?"

| Question | Start here |
|---|---|
| How is a single application scored end to end? | `hybrid_graphmcm_v3.py::compute_score_frame()`, then `fusion_classifier_v3.py::score_level_fusion()` |
| Why is isolated-node handling safe? | `hybrid_graphmcm_v3.py::HybridGraphMCM.encode_graph()` (§2) |
| Why does fusion use max, not a weighted sum? | `fusion_classifier_v3.py` docstring (§5) |
| Why isn't Deep SAD in fusion if it's the strongest signal? | `deepsad_detector_v3.py` docstring (§6) |
| Where do thresholds come from? | `evt_scorer_v3.py::_fit_evt()` (§7) |
| How does a number get from a score to a sentence on a card? | `xai_layer_v3.py` → `xai_card_html_v3.py` (§8) |
| What's actually locked vs. still open? | `docs/AGENTS.md` §4 (hard stops) and §7 (open decisions) |
