# ML Codebase Guide — NIC Scholarship Fraud Detection (V3/V4)

<!-- OWNER: not lead-owned — explanatory reference, safe to extend. -->
<!-- Scope: the detection/ML core only (src/*_v3.py + main_v3.py). API, -->
<!-- serving, console, and Postgres-migration code are covered in -->
<!-- docs/TECHNICAL_REFERENCE_AND_SCALING.md and docs/OPERATIONS_RUNBOOK.md. -->

## What this document is for

Say you've just been handed this codebase and asked to explain it to someone
else. That's the situation this guide is written for. It walks through the
ML-relevant files one at a time, in the order the pipeline actually runs
them, and for each one covers the functions and classes that matter, real
snippets from the code, and — this is the part that matters most — *why*
each piece is built the way it is.

That "why" is usually the hardest thing to recover just by reading code. A
lot of what looks like an arbitrary constant, or a strange one-line comment,
is actually the fossil record of an ablation that failed in some specific,
informative way before the current version was settled on. This guide tries
to preserve that history alongside the code itself, so you're not left
guessing why `root_weight=False` or `LAMBDA_EDGE_SCORE = 0.0` are there.

If you want the architecture explained in prose first — what the system does
before you look at how — start with `docs/AGENTS.md` and Part I of
`docs/TECHNICAL_REFERENCE_AND_SCALING.md`. This guide assumes you already
have that picture and takes you into the code itself.

**How to read it**: the sections below follow the pipeline's actual
execution order (it mirrors `PIPELINE_STEPS` in `main_v3.py`). Each
detector's section is mostly self-contained, but the fusion and XAI sections
near the end assume you've read all three detector sections first, since
they depend on all three.

---

## 0. Orchestration and configuration

### `main_v3.py` (project root, ~344 lines)

This is the pipeline's entry point. The function that matters most is
`run_pipeline(steps=None, smoke_test=False, cycle="manual")` at line 165,
which runs all 14 steps in order: feature engineering, graph building,
synthetic exposure, Hybrid GraphMCM training, relation ablation, Subspace
IF, dense-block, Deep SAD, EVT, self-training, fusion, XAI, and evaluation
— all wrapped inside an MLflow experiment run backed by SQLite, so it keeps
working no matter what directory you launch it from. `should_run(name)`
gates each of those steps individually, so you can run a subset with
something like `--steps=build_base,train_hybrid`.

One thing worth noticing here: line 17 puts a hard stop directly into the
orchestrator's code, not just into a policy document —

```python
HARD STOP: self-training round advancement requires Phase D PR-AUC check by project lead.
           This orchestrator always runs round=0. Do not modify to auto-advance.
```

That's a deliberate choice. A policy doc can get skimmed past by someone
editing code six months from now; a docstring sitting right at the top of
the function they'd have to change is much harder to miss.

### `src/config_v3.py` (~300 lines)

Every hyperparameter, every feature list, every architecture switch lives
here — this is the single source of truth. Almost nothing else in the
codebase hard-codes a number; everything imports from this file instead.
If you're ever asking "what does the system currently do," this is the file
to read first, because it's where every locked decision and every
still-open ablation switch (`V4_ENCODER_ARCH`, `V4_TOPO_EXPOSURE`,
`V4_RING`, and so on) actually lives — usually with a one-line comment
citing the ablation that settled it.

Two constants here are worth understanding right away, because they show up
everywhere downstream:

```python
LAMBDA_EDGE = 0.3          # training-loss weight — edge-prediction head still trains
LAMBDA_EDGE_SCORE = 0.0    # inference SCORE weight — decoupled from the above
```

Why two constants for what sounds like one knob? `edge_pred_error` — how
well the model predicts which relations a node is connected by — turned out
to have no usable ranking signal on any of the three designed relational
fraud categories: held-out PR-AUC of 0.011–0.023, which is at or below what
a random ranking would score. Worse, on one category it actively diluted the
other detector term's real signal, dropping it from 0.268 standalone to
0.017 once combined. Ripping the edge-prediction head out of training
entirely would have been the bigger, riskier fix, and it would have cost the
head's value as an XAI signal too. So instead, the head keeps training, but
its contribution to the actual score is zeroed out — which is why there are
two separate constants instead of one.

Also worth flagging: `FUSION_COMPONENTS = ("subspace", "dense_relational",
"hybrid")` around line 286 documents the three — and only three — inputs to
the locked fusion. More on that in §5.

---

## 1. Feature engineering

### `src/tabular_feature_engine_v3.py` (~398 lines)

The build happens in two stages. `build_base()` (line 263) produces a
63-feature CSV, before any graph-degree features exist yet.
`add_degree_features()` (line 305) then merges in the five relation-degree
columns from the graph builder to produce the final 44-feature model input
— after the 24 nominal-identifier columns get dropped here — along with
`data/processed/v3_feature_schema.json`. That schema file is what every
other module reads feature order from, so the ordering never has to be
hard-coded twice.

There's also a parallel Postgres-backed path for the V4-Scale migration:
`build_base_pg()` (line 282) and `apply_stored_scaling()` (line 233).
Same output, just sourced from Postgres instead of a file, and verified
bit-exact against the file path (see `docs/IMPLEMENTATION.md` Gate 4).
`apply_stored_scaling()` deliberately **raises** rather than silently
refitting when scaler params for a given schema version are missing — that's
hard stop #11: scaler params get fit once and persisted, never refit on a
scoring batch, because refitting on a batch's own statistics is a data leak.

One precision detail is worth calling out (around lines 251–253), because it
looks like paranoia right up until you've been burned by it:

```python
# numpy, not pandas Series arithmetic: pandas routes >10k-element ops
# through numexpr, which rounds the mul/add chain 1 ULP differently --
# numpy x*scale_+min_ is bit-identical to sklearn's transform.
```

That's exactly why the Postgres-path parity gate in `docs/IMPLEMENTATION.md`
Step 4 could claim genuinely **bit-exact** results, rather than "close
enough." This codebase cares about floating-point reproducibility down to
the ULP in the places where two code paths are supposed to be identical —
because a silent divergence there wouldn't throw an error, it would just let
scores quietly drift apart over time.

### `src/graph_builder_v3.py` (~228 lines)

This file builds the five-relation identity graph as a PyG `HeteroData`
object: any two applications that share a raw value — mobile number, IP,
father's name, mother's name, or pincode — get an edge between them.
`build_graph()` (line 58) is the main entry point; the actual pairing logic
lives in `_build_edges()` (line 38), which currently builds a full clique
per shared value (so cost grows O(n²) per group).

For scale, there's `build_graph_pg()` (line 178), the SQL equivalent, which
adds hub-capping: `_edges_from_groups()` (line 139) turns oversized groups
into a star instead of a clique once they exceed `k_cap`, and skips groups
entirely once they exceed a `ceiling`. That ceiling isn't hand-picked —
`derive_group_ceiling()` (line 170) computes it from the *observed*
group-size distribution (the 99.9th percentile, by default), which is what
hard stop #1 requires: no rule-like threshold set against a domain concept.

Both caps default to `None` — off — and that's deliberate, not an
oversight. It's called out explicitly in the code:

```python
# BOTH CAPS DEFAULT OFF ... K_CAP is open decision #1 (lead-owned,
# needs 3.5M profiling)
```

The machinery for capping is built and validated, but choosing the actual
cap value needs real group-size data at 3.5M scale, which doesn't exist yet
on the 15k dataset. Picking a number now, without that data, would be
exactly the kind of hand-picked threshold hard stop #1 rules out.

### `src/synthetic_exposure_builder_v3.py` (~308 lines)

This builds the **programmatic** LOE (exposure) set — the examples the
detector is trained to push away from. It's 750 rows across five
archetypes (IP concentration, mother-name collision, fee inflation, age
violation, income violation), each one a real row's features perturbed on
that archetype's target columns. `build_exposure_set()` (line 194) is the
main function. `_add_context_noise()` (line 47) additionally perturbs a
random 25% of the *non-target* columns, so exposure rows don't end up
suspiciously identical to a real row everywhere except one signal — if they
were, the model could learn "identical except one field" as its tell,
rather than the actual shape of the fraud. `build_topology_exposure()`
(line 224) is the graph-structured counterpart: it builds synthetic
connected cliques that feed the RGCN's relational LOE term, saved to
`synthetic_exposure_graph_v3.pt`.

Why isn't any of this a GAN? That's hard stop #7. CTGAN, TVAE, and
copula-based generators were all tried and explicitly rejected. With only
15k real applicants, a learned generative model risks memorizing and
leaking real individuals' data into the synthetic fraud examples — and
there's something uncomfortably circular about a rule-free system learning
"what fraud looks like" from a generative model that was itself fit on the
same population it's supposed to be catching fraud in. Perturbing real rows
directly is auditable and bounded in a way CTGAN-family output just isn't.

---

## 2. Detector 1 — Hybrid GraphMCM (RGCN)

### `src/hybrid_graphmcm_v3.py` (~1,220 lines — the largest, most important file)

This is the core learned detector. The idea: reconstruct each application's
own (masked) features, conditioned on its graph neighborhood. An
application the model can't reconstruct well — or whose neighborhood looks
unlike its own declared attributes — scores as anomalous. Training happens
in two stages: Stage 1 warms up a DeepSVDD-style centroid together with a
graph-side LOE push-away term; Stage 2 is free joint feature/edge
reconstruction with a lighter, but *persistent*, LOE term.

**`RGCNEncoder`** (class, line 90) is two `RGCNConv` layers:

```python
self.conv1 = RGCNConv(N_FEATURES, GRAPH_HIDDEN, num_relations=N_EDGE_TYPES, aggr="add", root_weight=False)
self.conv2 = RGCNConv(GRAPH_HIDDEN, GRAPH_EMB_DIM, num_relations=N_EDGE_TYPES, aggr="add", root_weight=False)
```

`forward()` (line 101) just runs `h = tanh(conv1(...)); h = tanh(conv2(...))`.

The `root_weight=False` setting is arguably the single most important
non-obvious line in the whole codebase, and it's commented in place at
lines 87–99. By default, `RGCNConv` sets `root_weight=True`, which adds a
learned self-transform of each node's *own unmasked* features straight into
its output embedding — completely bypassing the masked-channel (MCM)
mechanism that `_apply_masks()` is supposed to enforce. In other words, with
the default left on, the "graph-conditioned" embedding could just smuggle
in the answer the model is supposed to be predicting from context alone.
Turning it off measurably mattered: on a stress test, overall PR-AUC went
0.153 → 0.201, mobile-ring 0.029 → 0.078, IP-ring 0.032 → 0.055, with no
regression at low degree.

**`HybridGraphMCM`** (class, line 219) implements the masked-channel
mechanism itself:

```python
self.mask_logits = nn.Parameter(torch.randn(MASK_NUM, N_FEATURES))   # K=8 learned masks
...
def _apply_masks(self, x: torch.Tensor) -> torch.Tensor:
    """Average over K soft-masked versions of x."""
    masks = torch.softmax(self.mask_logits, dim=1)  # (K, N_FEATURES)
    masked = x.unsqueeze(0) * masks.unsqueeze(1)     # (K, B, N_FEATURES)
    return masked.mean(dim=0)                         # (B, N_FEATURES)
```

Eight learned, softmax-normalized masks get averaged together. The model
itself learns *which* features to hide before trying to reconstruct them
from context — rather than someone hand-picking a masking scheme up front.

`encode_graph()` (line 260) handles isolated nodes — applications with zero
edges. Rather than let the RGCN produce something meaningless off an
edgeless input, a node with no edges gets substituted with a **trainable**
`isolated_embedding` parameter:

```python
def encode_graph(self, x, edge_index_list, edge_type_tensor, isolated_mask) -> torch.Tensor:
    h = self.encoder(x, edge_index_list, edge_type_tensor)
    iso_emb = self.isolated_embedding.unsqueeze(0).expand(h.shape[0], -1)
    mask_exp = isolated_mask.unsqueeze(1).expand_as(h)
    return torch.where(mask_exp, iso_emb, h)
```

This is *why* isolated nodes aren't simply broken by this detector — they
get a real, trained fallback embedding instead of a zero vector or garbage
from an encoder that has no neighbors to aggregate over.

**`compute_score_frame()`** (line 493) is the single source of truth for the
score formula. Training, incremental fine-tuning, and the API's read-only
scoring path all call this same function, so the three can never silently
diverge from each other:

```python
per_feat_err       = (pred_x - x_all).abs()          # (N, N_FEATURES)
feature_pred_error = per_feat_err.mean(dim=1)        # (N,)
...
edge_pred_error = F.binary_cross_entropy(edge_prob, target, reduction="none").mean(dim=1)

# LAMBDA_EDGE_SCORE (0.0), not LAMBDA_EDGE (0.3, still the training-loss weight)
hybrid_anomaly_score = feature_pred_error + LAMBDA_EDGE_SCORE * edge_pred_error
```

Why is `feature_pred_error` just a per-feature mean absolute error, and not
something more sophisticated? Because it's directly interpretable — the XAI
layer pairs each feature's predicted value against its actual declared
value and states something like "expected X, declared Y" with a direction.
That only works cleanly because the error metric *is* a literal difference
in the same scaled feature space, not a compressed embedding distance that
would need translating back into something human-readable.

**`train()`** (line 737) runs the two-stage loop:

- **Stage 1** (`EPOCHS_STAGE1`): `loss = svdd_loss + loe`, where
  `svdd_loss = torch.norm(h_n - centroid).mean()` pulls real embeddings
  toward a DeepSVDD centroid, and `loe` (`_loe_loss()`, line 467) pushes
  synthetic exposure embeddings away, with a weight that decays over the
  stage: `lam_t = LAMBDA_EXPOSURE * (1 - epoch/epochs_s1)`.
- **Stage 2** (`EPOCHS_STAGE2`): `loss = feat_loss + LAMBDA_EDGE * edge_loss
  + loe_s2` — free joint reconstruction, but now with a *persistent*
  (non-decaying) LOE term, weighted by `LOE_STAGE2_WEIGHT`.

Why did Stage 2 need its own persistent LOE term? This is the kind of
regression you only catch by actually testing, not by reasoning about the
math on paper. Stage 2 originally had zero exposure term at all — pure
reconstruction. But 120 epochs of unconstrained reconstruction can slowly
re-absorb whatever separation Stage 1 built between real and
synthetic-fraud embeddings, because dense synthetic cliques are,
mechanically, *easy* to reconstruct. (That's the same reason
`dense_block_detector_v3.py` exists as a separate specialist — see §3.)
Adding a light, non-decaying LOE weight in Stage 2 keeps that separation
from silently eroding over the longer training stage.

**`_loe_loss()`** (line 467) is a hinge loss on distance-to-centroid:

```python
clamp(margin - dist, min=0).mean() * lam
```

The `margin` here is **data-derived** (`_derive_loe_margin()`, line 447),
replacing an earlier fixed constant, `LOE_MARGIN=2.0`. That fixed value
turned out to be roughly 3x smaller than the real population's own median
embedding-to-centroid distance at `GRAPH_EMB_DIM=64` (median ≈ 5.9) — which
meant exposure examples were already "past the margin," and contributing
zero loss, before training even started. An earlier `exp(-sqrt(dist))` loss
formulation was tried before this and rejected outright: it saturated to
roughly zero within the first few epochs, at both 30- and 150-epoch
budgets, meaning it wasn't a usable training signal at any reasonable
schedule.

**`compute_relation_ablation()`** (line 618) re-scores the *already
trained, locked* checkpoint five separate times, masking out one relation's
edges from `edge_type_tensor` on each pass, to see how much each relation's
presence changes the reconstruction. This is strictly XAI-only — it never
feeds a threshold or the fusion score. Its whole job is to narrate, on a
reviewer card, something like "removing this relation would have changed
the neighborhood-expectation this much."

**`train_incremental()`** (line 1074) is the yearly CPU fine-tune path. By
default it freezes the RGCN encoder (`freeze_rgcn=True`) and only updates
the MLP predictor heads — unless there are 50 or more confirmed-fraud
examples on hand (see `retraining_orchestrator.py`, §7). Below that count,
there just isn't enough signal to safely fine-tune the graph encoder itself
without risking overfitting to a small handful of labels.

---

## 3. Detector 2 — Dense-block (FRAUDAR-style)

### `src/dense_block_detector_v3.py` (~220 lines)

This is a structural specialist built for exactly the case Hybrid GraphMCM
is weakest on: dense fraud rings reconstruct *too easily*. A tight clique's
members all look like each other, so a graph-conditioned reconstruction
model finds nothing surprising about any of them. This module instead runs
Charikar greedy peeling — repeatedly removing the minimum-weighted-degree
node and tracking density at each step — separately per relation (mobile,
IP), then combines the two relations via a priority-weighted max.

`_charikar_peeling()` (line 40) assigns each node the **prefix max**
density over the peeling steps it was present for. This was verified on a
triangle-plus-pendant toy case (see the comment at lines 91–95): prefix max,
not suffix max, is the version that correctly scores a node that was still
attached when the graph was at its densest — even if a later peeling step,
after that node had already been removed, reached an even higher density
among whatever core remained.

`dense_block_scores()` (line 107) produces one score per relation, plus the
combined `dense_block_score_relational`, using
`DENSE_BLOCK_RELATION_WEIGHTS = {0: 0.3, 1: 1.0}` (mobile : IP, defined in
`config_v3.py`).

Why weight IP more than 3x mobile, instead of weighting them equally? Equal
weighting was actually tried first, and it produced a *higher* overall
PR-AUC (0.268) — but it let ordinary, non-fraud mobile/pincode density
outrank true IP-ring members, which collapsed IP-specific PR-AUC down to
0.067. Since IP sharing is the dominant real fraud vector in this domain,
that regression was judged unacceptable even though the aggregate number
looked better on paper. It's a good reminder that the single best-looking
number isn't always the right one to optimize for, especially when it's
hiding a regression on the exact case that matters most.

---

## 4. Detector 3 — Subspace Isolation Forest

### `src/subspace_if_v3.py` (~96 lines)

This is the backbone detector: unsupervised, with no graph dependency, so
it's the one detector that still works for isolated nodes with zero edges.
It fits one independent `IsolationForest` per feature subgroup
(`financial`, `identity`, `network` — see `config_v3.SUBSPACE_GROUPS`),
then aggregates across groups by taking the per-group max.

```python
clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_SEED)
clf.fit(X)

# decision_function returns higher = more normal; negate so higher = more anomalous
raw_scores = clf.decision_function(X)
anomaly_scores = -raw_scores
```

That comment is doing real work. sklearn's `IsolationForest.decision_function`
convention — higher means more normal — is the exact opposite of this
codebase's own convention (hard stop #3: higher always means more
anomalous, everywhere). Miss that negation and every subspace score gets
silently inverted, with no error raised anywhere. The model would still
"work," just backwards — exactly the kind of bug that stays invisible until
someone notices the supposedly-safest applications are the ones getting
flagged as most suspicious.

`compute_subspace_if_scores()` (line 24) is a **pure function** — no file
I/O — and it's reused unchanged by both the committed pipeline
(`run_subspace_if()`) and the read-only cohort-preview API path. That means
a preview score is computed by the exact same code as a committed one,
never a separate re-implementation that could quietly drift from it.

---

## 5. Fusion (LOCKED)

### `src/fusion_classifier_v3.py` (~122 lines)

This is the single point where the three production detectors get combined.
The file itself is short, because the actual decision is simple — nearly
all of the complexity here is in *why* it's this simple:

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

Why max, and not a weighted sum — or, for that matter, a learned combiner
at all? An earlier additive weighted-sum approach diluted whichever single
detector had actually found a given fraud shape. On a mobile-ring stress
test, for instance, Subspace IF alone scored 0.674 PR-AUC, but the
weighted-sum *fused* score dropped all the way to 0.349 — because averaging
in two detectors that had no signal on that specific ring dragged the one
detector that did have signal back down with them. Switching the
combination function to max raised overall PR-AUC from 0.403 to 0.447.

This was a separate, earlier decision from the LightGBM-learned-combiner
rejection you'll find in `docs/HISTORY.md`. That one was about LightGBM as
a *learned* fusion layer destroying 14 real positive labels by learning the
wrong decision boundary from too few examples — a different failure mode
from the additive-sum dilution problem this file's docstring is describing.

`FUSION_COMPONENTS` in `config_v3.py` documents that these three are the
**only** inputs. Deep SAD (`deepsad_detector_v3.py`, §6) is deliberately
left out — it was tested in a candidate 4-way fusion and rejected on the
evidence, which the next section covers.

---

## 6. Supplementary / XAI-only signal — Deep SAD

### `src/deepsad_detector_v3.py` (~173 lines)

This is a second, architecturally independent detector — its own two-layer
RGCN encoder, its own checkpoint (`models/deepsad_v3.pth`) — implementing
Deep SAD (Ruff et al., ICLR 2020). It has no reconstruction loss at all,
just a center-pull/exposure-push objective:

```python
loss_normal  = dist_real.mean()
loss_anomaly = (DEEPSAD_ETA / (dist_synth + 1e-6)).mean()
loss = loss_normal + loss_anomaly
```

If it's not part of fusion, why does it exist at all? Because it's
genuinely the strongest *single* relational signal found anywhere in this
codebase's stress testing — 0.201 overall, 0.093 mobile-ring, 0.050
IP-ring, each individually higher than any of the three fusion detectors
scored alone on those same categories. It was tested in a candidate 4-way
max fusion and **rejected on evidence, not by default**: it only won the
fusion argmax — was the single highest of the four scores — on fewer than
1% of nodes. The existing three-detector trio already covers its
specialties too well for a fourth input to move the aggregate result.

So it stays XAI-card-only. Reviewers see it as a "second opinion
(supplementary)" signal on the card, but it never gates a threshold or a
fusion decision. It's a good illustration of a general point: a component
being individually strong doesn't automatically mean it belongs in the
ensemble. The test that actually matters is marginal contribution, not
standalone performance.

---

## 7. Thresholds, self-training, retraining

### `src/evt_scorer_v3.py` (~152 lines)

This fits a Generalized Pareto Distribution to the extreme right tail of
each score distribution (Peaks-Over-Threshold), which gives
statistically-derived flagging thresholds instead of hand-picked cutoffs.
This is the *only* thresholding mechanism the project allows (hard stop
#1).

```python
Q            = 0.002   # false-positive rate (only human-set value in this module)
```

Everything else in this module — where the tail starts, the GPD shape and
scale, the final threshold — is fit from data. `_fit_evt()` (line 43)
jitters scores with tiny noise before fitting, which breaks exact ties that
would otherwise distort the tail fit, and falls back to an empirical
quantile if either the fitted shape parameter falls outside
`[EVT_SHAPE_MIN, EVT_SHAPE_MAX] = [-0.5, 1.0]` or there are fewer than 10
exceedances to fit against. A GPD fit on too few points, or with an invalid
shape, just isn't trustworthy — so the code has an explicit fallback rather
than silently trusting a bad fit.

### `src/self_training_loop_v3.py` (~198 lines)

Round 0 promotes an application to pseudo-positive only if it clears at
least `MIN_SIGNALS_FOR_PROMOTION = 2` of 5 independent EVT tail thresholds:

```python
signal_count = sum(...)   # 5 independent EVT signal flags
promoted_mask = signal_count >= MIN_SIGNALS_FOR_PROMOTION
```

Why require two signals instead of just one? It's cheap insurance against
confirmation bias from single-signal noise. The code comment gives a
concrete example: an income of ₹5 is almost certainly a data-entry error,
not fraud, but it would trigger `EVT_FINANCIAL` on its own while looking
completely normal on every other dimension. Requiring agreement across
independent signals guards against exactly the failure mode the project's
own "Known Structural Weaknesses" table (in `CLAUDE.md`) warns about: "if
the EVT tail is data-entry errors, the classifier anchors on typos."

The module carries its own hard stop too (lines 16–17), mirrored in
`main_v3.py`: rounds do not advance automatically. Each round requires a
Phase D PR-AUC check by the project lead before its label set is used for
training.

### `src/retraining_orchestrator.py` (~311 lines)

This is the decision point for the yearly retrain cycle. `_check_drift()`
(line 74) runs a KS test comparing this cycle's `hybrid_anomaly_score`
distribution against the previous cycle's; a p-value below
`DRIFT_KS_THRESHOLD` (0.01) recommends a full retrain instead of an
incremental one.

```python
freeze_rgcn = n_confirmed < 50
```

Why 50? Below 50 confirmed-fraud examples, there isn't enough real signal to
safely fine-tune the graph encoder's weights without risking overfitting to
a small, possibly-unrepresentative handful of labels — so only the MLP
predictor heads get updated. Above 50, the encoder itself unfreezes. It's
the same underlying caution as the self-training two-signal rule: don't let
the model update itself on too little evidence.

### `src/checkpoint_manager.py` (~164 lines)

`validate_and_hotswap()` (line 61) is the **only** authorized path for
replacing the live checkpoint — this is hard stop #9, never `torch.save`
directly onto the live path. It validates that the checkpoint dict has
exactly `{model_state_dict, centroid, config}`, with `config`'s
`N_FEATURES`, `GRAPH_EMB_DIM`, and `N_EDGE_TYPES` matching the currently
running `config_v3.py`, before it touches anything. That means a
dimension-mismatched checkpoint — say, one left over from a stale feature
schema after a features change — can never overwrite the live model.
Validation failure leaves the live checkpoint completely untouched, backed
up, and versioned.

---

## 8. Explainability (XAI)

### `src/xai_layer_v3.py` (~1,555 lines) + `src/xai_card_html_v3.py` (~1,358 lines)

The split between these two files is clean and deliberate:
`xai_layer_v3.py` computes every number a reviewer card shows;
`xai_card_html_v3.py` renders those numbers into HTML and computes nothing
new. That separation matters for auditability — if a claim on a card turns
out to be wrong, the bug lives in exactly one of these two files, depending
on whether it's a wrong *number* or a wrong *rendering*.

The project's XAI design rule is stated right in the module docstring
(`xai_layer_v3.py`, lines 6–17): no hand-set narrative thresholds. The only
numeric gates ever quoted are EVT-derived; everything else gets stated as a
computed percentile.

`build_fusion_contributions()` (`xai_layer_v3.py`, line 387) exists because
max-fusion has no natural notion of "percentage contribution" the way a sum
would. So attribution here is exact instead: whichever detector's own
minmax value equals the fused max **is** the driver, full stop, plus a
`margin_over_next` value showing how decisively it won. That's why the
fusion section of a reviewer card shows a "WON DRIVER" badge on exactly one
detector, rather than a stacked-percentage bar chart — the underlying math
doesn't support a percentage breakdown, so the UI doesn't pretend it does.

`DETECTOR_PILL["deepsad"]` (`xai_card_html_v3.py`, line 71) is deliberately
excluded from `DETECTOR_ORDER` (the three locked fusion inputs), so the
fusion-composition footer on a card always shows exactly the three real
fusion inputs. Deep SAD still gets its own visual identity elsewhere on the
card, clearly labeled as a non-fusion "second opinion."

### `src/evaluate_model_v3.py` (~547 lines)

This is the synthetic evaluation harness, independent of training. It
injects five fresh (unseen-seed) archetype anomalies and checks PR-AUC
against hard regression floors (`V2_FLOORS`). There's a subtlety worth
knowing about here, spelled out in the module docstring (lines 9–20): every
injected eval node is **isolated** — zero edges — and Hybrid GraphMCM's
`isolated_embedding` is one fixed learned vector shared across every
isolated node. That means feature-reconstruction error is nearly *constant*
across different injected anomalies, and can't actually rank them against
each other. The harness works around this by scoring each category with the
one Subspace IF group that targets its archetype instead — IP concentration
maps to the "network" group, for example — rather than using the hybrid
score directly, for this specific evaluation. It's a good illustration of
why understanding a detector's blind spot (isolated nodes, in this case)
matters even when you're just writing its test harness.

---

## 9. Stores (JSON, dual-written to Postgres)

These hold state, not ML logic, so this section is brief — but every
detector above reads its labels from one of these:

- **`src/confirmed_fraud_store.py`** (~209 lines) — supervisor-confirmed
  fraud/false-positive labels. `get_exposure_tensor()` (line 168) feeds real
  confirmed-fraud examples into the LOE push-away term alongside the
  synthetic exposure set, once at least 5 real examples exist.
- **`src/confirmed_fraud_graph_store.py`** (~306 lines) — confirmed fraud
  *ring* patterns, moving through `FLAGGED → CONFIRMED → SELECTED →
  PROMOTED`. `promote()` (line 114) appends a real ring's edges into the
  topology exposure set, so the RGCN's relational LOE term actually learns
  from it.
- **`src/model_registry.py`** (~136 lines) — the MLflow-free run-tracking
  log (`log_run()`, line 69) behind the console's Run History panel.

### `src/db/` — Postgres-side SQL modules (V4-Scale migration, brief)

- `bootstrap.py` — one-shot container startup: schema plus primary ingest.
- `connection.py` — pooled connections, `.env` config.
- `features.py` — SQL replicas of the pandas feature/graph logic, built to
  be bit-exact with `tabular_feature_engine_v3.py` and `graph_builder_v3.py`
  (see `docs/MAINTAINER_PLAYBOOK.md` Recipe 6 for the dual-path discipline
  this requires).
- `ingest.py` — stages raw batches and populates derived tables on
  Evaluate/Merge, the admin-gated contract described in
  `docs/IMPLEMENTATION.md` Step 3.
- `migrate.py` — schema migration runner.
- `reads.py` — the Postgres-vs-file read toggle used by the API.
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
