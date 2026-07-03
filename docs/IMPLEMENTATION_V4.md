# IMPLEMENTATION_V4.md — HAN + Topology Exposure + Supervisor Cycle

> **Audience:** the coding agent implementing V4 on branch `v4-han-graphmcm`.
> **Reviewer:** Claude (this repo's assistant) reviews every diff; the two
> ML-core tasks (T3 HAN encoder, T2 topology exposure) get the closest scrutiny.
> **Authority:** project-lead directed. `src/*.py` edits ARE allowed for this
> work. Decision records: `docs/AGENTS.md` Appendix F (ADR-015, ADR-016).

---

## 0. Read first — hard guardrails (breaking any of these fails review)

1. **Do NOT rename anything.** All files stay `*_v3.py`; all routes stay
   `/v3/...`; checkpoint stays `models/hybrid_graphmcm_v3.pth`; MLflow
   experiment stays `nic-fraud-detection-v3`. "V4" is a capability label only.
2. **The output contract is frozen.** `hybrid_graphmcm_v3.compute_score_frame()`
   must keep producing exactly these columns, same dtypes, same normalization:
   `application_id, hybrid_anomaly_score, feature_pred_error, edge_pred_error,
   per_feature_error_json, per_feature_predicted_json`. The anomaly formula
   stays `feature_pred_error + LAMBDA_EDGE * edge_pred_error`.
3. **`h_N(i)` stays 64-dim.** The graph encoder output shape is
   `(N, GRAPH_EMB_DIM)` where `GRAPH_EMB_DIM=64`. The `_check_shape(h_n,
   (None, GRAPH_EMB_DIM), "h_N(i)")` assertion in `forward()` must still pass.
4. **Isolated-node fallback is bit-for-bit preserved.** `encode_graph()` keeps
   `torch.where(isolated_mask, isolated_embedding, h)`. Real degree-0 nodes get
   the trainable `isolated_embedding`, unchanged.
5. **No raw embeddings leave `hybrid_graphmcm_v3.py`** (hard stop #2). Only
   `hybrid_anomaly_score`, `feature_pred_error`, `edge_pred_error`, and — new
   for V4 — **attention weights** (per-relation `beta_r`, top-k node-level
   `alpha`) may be exported. Attention weights are interpretable scalars, NOT
   the 64-dim embedding. Never export `h_n` or `concat`.
6. **No domain-threshold rules** (hard stop #1). No `age > 35`, no rule codes.
   Only EVT-derived or learned-from-exposure thresholds.
7. **`sanity` column is never used** (hard stop #4). It is already dropped;
   keep it dropped.
8. **No auto-retrain.** The supervisor cycle stages patterns and lets a human
   trigger one batched retrain (hard stops #5, #7). Never loop retrains.
9. **Seed everything.** `RANDOM_SEED=42` is already set in every module. Any
   new randomness (eval injection, exposure subgraph sampling) must seed from
   `config_v3` constants so runs are reproducible and comparable.
10. Run everything **from the project root**, e.g. `.\.venv\Scripts\python.exe
    src/evaluate_model_v3.py`. Paths are relative to root.

---

## 1. The ablation this whole branch exists to produce

Same seed, same eval harness, per-category PR-AUC. Build order is chosen so
each step yields one clean delta:

| # | Config | Encoder | Exposure | Measures |
|---|--------|---------|----------|----------|
| 1 | baseline | RGCN | feature-vector | current V3 (already trained) |
| 2 | topology-exposure | **RGCN** | **topology** | topology exposure's effect on the existing encoder |
| 3 | HAN | **HAN** | topology | HAN's effect on top of topology exposure |

Δ(2−1) = topology exposure's contribution. Δ(3−2) = HAN's contribution.

**CRITICAL — the current evaluator cannot measure any of this.** In
`evaluate_model_v3.py`, the category PR-AUC table is computed **only** from a
subspace IsolationForest; the graph model is used solely for the Level-3
edge-dropout `score_retention`. Every injected anomaly is an **isolated** node,
so the graph stream returns the fixed `isolated_embedding` and cannot rank
them. Therefore **T1 (a graph-sensitive eval) must be built and merged before
T2's comparison means anything.** Do T1 first.

Each config's results are saved as a JSON under `outputs/ablation/` (see T1)
so the three can be diffed without rerunning.

---

## 2. New config constants (add to `src/config_v3.py`, do not touch existing)

```python
# ── V4: HAN encoder (ADR-015) ───────────────────────────────────────────────
ARCH_VERSION         = "han_v1"   # checkpoint architecture tag (hard stop #15)
ATTN_HEADS           = 4          # node-level GAT heads per relation
ATTN_LEAKY_SLOPE     = 0.2        # LeakyReLU slope in node-level attention
SEMANTIC_ATTN_HIDDEN = 32         # hidden dim of the semantic-attention MLP

# ── V4: topology synthetic exposure (ADR-016) ───────────────────────────────
TOPO_EXPOSURE_ENABLED   = True    # master switch; False reproduces config-1 exposure
N_TOPO_CLUSTERS         = 50      # synthetic connected fraud clusters per rebuild
TOPO_CLUSTER_SIZE_RANGE = (6, 40) # nodes per synthetic cluster (min, max)

# ── V4: connected-cluster evaluation (T1) ───────────────────────────────────
EVAL_CONNECTED_N_CLUSTERS   = 30  # injected connected clusters per category
EVAL_CONNECTED_SIZE_RANGE   = (6, 40)
```

`RANDOM_SEED` is reused everywhere; do not add new seed constants.

---

## T1 — Connected-cluster evaluation harness (DO THIS FIRST)

**File:** `src/evaluate_model_v3.py` (extend; do not remove the existing
isolated-node Level-1/2/3 code — the V3 baseline table must stay intact and
auditable).

**Goal:** measure the *graph model's* ability to rank **connected** fraud, so
HAN and topology exposure become visible.

**Add a function** `evaluate_connected() -> dict[str, float]`:

1. Load `x_all`, the graph, and the live model exactly as `evaluate()` does.
2. For each category in `{IP_CONCENTRATION, MOTHER_NAME_COLLISION,
   FEE_INFLATION, AGE_VIOLATION, INCOME_VIOLATION}`:
   - Build `EVAL_CONNECTED_N_CLUSTERS` synthetic **connected** clusters. Each
     cluster is a set of new nodes (size drawn from `EVAL_CONNECTED_SIZE_RANGE`,
     seeded `np.random.default_rng(EVAL_SEED + category_index)`) whose feature
     vectors are built by the SAME per-category perturbation used for injection
     today (reuse `INJECTION_FNS[category]`), AND which are wired together with
     edges of the category's signature relation:
       - IP_CONCENTRATION → `shares_ip` clique
       - MOTHER_NAME_COLLISION → `shares_mother_name` clique
       - FEE_INFLATION / INCOME_VIOLATION / AGE_VIOLATION → `shares_pincode`
         clique (these are tabular archetypes; the clique tests whether the
         graph stream *also* lifts them, but the subspace IF remains their
         primary detector — report both).
   - Append these nodes+edges to a COPY of the graph tensors (never mutate the
     canonical `identity_graph_v3.pt`). Recompute `isolated_mask` for the
     augmented graph.
   - Score ALL nodes with `compute_score_frame(...)` (the real model path).
   - Labels: 1 for injected cluster nodes, 0 for the 15,000 real nodes.
   - `pr_auc = average_precision_score(labels, hybrid_anomaly_score)`.
3. Print a table mirroring the existing one, plus a `graph_lift` column =
   connected PR-AUC − the isolated-node PR-AUC for that category.
4. Return `{f"conn_pr_auc_{cat.lower()}": ...}` merged with a
   `mean_conn_pr_auc`.

**Add** a `--connected` CLI flag and an `--ablation-tag <name>` flag in
`__main__`. When `--ablation-tag` is given, write the full metrics dict to
`outputs/ablation/<tag>.json` (create the dir). Tags used by this plan:
`config1_rgcn_feature`, `config2_rgcn_topo`, `config3_han_topo`.

**Acceptance:** `evaluate_connected()` runs on the current RGCN checkpoint and
prints non-degenerate (varying) PR-AUC across categories. `evaluate()` output
is byte-identical to before (regression check).

**Then capture config-1 baseline:** retrain the current RGCN model unchanged
(`python main_v3.py` or `python -m src.hybrid_graphmcm_v3`) with seed 42, run
`evaluate --connected --ablation-tag config1_rgcn_feature`. This is the honest
config-1 row measured under identical conditions.

---

## T2 — Topology synthetic exposure (graph injection into Stage 1)

**Files:** `src/synthetic_exposure_builder_v3.py` (add topology clusters),
`src/hybrid_graphmcm_v3.py` (consume them with edges intact in Stage 1).

**The problem being fixed:** today `_get_synth_h()` force-isolates every
exposure node (`synth_isolated = torch.ones(...)`), so `encode_graph()` returns
the single `isolated_embedding` for all of them — the graph stream learns
nothing about fraud topology from exposure. T2 makes exposure nodes carry real
edges through the encoder.

**T2a — build connected exposure clusters.**
Add `build_topology_exposure()` to `synthetic_exposure_builder_v3.py`:
- Produce `N_TOPO_CLUSTERS` clusters. Each: size in `TOPO_CLUSTER_SIZE_RANGE`,
  feature vectors from the existing archetype perturbations, wired as a clique
  (or near-clique) on that archetype's signature relation.
- Save to `data/processed/synthetic_exposure_graph_v3.pt` as a dict:
  `{"x": FloatTensor[M, 68], "edge_index": LongTensor[2, E],
    "edge_type": LongTensor[E], "cluster_id": LongTensor[M]}`.
  (M = total exposure nodes across clusters.) This is SEPARATE from the
  existing `synthetic_exposure_set_v3.pt` (68-dim vectors) which stays for
  backward compatibility / the feature-exposure baseline.

**T2b — consume topology exposure in Stage 1.**
In `hybrid_graphmcm_v3.py`:
- Add `_get_synth_h_topology(model, topo_pack, device) -> Tensor[M, 64]`: runs
  the exposure nodes through `model.encode_graph(x_topo, [topo_edge_index],
  topo_edge_type, isolated_mask_topo)` where `isolated_mask_topo` is computed
  from the topo edges (nodes with edges are NOT isolated → they flow through the
  real encoder, giving topology-derived embeddings). Cluster-internal edges
  only; do NOT connect exposure nodes to the 15,000 real nodes.
- In `train()` Stage 1: if `TOPO_EXPOSURE_ENABLED`, use `_get_synth_h_topology`
  for the LOE term instead of (or in addition to) the isolated
  `_get_synth_h`. Keep the SVDD term on real nodes unchanged. The LOE loss
  function `_loe_loss` is reused as-is.
- **Centroid contamination guard:** exposure clusters must NOT enter
  `init_centroid()`. The centroid is computed on the 15,000 real nodes only
  (it already is — just do not add exposure nodes there).

**Config switch:** `TOPO_EXPOSURE_ENABLED=False` must exactly reproduce the
old behavior (feature-vector exposure via `_get_synth_h`) so config-1 is
reproducible from the same code.

**Acceptance:**
- With `TOPO_EXPOSURE_ENABLED=False`, `train()` produces a checkpoint whose
  `evaluate --connected` matches config-1 within noise.
- With `True`, a full retrain + `evaluate --connected --ablation-tag
  config2_rgcn_topo` runs and writes the JSON. Print `h_synth.std(0).mean()`
  in Stage 1 for both settings: it should be ~0 when False (collapse confirmed)
  and clearly >0 when True (topology reaches the encoder). This one-line print
  is the evidence that T2 actually does something.

---

## T3 — HAN encoder (replaces RGCNEncoder, same seams)

**File:** `src/hybrid_graphmcm_v3.py` only (+ config constants from §2).

Replace `class RGCNEncoder` with `class HANEncoder(nn.Module)` exposing the
**same forward signature** so `encode_graph()` and everything downstream is
untouched:

```python
def forward(self, x, edge_index_list, edge_type_tensor) -> Tensor  # (N, 64)
```

**Internal design (ADR-015):**
- **Level 1 — node-level attention (GAT) per relation.** For each of the
  `N_EDGE_TYPES=5` relations, a GAT-style layer with `ATTN_HEADS=4` heads and
  `LeakyReLU(ATTN_LEAKY_SLOPE)`. Two stacked layers overall to match RGCN depth
  (68→128→64 equivalent: project to `GRAPH_HIDDEN=128` then `GRAPH_EMB_DIM=64`).
  Add self-loops per relation so nodes with no neighbors in a relation still get
  a defined representation. You MAY use `torch_geometric.nn.GATConv` per
  relation (it is already a dependency via RGCNConv). Produce one embedding
  `z_r ∈ R^{N×64}` per relation.
- **Level 2 — semantic attention across relations.** For each relation compute
  a scalar importance `w_r = q^T · tanh(W · mean_i(z_r[i]) + b)` via an MLP with
  hidden dim `SEMANTIC_ATTN_HIDDEN=32`; softmax over the 5 relations →
  `beta_r`. Fuse: `h = Σ_r beta_r · z_r`. Output `h ∈ R^{N×64}`.
- **Store attention for export** (do NOT return embeddings): keep
  `self.last_beta_r` (Tensor[5], the semantic weights, or [N,5] if per-node)
  and a method `top_alpha(node_idx, k)` returning that node's top-k neighbors
  by node-level attention with their relation + weight. These feed T4. They are
  weights, not embeddings — hard-stop-#2 compliant.

**Preserve exactly:** the `HybridGraphMCM.__init__` attribute name can change
from `self.rgcn` to `self.encoder`, BUT then update the two references in
`train_incremental()` (`model.rgcn.parameters()`) and any doc/comment. Keeping
`self.rgcn` as the attribute name (holding a HANEncoder) is also acceptable and
lower-risk — pick one and be consistent. `encode_graph()` body, the isolated
fallback, `forward()`, `compute_score_frame()`, `init_centroid()` stay
unchanged except that they now call the HAN encoder.

**Acceptance:**
- Unit check: `h_N(i)` shape is `(N, 64)`; `_check_shape` passes.
- Isolated nodes still receive `isolated_embedding` (parity test: force a node
  isolated, confirm its `h_n` equals `isolated_embedding`).
- Full retrain with `TOPO_EXPOSURE_ENABLED=True`, then `evaluate --connected
  --ablation-tag config3_han_topo`.
- Print aggregate `beta_r` (mean±std per relation) at end of training.

---

## T4 — Attention attribution in XAI ("which edges did it focus on")

**File:** `src/xai_layer_v3.py` (extend the existing evidence-first narratives;
do not change training code).

For each explained application, add to its card:
- `attention.beta_r`: the semantic weights over the 5 relations for that node
  (the "relation mix", e.g. `{"shares_ip": 0.81, ...}`).
- `attention.top_edges`: top-k neighbors by node-level `alpha`, each
  `{application_id, relation, weight}` — resolved to real application IDs like
  the existing `top_graph_neighbors` field.
- One sentence in `narrative` composed deterministically from the above, e.g.
  "The model weighted this application's shared-IP links most heavily (β=0.81),
  focusing on 3 co-IP applications." Same evidence ⇒ same words (auditable).

Source the weights by calling the model's `last_beta_r` / `top_alpha()` on the
scored graph. Respect hard stop #2: only weights, never embeddings.

**Acceptance:** `explanation_cards_v3.json` regenerates with the new
`attention` object; existing fields unchanged; narrative still deterministic.

---

## T5 — Checkpoint ARCH_VERSION (schema safety)

**Files:** `src/hybrid_graphmcm_v3.py` (both save sites: `train()` and
`_score_and_save()`), `src/checkpoint_manager.py`.

- Add `"ARCH_VERSION": ARCH_VERSION` to the `config` dict in every
  `torch.save(...)` in `hybrid_graphmcm_v3.py`.
- In `checkpoint_manager.py`: add `"ARCH_VERSION"` to `_REQUIRED_CONFIG` and,
  in `_validate`, raise if `cfg["ARCH_VERSION"] != config_v3.ARCH_VERSION`.
  Import `ARCH_VERSION` from `src.config_v3`.
- Effect: an old RGCN checkpoint (no `ARCH_VERSION`, or a different value) fails
  `validate_and_hotswap()` by design — do not coerce it. Message must name the
  mismatch clearly.

**Note:** `validate_and_hotswap` loads with `weights_only=True`; `ARCH_VERSION`
is a str, which is safe. Keep config values to str/int/float only.

**Acceptance:** saving a HAN checkpoint then hot-swapping succeeds; feeding the
pre-V4 `.bak` (RGCN) through `validate_and_hotswap` raises a clear
ARCH_VERSION error.

---

## T6 — Visualization + supervisor review cycle (LAST; after T1–T3 prove value)

Only start once the ablation shows topology exposure helps. Read the actual
handler files before editing: `src/api/handlers/{monitoring,supervisor,
training}.py`, `src/api/schemas.py`, `src/confirmed_fraud_store.py`, `main_v3.py`.

**T6a — `src/topology_view.py` (new, read-only).**
- `extract_ego(app_id, hops=1, node_cap=50) -> dict`: BFS the ego-graph from
  `identity_graph_v3.pt`; keep real application_ids; cap at `node_cap` by risk
  score (from `outputs/hybrid_scores_v3.csv`) with a `shown/total` field; 1-hop
  only. Include all typed edges among the kept nodes.
- `render_svg(ego) -> str` and `render_html(ego) -> str`: self-contained
  (inline SVG + a numpy/JS force layout — no external deps, no CDN). Nodes
  colored by risk, edges by relation, center ringed. HTML is interactive
  (hover), SVG is static. (A working reference prototype exists — ask the
  reviewer for `scratchpad/render_html.py` from the design session.)

**T6b — pattern lifecycle store: `src/confirmed_fraud_graph_store.py` (new).**
Mirror `confirmed_fraud_store.py` but persist subgraphs + lifecycle state:
`FLAGGED → CONFIRMED → SELECTED → PROMOTED / REJECTED`. Functions:
`add_confirmed_pattern(app_id, fraud_type, subgraph, confirmed_by)`,
`list_pending()`, `select(pattern_ids)`, `promote(pattern_ids)` (writes the
selected subgraphs into the topology exposure set and returns them for the
retrain), `count_pending()`. Anonymization: keep IDs (project-lead decision);
document that the store therefore holds PII.

**T6c — API endpoints** (extend handlers; do not change existing endpoint
logic):
- `GET /v3/monitoring/{app_id}/topology` → returns interactive HTML.
- `GET /v3/supervisor/patterns` → pending count + list.
- `POST /v3/supervisor/patterns/confirm` → FLAGGED→CONFIRMED (stores subgraph).
- `POST /v3/supervisor/patterns/promote` → SELECTED subset → rebuild topology
  exposure → dispatch ONE retrain via the existing Celery task. Human-triggered
  only; append to the drift/decision audit log like ADR-014 does.

**T6d — MLflow audit** (`main_v3.py`): for each top-suspicious app, render a
**static SVG** (not HTML — MLflow's viewer sandboxes JS) and
`mlflow.log_artifact(..., artifact_path="topology")`. On promote, log which
pattern IDs + reviewer + resulting exposure-set version as run tags/params.

**T6e — curl / API testing guide:** add a §to `docs/API_TESTING_GUIDE.md` with
the new endpoints. Do not change existing curl examples.

---

## 7. Final validation checklist (reviewer will verify)

- [ ] `evaluate()` (isolated-node) output unchanged vs pre-V4 (regression).
- [ ] `outputs/ablation/config1_rgcn_feature.json`,
      `config2_rgcn_topo.json`, `config3_han_topo.json` all present.
- [ ] Stage-1 `h_synth.std` print: ~0 for feature-exposure, >0 for topology.
- [ ] Isolated-node parity test passes under HAN.
- [ ] `hybrid_scores_v3.csv` schema byte-identical (6 columns as in §0.2).
- [ ] Old RGCN checkpoint fails ARCH_VERSION validation.
- [ ] No `h_n` / `concat` / embedding tensor returned outside the model file.
- [ ] `explanation_cards_v3.json` gains `attention` object, rest unchanged.
- [ ] No `_v3`→`_v4` renames anywhere; no `/v3`→`/v4` route changes.
- [ ] All new randomness seeded from `RANDOM_SEED`.

## 8. Suggested commit sequence (one concern per commit)

1. `feat: connected-cluster eval harness (T1) + config-1 baseline JSON`
2. `feat: topology synthetic exposure into Stage 1 (T2) + config-2 results`
3. `feat: HAN two-level attention encoder (T3) + config-3 results`
4. `feat: attention attribution in XAI (T4)`
5. `feat: ARCH_VERSION checkpoint validation (T5)`
6. `feat: topology view + supervisor review cycle + MLflow audit (T6)`

Report the three ablation JSONs to the reviewer after commit 3 — that is the
"which component helped by how much" answer, and nothing after it should start
until those numbers are reviewed.
