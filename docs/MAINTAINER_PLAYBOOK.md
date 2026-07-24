# MAINTAINER_PLAYBOOK.md — Task-Oriented Recipes for Future Maintainers

<!-- Added 2026-07-23. Companion to AGENTS.md (the contract/rules) and
     TECHNICAL_REFERENCE_AND_SCALING.md (the deep how-it-works reference).
     This file answers "I need to do X — which files, in what order, what
     can I break?" It does not restate the rules; it points at them. -->

This file is for whoever maintains this system after the original author is
gone. It assumes you've read `AGENTS.md` (the contract — hard stops, module
ownership, what's locked) and know roughly what the pipeline does
(`docs/PROJECT_OVERVIEW.md` if you don't). What follows is "how do I actually
make this specific kind of change without breaking something two files away."

Before you touch any of these recipes, read `AGENTS.md` §4 (hard stops) and
§6 (when to stop and ask). Nothing below overrides those — if a recipe ever
conflicts with a hard stop, the hard stop wins, and the right move is to stop
and ask rather than push through.

One habit worth adopting from the start: use `src/interfaces/` (see
`AGENTS.md` §3) whenever you're importing a layer's functions into new code.
It's the stable front door, and it also means the file lists below stay
accurate even if a concrete `_v3` file is later reorganized.

---

## Recipe 1 — Adding a new tabular feature

Here's the order you'll actually touch files in, and why each step exists.

1. Start in `src/tabular_feature_engine_v3.py::_engineer_features()` and add
   the derivation. If it's a per-row scalar, you can just write it directly.
   But if it's a cross-row aggregate — a groupby/count/rank over the whole
   population — you also need step 2, no way around it.
2. For a cross-row aggregate, go add the SQL equivalent in
   `src/db/features.py::aggregate_features()`. This is the file most people
   forget. The scale path (step 4 of `IMPLEMENTATION.md`, now cut over)
   computes aggregates in SQL, not pandas — see Recipe 6 below for why these
   two must match exactly.
3. In `src/config_v3.py`, a few things may apply depending on what the
   feature is for. If it should feed the model, you usually don't need to do
   anything here — unless it's an identifier/code column that should be
   *excluded*, in which case add it to `IDENTIFIER_FEATURES` instead (it gets
   dropped from `N_FEATURES`, but its sharing signal still flows through the
   graph). If it belongs in a subspace-IF group (financial/identity/network),
   add it to the matching list in `SUBSPACE_GROUPS`. And if it needs log1p
   compression (skewed monetary values, etc.), add it to `LOG1P_COLS`.
4. Touch `deploy/postgres/schema.sql` only if you're changing the feature
   *count*: `features.vec` has `CHECK (cardinality(vec) = 44)`. Bumping
   `N_FEATURES` in `config_v3.py` without a matching schema migration will
   make every scored batch fail that constraint. See Recipe 4 for how to
   handle that migration properly.
5. Regenerate the held-out gate bundle. `src/deploy_gate.py` (Recipe 3)
   versions its held-out evaluation set by `schema_version = "v3_<N_FEATURES>"`
   — changing the feature count or list changes that string, and the gate
   will refuse to run (fail closed) until you re-run
   `python -m src.build_held_out_set`. Do this in the same change as the
   feature edit, not as an afterthought before the next deploy — an engineer
   six months from now bumping a feature count has no reason to know the gate
   bundle exists unless this step is part of the recipe.
6. Finally, check `src/xai_layer_v3.py` if the feature should be narrated on
   evidence cards — the `xai-narration-policy` behavior applies here
   (identifiers are never spoken; continuous + network-disagreement binaries
   are). New identifiers need no XAI work at all, since they're excluded by
   construction via `IDENTIFIER_FEATURES`.

To validate, re-run `evaluate_model_v3.py::evaluate()` /
`evaluate_connected()` (via `src.interfaces.evaluate`) and compare against
the v2 floors — that's the existing pass bar, not a new one you invent. Check
Gate 4 in `IMPLEMENTATION.md` for what "the SQL and pandas paths must match"
looks like in practice: bit-exact where deterministic, tolerance-documented
where float summation order legitimately differs (for example, `PERCENTILE_CONT`
vs. pandas median).

What can silently break here: adding a feature changes `N_FEATURES`, which is
part of every checkpoint's validated config (`checkpoint_manager`, hard stop
9: `{model_state_dict, centroid, config}` must contain `N_FEATURES`,
`GRAPH_EMB_DIM`, `N_EDGE_TYPES`). A checkpoint trained on the old width will
be rejected by the loader against the new width — that's correct behavior,
not a bug, but it does mean you must retrain before the new feature is live,
not just re-run scoring.

---

## Recipe 2 — Adding a new detector (as a fusion input, or XAI-only)

Before writing any code, decide: is this a fusion input, or an XAI-only
signal? That's not really a coding question — it's an evidence question, and
the project has a recorded precedent for rejecting a 4th fusion input (Deep
SAD, `AGENTS.md` §1, 2026-07-22). A candidate detector can be genuinely
useful for XAI narration while adding nothing to the *fused* score, because
`max()` fusion only listens to whichever detector already wins the argmax
for that node's fraud category. So test before you assume "another signal =
better fusion."

Once you know which one you're building, here's the order:

1. Write the new detector module (`src/<name>_detector_v3.py`), following
   the `_v3` naming convention still in force — see `AGENTS.md` §1 naming
   rule; don't invent a new suffix scheme. Score direction must be higher =
   more anomalous (hard stop 3), and if you ever have to invert a score,
   document that inversion right at the point it happens.
2. Add a matching `src/interfaces/<name>.py` re-export module (see
   `AGENTS.md` §3) so other code has a stable import path from day one.
3. Wire the pipeline step into `main_v3.py` — look at how `run_dense_block`,
   `run_deepsad`, etc. are wired in already, and follow that pattern for
   where between existing steps yours belongs.
4. Add an `<NAME>_ENABLED` env-gated flag to `src/config_v3.py`, following
   the existing pattern (`DEEPSAD_ENABLED`, `DENSE_BLOCK_ENABLED`), plus the
   detector's own hyperparameters, namespaced with the detector's name
   prefix.
5. Run the ablation before touching fusion at all. This is a project rule,
   not a suggestion: compute the candidate's standalone PR-AUC per fraud
   category, then re-score with it added as a 4th `max()` input, and compare
   both against the current 3-way fusion — on the same held-out/stress
   population, same seed. `outputs/stress_testing_1_*` and
   `outputs/ablation/locked_fusion_validation.json` are the pattern to
   follow (see `HISTORY.md` for the Deep SAD numbers as a worked example of
   "tested and rejected, with the argmax-win-rate reasoning spelled out").
   Do not add a detector to `FUSION_COMPONENTS` without this step — it's how
   the Deep SAD rejection was actually decided, not a formality.
6. If, and only if, the ablation shows a real, non-noise-level improvement,
   `src/fusion_classifier_v3.py::score_level_fusion()` gains the new
   `minmax()` term inside the `max()`, and `config_v3.FUSION_COMPONENTS`
   gets the new name appended. This is a locked-architecture change — per
   `AGENTS.md` §6, this is exactly the kind of thing to stop and confirm
   with the lead before merging, even with good ablation numbers in hand.
7. If the detector stays XAI-only, which is the likely outcome on
   precedent, wire it into `src/xai_layer_v3.py` as a supplementary signal
   only — the way `center_dist_score` is surfaced is the template
   ("supplementary signal, >75th pct," not a driver of the score itself).

The part of your original ask that matters most is probably this fusion
weight caution: `AGENTS.md` §1 is explicit that this pipeline is not a
flawless architecture search. The current `max()` fusion replaced a
weighted-sum that diluted strong single-detector signals, which itself
replaced a learned LightGBM combiner that got destroyed by 14 positive
labels. Every one of those transitions was decided by re-running the *same*
ablation harness against the *same* stress population and comparing
per-category PR-AUC, not by reasoning about it in the abstract. So if you're
tempted to hand-tune a weight — don't. There are no per-component weights in
the locked fusion by design (`FUSION_W_*` are retired), and reintroducing
them requires the same ablation-and-lead-confirmation process as adding a
detector.

---

## Recipe 3 — Running the deploy gate before promoting a checkpoint

Never call `checkpoint_manager.validate_and_hotswap()` directly on a newly
trained checkpoint without gating it first. `checkpoint_manager` only checks
*shape* — does this checkpoint match `N_FEATURES`/`GRAPH_EMB_DIM`/
`N_EDGE_TYPES`/`ARCH_VERSION` — it says nothing about whether the new
checkpoint is actually *better* than what's live. `src/deploy_gate.py` is
that missing quality check, and it's a separate manual step on purpose (not
wired into `checkpoint_manager` itself), so a bad candidate can never reach
the atomic-swap code path.

```bash
# 1. If this is the first gate run after a feature/schema change, build the
#    held-out bundle first (fails otherwise — see Recipe 1 step 5):
.venv/Scripts/python.exe -m src.build_held_out_set

# 2. Gate the candidate checkpoint (produced by training, sitting at a temp
#    path, NOT yet live):
.venv/Scripts/python.exe -m src.deploy_gate --candidate models/incoming_<ts>_<uuid>.pth --cycle 2026H2

# 3. Only if the gate exits 0 (PASS), promote:
python -c "from src.checkpoint_manager import validate_and_hotswap; from pathlib import Path; \
  validate_and_hotswap(Path('models/incoming_<ts>_<uuid>.pth'), cycle='2026H2', source_ref='<git-ref>')"
```

What the gate actually checks: it re-scores the candidate AND the
currently-live checkpoint, in the same run, against a held-out set the model
never trained on (built from `data/uploads/stress_testing_1.csv` — the same
50k synthetic ground-truth cohort used for this project's fusion/dense-block
ablations). It compares per-fraud-category PR-AUC, and the candidate must
not regress beyond the documented ±0.03–0.04 noise floor on *any* category,
even if its overall/average number looks better — that's Recipe 5's lesson
in practice: overall-only comparisons hide exactly this failure mode. Every
run, pass or fail, is logged to `outputs/model_registry.json`
(`run_type: "deploy_gate"`) with the git commit, `schema_version`, held-out
bundle version, and both checkpoints' full metrics, so there's an audit
trail of what was ever allowed to promote.

If the gate fails, the live checkpoint is untouched — the gate never calls
`validate_and_hotswap` itself. Read the printed per-category deltas; which
category regressed tells you which recent change to suspect (a new feature,
a detector change, a fusion edit) before you go looking further.

---

## Recipe 4 — Modifying the Postgres schema

1. Never edit `deploy/postgres/schema.sql` in place for a live-running
   system — hard stop 14 requires versioned migrations. Check
   `src/db/migrate.py` for the migration runner pattern and add a new
   numbered migration file; `schema.sql` stays the cumulative reference.
2. Every SQL query touching the changed table lives in `src/db/` (hard stop
   14 — no inline SQL anywhere else). Grep `src/db/*.py` for the table name
   before assuming you've found every consumer.
3. If you're changing `features.vec`'s cardinality (i.e. `N_FEATURES`
   changed per Recipe 1), the `CHECK (cardinality(vec) = 44)` constraint must
   be updated in the same migration — and every row already in the table
   becomes invalid under the new constraint. This is a **backfill problem**,
   not just a DDL change. Plan the backfill (recompute + rewrite `vec` for
   existing rows) before applying the constraint change, or the migration
   will fail against non-empty tables.
4. Dual-write discipline (hard stop 13) still applies to any new
   column/table you add that has a JSON-file predecessor or file-based
   ancestor — don't make Postgres authoritative for a new field until you've
   demonstrated parity the same way steps 1–3 of `IMPLEMENTATION.md` did
   (see those steps' gate-evidence blocks for the actual comparison
   methodology: field-level diff, round-trip verified through the public
   API, not just "looks right").
5. Run the round-trip / parity check pattern from Gate 0/1 (one synthetic
   row per table, field-level comparison) before considering the migration
   done.

---

## Recipe 5 — Running ablations for weight/combination changes

This project holds a specific evidence bar for changing anything
score-combination-related — fusion weights, dense-block relation weights,
subspace group membership. Here's what that bar looks like in practice:

1. Name the baseline explicitly (quantitative claims protocol, `AGENTS.md`
   §5 / `CLAUDE.md`) — which file, which commit, which run.
2. Use a population with ground truth, not the raw 15k (it has no confirmed
   fraud rings at meaningful scale). The `stress_testing_1`-style synthetic
   generator (see `outputs/stress_testing_1_*` and the generator script kept
   "for the record" per the git log) samples with replacement to build
   larger structures with known fraud rings, specifically so ring-detection
   PR-AUC is measurable per category (mobile-ring, IP-ring, income, etc.).
3. Seed everything, and mind the ±0.03–0.04 GPU scatter-add noise floor —
   use CPU scoring or deterministic algorithms if the effect you're
   measuring is smaller than that.
4. Compare per-category, not just overall. The dense-block and fusion
   history in `AGENTS.md` §1 is full of changes that improved the *overall*
   number while quietly collapsing one category — equal-weighting
   dense-block relations gained overall PR-AUC but collapsed IP-ring
   detection from 0.2199 to 0.067. Overall-only comparisons hide exactly
   this failure mode.
5. Log the result as "proposed, pending." A number generated in the same
   session as the change cannot itself settle whether the change is good
   (protocol rule 5) — get a second look, ideally on a second seed or a
   second stress population, before calling it adopted.
6. If numbers conflict with `HISTORY.md` or an ablation JSON, stop. Don't
   reconcile by narrative — re-derive under matching conditions (same fusion
   formula, same encoder config; `AGENTS.md` §7 open decision 6 is a live
   example of a fusion validation that predates later encoder fixes and can
   no longer be cited as validating the current system).

---

## Recipe 6 — Changing data ingestion / preprocessing (the SQL + Python dual-path problem)

This is the one you specifically flagged, and it's real: since step 4 of
`IMPLEMENTATION.md`, feature engineering has two parallel implementations
that must produce identical output.

| Concern | Python (file path) | SQL (scale path) |
|---|---|---|
| Row cleaning / per-row scalars | `tabular_feature_engine_v3.py::_load_and_clean()`, `_engineer_features()` | `src/db/features.py::fetch_raw_frame()` reconstructs an identical pandas frame, then reuses the **same** `_engineer_features()` — per-row logic is NOT duplicated |
| Cross-row aggregates (counts, ranks, percentiles) | pandas groupby/transform, inline in `_engineer_features()` when `sql_agg is None` | `src/db/features.py::aggregate_features()` — hand-written SQL replicating the pandas semantics **exactly**, including pandas' specific rank formula `(RANK + (ties−1)/2) / n` |
| Graph edges | `graph_builder_v3.py::build_graph()` (in-memory adjacency) | `graph_builder_v3.py::build_graph_pg()` — SQL self-joins on `identity_keys`, hub-capped |
| Scaling | `MinMaxScaler.fit_transform()` | `apply_stored_scaling()` reapplies **persisted** `scale_/min_` params — never refits (hard stop 11) |

The critical thing to understand is that per-row logic is *shared* — the SQL
path reconstructs a pandas frame via CSV round-trip and calls the same
`_engineer_features()` — so a per-row change only needs editing once. Only
the cross-row aggregates are genuinely duplicated: one pandas
implementation, one SQL implementation, and they have to agree on every
value. This is exactly Recipe 1 step 2 above, but the point here is the
*discipline*, not just the file list:

1. Never change one without the other in the same commit. If you add or
   modify a cross-row aggregate, `_engineer_features()`'s pandas branch and
   `aggregate_features()`'s SQL both change together. A one-sided change
   passes tests against whichever path you tested and silently diverges on
   the other — and nothing will tell you until Gate-4-style bit-comparison
   is re-run.
2. Check the docstring in `src/db/features.py` before writing new SQL — it
   documents which columns are exact matches vs. tolerance-compared (e.g.
   `PERCENTILE_CONT` vs. pandas median differs by one float op; that's
   accepted and documented, not a bug to chase).
3. Re-run the Gate 4 comparison after any change here — bit-for-bit where
   deterministic, documented tolerance where not. This isn't a nice-to-have
   regression test; it's the mechanism that catches path divergence before
   it reaches production, since the two paths are used at different times
   (file path historically, SQL path is the live one now) and a silent
   mismatch would otherwise only surface as an unexplained score drift.
4. Graph changes follow the same rule — `build_graph()` and
   `build_graph_pg()` must agree on adjacency. The hub-cap machinery
   (`derive_group_ceiling()`) lives once in `graph_builder_v3.py` and is
   called from both paths, so a hub-cap logic change is automatically
   shared — but if you change which relations feed the graph, or add a 6th
   relation, both `identity_keys` (schema — Recipe 4) and both builder
   functions need it.
5. Ingestion contract stays raw-in. Per `IMPLEMENTATION.md` step 3, there's
   no upstream preprocessing on data landing in `applications` — the schema
   binds raw CSV shape only, and derived tables (`identity_keys`, `features`,
   `scores`) populate only on admin-triggered actions (Evaluate/Merge). If
   your ingestion change adds any transformation before rows hit
   `applications`, that's a contract violation, not a preprocessing detail —
   stop and ask (`AGENTS.md` §6: Postgres schema changes are lead-review
   territory).

---

## Quick-reference: "I want to change X, what do I touch?"

| Change | Primary file(s) | Also check |
|---|---|---|
| New per-row feature | `tabular_feature_engine_v3.py` | XAI narration policy if it should be spoken |
| New cross-row aggregate | `tabular_feature_engine_v3.py` **and** `src/db/features.py` | Gate-4-style bit comparison |
| New detector | new `_v3` module + `src/interfaces/` entry + `main_v3.py` wiring | Ablation (Recipe 5) before any fusion change |
| Fusion formula/weights | `fusion_classifier_v3.py`, `config_v3.FUSION_COMPONENTS` | Ablation (Recipe 5) + lead confirmation (locked architecture) |
| Promoting a newly trained checkpoint | `src/deploy_gate.py` then `checkpoint_manager.validate_and_hotswap()` | Recipe 3 — held-out bundle must match current `schema_version` |
| New Postgres column/table | `deploy/postgres/schema.sql` via migration + `src/db/` queries | Dual-write parity if there's a file predecessor |
| Feature count (`N_FEATURES`) | `config_v3.py` | `features.vec` CHECK constraint + backfill + retrain |
| New identity relation (6th edge type) | `config_v3.EDGE_TYPES`, `graph_builder_v3.py` (both paths), `identity_keys` schema | `N_EDGE_TYPES`, dense-block relation gate, checkpoint config validation |
