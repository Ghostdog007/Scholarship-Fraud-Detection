# NIC Scholarship Fraud Detection — Complete Technical Reference & 30–40 Lakh Scaling Blueprint

<!-- VERSION: 2.0 | DATE: 2026-07-21 | AUDIENCE: project lead / implementing engineer / new contributor -->
<!-- Companion docs: docs/AGENTS.md (architecture contract), docs/IMPLEMENTATION.md (migration steps
     + gate evidence), docs/HISTORY.md (how the detection architecture got locked, with metrics),
     docs/OPERATIONS_RUNBOOK.md (console operation), deploy/README.md (current deploy) -->

This document comes in three parts. Part I walks through how the system works today — every model, how each one reaches its conclusion, the exact feature schema, the backend, the frontend, and how a single application flows end to end. Part II covers the PostgreSQL + Kubernetes remodel: implemented and gate-tested, not a proposal, aimed at **30–40 lakh (3–4 million) applications** — the real schema, how Postgres talks to the console, how ingestion works, how external GPU-trained checkpoints get installed, and the measured scaling results. Part III gives one clear, evidence-backed answer to "how many applications can this system comfortably process," stated with its constraints rather than softened.

### How to use the SVG prompts in this document

Throughout the document you'll find blocks marked **`[SVG PROMPT]`**. These aren't images — they're specifications for a diagram a designer, or an LLM/diagramming tool, should render. Each one names the exact elements, labels, and data the diagram must contain, sourced only from what's documented around it (no invented numbers). Treat them as a to-do list for a visual pass over this document, not as decoration.

---

## 0. Project scope at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  RAW APPLICATIONS (console CSV, portal sync, or bulk COPY — raw      │
│  data_for_ml_model.csv schema only, no pre-engineering by senders)   │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
        PostgreSQL system of record  (Part II §11)
        applications · identity_keys · features · scores ·
        confirmed_fraud · loe_patterns · training_runs · drift_baselines
                                │
                                ▼
        DETECTION PIPELINE (Part I §6 — the fixed, locked architecture)
        44-feature engineering → 5-relation identity graph →
        3 detectors (Hybrid GraphMCM · Subspace IF · Dense-block) →
        EVT thresholds → locked score-level fusion → XAI cards
                                │
                                ▼
        CONSOLE (vanilla-JS, nginx-served, Part I §8)
        Review queue · Pattern queue (LOE) · Model audit & deploy (admin)
                                │
                                ▼
        MODEL LIFECYCLE (Part II §11.5)
        Incremental/full retrain (in-cluster) OR install a GPU-laptop-
        trained checkpoint (external ingestion, validated, atomic hot-swap)
```

What's fixed, and why you shouldn't relitigate it: the detection architecture — three detectors plus locked fusion weights — was settled through a six-mode head-to-head comparison recorded in `docs/HISTORY.md`, with the metrics that justify every weight sitting right there. This document's job is to explain that fixed system clearly and describe the scale remodel built around it, not to reopen model-architecture decisions.

What actually changed for scale is a short list, but each item is a real rewrite: file-based storage became PostgreSQL, whole-frame pandas became SQL-pushdown plus a persisted scaler, all-pairs graph edges became hub-capped edges, and full-graph training became exact-neighborhood mini-batch training. The model math itself — every loss function, every hyperparameter, every score formula — is byte-for-byte the same code as before the migration.

Who touches what:

| Layer | Technology | Owns |
|---|---|---|
| Storage | PostgreSQL 18 | Every application, feature vector, score, label, pattern, run record |
| Detection | PyTorch + PyG + scikit-learn (`src/*.py`) | The three detectors, EVT, fusion, XAI |
| API | FastAPI + Celery + Redis (`src/api/`) | HTTP surface, async training jobs |
| Console | Vanilla JS + nginx (`frontend/`) | Everything a reviewer or admin clicks |
| Orchestration | Docker Compose (dev) / k3s (prod) | Bringing all of the above up together |

---

# PART I — THE SYSTEM AS IT IS

## 1. One-paragraph summary

The system is a rule-free, unsupervised fraud detector. It takes raw scholarship application rows, engineers **44 numeric model features** per application, and builds a **5-relation identity graph** — shared mobile, IP, father-name, mother-name, pincode. Every application then gets scored by three detectors: a **Hybrid GraphMCM** (a masked-feature + RGCN two-stream reconstruction model), a **per-group subspace Isolation Forest**, and an **IP-gated dense-block detector**. The three scores are combined by a **locked score-level weighted fusion** into a single `final_risk_score` between 0 and 1, where higher always means more anomalous. **EVT (extreme value theory)** fits statistical thresholds to each score's tail, and a **human-gated self-training loop** turns EVT-tail cases into pseudo-labels only after a supervisor has reviewed them. An **XAI layer** produces evidence-first explanation cards, both JSON and interactive HTML, for every flagged application. There are no hand-written rules anywhere in this system — every threshold is either EVT-derived or learned from programmatically constructed synthetic exposure.

## 2. Pipeline dataflow (file-based core; Postgres wraps it — see Part II)

```
data/raw/data_for_ml_model.csv  (15,000 raw rows, 136 raw columns)
        │
        ▼  ①  src/tabular_feature_engine_v3.py :: build_base()
data/processed/engineered_features_v3_nodeg.csv        (63 base features)
        │
        ▼  ②  src/graph_builder_v3.py :: build_graph()
data/processed/identity_graph_v3.pt   (PyG HeteroData, 5 edge types)
data/processed/degree_features_v3.csv (5 per-relation degree counts)
        │
        ▼  ③  src/tabular_feature_engine_v3.py :: add_degree_features()
data/processed/engineered_features_v3.csv   (68 → identifier-drop → 44 features)
data/processed/v3_feature_schema.json       (the authoritative 44-name list)
        │
        ▼  ④  src/synthetic_exposure_builder_v3.py
data/processed/synthetic_exposure_set_v3.pt      (tabular LOE anomalies)
data/processed/synthetic_exposure_graph_v3.pt    (topology LOE clusters)
        │
        ▼  ⑤  src/hybrid_graphmcm_v3.py           (train Stage 1 + Stage 2, score)
outputs/hybrid_scores_v3.csv                 models/hybrid_graphmcm_v3.pth
        │
        ├──▼  ⑥  src/subspace_if_v3.py  → outputs/subspace_if_scores_v3.csv
        ├──▼  ⑦  src/dense_block_detector_v3.py  (IP-gated dense-block score)
        ├──▼  ⑧  src/evt_scorer_v3.py   → outputs/evt_thresholds_v3.json
        │
        ▼  ⑨  src/self_training_loop_v3.py  → outputs/pseudo_labels_v3.json  (HUMAN-GATED)
        ▼  ⑩  src/fusion_classifier_v3.py   → outputs/risk_scores_v3.csv
        ▼  ⑪  src/xai_layer_v3.py           → outputs/explanation_cards_v3.json
        ▼  ⑫  src/xai_card_html_v3.py       → outputs/cards/*.html  (served via API)
```

Every arrow here is a file contract — modules never share in-memory state. Since the V4-Scale migration, steps ①–④ also write into PostgreSQL (Part II §11.4), and steps ⑤–⑫'s outputs are mirrored into `scores`/`training_runs` for Postgres-backed console reads. But the file contracts above remain the authoritative pipeline; Postgres is a synchronized mirror, not a parallel universe.

> **[SVG PROMPT — Pipeline Architecture Diagram]**
> A left-to-right flow diagram with four horizontal bands:
> 1. **Ingestion** — a box "Raw CSV / portal sync / bulk COPY" feeding into a
>    cylinder labeled "PostgreSQL — applications, identity_keys".
> 2. **Feature + graph** — two parallel boxes "Feature engineering (44-dim)"
>    and "Identity graph (5 relations)", each with a small annotation "SQL-
>    pushdown at scale" / "hub-capped at scale", feeding into...
> 3. **Detection** — three parallel boxes side by side: "Hybrid GraphMCM
>    (RGCN, root_weight=False)", "Subspace Isolation Forest", "Dense-block
>    (mobile+IP, IP-weighted)", each with its one-line job description
>    from §6, converging into a single diamond "Locked score-level fusion"
>    with the formula `risk = minmax(max(subspace, dense_relational, hybrid))`
>    printed beside it. A fourth, visually distinct box "Deep SAD
>    (supplementary)" feeds ONLY into "XAI cards" directly, bypassing the
>    fusion diamond — tested as a 4th fusion input and rejected (no
>    improvement, see §6.6b).
> 4. **Output** — the fusion diamond feeds into "EVT thresholds", then
>    branches to "XAI cards" and "Human-gated self-training", both feeding
>    into a "Console" box at the far right.
> Use the existing console's dark theme (near-black background, cyan/orange
> accent) for consistency with the product itself. Label every arrow with the
> file or table it writes (e.g. `engineered_features_v3.csv` / `features` table).

## 3. Data ingestion & preprocessing — exact steps

### 3.1 Load & clean (`_load_and_clean`)

The raw CSV comes in with `low_memory=False` across 136 columns. From there, cleaning drops 16 all-null columns (`NULL_COLS_TO_DROP` in `config_v3.py`) — `updated_by, delete_record, deleted_by, delete_on, delete_ip_address, deleted_by_level, c_university_id, p_institution_id, x_institution_id, xii_institution_id, competitive_exam_score, xii_course_id, new_entitled_fee_amount_centre_share, sub_category_id, updated_by-2, updated_on-2` — and 7 duplicate columns (`DUPLICATE_COLS_TO_DROP`): `state_id, state_id-2, pfms_state_code, state_name-2, district_id, district_name-2, district_short_name`. High-nullity columns get filled with typed defaults instead of being dropped: `disability_percentage→0, disablity_type→0, orphan_flag→0, gaurdian_name→"", enroll_udid_no→0, ration_card_no→0, ration_card_member_no→0`. `application_id` is kept for row tracking but never enters the feature matrix, and `sanity` and `jwt` are excluded entirely — that's hard stop #4.

### 3.2 Feature engineering (`_engineer_features`)

A handful of scalar features are derived directly. `age_at_registration` is `(registered_date − date_of_birth) / 365.25`, clipped to be ≥ 0. `admission_fee, tution_fee, misc_fee, annual_family_income` are coerced to numeric, with NaN filled to 0 and clipped ≥ 0. `fee_income_ratio` is the sum of admission, tuition, and misc fees divided by `max(income, 1)`. `name_similarity_score` is a `difflib.SequenceMatcher` ratio between the applicant's name and the father's name, both lowercased and stripped first.

Three boolean identity-match features follow the same lowercase-stripped equality check: `is_applicant_name_eq_father`, `is_applicant_name_eq_mother`, `is_father_name_eq_mother`.

The cross-row aggregates are the scale-sensitive ones, since they're groupby-transforms: `mobile_application_count` counts rows sharing a mobile number, `ip_application_count` counts rows sharing an IP address, `mobile_unique_names` / `mobile_unique_fathers` count the number of distinct applicant/father names seen per mobile number, `institute_application_count` counts rows per `c_institution_id`, and `ip_to_mobile_ratio` is `ip_count / max(mobile_count, 1)`.

Two features are relative to district/state: `income_rank_in_district` is the percentile rank of income within `permanent_district_id`, and `income_deviation_from_state_median` is income minus the state median income (grouped by `domicile_state_id`).

Finally, a set of categoricals get binary-encoded as-is: `is_female`, `is_rural`, `is_urban`, `disability_flag`, `orphan_flag`, `hosteller`, `is_singlegirlchild`, `has_state_verify`.

### 3.3 Select & scale (`_select_and_scale`)

Non-numeric columns and anything with more than 50% nulls get dropped first. Then a `log1p` transform is applied to the four heavy-tailed money columns (`LOG1P_COLS`): `annual_family_income, admission_fee, tution_fee, misc_fee`. Remaining NaNs become 0, and finally a **MinMaxScaler** puts every feature into `[0, 1]`. At scale, the fitted scaler parameters are persisted and re-applied to every later batch rather than refit — see Part II §12.3; that change is a correctness fix as much as it is a scale one.

This stage's output is **63 base features**.

### 3.4 Degree merge & identifier drop (`add_degree_features`)

The five graph-degree features — `degree_shares_mobile/ip/father_name/mother_name/pincode` — are merged in next, each min-max scaled, bringing the count to **68 features**. From there, the **24 nominal identifier/code features** (`IDENTIFIER_FEATURES` in `config_v3.py`: mobile_no, aadhaar token, pincode, village/district/institution/course/university IDs, religion, marital_status, and so on) get dropped. They're nominal codes with no ordinal meaning, and the noid ablation (2026-07-15) showed dropping them causes no regression at either the detector or fused level — their sharing signal survives anyway, through the graph edges (which are built from the raw columns, not the model features) and through the count/degree features.

What's left is the final model input: **44 numeric features**, listed authoritatively in `data/processed/v3_feature_schema.json` (`N_FEATURES = 44`).

## 4. Identity graph construction (`graph_builder_v3.py`)

Every application becomes one node, with the 44-dim feature vector as its node features. On top of that sits **5 typed edge sets**, built from the raw columns rather than the model features: `shares_mobile, shares_ip, shares_father_name, shares_mother_name, shares_pincode`. For each raw column, rows are grouped by value, and every pair of rows sharing a value gets an edge — undirected, stored in both directions. At scale this becomes hub-capped (Part II §12.4), capped so a single shared value can never produce an unbounded edge count. The output is a PyG `HeteroData` object (`identity_graph_v3.pt`) plus per-node degree counts per relation (`degree_features_v3.csv`). All 5 relations are still built and stored here — `graph_builder_v3.py` is unchanged. Since 2026-08-03, though, the Hybrid GraphMCM's own RGCN encoder only reads 3 of them (`ENCODER_RELATION_IDS` in `config_v3.py`, §5 below); `shares_father_name`/`shares_mother_name` remain in `identity_graph_v3.pt` for `graph_viz_v3.py`'s XAI rings and `dense_block_detector_v3.py` (which only ever reads mobile/ip regardless).

## 5. Synthetic exposure (LOE) — how the model learns "what fraud looks like" without rules

`synthetic_exposure_builder_v3.py` **programmatically constructs** anomalies — hard stop #7 rules out a tabular GAN entirely, since CTGAN/TVAE were found to degrade fraud behavioral signals 24×. Two artifacts come out of this: a tabular exposure set (`synthetic_exposure_set_v3.pt`), where feature vectors are perturbed along fraud archetypes such as income inflation, fee manipulation, and identity collision patterns; and a topology exposure graph (`synthetic_exposure_graph_v3.pt`), roughly 50 synthetic connected clusters (6–40 nodes each, `N_TOPO_CLUSTERS`, `TOPO_CLUSTER_SIZE_RANGE`) injected as dense shared-attribute rings. Confirmed real patterns join here too: when a supervisor promotes a flagged ring through the Pattern queue's Promote action, or through CSV pattern intake, its real subgraph is appended as a topology-exposure cluster — this is the loop that turns investigations into training signal.

> **Naming note (corrected 2026-07-22):** this section previously expanded
> "LOE" as "Latent Outlier Exposure" — that is the name of a *different*,
> unrelated technique (Qiu et al., ICML 2022: jointly re-inferring which
> unlabeled real training points are secretly contaminated, via
> block-coordinate updates). This project's LOE is this project's own
> synthetic-exposure margin mechanism (below) — the two were never the same
> thing, and the Qiu et al. method was itself prototyped separately against
> `stress_testing_1` this session and rejected (not adopted; see README
> changelog). Do not conflate the two when reading older records.

During training, LOE pushes exposure samples' embedding-distance-to-centroid **up** past a data-derived margin (`_derive_loe_margin` — the 75th percentile of real nodes' own distance to the centroid, recomputed each epoch) via a hinge loss, while normal reconstruction pulls real-data errors **down** (`LAMBDA_EXPOSURE = 1.0`). This margin was changed 2026-07-22 from a fixed `LOE_MARGIN = 2.0`: measurement showed 2.0 was roughly 3x too small for this embedding scale, so the margin term had been contributing effectively zero gradient throughout training, since some point after RGCN adoption — the mechanism had silently stopped working. Stage 2 (free reconstruction) also gained a small **persistent** LOE term it previously lacked entirely (`LOE_STAGE2_WEIGHT = 0.15`, non-decaying), so the separation bought in Stage 1 can't be freely re-absorbed by 120 epochs of unconstrained reconstruction.

## 6. The detectors — how each model works, and how each one reaches a conclusion

### 6.1 Hybrid GraphMCM (`hybrid_graphmcm_v3.py`) — the relational detector

This detector runs two streams into one predictor. The **feature stream** does masked cell modeling (MCM): `MASK_NUM = 8` learned mask vectors sit over the 44-dim input, each hiding a learned subset of features, and the model has to predict the hidden values from the visible ones — per-feature reconstruction error then measures how "surprising" each declared value is. The **graph stream** is a 2-layer **RGCN** over 3 of the 5 typed edge sets (`shares_mobile`/`shares_ip`/`shares_pincode` — `ENCODER_RELATION_IDS` in `config_v3.py`; `shares_father_name`/`shares_mother_name` excluded since 2026-08-03, see §4 above and the README changelog) (`GRAPH_HIDDEN = 128` → `GRAPH_EMB_DIM = 64`), producing a 64-dim neighborhood embedding `h_N` per node. `N_EDGE_TYPES` stays 5 — the architecture (RGCN relation-basis count, edge-predictor output width) is unchanged; the two excluded relations simply never appear in the encoder's input, so their relation-basis weights stay at initialization. (A HAN encoder also exists, behind `V4_ENCODER_ARCH=han`, but it regresses −0.091 over 3 seeds, so RGCN stays the default.) Since 2026-07-22, `root_weight=False`: `RGCNConv` defaults to `root_weight=True`, which adds a learned self-transform of a node's own unmasked features directly into `h_N`, independent of the MCM masking described above — leaking self-signal around the intentional mask for every connected node (isolated nodes were already unaffected, since they're overridden by `isolated_embedding`). Turning it off makes `h_N` pure multi-relation neighbor aggregation, which is the actual MCM contract. This was validated on `stress_testing_1` (overall PR-AUC 0.153→0.201, mobile-ring 0.029→0.078, no low-degree regression) and on the real 15k set (5/5 V2 floors still pass, edge-dropout retention 2.34). A **fusion MLP** then concatenates the masked features with `h_N`, passes through `MLP_HIDDEN = 256` → `Z_DIM = 64`, and outputs a predicted feature vector plus edge-existence probabilities.

Here's how that architecture reaches a conclusion for one application. The model takes the application's 44-dim feature vector together with its position in the 5-relation graph, and runs the feature vector through all 8 learned masks, averaging the masked views — that average is the model's best guess at what a normal applicant with this partial information looks like. In parallel, the 2-layer RGCN runs over the applicant's typed neighbors — shared mobile, IP, names, pincode — producing a 64-dim summary of what this applicant's network looks like. Both pieces get concatenated and passed through the predictor MLP, which outputs a predicted version of every one of the 44 features and a predicted probability for each of the 5 edge types. Comparing predicted against declared gives two error terms: `feature_pred_error` is the mean absolute error across all 44 features, and `edge_pred_error` is the binary cross-entropy between predicted and actual edge presence. The final score is `hybrid_anomaly_score = feature_pred_error + LAMBDA_EDGE_SCORE·edge_pred_error`, with `LAMBDA_EDGE_SCORE = 0.0` since 2026-07-23 (it was 0.3 before that — so today the score is effectively `feature_pred_error` alone, since `edge_pred_error` showed no usable signal on any of its 3 designed relational categories and was diluting a real one; see `AGENTS.md` §1/§7 open decision 7). `edge_pred_error` is still computed and available for XAI even though it no longer feeds the score. **A high score means: given everything else about this applicant and their network, the model didn't expect the values they declared.** Note that `LAMBDA_EDGE_SCORE` is distinct from `LAMBDA_EDGE` below — this change decoupled the score weight from the training-loss weight, it didn't zero both.

Training runs in two stages, always with seed 42. **Stage 1** (80 epochs) is LOE pre-training against the synthetic exposure set, where the model learns a margin between normal geometry and fraud archetypes. **Stage 2** (120 epochs) is a joint objective on real data: `L = feature_reconstruction + LAMBDA_EDGE(0.3) · edge_reconstruction + LAMBDA_EXPOSURE(1.0) · LOE_margin_loss + DeepSVDD compactness`, where the centroid is the mean of the bottom-95%-norm embeddings (`CENTROID_CLEAN_PERCENTILE`). `LAMBDA_EDGE` here is the training loss weight, unchanged by the 2026-07-23 score-composition change above — the edge-prediction head still trains at full weight, LR is 1e-3, batch size 256, Adam.

The incremental fine-tune that runs after each cycle is a CPU update: 10 epochs at LR 1e-4, with the RGCN frozen — only the MLP head adapts to newly confirmed fraud.

One hard boundary matters here above all others: the 64-dim `h_N` embedding never leaves this module (hard stop #2). Only scalar scores and attention weights are ever exported.

### 6.2 Subspace Isolation Forest (`subspace_if_v3.py`) — the tabular backbone

This detector runs three **independent Isolation Forests**, one per semantic feature group (`SUBSPACE_GROUPS`):

| Group | Features |
|---|---|
| financial | annual_family_income, fee_income_ratio, income_rank_in_district, income_deviation_from_state_median, admission_fee, tution_fee, misc_fee |
| identity | name_similarity_score, is_father_name_eq_mother, is_applicant_name_eq_father, is_applicant_name_eq_mother, mobile_unique_names, mobile_unique_fathers |
| network | ip_application_count, ip_to_mobile_ratio, mobile_application_count, institute_application_count, degree_shares_ip, degree_shares_mobile, degree_shares_pincode |

An Isolation Forest reaches its conclusion by randomly partitioning the feature space: a point that's easy to isolate — few splits needed — is anomalous, while a point buried in a dense cluster, needing many splits, is normal. Running one IF per group above, independently, means an applicant who's only anomalous on, say, their financial numbers doesn't get diluted by being perfectly normal on identity/network features — a single 44-dim full-space IF would average that signal away. The combined `subspace_if_score` is the highest of the three group scores. This is the **dominant fusion component**: it wins 4 of the 5 fraud categories raw, and it's the only structural signal available for **isolated nodes** — applicants with a unique mobile and a unique IP, and therefore zero graph edges.

### 6.3 Dense-block detector (`dense_block_detector_v3.py`) — the shared-identity ring specialist

Reconstruction models have a specific blind spot: they smooth over dense cliques, because a tight fraud ring reconstructs *easily* and so weakens the relational signal (this is the MAR critique). The dense-block detector exists to attack exactly that blind spot.

It's gated to `shares_mobile` + `shares_ip` (`DENSE_BLOCK_RELATIONS = [0, 1]`) — it used to be `shares_ip` only, until extending to mobile+pincode was tested against `stress_testing_1` after the IP-only gate was found to score mobile-sharing rings near zero (PR-AUC 0.030). Each relation is peeled independently, then combined via an IP-priority-weighted max: `DENSE_BLOCK_RELATION_WEIGHTS = {mobile: 0.3, ip: 1.0}`. IP stays dominant, since it's the real population's primary fraud vector, while mobile contributes as a boost rather than an equal. Equal weighting was tried first and rejected — it gained more in aggregate, but let ordinary non-fraud density in mobile/pincode outrank true IP rings, collapsing IP PR-AUC from 0.220 to 0.067. Pincode was briefly added to the gate on 2026-07-22 (`DENSE_BLOCK_RELATIONS = [0, 1, 4]`, weight 0.2) and dropped the same day per lead direction: shared pincode reflects legitimate geographic clustering, not collusion, and isn't a valid fraud signal on its own for this detector — so it was reverted to mobile+ip only. The output columns are `dense_block_score_mobile/ip` (per-relation, kept for XAI transparency) plus `dense_block_score_relational` (the weighted max, which is what fusion actually consumes). Under the hood, a k-core prefilter narrows candidates, then greedy peeling extracts the dense blocks, using camouflage-resistant weighting `w = 1/log(deg + 5.0)` (`DENSE_BLOCK_CAMOUFLAGE_C`).

This detector doesn't look at any of the 44 features at all — it's purely graph structure, run separately on each of the 3 gated relations. For each relation it repeatedly removes the lowest-weighted-degree node from that relation's graph — FRAUDAR-style greedy peeling, with the complexity citation in §13.3 — until what's left is a provably dense subgraph. Every application inside a dense remainder gets a per-relation score proportional to how dense and camouflage-resistant that block is, and the three per-relation scores are min-max normalised and combined via the IP-weighted max described above. **A high `dense_block_score_relational` means this application is part of a mathematically dense cluster of applications sharing one identity value** — mobile or IP — exactly the "many students, one internet connection / one phone" signature a reconstruction model alone would miss.

### 6.4 EVT scorer (`evt_scorer_v3.py`) — statistical thresholds, not policy

This module fits a **Generalized Pareto Distribution** to each score's upper tail using peaks-over-threshold. The shape parameter must land in `[-0.5, 1.0]` (`EVT_SHAPE_MIN/MAX`), or the fit is rejected and an empirical quantile is used instead. The output, `evt_thresholds_v3.json`, holds the only numeric thresholds allowed anywhere in the system — hard stop #1.

### 6.5 Self-training loop (`self_training_loop_v3.py`) — human-gated pseudo-labels

Round 0 candidates are applications that exceed EVT thresholds on at least 2 independent signals (`MIN_SIGNALS_FOR_PROMOTION = 2` — single-signal OR-promotion turned out to be too noisy). The Round 0 classifier-agreement condition is code-enforced OFF, and every round requires a human PR-AUC check before its labels feed the next cycle — that's hard stop #5. The output is `pseudo_labels_v3.json`. Supervisor-confirmed fraud coming in from the console enters as hard labels, with sample weight `CONFIRMED_WEIGHT = 3.0`.

### 6.6 Fusion (`fusion_classifier_v3.py`) — LOCKED score-level max

> **Fusion history — read this if your records mention LightGBM or a
> weighted sum.** LightGBM was the *first* fusion layer, superseded by a
> **weighted sum** (`1.0·subspace + 0.5·dense_ip + 0.3·hybrid`), itself
> superseded 2026-07-22 by the **unweighted max** described below. Any
> record showing LightGBM, or a `+` between detector terms, is stale.
> Source: `docs/HISTORY.md` and `outputs/ablation/locked_fusion_validation.json`.

The original LightGBM stacker was removed because, with only 14 positives, the meta-learner had essentially no signal to fit combination weights on — it destroyed calibrated components, dropping subspace PR-AUC from 0.966 to 0.315 and RGCN IP from 0.51 to 0.169. The weighted-sum replacement that followed was itself replaced 2026-07-22, after a `stress_testing_1` ablation showed it diluting strong signals: on every category where one detector actually had signal, the summed fusion scored *worse* than that detector alone — for example, mobile-ring came in at 0.674 for subspace alone versus 0.349 for the summed fusion, because the other two detectors' near-random noise on that category dragged the strong one down. The current, locked replacement is:

```
final_risk = minmax( max( minmax(subspace_if_score),
                           minmax(dense_block_score_relational),
                           minmax(hybrid_anomaly_score) ) )
```

There's no per-component weight — `FUSION_W_*` has been retired, and `FUSION_COMPONENTS = ("subspace", "dense_relational", "hybrid")` in `config_v3.py` just names the 3 inputs, not weights. The conclusion is reached like this: each of the three raw scores is independently rescaled to [0,1] across the current population, and the fused score is whichever one is highest for that application — attribution is therefore exact and binary (the "driver"), not a proportional share. There's still no learned combination step and no retraining needed for fusion itself, since it's pure arithmetic. Overall PR-AUC on `stress_testing_1` moved 0.403 (sum) → 0.447 (max) at introduction, and now sits at 0.418 with the current root_weight-fixed Hybrid GraphMCM — these numbers move as upstream detectors change, so re-derive before citing. Output: `outputs/risk_scores_v3.csv`.

### 6.6b Deep SAD (`deepsad_detector_v3.py`) — supplementary, NOT in fusion

A 4th detector exists, but it's deliberately kept outside the fusion above: a separate 2-layer RGCN encoder, with its own checkpoint (`models/deepsad_v3.pth`), trained with a Deep SAD objective (Ruff et al., ICLR 2020) — pulling real nodes toward a learned normal center, and pushing topology-exposure's synthetic archetypes away via an inverted-distance term, with **no reconstruction loss** at all. That absence of a reconstruction term is the point: it doesn't inherit the MAR smooth-over-dense-cliques failure mode the way `hybrid_graphmcm_v3` partially does. It was validated on `stress_testing_1` as the single strongest relational signal found this session — 0.201 overall, 0.093 mobile-ring, 0.050 IP-ring, against `hybrid_anomaly_score`'s 0.153, 0.029, and 0.032 on the same categories. And yet it was **tested directly as a 4th fusion input and rejected** (2026-07-22): a candidate `max(subspace, dense_relational, hybrid, center_dist_score)` scored 0.4181 against the locked 3-way's 0.4182 on `stress_testing_1` — noise-level, not an improvement, because Deep SAD only won the argmax on fewer than 1% of nodes. The existing trio already covers its specialty categories too well for a 4th max-input to matter. So instead, `center_dist_score` is surfaced on XAI cards (§6.7) as a supplementary signal — shown when it's above the 75th population percentile — with an explicit "does not drive the fused score" note, and it's exported as a `deepsad_percentile` scorecard column. It reads the `deepsad` pipeline step (`main_v3.py`, between `dense_block` and `evt`); its own `CENTROID_CLEAN_PERCENTILE=95` clean-population heuristic was itself prototyped for a contamination-aware replacement — Qiu et al.'s Latent Outlier Exposure, unrelated to this project's own LOE, see §5 — and that replacement was tested and not adopted (see the README changelog for the alpha-sensitivity result).

### 6.7 XAI layer (`xai_layer_v3.py` + `xai_card_html_v3.py`)

Every flagged application gets an evidence-first JSON card: ranked reason codes, per-feature declared-vs-model-predicted values pulled from the detector's per-feature error export, a closed-form fusion attribution (exact, not a SHAP approximation — since fusion is max, this means a single argmax DRIVER plus a margin over the next-highest detector, not a proportional split), a supplementary Deep SAD center-distance signal when it's elevated (above the 75th percentile, explicitly marked as not driving the fused score), EVT threshold context, and a `model_trace` recording which checkpoint or component produced each line.

A few enhancements landed on 2026-07-22, all worth knowing about individually. **RGCN per-relation ablation** (`hybrid_graphmcm_v3.compute_relation_ablation()`) exists because the production RGCN encoder has no learned attention — unlike the rejected HAN path, whose `beta_r`/`top_alpha` stay dormant — so "expected this based on neighbours" previously had no way to say *which* relation drove that expectation. This fills the gap post-hoc with 5 extra full-graph forward passes, one per edge type, masked via `edge_type_tensor` and not retrained, comparing each node's feature-reconstruction error with versus without that relation's edges. The relation whose removal improves the fit most gets narrated as the "Neighbourhood-expectation driver" and shown as a bar on the Signal drivers tab — it's XAI-only, and never feeds fusion or a threshold.

**EVT empirical-rate framing** means trigger sentences now cite the actual measured flagged rate for that signal's fitted threshold — `n_flagged` over population size, computed in every `evt_scorer_v3._fit_evt()` branch including both fallbacks — rather than just the aspirational target `Q` the tail was fit towards. That gives a stronger, per-signal justification, something like "this pattern was this extreme in 31 of 15,000 applications, ~0.21%," instead of one blanket target rate for the whole card.

**Dense-block core highlighting on the 3D ring** used to show every shares-X neighbour uniformly; now a gold diamond outline marks nodes that are part of the actual Charikar-peeled dense core (`dense_block_score_relational > 0`), distinguishing them from incidental neighbours that share the same identity value but aren't part of the anomalous structure. This lives in `xai_card_html_v3._dense_core_app_ids()`, mtime-cached like the existing graph-context cache.

**Per-relation edge toggle on the 3D ring** gives each relation its own Plotly trace with an explicit `itemclick="toggle"` / `itemdoubleclick="toggleothers"` legend — previously this relied on Plotly's default behavior, now it's made explicit — so a reviewer can hide or isolate relations to declutter a dense clique. This fix rode alongside a real rendering bug in the same function: `_figure_for_ring` used to keep only the first relation seen for a node pair (`edge_rel.setdefault`), so a pair sharing both `shares_ip` and `shares_mother_name`, say, would only ever draw the IP edge — `shares_mother_name` was silently shadowed whenever a lower-`EDGE_TYPES`-index relation also connected the same two nodes. The fix tracks a set of relations per pair and draws one line per relation that actually connects it, which also means the toggle now works correctly for such pairs — previously, toggling off the shadowing relation wouldn't have revealed the hidden one, because it was never rendered in the first place.

**Cohort-preview signal drivers** means `POST /evaluate-dataset` now also computes subspace IF (`subspace_if_v3.compute_subspace_if_scores`, refactored out as a reusable pure function) and dense-block scores over the batch's merged population, plus a preview fusion score via `fusion_classifier_v3.score_level_fusion` / `xai_layer_v3.build_fusion_contributions` — the same functions the committed pipeline uses, so there's no separate logic to drift out of sync. A staged cohort card's Signal drivers tab used to be empty by design, since only the raw hybrid score existed pre-commit; now the same bars and fusion-composition footer render, renormalised over the cohort's own population — still clearly labeled PREVIEW, still not `risk_score_v3`, and still without EVT triggers, since those are fitted against the canonical population rather than a moving cohort.

The narration policy is simple: cards narrate only continuous features and network-DISAGREEMENT binaries, nominal identifiers are never spoken, and the whole layer is presentation-only — XAI never gates a score. The HTML reviewer cards themselves are interactive: a gauge, comparison bars, and an identity network, with lazy-loaded Plotly 3D identity rings and flat ego-graphs, served via API at `/card`, `/ring`, `/topology`.

> **[SVG PROMPT — Explanation / Reviewer Card Layout]**
> A two-column mockup matching the console's dark theme (near-black
> background `#0d1117`, cyan `#4cc9f0` accents, risk colors: high `#e5383b`,
> medium `#f4a261`, low `#2c7da0`). Left column: a tab bar with "Identity
> network" (active) / "Signal drivers"; below it, a small force-directed
> graph — one center node (the applicant, larger, orange ring) connected by
> pink edges to up to 6 neighbor nodes (smaller, colored by their own risk),
> a legend for edge-type colors (mobile/ip/father_name/mother_name/pincode),
> and a caption line like "shares IP with N other applications — more
> connected than X% of applicants." Right column, top to bottom: a circular
> risk gauge (0–1, conic gradient by risk color) with headline text; a
> "Why it flagged — ranked reason codes" numbered list (2–3 entries, each
> with a colored source-model pill: red "Tabular subspace", pink "Shared-
> identity dense-block", cyan "Relational RGCN"); an expandable "What's
> happening in each field" accordion showing one open field with two
> horizontal bars (declared value vs. model-expected value, ± signed, red vs.
> blue) and an explanatory sentence; a "How this score was produced" section
> with three percentage bars (one per fusion detector, colored to match their
> pills, the single argmax marked DRIVER — not a proportional share, since
> fusion is max not sum), plus a 4th, visually distinct purple bar for "Deep
> SAD (supplementary)" shown only when elevated, explicitly captioned as not
> driving the fused score; a reviewer-decision form (name field, fraud-type
> dropdown, Confirm/Mark-false-positive/Undo buttons) at the bottom.

> **[SVG PROMPT — 3D Identity Ring]**
> A 3D scatter/network render (Plotly-style) on a near-black background:
> the center application as a large red sphere, its identity-ring neighbors
> as smaller spheres colored on a teal-to-red risk gradient, connected by
> thin lines colored by relation type (mobile=teal, ip=pink, father_name=
> green, mother_name=purple, pincode=gold) per the legend already used in
> the console. Include an axis-box wireframe (as Plotly 3D does by default),
> a title reading "Identity ring — {application_id}    N nodes · M edges ·
> risk R.RRR", and a small side legend matching the console's relation-color
> key. Show one dense, tightly-clustered ring of ~40–50 nodes (illustrating
> a real shared-IP fraud ring) to convey scale, not a sparse toy example.

## 7. Backend (FastAPI + Celery + Redis)

`src/api/` is served by `nic-api`, with jobs executed by `nic-worker` (Celery, `concurrency=1`, replicas fixed at 1 as a hard stop — training jobs write fixed output paths, and two workers would corrupt each other).

| Area | Endpoints (prefix `/v3/...`) |
|---|---|
| Review queue | `GET /top-suspicious` (paged), `GET /{app_id}/card`, `/ring`, `/topology`, `/export` |
| Cohort preview | `GET /cohorts`, `/cohort/{name}/top-suspicious`, per-app card/ring/topology/export, `export-bulk`, `export-selected`, `POST /cohort/{name}/delete` |
| Supervisor labels | `POST /confirm-fraud`, `/mark-false-positive`, `/clear-label`, `/confirm-batch` (batch label + optional retrain) |
| LOE patterns | `GET /patterns`, `/patterns/all`, `/patterns/coverage/{app_id}` (dedup banner), `POST /patterns/confirm`, `/patterns/promote`, `/patterns/delete` |
| Pattern CSV intake | `POST /pattern/test` (read-only scoring of an uploaded ring), `POST /pattern/ingest` (permanent ingest + topology-exposure + fine-tune) |
| Dataset intake | `POST /upload-dataset`, `POST /evaluate-dataset` (read-only cohort scoring + drift p-value), `POST /push-dataset` (2026-07-23: CSV-free portal/ETL intake — server-side path, not inline rows, stages + auto-evaluates async, no merge/retrain), `POST /decision` (merge / retrain, human-gated) |
| Training | `POST /incremental`, `/full` (`data_source=file\|postgres` param added 2026-07-23 — `postgres` reads every merged batch straight from Postgres, no CSV), `GET /jobs/{job_id}` |
| Model lifecycle | `GET /checkpoint-info`, `/registry`, `POST /upload-checkpoint` (external GPU model ingestion — Part II §11.5), `/pull-checkpoint`, `/rollback` |
| Monitoring | `GET /drift` (KS on score distribution, alert at p < 0.01), `/drift-explain` (feature-level KS over the 44 model features), `/fraud-store-summary`, `/stats`, `/dataset-xai`, `GET /health`, `/ready` |

As for persistence today: PostgreSQL is the default read path (Part II §11.4), while files remain the write-authoritative source during the migration. `NIC_READS_FROM_PG=0` forces the file path, and any Postgres failure falls back to files automatically — never a hard error. Redis holds only Celery job state.

## 8. Frontend (vanilla JS console, nginx-served)

The console lives at a single origin, `http://<host>:8080/`, with nginx proxying `/v3/*` to the API. It has three tabs — full operator detail is in `docs/OPERATIONS_RUNBOOK.md`. The **Review queue** shows ranked flagged applications (50 per page, roughly 500 carded), with a dataset switcher between the primary 15k population and evaluated cohorts (read-only, pre-fusion), multi-select triage (batch label/retrain, flag-as-ring for LOE, export selected), a reviewer card with a 3D ring and ego-graph, and an "already flagged?" IP-cluster dedup banner. The **Pattern queue (LOE)** holds pending flagged rings, with a Promote action that appends the real subgraph to topology exposure and dispatches an incremental retrain, plus a persistent flagged history tracking promoted/rejected state. **Model audit & deploy (admin)** carries a status strip, a drift explanation, a 4-step deployment loop — Intake (cohort CSV or fraud-pattern CSV) → Evaluate → Decide (human gate) → Watch — run history, an "Install pretrained checkpoint" widget for external GPU model ingestion, and checkpoint rollback.

CSV-intake flows are unchanged by the Postgres migration — CSV upload is just one of several ingestion paths into PostgreSQL now, not a replaced feature.

## 9. How one application batch is processed (operational sequence)

A batch cycle starts with the incoming CSV verified against the raw schema (Intake step 1), followed by submission of the previous cycle's confirmed fraud — console labels or pattern promotion — which feeds LOE exposure, hard labels, and the 3× fusion weight. Next comes a **drift check**: a KS test of the new batch's score distribution against the previous cycle (`DRIFT_KS_THRESHOLD = 0.01`). A p-value below 0.01 recommends a full retrain; otherwise an incremental retrain (10 epochs, MLP-only) is enough. The model update itself is either an in-cluster incremental/full retrain, or an externally GPU-trained checkpoint installed via the admin upload widget (Part II §11.5) — either way, it goes through the human-gated Decide step. Then the full scoring pipeline runs end to end (features → graph → detectors → EVT → fusion → XAI), and reviewers triage the ranked queue, with the EVT-tail sample getting human review before any self-training round is allowed to advance.

---

# PART II — POSTGRESQL + KUBERNETES REMODEL (implemented, gate-tested)

> **Scale target, stated once:** 30–40 **lakh** = **3.0–4.0 million**
> applications. All sizing in Part II assumes **≤ 4M rows**. A 30–40 *million*
> target would invalidate the single-node PostgreSQL and k3s pod sizing below —
> at that order, single-primary write throughput becomes the bottleneck before
> the GNN does, and a separate design round would be required.

## 10. Scale delta and what actually breaks

Going from 15k to 3.5M rows — roughly 233× — breaks four specific things, while everything else scales linearly and still fits the server:

| # | Component | Why it breaks at 3.5M | Fix (section) | Status |
|---|---|---|---|---|
| 1 | Pandas whole-file feature engineering | Raw CSV ~4–6 GB; groupby-transforms over 3.5M rows in one frame ≈ 20–30 GB peak | SQL-pushdown feature engineering (§12.2) | ✅ implemented, bit-exact vs. file pipeline on 15k |
| 2 | Pairwise edge construction | O(k²) per shared value; hundreds of millions to billions of edges | Hub-capped star/ceiling topology (§12.4) | ✅ implemented, verified on synthetic 1M |
| 3 | Full-graph RGCN training | Full-batch message passing over 3.5M nodes cannot fit 64 GB CPU RAM | Exact-neighborhood mini-batch training (§13.2) | ✅ implemented, bit-exact vs. full-graph on 15k |
| 4 | File-based stores (CSV/JSON) | 3.5M-row CSVs re-read per request; JSON stores unindexed | PostgreSQL system of record (§11) | ✅ implemented, all 5 migration steps gate-passed |

Everything else is a non-problem at this scale: subspace IF (sklearn handles 3.5M × 7 easily), EVT (fits on score vectors, and is actually *more* reliable at this scale — §13.1), fusion (vector arithmetic), XAI card generation (only for the flagged tail, and lazy), and the console itself (already paginated).

## 11. PostgreSQL as the system of record

### 11.1 The schema, as actually implemented (`deploy/postgres/schema.sql`)

Every table below is live — this is not a design draft. It's idempotent (`CREATE TABLE IF NOT EXISTS`), so it applies safely on every container start.

```sql
-- Every ingestion path lands rows under a batch (staged → evaluated → merged).
CREATE TABLE batches (
    batch_id    SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('primary', 'cohort', 'pattern')),
    row_count   INT,
    status      TEXT NOT NULL DEFAULT 'staged'
                CHECK (status IN ('staged', 'evaluated', 'merged')),
    drift_p     DOUBLE PRECISION,   -- KS p-values reach 1e-85+ — REAL underflows
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- application_id is TEXT: real portal IDs are alphanumeric (e.g. 'AS202526000000139').
CREATE TABLE applications (
    application_id  TEXT PRIMARY KEY,
    batch_id        INT NOT NULL REFERENCES batches(batch_id),
    raw             JSONB NOT NULL,   -- the full raw row, lossless
    source          TEXT NOT NULL DEFAULT 'csv_upload'
                    CHECK (source IN ('csv_upload', 'portal_sync', 'pattern_csv', 'bulk_copy')),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The 5 identity relations, normalised at ingest, indexed. An "edge" is a
-- shared value here — this is what makes ego-graph/ring queries indexed
-- lookups instead of an in-memory graph scan (§12.5).
CREATE TABLE identity_keys (
    application_id    TEXT PRIMARY KEY REFERENCES applications(application_id),
    mobile_no         TEXT,  ip_address        TEXT,
    father_name_norm  TEXT,  mother_name_norm  TEXT,
    pincode           TEXT
);
-- + one B-tree index per column above

CREATE TABLE features (            -- the 44-dim engineered vector
    application_id  TEXT PRIMARY KEY REFERENCES applications(application_id),
    batch_id        INT NOT NULL REFERENCES batches(batch_id),
    schema_version  TEXT NOT NULL,
    vec             REAL[] NOT NULL CHECK (cardinality(vec) = 44)
);

-- Persisted MinMaxScaler parameters (hard stop 11: fit once, never refit).
-- scale_factor/offset_ are sklearn's fitted scale_/min_ verbatim, so a
-- persisted apply reproduces fit_transform bit-for-bit (Gate 4 evidence).
CREATE TABLE feature_scaling (
    schema_version  TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    col_min         DOUBLE PRECISION NOT NULL,
    col_max         DOUBLE PRECISION NOT NULL,
    scale_factor    DOUBLE PRECISION NOT NULL,
    offset_         DOUBLE PRECISION NOT NULL,
    log1p           BOOLEAN NOT NULL DEFAULT FALSE,
    fitted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (schema_version, feature_name)
);

-- The three FUSED detector scores + fusion, plus Deep SAD (supplementary,
-- XAI-only -- see §6.6b -- deliberately not part of final_risk_score).
-- DOUBLE PRECISION (not REAL): float32 round-trip would change the JSON
-- representation of every score, breaking byte-identical API payload parity
-- (Gate 2 requirement).
CREATE TABLE scores (
    application_id       TEXT NOT NULL REFERENCES applications(application_id),
    batch_id             INT NOT NULL REFERENCES batches(batch_id),
    model_version        TEXT NOT NULL,
    hybrid_anomaly_score DOUBLE PRECISION, feature_pred_error DOUBLE PRECISION,
    edge_pred_error      DOUBLE PRECISION,
    subspace_if_score    DOUBLE PRECISION, group_scores JSONB,
    dense_block_score_relational DOUBLE PRECISION,
    dense_block_score_mobile DOUBLE PRECISION,  -- per-relation, XAI transparency only
    dense_block_score_ip     DOUBLE PRECISION,
    -- (dense_block_score_pincode dropped 2026-07-22: pincode removed from
    -- the dense-block gate — not a valid fraud signal on its own)
    center_dist_score    DOUBLE PRECISION,  -- Deep SAD, supplementary, NOT in final_risk_score
    final_risk_score     DOUBLE PRECISION, label_source TEXT,
    risk_bucket          TEXT CHECK (risk_bucket IN ('High', 'Medium', 'Low')),
    feature_errors       JSONB,    -- per-feature error vector (XAI)
    predicted_values     JSONB,    -- model-expected values (XAI)
    PRIMARY KEY (application_id, batch_id, model_version)
);
CREATE INDEX idx_scores_queue ON scores (batch_id, final_risk_score DESC);  -- the queue query

-- Supervisor hard labels. Mirrors the JSON store field-for-field.
CREATE TABLE confirmed_fraud (
    application_id  TEXT PRIMARY KEY,
    label           TEXT NOT NULL CHECK (label IN ('confirmed', 'false_positive')),
    fraud_type      TEXT, confirmed_by TEXT, cycle TEXT,
    feature_vec     REAL[] CHECK (feature_vec IS NULL OR cardinality(feature_vec) = 44),
    confirmed_at    TEXT, notes TEXT
);

-- Flagged fraud rings. state follows the console's CONFIRMED -> SELECTED ->
-- PROMOTED / REJECTED lifecycle exactly.
CREATE TABLE loe_patterns (
    pattern_id      TEXT PRIMARY KEY,   -- 'pat_<hex>'
    center_app_id   TEXT, fraud_type TEXT,
    state           TEXT NOT NULL
                    CHECK (state IN ('CONFIRMED', 'SELECTED', 'PROMOTED', 'REJECTED')),
    subgraph        JSONB,   -- {"nodes": [...], "edges": [...]} — structure only, NO embeddings
    exposure        JSONB,   -- promote() outcome
    confirmed_by TEXT, notes TEXT, created_at TEXT, updated_at TEXT
);

-- EVT thresholds — the only numeric thresholds allowed anywhere (hard stop 1).
CREATE TABLE evt_thresholds (
    score_name TEXT NOT NULL, model_version TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    gpd_shape DOUBLE PRECISION, gpd_scale DOUBLE PRECISION,
    method TEXT NOT NULL CHECK (method IN ('gpd', 'empirical_quantile')),
    fitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (score_name, model_version)
);

-- Training/audit history. Mirrors model_registry.json field-for-field.
CREATE TABLE training_runs (
    run_id TEXT PRIMARY KEY, ts TEXT NOT NULL, run_type TEXT NOT NULL,
    cycle TEXT, smoke_test BOOLEAN NOT NULL DEFAULT FALSE, status TEXT NOT NULL,
    params JSONB, metrics JSONB, checkpoint JSONB
);

-- Yearly-cycle drift baselines. One row per kind, overwritten each cycle
-- (same behaviour as the JSON files it replaces).
CREATE TABLE drift_baselines (
    baseline_kind TEXT PRIMARY KEY CHECK (baseline_kind IN ('scores', 'features')),
    payload JSONB NOT NULL, saved_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE schema_migrations (   -- versioned migrations ledger (hard stop 14)
    version INT PRIMARY KEY, filename TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Two corrections were made during implementation, worth recording here so nobody re-derives the wrong version later: `application_id` is `TEXT`, not `BIGINT`, because real IDs are alphanumeric; and the three mirror tables (`confirmed_fraud`, `loe_patterns`, `training_runs`) match their JSON predecessors' field shapes exactly, rather than a "cleaner" redesign — this is what lets the dual-write parity gate be an exact field-for-field comparison instead of a lossy approximation.

> **[SVG PROMPT — Postgres Entity-Relationship Diagram]**
> An ER diagram with 10 entities matching the schema above. Draw
> `batches` at the top, with arrows (1-to-many) down to `applications`;
> `applications` fans out (1-to-1) to `identity_keys` and `features`, and
> (1-to-many) to `scores`. `feature_scaling` and `evt_thresholds` are drawn
> off to one side, keyed by `schema_version`/`model_version` rather than by
> application (no FK to `applications`). `confirmed_fraud` and `loe_patterns`
> sit independently (no FK — labels can exist before an app is fully
> ingested). `training_runs` and `drift_baselines` sit at the bottom as
> "audit trail" tables. Use crow's-foot notation for cardinality. Color the
> five tables that hold per-application data (`applications`,
> `identity_keys`, `features`, `scores`, plus `confirmed_fraud`) one color,
> and the four "system/audit" tables (`batches`, `feature_scaling`,
> `evt_thresholds`, `training_runs`, `drift_baselines`) another.

### 11.2 `src/db/` — the only module that touches SQL

Every table above is read and written exclusively through `src/db/` — hard stop 14, no inline SQL anywhere else in the codebase:

| File | Owns |
|---|---|
| `connection.py` | Connection pool, `.env`-driven config |
| `migrate.py` | Applies `schema.sql` + versioned migrations |
| `bootstrap.py` | One-shot startup: migrate → ingest primary batch → replay JSON stores |
| `ingest.py` | Primary-batch ingest + the staged-batch lifecycle (`stage_raw_csv`, `evaluate_batch`, `merge_batch`, `delete_staged_batch`) |
| `reads.py` | Payload-exact read mirrors: `top_suspicious`, `fraud_store_summary`, `n_scored`, `ego_neighbors`, `induced_subgraph_edges`, `risk_scores_for` |
| `features.py` | SQL-pushdown aggregates, persisted-scaler save/load, hub-capped edge groups |
| `stores.py` | Dual-write mirrors for the three JSON stores |
| `drift.py` | Dual-write mirror for the yearly drift baselines |

### 11.3 `db-init` — Postgres is populated by default, not just reachable

A one-shot container service (`docker-compose.yml`) runs `python -m src.db.bootstrap` on every stack startup: it applies the schema, ingests the primary dataset, and replays the confirmed-fraud/pattern/run-history JSON stores into Postgres. `nic-api` and `nic-worker` wait for it (`depends_on: db-init: condition: service_completed_successfully`) before accepting a request. In practice this means Postgres is schema-current and populated the moment the API comes up — not an empty database that merely happens to be reachable. The whole thing is idempotent, so it safely reruns on every restart and keeps Postgres in sync with whatever is currently in `data/`/`outputs/`.

> **[SVG PROMPT — db-init Bootstrap Sequence]**
> A vertical sequence diagram with four lifelines: `docker compose up`,
> `postgres`, `db-init`, `nic-api`/`nic-worker`. Steps: (1) `postgres`
> starts, becomes healthy (health-check icon); (2) `db-init` starts, calls
> `apply_schema()` then `apply_migrations()` against `postgres`; (3) `db-init`
> calls `ingest_primary()` (arrow labeled "reads data/raw + data/processed +
> outputs/*.csv, writes applications/identity_keys/features/scores"); (4)
> `db-init` calls `replay_all()` (arrow labeled "reads the 3 JSON stores,
> writes confirmed_fraud/loe_patterns/training_runs"); (5) `db-init` exits 0;
> (6) only now do `nic-api`/`nic-worker` lifelines start, gated by a dashed
> "depends_on: service_completed_successfully" annotation. End with a small
> callout: "Idempotent — reruns safely on every `docker compose up`."

### 11.4 How Postgres interacts with the console — the read/write mechanism

On the read side, Postgres is the default and files are the fallback. The review queue, status tiles, and model-stats strip call `src/db/reads.py` functions first; `NIC_READS_FROM_PG` (default `1`) gates this, and setting it to `0`, or any Postgres query raising an exception, falls back to the original file-parsing code path automatically — the console never hard-fails because of a database hiccup. Reviewer cards, 3D rings, and ego-graphs for the committed population read from `identity_keys` via two indexed queries (`ego_neighbors`, `induced_subgraph_edges`) instead of loading the multi-million-node `.pt` graph into API memory — this was proven to produce identical node/edge sets to the old graph-file path across 150 sampled ego-graphs and 60 rings. Reviewer cards for an evaluated cohort (a pre-commit preview) read from the cohort's staged graph bundle the same way, so the cohort card gets the same identity-network visual as the committed-population card.

On the write side, the migration is running dual-write, with Postgres becoming authoritative at cut-over. Every write to the confirmed-fraud store, the LOE pattern store, and the training-run registry goes to its JSON file **and** to Postgres — the file write is authoritative, the Postgres write is best-effort, and if it fails, a warning is logged but the request still succeeds. The staged-batch lifecycle is how CSV intake interacts with Postgres specifically: a console CSV upload lands its raw rows in `applications` under a new `batches` row with `status='staged'`, with `identity_keys`, `features`, and `scores` deliberately left empty at this point — that's the lead-set ingestion contract, nothing is derived until an admin acts. Clicking **Evaluate** populates those three tables for that batch, tagged with a preview `model_version`, as a read-only, pre-fusion scoring pass — the Postgres `scores` row stays hybrid-only. The file-based staged bundle (`outputs/staged_scores_<name>.csv`) additionally carries a subspace IF / dense-block / preview-fusion breakdown since 2026-07-22 (§6.7), computed over the batch's merged population; that extra breakdown is what the console's Signal drivers tab reads for a cohort, and it isn't yet dual-written to Postgres. Clicking **Decide → Merge** flips `status='merged'`, which is permanent, and the only state from which training pulls data. A merged batch's rows can never be deleted through the console — only staged/evaluated batches can, via **Remove cohort**, which now warns the user that everything about that cohort (cards, rings, scores, evidence, the uploaded CSV) will be discarded, then cleans both the files and the Postgres rows.

> **[SVG PROMPT — Staged-Batch Lifecycle / CSV Intake Sequence]**
> A four-stage horizontal state diagram: **Staged** (icon: raw CSV rows into
> a database cylinder, "identity_keys/features/scores EMPTY") →
> **Evaluated** (icon: gears turning, "identity_keys + features + pre-fusion
> scores populated, model_version='staged_<name>'") → **Merged** (icon: a
> padlock, "status='merged', permanent, training-visible") — with a fourth,
> parallel dead-end branch off "Staged" or "Evaluated" labeled **Removed**
> (icon: a trash can, "files + Postgres rows deleted — refused if merged").
> Annotate the Evaluated→Merged arrow with "admin: Decide → Merge (human
> gate)" and the Staged→Evaluated arrow with "admin: Evaluate button".

### 11.5 External GPU-trained model ingestion — installing a checkpoint from outside the cluster

The server (`nic-worker`) is CPU-only by design (ADR-010). Full retrains are possible in-cluster, but take hours-to-days at scale (§13.2). For a faster turnaround — or simply because a GPU laptop happens to be available — a checkpoint trained **entirely outside this system** can be installed without ever running training in the cluster.

The mechanism runs end to end like this. First, **training anywhere**: any machine running `src/hybrid_graphmcm_v3.py`'s `train()` (or `train_incremental()`) against the same `config_v3.py` produces a checkpoint with the exact schema the validator requires — this isn't a separate export format, it's the same `torch.save(...)` call the in-cluster trainer uses. Then comes **upload**: from the admin console's Model audit & deploy tab, the "Install pretrained checkpoint (.pth)" widget lets an admin choose the `.pth` file, give it a cycle label and a source note (say, "gpu-laptop full retrain, seed 42"), and click Upload & validate. This calls `POST /v3/training/upload-checkpoint` (multipart), which writes the file to a temp path (`models/incoming_<timestamp>_<uuid>.pth`) — never the live path — and queues a Celery validation job. **Validation** (`checkpoint_manager.validate_and_hotswap`) then rejects the checkpoint, leaving the live model completely untouched, unless it contains exactly `{model_state_dict, centroid, config}`, where `config` contains `N_FEATURES`, `GRAPH_EMB_DIM`, `N_EDGE_TYPES`, and `ARCH_VERSION` matching this deployment's `config_v3.py` exactly — hard stops #9/#15. A checkpoint trained with a different feature count, a different embedding dimension, or the wrong encoder architecture is rejected outright; there's no partial-compatibility mode. On success, an **atomic hot-swap** follows: the current live checkpoint is backed up (`models/hybrid_graphmcm_v3.pth.bak`), a versioned copy is kept (`models/checkpoints/hybrid_v3_<cycle>_<run_id>.pth`, last 5 retained), and the new checkpoint is atomically renamed into the live path — the API never serves a half-written checkpoint file. If the newly installed model turns out to perform badly, **Rollback** (same admin panel) restores any of the last 5 versioned checkpoints by path, through the same atomic-rename mechanism.

Why is this safe to do casually? Because validation happens before anything about the live system changes, so a bad or incompatible upload has zero blast radius beyond a rejected HTTP response. The mechanism is identical whether the checkpoint came from an in-cluster full retrain, an incremental fine-tune, or a laptop with a GPU that trained the exact same architecture on a downloaded copy of the data.

> **[SVG PROMPT — External Checkpoint Ingestion Sequence]**
> A sequence diagram with three lifelines: "GPU laptop (external)",
> "Admin console", "nic-api / checkpoint_manager". Steps: (1) GPU laptop runs
> `train()` against `config_v3.py`, produces `hybrid_v3_custom.pth`
> (annotate: "same {model_state_dict, centroid, config} schema as in-cluster
> training — no special export format"); (2) admin drags the file into the
> "Install pretrained checkpoint" widget, fills cycle + source note, clicks
> Upload; (3) arrow to nic-api: "POST /upload-checkpoint (multipart)"; (4)
> nic-api writes to a temp path (icon: a file with a dashed border, labeled
> "NOT the live path"); (5) a branch: green path "config matches exactly →
> backup live → versioned copy → atomic rename → LIVE" vs. red path "config
> mismatch → 422 rejected → live model UNCHANGED"; (6) a small inset showing
> the Rollback flow as the mirror image of step 5's green path, restoring
> from `models/checkpoints/`.

## 12. Ingestion & preprocessing at scale

### 12.1 Ingest

The ingestion contract is lead-set: every sender — console CSV upload, portal sync, bulk COPY — delivers rows in the raw schema, the shape of `data_for_ml_model.csv`, nothing more. Senders never pre-engineer features or normalise identities. Raw rows land in `applications` under a staged batch, and all preprocessing is this system's job; the derived tables are populated per-batch only when the admin triggers it through the console (Evaluate for read-only scoring, Merge/retrain for permanent).

`COPY` (or `psycopg`'s `copy_expert`) loads roughly 3.5M rows in single-digit minutes. For reference, pganalyze measured 14 s via COPY versus roughly 9,000 s via single-row INSERTs for 10M rows, and Tiger Data's benchmark establishes a roughly 100,000 rows/s sustained baseline for plain COPY.

⚠ These numbers are ingest throughput only — they say nothing about concurrent read/write contention while `nic-api` pods serve the review queue during a merge or retrain window. At ≤ 4M batch-cadence writes this stays manageable.

### 12.2 Feature engineering: SQL-pushdown (implemented, bit-exact)

The cross-row aggregates from §3.2 are pushed down to SQL (`src/db/features.py :: aggregate_features()`): window/group queries replicate pandas' groupby-transforms exactly, including the trickiest two — `income_rank_in_district` (pandas average-rank: `(RANK + (ties−1)/2)/n`, via `RANK() OVER` plus a tie-count window) and `income_deviation_from_state_median` (via `PERCENTILE_CONT(0.5) WITHIN GROUP`). Per-row scalar features — age, ratios, name similarity — still run in pandas on the reconstructed raw frame; the one Python holdout is `difflib` name-similarity, since `pg_trgm` equivalence is still an open decision (§15 #3).

This path is verified: 63/63 base-feature columns and 44/44 final-feature columns are **bit-exact** between the SQL-pushdown path and the canonical file pipeline on the 15k dataset (`IMPLEMENTATION.md` Gate 4).

### 12.3 Persisted scaling parameters (implemented — a correctness fix, not just scale)

The old `MinMaxScaler.fit_transform` on the scored population leaked batch statistics. Now `feature_scaling` stores each feature's exact fitted `scale_`/`min_` — sklearn's own transform coefficients — under a `schema_version`, and `apply_stored_scaling()` re-applies them to any later batch/cohort. It refuses to run, raising an error, if the params are missing rather than silently refitting — hard stop 11. This has been verified within 1 ULP of the original `fit_transform`, a hex-traced single-rounding/FMA-context difference rather than a logic bug (Gate 4).

### 12.4 Edge construction: hub-capped (implemented, verified on synthetic 1M)

"All pairs sharing a value" is replaced with a hub-capped topology (`src/graph_builder_v3.py :: _edges_from_groups`). Groups of size ≤ `k_cap` still get a full clique, unchanged signal. Groups larger than `k_cap` get a star to the group's highest-degree member instead — O(k) edges rather than O(k²), with the same connectivity and degree signal. And groups above a statistical ceiling, derived from the observed group-size distribution's high percentile and never hand-picked (hard stop 1), skip edges entirely — mega-groups such as an ISP's NAT IP shared by thousands are shared-infrastructure noise, not rings, and the count/degree features preserve the size signal regardless. `k_cap` and the ceiling are still open decision #1 (§15); they need a profiling query against real 3.5M data, and the values used in scale testing (`k_cap=50`) were test parameters, not production settings.

This was verified on the synthetic-1M scale test: at `k_cap=50`, 1,657 groups were correctly starred, and edge counts dropped as designed — one relation went from 104k to 10.7k undirected edges in the earlier 15k cap-smoke test.

### 12.5 Ego-graph serving from Postgres (implemented — replaces the `.pt` graph for the console)

The 3D ring / ego-graph endpoints only ever need a 1–2-hop neighbourhood, so they run as two indexed lookups against `identity_keys` (`src/db/reads.py :: ego_neighbors`, `induced_subgraph_edges`) rather than holding a multi-million-node graph in API memory. This has been verified identical to the old `.pt`-graph adjacency across 150 ego-graphs and 60 rings sampled from the top-risk and random populations.

## 13. Model layer at scale

### 13.1 Subspace IF, EVT, fusion — unchanged logic, chunked I/O, EVT actually improves

The Isolation Forest fits on a uniform sample — 500k rows is statistically ample — and scores in chunks, taking minutes on 16 vCPU. EVT's reliability actually *improves* at scale: GPD parameter estimation is sample-size sensitive (Hosking & Wallis 1987), and below n≈500 exceedances, maximum-likelihood estimation is unreliable, with method-of-moments / probability-weighted-moment estimators preferred instead — the 15k dataset's ~15 tail points sit deep in that fragile regime. At 3.5M rows, the same 99.9th-percentile threshold yields roughly 3,500 exceedances, comfortably past the stable-estimation line, so expect fewer `EVT_SHAPE_MIN/MAX` rejections, not more. Fusion itself is unchanged — a three-vector weighted sum — just writing to `scores` instead of a CSV.

### 13.2 Hybrid GraphMCM: exact-neighborhood mini-batch training (implemented, adopted, measured)

Full-graph forward passes are replaced with PyG `NeighborLoader` batching. The fan-out ablation (open decision #2, closed 2026-07-21) tested every truncating fan-out — [15,15], [25,10], [15,10], [50,50] — and every one deviated up to 0.41 from full-graph scores, against a noise-floor bar of 0.03–0.04. Exact-neighborhood batching (fanout `(-1,-1)`), by contrast, reproduced full-graph scores bit-for-bit, with a max deviation of 0.00000 on 15k — a 2-layer RGCN only ever sees the 2-hop neighborhood, so batching it exactly changes nothing. This was adopted as the production sampled path. Memory stays bounded because the *graph* is hub-capped (§12.4), not because the fan-out is truncated — that's what makes exact-neighborhood batching viable at all.

Retrain time was measured on a synthetic 1M-node population (hub-capped test graph, exact-neighborhood batching, CPU 8 threads — the server thread config):

| Measurement | Value |
|---|---|
| Population | 1,000,000 nodes, 44 features, 17.56M directed edges |
| Stage 1 epoch | 406 s |
| Stage 2 epochs | 639 s, 550 s (mean 595 s) |
| Scoring the full 1M | 56 s |
| Peak RSS | **3.53 GB** |

Extrapolating to the real training schedule — 80 Stage-1 epochs plus 120 Stage-2 epochs — gives 80×406s + 120×595s ≈ 28.9 h at 1M, which projects to roughly 101 h (about 4.2 days) at 3.5M under linear row-count scaling. This exceeds an earlier informal 24–48 h guess by 2–4×, reported here as measured, not softened. A few caveats apply: this was measured on a laptop CPU, not the production server, so it should be re-measured there; batch size and threading were untuned, so real headroom likely exists; and linear scaling is an approximation, since edge growth could be superlinear at 3.5M. Memory is comfortably not the bottleneck — 12 GB extrapolated at 3.5M, well inside the `nic-worker` pod budget (§14) — training *time* is the real planning constraint. The incremental fine-tune (MLP-only, RGCN frozen, 10 epochs) is unaffected by any of this and stays cheap regardless, and so is installing an externally GPU-trained checkpoint (§11.5), which bypasses in-cluster training time entirely.

### 13.3 Dense-block detector

This runs k-core plus peeling on the IP relation only, with unchanged logic. Near-linear greedy peeling is validated in the literature on a real 1.47-billion-edge graph (FRAUDAR, Hooi et al., KDD 2016), and k-core decomposition itself is O(V+E) (Batagelj & Zaveršnik). The §12.4 frequency ceiling is what keeps the IP edge set in the regime this literature was validated at — the ceiling and the peeling are one design, not two independent choices.

### 13.4 What does NOT change

The 44-feature schema and every hyperparameter in `config_v3.py` stay exactly as they are. Score semantics (higher = anomalous), file/table contract discipline, the checkpoint schema `{model_state_dict, centroid, config}`, and `checkpoint_manager`'s atomic hot-swap are all untouched. And every hard stop remains in force — no rules, no embeddings out of the detector, human-gated self-training, programmatic-only exposure.

## 14. Kubernetes deployment on the 16 vCPU / 64 GB server

| Pod | Replicas | Request | Limit | Notes |
|---|---|---|---|---|
| `postgres` | 1 | 2 vCPU / 8 GB | 4 vCPU / 16 GB | local-path PV; `shared_buffers` 4 GB, `work_mem` 256 MB |
| `nic-api` | 2 | 1 vCPU / 2 GB | 2 vCPU / 4 GB | queue/card queries hit Postgres, not files |
| `nic-worker` | **1 (fixed)** | 6 vCPU / 24 GB | 12 vCPU / 40 GB | training + scoring jobs |
| `redis` | 1 | 0.25 vCPU / 256 MB | 0.5 vCPU / 512 MB | Celery broker |
| `nginx` | 1 | 0.25 vCPU / 256 MB | 0.5 vCPU / 512 MB | front door, static console |

The worst case — a full retrain while still serving — comes out to roughly 6 + 2×1 + 2 + 0.5 vCPU in requests, and roughly 24 + 4 + 8 + 1 GB ≈ 37 GB in memory, which fits inside 64 GB with headroom left for page cache. During the retrain window, API latency is unaffected, because readers hit Postgres, not the worker.

For local development, the docker-compose equivalent runs `postgres` on host port **5433** (not 5432, to avoid colliding with a locally-installed PostgreSQL), plus the one-shot `db-init` bootstrap service (§11.3). One known gotcha: rebuilding just `nic-api`/`nic-worker` without recreating `nginx` leaves nginx holding the old container IP, and every request 502s until `docker compose restart nginx`.

For storage, local-path PVs back `postgres-data`, `models/` (checkpoints), and `outputs/cards`. Nightly `pg_dump` plus a checkpoint copy to off-server storage is the minimum backup discipline.

All 5 migration steps are implemented and gate-passed, with `src/` model code left untouched until step 4 — full evidence lives in `IMPLEMENTATION.md`:

1. ✅ Postgres schema + `src/db/` stand up (Gate 0).
2. ✅ Dual-write the three JSON stores (Gate 1).
3. ✅ Serve reads from Postgres (Gate 2).
4. ✅ Ingestion lands in Postgres — staged-batch lifecycle (Gate 3).
5. ✅ SQL-pushdown features + hub-capped graph + exact-neighborhood training +
   1M scale test (Gates 4, 5a, 5b).

## 15. Open decisions — lead-owned, not resolved autonomously

**1. K_CAP and the group-size frequency ceiling** (§12.4) is still open, though the profiling query itself is built and dry-run tested (`scripts/profile_group_sizes.py`, rerunnable in one line against the real 3.5M ingest). A dry run on the current 15k population (2026-07-21) produced:

   | Relation | Groups (size≥2) | Max size | p99.9 | Raw clique edges | @k_cap=50 |
   |---|---|---|---|---|---|
   | shares_mobile | 63 | 6 | 6 | 83 | 68 (−18%) |
   | shares_ip | 1,534 | 39 | 27 | 7,202 | 6,083 (−16%) |
   | shares_father_name | 1,116 | 36 | 32 | 9,151 | 7,993 (−13%) |
   | shares_mother_name | 1,170 | 110 | 61 | 38,146 | 28,668 (−25%) |
   | shares_pincode | 2,026 | 152 | 73 | 104,081 | 72,852 (−30%) |

   This isn't the production value — at 15k the largest group is 152 members, but at 3.5M (233×) group sizes will be materially larger, since more people can plausibly share one pincode, one common surname, or one NAT-gateway IP. So both the percentile-derived ceiling and the edge-reduction curve above need to be re-run on the real ingest before a K_CAP is chosen. What this run does validate is that the query is correct, fast (single-digit seconds at 15k), and that the star-capping tradeoff is visible and inspectable per relation — the `pincode`/`mother_name` relations carry the heaviest tail and will need the ceiling most.

   There's also concrete downstream evidence for why this matters beyond edge count and compute (2026-07-22): a `stress_testing_1` artifact — a 382-node shares_ip structure, produced by that dataset's 50k-from-15k with-replacement sampling (see `scripts/generate_stress_test_dataset.py`'s own docstring, which flags this as an intentional stress case for the not-yet-hub-capped file-based graph builder) — demonstrated a *second* consequence of oversized groups beyond edge-count blowup. `dense_block_detector_v3.py`'s per-relation min-max normalization uses the single densest structure in the population as its scaling anchor, and that one 382-node structure compressed every genuine IP-fraud ring's score into 0.3–0.5, with true rings never reaching 1.0 — which measurably hurt fused IP-cluster detection (PR-AUC 0.095→0.055 in that ablation), since max-fusion favors whichever detector gets closest to 1.0. This is confirmed absent from real 15k production data today (max shares_ip degree is 38, consistent with the p99.9=27 / max=39 profiling row above), so it isn't an active production bug — but it does mean the eventual K_CAP design should also cap what a single structure is allowed to set as the density-score normalization anchor, not only the raw edge/fan-out count.

**2.** ~~NeighborLoader fan-out~~ — **CLOSED.** Exact-neighborhood batching adopted (§13.2).

**3. `pg_trgm` vs `difflib`** for name similarity still needs an equivalence check; they are not identical metrics.

**4. Postgres HA** — a single node is fine for this server, but whether a warm standby is required by NIC ops policy is still to be decided.

**5. Batch cadence** — one 3.5M yearly batch versus rolling monthly cohorts changes the drift-check and retrain calendar, and interacts directly with §13.2's ~101 h projected full-retrain window: a yearly cadence needs that much contiguous downtime-tolerant scheduling, while rolling cohorts would need incremental fine-tunes (cheap) far more often than full retrains.

---

# PART III — CAPACITY ASSESSMENT

How many applications can this system comfortably process, given what has actually been measured rather than projected?

| Constraint | Measured / verified value | Comfortable ceiling implied |
|---|---|---|
| Memory (training) | 3.53 GB peak at 1M nodes | ~12 GB extrapolated at 3.5M — **not the bottleneck**; the `nic-worker` pod budget (24–40 GB) has 2–3× headroom even at 3.5M |
| Memory (serving) | Ego-graph/ring queries are two indexed lookups, no graph in API memory | Scales with query load, not population size — effectively unbounded within Postgres's own limits |
| Ingest throughput | 14 s per 10M rows via COPY (external benchmark, reproduced at our scale in testing) | 3.5M rows in single-digit minutes |
| Feature engineering | Bit-exact SQL-pushdown, chunked, <1 GB peak (vs. 20–30 GB pandas) | No known ceiling below 3.5M; not yet tested beyond it |
| Edge construction | Hub-capped, verified on synthetic 1M (17.56M directed edges, 3.53 GB) | Scales with the *capped* edge count, not raw shared-value collisions — the actual ceiling depends on the still-open K_CAP decision (§15 #1) |
| Full retrain wall-clock | 28.9 h measured at 1M (laptop CPU) → ~101 h projected at 3.5M | **This is the real constraint**, not memory. A 3.5M full retrain needs a multi-day maintenance window, tolerable only on a yearly/twice-yearly cadence |
| Incremental fine-tune | 10 epochs, MLP-only, RGCN frozen — unaffected by any of the above | Cheap regardless of population size; the practical tool for frequent updates |
| External checkpoint install | Validation + atomic hot-swap, no training at all | Removes the retrain-time constraint entirely when a GPU machine is available |

The honest answer is that this system, as built and measured, comfortably handles the full 30–40 lakh (3–4 million) target on the specified 16 vCPU / 64 GB server, with memory to spare. The one real constraint is training time, not capacity — a full retrain at 3.5M is a multi-day operation on CPU, which is why the operational design (§9, §11.5) offers two paths around it: either schedule full retrains on a yearly cadence with a matching maintenance window, using cheap incremental fine-tunes in between, or train on external GPU hardware and install the checkpoint in minutes via the validated upload mechanism. Nothing measured so far suggests a hard ceiling below 3.5M — the open items (real K_CAP from live data, server-side re-measurement of retrain time, batch cadence policy) refine *how well* it runs at that scale, not *whether* it can.

---

## 16. External references (re-verified in-session 2026-07-21)

Citations were first proposed by an external evidence review, then independently re-verified in-session, with abstracts and pages fetched directly. Three of the original reviewer's attributions required correction — see the ✗→fixed rows below. Claims that couldn't be sourced are marked as unvalidated projections at their point of use (§13.2's retrain wall-clock estimate).

| Claim (section) | Source | Re-verified |
|---|---|---|
| Shared-attribute fraud graphs produce power-law fan-out / dense components; hub-capping is the standard mitigation (§12.4) | 2026 shared-infrastructure fraud-graph benchmark (arXiv, per external review) | not re-fetched |
| CPU minibatch GNN benchmark closest to ours — **32-node distributed** x86, Papers100M, epoch times / relative speedups only (§13.2) | DistGNN-MB, [arXiv:2211.06385](https://arxiv.org/abs/2211.06385) | ✅ (corrected: distributed cluster, not single-node; Products not confirmed) |
| CPU→GPU data copy identified as dominant hybrid-training bottleneck — **no percentage published**; a circulated "60–80 %" figure is not in the paper (§13.2) | Global Neighbor Sampling, [arXiv:2106.06150](https://arxiv.org/abs/2106.06150) | ✗→fixed |
| Multiplicative fan-out / neighborhood explosion; per-edge-type sampling on hetero graphs (§13.2 background) | PyG NeighborLoader docs; Kumo.ai PyG production guide | not re-fetched |
| Adaptive / degree-aware fan-out beats flat: 12.6× Reddit speedup; F1 73.78→76.88 ogbn-products (§13.2 background) | DAFOS, [arXiv:2507.08845](https://arxiv.org/abs/2507.08845) **only** | ✅ exact; ✗→fixed (a second paper originally cited here was full-batch and had no fan-out — removed) |
| GPD small-sample estimation: below n≈500 exceedances **MLE** is unreliable and MOM/PWM are preferred; large tails are well-behaved (§13.1) | Hosking & Wallis 1987, [Technometrics 29:339–349](https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488243) | ✗→fixed (original phrasing inverted the finding — PWM is the *small-sample recommendation*, not the unstable estimator) |
| Near-linear greedy peeling, validated on a 1.47B-edge Twitter graph; 4031×4313 dense subgraph found (§13.3) | FRAUDAR, Hooi et al., KDD 2016, [DOI 10.1145/2939672.2939747](https://dl.acm.org/doi/10.1145/2939672.2939747), [CMU PDF](https://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd16-fraudar.pdf) | ✅ exact |
| k-core decomposition is O(V+E) (§13.3) | Batagelj & Zaveršnik, arXiv:cs/0310049 | established result, not re-fetched |
| COPY: 10M rows in ~14 s vs ~9,000 s single-row inserts (~643×); ~100k rows/s sustained (§12.1) | [pganalyze](https://pganalyze.com/blog/5mins-postgres-optimizing-bulk-loads-copy-vs-insert); [Tiger Data](https://www.tigerdata.com/blog/benchmarking-postgresql-batch-ingest) | ✅ exact (URLs verified) |
| Single-primary Postgres limits are write-side, not read-side (Part II scale note) | OpenAI Postgres-scaling engineering account (per external review) | not re-fetched |
