# Model Architecture Review — NIC Scholarship Fraud Detection (v3 / V4 layer)

**Purpose.** A component-by-component walkthrough of the detection stack: how each
piece works, *what kind of fraud it is built to catch*, and *what signal it emits*.
Written to be turned into a polished review deck — each section carries an **SVG
PROMPT** you can paste into a project/image tool to generate the figure.

**Source of truth.** All performance numbers trace to `docs/AGENTS.md` Appendix H
and `outputs/ablation/tier_comparison.json`. Where a number appears here it is
quoted *from* those, not re-derived. Do not restate them elsewhere from memory.

**One-line mental model.** *Several specialised detectors, each catching what the
others structurally cannot, combined by a transparent weighted sum — not a learned
tree.*

---

## 0. The stack at a glance

```
raw CSV ──► feature engine ──► graph builder ──► synthetic exposure
                                     │
        ┌────────────────────────────┼───────────────────────────┐
        ▼                            ▼                            ▼
  Subspace IF              Hybrid GraphMCM (RGCN)         Dense-block (FraudAR, IP)
  tabular backbone         relational backbone            IP-ring specialist
        │                            │                            │
        └──────────────┬─────────────┴──────────────┬─────────────┘
                       ▼                             │
             Weighted score-level fusion  ◄──────────┘   (Deviation layer: dormant)
                       │
                       ▼
                 EVT threshold ──► suspicious set ──► XAI cards ──► supervisor
                       │                                                 │
                       └──────── AAD feedback (human-gated) ◄────────────┘
                                        │
                 Standing audit signal: Ring classifier (not fused)
```

> **SVG PROMPT — "System pipeline (with feature routing)"**
> Create a clean, left-to-right horizontal flow diagram on a dark background
> (#0d1117), cyan (#39c5cf) accents, rounded rectangle nodes with thin borders.
> Intake stage in order: "Raw CSV (136 raw cols)" → "Clean (drop null/dup cols,
> fill nullity)" → "Engineer (18 derived)" → "Scale (log1p + MinMax)" →
> "44-feature matrix". Then "Graph Builder (5 edge types) → +5 degree features"
> and "Synthetic Exposure". From the 44-feature matrix draw THREE labelled
> routing arrows into three parallel detector boxes stacked vertically:
> a thick green arrow "all 44 feats + graph" → "Hybrid GraphMCM / RGCN
> (relational backbone)"; a thin green arrow "20 of 44 feats (fin/id/net groups)"
> → "Subspace IF (tabular backbone)"; and an amber arrow "0 feats — shares_ip
> edges only" → "Dense-block FraudAR (IP specialist)". They converge into one box
> "Weighted Score-Level Fusion (1.0 / 0.5 / 0.3)". Then "EVT Threshold" →
> "Suspicious Cards" → "Supervisor". A dashed feedback arrow loops from Supervisor
> back to the detectors labelled "AAD feedback (human-gated)". Show a detached box
> "Ring Classifier — standing audit, not fused" off to the side connected by a
> dotted line. Emphasise the three routing labels — the point is that only the
> GraphMCM sees all 44. Minimal, technical, legible labels.

---

## A. Feature Preprocessing — *from raw application to 44 numbers*

Before any detector runs, every raw scholarship application is rewritten into a
fixed **44-column numeric scorecard**. This section is the ground truth for *what*
we measure, *how* it is preprocessed, and *which detector reads which subset* —
because, crucially, **the three detectors do not all see the same features.**

> **Adopted 2026-07-15 (68 → 44):** the 24 nominal identifier/code features
> (`IDENTIFIER_FEATURES` in `config_v3.py` — phone/Aadhaar tokens, course/district/
> institution IDs, verifier codes, nominal categoricals) were dropped from the model
> feature set. A scaled identifier has no ordinal meaning for the reconstruction
> detector and they polluted the XAI narratives; the noisy-feature ablation showed
> the drop is safe (no regression at detector or fused level). The *sharing* signal
> is preserved — graph edges are built from RAW columns, and degree/count features
> are kept. Schema records the removed set under `dropped_from_model`.

**File:** `src/tabular_feature_engine_v3.py` (+ `src/graph_builder_v3.py` for the
5 degree features) → `data/processed/engineered_features_v3.csv` (15,000 × 44) and
`data/processed/v3_feature_schema.json` (the canonical name list).

### A.1 The preprocessing chain (deterministic, no rules)

Every step is a measurement or an encoding — **no numeric threshold is ever set
against a domain concept** (hard stop #1). The `sanity` and `jwt` columns are
dropped and never used (hard stop #4).

| # | Step | Function / config | What it does |
|---|---|---|---|
| 1 | Drop dead columns | `_load_and_clean`, `NULL_COLS_TO_DROP` / `DUPLICATE_COLS_TO_DROP` | removes 24 explicitly-listed dead columns (16 all-null audit fields + 8 duplicated state/district IDs); non-numeric and >50%-null columns are dropped later in steps 7–8, taking the raw 136 down to 63 base features |
| 2 | Fill high-nullity fields | `_load_and_clean` `high_nullity_fill` | disability / orphan / guardian / ration fields filled with `0` or `""` so they don't drop out |
| 3 | Datetime → age | `_engineer_features` | parses `registered_date`, `date_of_birth` → `age_at_registration` (years, clipped ≥0) |
| 4 | Derive engineered features | `_engineer_features` | 18 derived measures (ratios, cross-row counts, name-match flags — see A.2) |
| 5 | Coerce booleans to int | `_engineer_features` | `disability_flag`, `orphan_flag`, `hosteller`, `is_singlegirlchild` → {0,1} |
| 6 | `log1p` compression | `_select_and_scale`, `LOG1P_COLS` | `annual_family_income`, `admission_fee`, `tution_fee`, `misc_fee` (heavy right tails) |
| 7 | Fill + drop >50%-null | `_select_and_scale` | remaining NaNs → 0; any feature >50% null is dropped |
| 8 | MinMax scale to [0,1] | `_select_and_scale` | `MinMaxScaler` over the 63 base features |
| 9 | Merge + scale degree feats | `add_degree_features`, `DEGREE_FEATURES` | 5 `degree_*` columns from the graph builder, each MinMax-scaled, appended (→ 68), **then the 24 nominal identifiers are dropped → 44** (see A.4) |

Output: `engineered_features_v3.csv` (**44** numeric columns + `application_id`).
Score direction is not set here — these are just comparable numbers.

### A.2 The feature inventory — *original 68 (pre-migration reference)*

> **Note (2026-07-15):** the model now uses **44** features. The list below is the
> original 68-feature set, kept as the before/after reference. The **24 struck from
> the model** are the nominal identifiers in **A.4** (`mobile_no`, `permanent_pincode`,
> all `*_id`/`*_code`, verifier codes, `religion`/`category_id`/`marital_status`/…);
> everything else remains. The current 44 = these 68 minus those 24.

The original 68 decompose as **45 raw pass-through + 18 engineered-derived + 5 graph-degree**.

**Raw pass-through (45)** — coerced/scaled but not recomputed (★ = dropped from the model in A.4):
`domicile_state_id`, `category_id`, `disability_flag`, `disablity_type`,
`disability_percentage`, `orphan_flag`, `hosteller`, `annual_family_income`,
`marital_status`, `parent_occupation`, `permanent_district_id`,
`permanent_pincode`, `mobile_no`, `application_level`, `religion`, `admission_fee`,
`tution_fee`, `pre_post_matric`, `inst_verify_by`, `modeofstudy`, `misc_fee`,
`state_verify_by`, `entitled_fee_amount`, `entitled_lumpsump_amount`,
`pay_amt_centre_shr`, `pay_amt_state_shr`, `is_singlegirlchild`,
`aadhaar_vault_ref_token`, `parents_not_alive`, `sub_district_id`, `village_id`,
`c_institution_id`, `c_course_id`, `c_course_year`, `p_university_id`,
`p_course_id`, `p_course_year`, `p_percentage`, `x_university_id`, `x_course_year`,
`x_percentage`, `competitive_exam_year`, `admission_year`, `pfms_district_code`,
`lgd_district_code`.

**Engineered-derived (18)** — the measurements that actually carry fraud signal:

| Feature | Meaning |
|---|---|
| `age_at_registration` | years between DOB and registration |
| `fee_income_ratio` | (admission+tuition+misc fee) ÷ family income |
| `name_similarity_score` | fuzzy ratio, applicant name vs father name |
| `is_applicant_name_eq_father` | applicant name == father name (0/1) |
| `is_applicant_name_eq_mother` | applicant name == mother name (0/1) |
| `is_father_name_eq_mother` | father name == mother name (0/1) |
| `mobile_application_count` | # applications sharing this mobile |
| `ip_application_count` | # applications sharing this IP |
| `mobile_unique_names` | distinct applicant names on this mobile |
| `mobile_unique_fathers` | distinct father names on this mobile |
| `institute_application_count` | # applications from this institute |
| `ip_to_mobile_ratio` | IP-count ÷ mobile-count (concentration) |
| `income_rank_in_district` | percentile rank of income within district |
| `income_deviation_from_state_median` | income − state median income |
| `is_female` / `is_rural` / `is_urban` | binary encodings of gender / locality |
| `has_state_verify` | state-verifier field present (0/1) |

**Graph-degree (5)** — produced by `graph_builder_v3.py`, merged in step 9:
`degree_shares_mobile`, `degree_shares_ip`, `degree_shares_father_name`,
`degree_shares_mother_name`, `degree_shares_pincode` (per-relation connectivity).

### A.3 Which features go into which detector — *the key routing fact*

**The three detectors read different subsets. Only the Hybrid GraphMCM sees all
44.** This is the single most important thing to state before the component
walkthrough:

| Detector | Feature input | Count | Note |
|---|---|---|---|
| **Hybrid GraphMCM** | **all 44** (masked-prediction over the full vector) + the 5-edge graph | **44** + graph | the only detector that sees every feature |
| **Subspace IF** | financial + identity + network groups only (`SUBSPACE_GROUPS`) | **20** | the other **24 features never reach it** |
| **Dense-block (FraudAR)** | **none** — operates purely on `shares_ip` edge structure | **0** (+1 edge type) | unsupervised, structure-only |

The 20 features the Subspace IF reads, by group (`config_v3.py::SUBSPACE_GROUPS`):

- **financial (7):** `annual_family_income`, `fee_income_ratio`,
  `income_rank_in_district`, `income_deviation_from_state_median`, `admission_fee`,
  `tution_fee`, `misc_fee`
- **identity (6):** `name_similarity_score`, `is_father_name_eq_mother`,
  `is_applicant_name_eq_father`, `is_applicant_name_eq_mother`,
  `mobile_unique_names`, `mobile_unique_fathers`
- **network (7):** `ip_application_count`, `ip_to_mobile_ratio`,
  `mobile_application_count`, `institute_application_count`, `degree_shares_ip`,
  `degree_shares_mobile`, `degree_shares_pincode`

**Consequence for the review:** the 48 non-grouped features (most raw
pass-through columns — course/university IDs, verifier codes, category, geography,
entitled-fee amounts) influence risk **only** through the GraphMCM's reconstruction
error. If the GraphMCM were retired, those 48 features would carry no direct signal
at all — one more reason RGCN retirement was disproven (Appendix H).

> **SVG PROMPT — "Feature routing (44 → 3 detectors)"**
> Dark background (#0d1117). On the left, a vertical stack labelled "44 features"
> split into three colour bands: a large grey band "48 — raw pass-through (unrouted
> to subspace)", and inside it highlight three small coloured chips "financial 7 /
> identity 6 / network 7" plus a cyan chip "5 degree". Draw arrows: ONE thick green
> arrow from the WHOLE stack to "Hybrid GraphMCM — all 44 + graph"; a thin green
> arrow from ONLY the 20 grouped chips to "Subspace IF — 20 feats"; and a separate
> amber arrow originating from a small "shares_ip edges" icon (NOT from the feature
> stack) to "Dense-block — 0 feats, structure only". Caption: "only the GraphMCM
> sees every feature; 48 features reach risk solely through it". Legend green =
> tabular, amber = structural. Clean, technical.

---

## A.4 Feature-set migration: 68 → 44 (adopted 2026-07-15)

The model feature set was reduced from **68 to 44** by dropping 24 nominal
identifier/code features. This section is the permanent before/after record: what
the old 68 were, exactly which 24 left, why, and the measured effect. The old
68-feature artifacts (CSV, schema, exposure tensors, all 4 checkpoints) are
preserved under `backup_68feat/`; `V4_N_FEATURES=68` reconstructs the old model.

### A.4.1 The 24 dropped — and why each is noise for the detector

Governing reason: these are **nominal identifiers/codes**, and MinMax-scaling them
imposes a fake ordering (course-ID 5000 is not "more" than 4000; a phone number has
no ordinal expectation). A reconstruction detector cannot use nominal identity, and
they dominated ~74% of XAI card headlines with meaningless "expected mobile_no ≈
0.75" prose. Their *sharing* signal is **not** lost — graph edges are built from the
RAW columns, and degree/count features (`degree_shares_*`, `*_application_count`)
are kept.

| Group | Dropped features | Why removed |
|---|---|---|
| **Random tokens** | `mobile_no`, `aadhaar_vault_ref_token` | opaque/arbitrary strings — no predictable structure exists |
| **Geographic codes** | `permanent_pincode`, `village_id`, `sub_district_id`, `permanent_district_id`, `pfms_district_code`, `lgd_district_code`, `domicile_state_id` | nominal location labels; scaled magnitude meaningless; co-location captured by `shares_pincode` edge + `degree_shares_pincode` |
| **Institution/course IDs** | `c_institution_id`, `c_course_id`, `p_university_id`, `p_course_id`, `x_university_id` | nominal registry numbers; concentration captured by `institute_application_count` |
| **Verifier account codes** | `inst_verify_by`, `state_verify_by` | nominal officer/account codes; "was it verified" kept as `has_state_verify` |
| **Nominal categoricals** | `category_id`, `religion`, `marital_status`, `parent_occupation`, `disablity_type`, `application_level`, `modeofstudy`, `pre_post_matric` | semantic categories, not magnitudes; some legally wrong to treat as ordinal; disability *fact/severity* kept via `disability_flag` + `disability_percentage` |

### A.4.2 The 44 kept (model feature set)

Financial/entitlement (12): `annual_family_income`, `admission_fee`, `tution_fee`,
`misc_fee`, `entitled_fee_amount`, `entitled_lumpsump_amount`, `pay_amt_centre_shr`,
`pay_amt_state_shr`, `fee_income_ratio`, `income_rank_in_district`,
`income_deviation_from_state_median`, plus percentages `p_percentage`, `x_percentage`.
Temporal (5): `c_course_year`, `p_course_year`, `x_course_year`,
`competitive_exam_year`, `admission_year`, `age_at_registration`.
Identity/name (6): `name_similarity_score`, `is_applicant_name_eq_father`,
`is_applicant_name_eq_mother`, `is_father_name_eq_mother`, `mobile_unique_names`,
`mobile_unique_fathers`.
Network/count (6): `mobile_application_count`, `ip_application_count`,
`ip_to_mobile_ratio`, `institute_application_count`.
Binary flags (7): `disability_flag`, `orphan_flag`, `hosteller`, `is_singlegirlchild`,
`parents_not_alive`, `is_female`, `is_rural`, `is_urban`, `has_state_verify`.
Degree (5, from graph): `degree_shares_mobile`, `degree_shares_ip`,
`degree_shares_father_name`, `degree_shares_mother_name`, `degree_shares_pincode`.
(Exact canonical list: `data/processed/v3_feature_schema.json` → `features`; the
removed set is under `dropped_from_model`.)

### A.4.3 The measured improvement (source: `outputs/ablation/`)

Connected-cluster PR-AUC, seeds 42/43/44, `src/ablation_noid_v3.py`:

| Setting | full (68) | noid (44) | Δ | Note |
|---|---|---|---|---|
| topo-ON, **locked fusion** (production) | 0.6594 | **0.6806** | **+0.021** | no category regresses; within ±0.03–0.04 noise floor |
| topo-ON, hybrid detector only | 0.2266 | **0.3036** | **+0.077** | clears the noise floor; positive on all 5 categories |
| topo-OFF, hybrid only | 0.2128 | 0.2653 | +0.053 | **IP_CONCENTRATION −0.159 here was a topo-OFF ARTIFACT** — reverses to +0.062 topo-ON |

**Verdict:** safe (no regression at detector or fused level), modestly helpful at
the detector, marginal after fusion, and removes 24 uninterpretable features. Status
**proposed/pending** (H.6 gate wants a 4th seed); the fused delta sits inside the RGCN
scatter-add noise floor. Current 44-feature model passes all 5 isolated-node V2 floors
(`evaluate_model_v3.evaluate()`). Full data: `outputs/ablation/noid_fused_confirmation.json`,
`outputs/ablation/noid_ablation.json`.

---

## 1. Feature Engine — *the intake clerk*

**File:** `src/tabular_feature_engine_v3.py` → `engineered_features_v3.csv` (15,000 × 44)

**Analogy.** A meticulous clerk who takes each raw application and rewrites it into
a standard 44-field scorecard — every applicant described in the exact same
vocabulary so the detectors can compare like with like.

**How it works.** Deterministic numeric feature engineering over the raw scholarship
fields. No rules, no policy thresholds — just measurements (ratios, counts,
encodings) that turn messy raw text/numbers into 44 comparable numeric columns.
The `sanity` column is *never* used (hard stop #4).

**What it "catches".** Nothing on its own — it is the substrate. But the *quality*
of these 44 features sets the ceiling for everything downstream.

**Signal out.** `engineered_features_v3.csv` (44 numeric columns + `application_id`)
and `v3_feature_schema.json` (the names + exclusions).

> **SVG PROMPT — "Feature engine (raw → 44, routed)"**
> A single messy paper form on the left with mixed fields (text, dates, amounts)
> flowing through a funnel labelled "Feature Engine" into a tidy grid on the right.
> Colour the grid to show the composition: 45 grey cells "raw pass-through", 18
> green cells "engineered (ratios, counts, name-match)", 5 cyan cells "graph
> degree". Below the grid, three short arrows fan out labelled "→ GraphMCM: all 44",
> "→ Subspace IF: 20", "→ Dense-block: 0 (edges only)". Dark theme, cyan funnel.
> Caption: "raw application → 44 comparable numbers, routed to three detectors".

---

## 2. Graph Builder — *the relationship mapper*

**File:** `src/graph_builder_v3.py` → `identity_graph_v3.pt`, `degree_features_v3.csv`

**Analogy.** An investigator pinning photos to a corkboard and drawing string
between any two applicants who share something they shouldn't independently share —
same phone, same IP, same parent name, same pincode.

**How it works.** Builds a PyG `HeteroData` identity graph with **5 edge types**:
`shares_mobile`, `shares_ip`, `shares_father_name`, `shares_mother_name`,
`shares_pincode`. Also emits **degree features** (how connected each node is per
relation) — these feed the tabular detectors so even nodes with no useful text
still carry a "how crowded is your neighbourhood" number.

**What it catches.** Nothing yet — it is the map the relational detectors read.
Its most important property: a node with **no shared identifiers becomes an
isolated node** (the known blind spot — see §5, subspace IF picks these up).

**Signal out.** The graph tensor + per-node, per-relation degree counts.

> **SVG PROMPT — "Identity graph"**
> A node-link network on dark background. ~12 applicant nodes (circles). Draw
> coloured edges in 5 distinct colours with a legend: shares_mobile, shares_ip,
> shares_father_name, shares_mother_name, shares_pincode. Show one tight cluster
> of 5 nodes densely interlinked by shares_ip (highlight it), and 2 lonely nodes
> with no edges labelled "isolated node — no structural signal". Clean, technical.

---

## 3. Synthetic Exposure — *the flight simulator*

**File:** `src/synthetic_exposure_builder_v3.py` → `synthetic_exposure_set_v3.pt`,
`synthetic_exposure_graph_v3.pt`

**Analogy.** A flight simulator that shows the detector *programmatically
constructed* examples of what tampering looks like — both at the row level
(a doctored application) and the topology level (a fabricated ring) — before it
ever sees the real cohort, so it knows the shape of trouble.

**How it works.** Programmatically degrades clean records into fraud-shaped
archetypes (tabular exposure) **and** plants synthetic clusters/edges (topology
exposure). Never a GAN/CTGAN/TVAE (hard stop #7 — composite degradation is 24×+
stronger on behavioural signals). This is the Stage-1 teaching material.

**What it catches.** Indirectly — it is what lets the Hybrid GraphMCM learn a
margin around "normal" (LOE margin = 2.0) instead of memorising the training set.
The topology-exposure layer is validated at **+0.148 LOE** (AGENTS.md H.9).

**Signal out.** Two `.pt` tensors consumed only by training.

> **SVG PROMPT — "Synthetic exposure"**
> Split panel. Left "Tabular exposure": a clean row morphing into a red doctored
> row (one field inflated). Right "Topology exposure": a few normal nodes with a
> synthetic red ring dropped in. Header banner "programmatic, NOT a GAN". Dark
> theme, red = injected fraud, green = clean.

---

## 4. Hybrid GraphMCM — *the context-aware auditor* (relational backbone)

**File:** `src/hybrid_graphmcm_v3.py` → `hybrid_scores_v3.csv`,
`models/hybrid_graphmcm_v3.pth`

**Analogy.** An auditor who does not judge you in isolation. Before deciding
whether your declared numbers are plausible, they look at *everyone you are
connected to* — and ask, "given this applicant's neighbourhood, are these declared
values what I'd expect?"

**How it works — two streams:**
- **Feature stream (masked prediction):** hides `K=8` blocks of the 44 features and
  reconstructs them. Large reconstruction error = the declared values are internally
  surprising → `feature_pred_error`.
- **Graph stream (RGCN):** aggregates *neighbours' features* over the 5 edge types
  into a 64-d context vector `h_N`, which conditions the reconstruction; also
  predicts edge probabilities → `edge_pred_error`. (No raw embedding ever leaves the
  module — hard stop #2.)

`hybrid_anomaly_score = feature_pred_error + 0.3 · edge_pred_error`

**What it catches — the unique slot.** *"Normal alone, abnormal in context."* An
application whose own numbers look fine but are **inconsistent with the applicants
it shares an IP / parent-name / pincode with.** No other detector sees features
*and* graph context together.

**What it does NOT catch (be explicit in the review).** Dense fraud rings. A
reconstruction model reconstructs a dense clique *easily* (neighbourhood smoothing)
→ low error → low score. Feeding it more rings makes this **worse**, not better.
That structural limit is *why* the dense-block detector (§6) exists.

**Signal out.** `hybrid_anomaly_score`, `feature_pred_error`, `edge_pred_error`
(all scalars; higher = more anomalous). Raw IP 0.51 / MOTHER 0.45; RGCN retirement
was tested and **disproven** — removing it regresses (AGENTS.md Appendix H).

> **SVG PROMPT — "Hybrid GraphMCM two streams"**
> Two horizontal lanes merging. Top lane "Feature stream": a 44-cell row with 8
> cells masked (grey), an arrow to a reconstruction, and a red "error" gauge.
> Bottom lane "Graph stream (RGCN)": a centre node pulling arrows from 4 neighbour
> nodes into a box "neighbour context h_N", plus an "edge error" gauge. Both lanes
> merge into "hybrid_anomaly_score = feature_err + 0.3·edge_err". Dark, cyan.
> Add a small caption box: "catches: features inconsistent with your neighbours".

---

## 5. Subspace Isolation Forest — *the specialist panels* (tabular backbone)

**File:** `src/subspace_if_v3.py` → `subspace_if_scores_v3.csv`

**Analogy.** Three specialist review panels — a **finance** panel, an **identity**
panel, a **network** panel — each judging only the fields it is expert in, so a
weird income can't be washed out by ten perfectly normal unrelated fields.

**How it works.** Isolation Forest run *per feature group* (financial / identity /
network) instead of over all 44 dims at once. Isolating an outlier in a focused
subspace takes fewer random splits → higher score. Crucially, it needs **no
graph** — so it is the one detector that still works on **isolated nodes**.

**What it catches.** Per-group tabular outliers: income violations, fee inflation,
identity oddities — *including applicants with zero edges* that the relational
detectors are blind to. It is the **single strongest component** (raw mean 0.727;
INCOME 0.966, FEE 0.916 — AGENTS.md).

**Blind spot.** IP concentration / dense rings (a structural, not tabular, signal)
— stuck ~0.327 alone. That gap is filled by §6.

**Signal out.** `subspace_if_score` (scalar, higher = more anomalous).

> **SVG PROMPT — "Subspace IF (reads 20 of 44)"**
> Three vertical panels side by side labelled "Financial (7 feats)", "Identity
> (6 feats)", "Network (7 feats)", each showing a small scatter with one red
> outlier being isolated by a few dashed split lines. To the left, a faded stack
> "44 features" with only 20 chips lit up feeding the panels and 24 chips greyed
> out labelled "not read by subspace IF". Below, the three panels feed a combined
> bar "subspace_if_score = max(group scores)". Note tag on the side: "works even
> with no graph edges". Dark theme, green accent.

---

## 6. Dense-block Detector (FraudAR-style) — *the density inspector* (IP specialist)

**File:** `src/dense_block_detector_v3.py` → `dense_block_scores_v3.csv`

**Analogy.** A building inspector who ignores individuals and looks only for
*rooms that are impossibly crowded* — too many applicants packed onto one IP to be
a coincidence — and is not fooled when fraudsters pad the room with a few legit
faces (camouflage).

**How it works.** Charikar **greedy peeling** with **camouflage-resistant** column
weighting (the FraudAR mechanism, Hooi et al.), run **only on `shares_ip`**
(`DENSE_BLOCK_RELATIONS=[1]`). It repeatedly strips the least-connected node and
records the densest sub-block each node belonged to. **Unsupervised** — no labels,
no training, deterministic, self-terminating, unthresholded. It is *not* run on
`shares_mother_name` / `shares_pincode` (those are legitimately dense → false
positives).

**What it catches.** **Dense IP rings** — exactly the case the reconstruction
backbone smooths away. Raw IP **0.713**, filling subspace IF's one blind spot.

**Key review point.** This is **not** a learner and it is **not** the ring
classifier. It carries you at **cold start with zero confirmed labels**. It does
not improve as confirmations accumulate — that is the ring classifier's job (§11).

**Signal out.** `dense_block_score_ip` (scalar, higher = more anomalous).

> **SVG PROMPT — "Dense-block peeling"**
> A dense cluster of ~8 nodes on one shared IP, with 2 pale "camouflage" legit
> nodes attached. Show 3 peeling steps (arrows) stripping outer nodes until a red
> dense core remains, labelled "densest block". Side note: "unsupervised · IP only
> · camouflage-resistant". Amber accent, dark background.

---

## 7. Deviation Layer — *the apprentice* (wired, DORMANT)

**File:** `src/deviation_layer_v3.py` → `outputs/deviation_scores_v3.csv`

**Analogy.** An apprentice who *could* learn directly from confirmed cases, but
right now hasn't seen enough real confirmed fraud in any category to be trusted, so
it studies the simulator instead and stays quiet.

**How it works.** DevNet/PReNet-style weak-supervision deviation network, made
leakage-safe by out-of-fold stacking. Per-category cold-start (hard stop #19): a
category only uses real-confirmed anomalies once it crosses
`DEV_MIN_CONFIRMED_PER_CATEGORY`; below that it falls back to synthetic archetypes.
Currently synthetic-only → **DORMANT** (`DEVIATION_LAYER_ENABLED=0`). Exposure layer
validated at +0.092 LOE (H.9).

**What it will catch (once active).** Category-specific fraud that resembles past
*confirmed* cases — a supervised complement that sharpens as labels grow.

**Signal out.** `deviation_score` + `evidence_source` (dormant today).

> **SVG PROMPT — "Deviation layer dormant"**
> A greyed-out neural-net box labelled "Deviation Net (DevNet/PReNet)" with a
> "DORMANT" stamp, a small progress bar "confirmed labels per category" mostly
> empty, and a note "activates per category once labels cross threshold". Muted
> palette, one amber highlight on the stamp.

---

## 8. EVT / SPOT Thresholding — *the tide-line surveyor*

**File:** `src/evt_scorer_v3.py` → `evt_thresholds_v3.json`

**Analogy.** Rather than a manager decreeing "flood line = 15", a surveyor fits the
shape of the *tail* of the water-level distribution and lets the data say where
"extreme" begins.

**How it works.** Fits a Generalised Pareto Distribution to the upper tail of the
fused score to derive a **data-driven** threshold (GPD shape constrained to
[-0.5, 1.0]). This is the *only* sanctioned numeric threshold in the system — the
one place a cut-off is allowed, precisely because it is learned from the tail, not
set against a domain concept (hard stop #1).

**What it catches.** Converts a continuous risk score into the **suspicious set**
that goes to reviewers.

**Known fragility.** If the tail is discontinuous, the threshold can explode or
collapse; if the tail is data-entry noise rather than fraud, self-training can
anchor on typos (see §9, §MAR). Human gate is the safeguard.

**Signal out.** `evt_thresholds_v3.json`.

> **SVG PROMPT — "EVT tail"**
> A score histogram (dark, cyan bars) with a smooth GPD curve fitted over the far
> right tail, a vertical dashed line marking the EVT threshold, and everything to
> its right shaded red "suspicious set". Caption: "threshold learned from the tail,
> not decreed".

---

## 9. Self-Training Loop — *the probation officer* (human-gated)

**File:** `src/self_training_loop_v3.py` → `pseudo_labels_v3.json`

**Analogy.** A probation officer who may *propose* promoting a suspected case to a
training label — but can never do it alone. Every promotion needs a sign-off, and
rounds never advance automatically.

**How it works.** Promotes high-confidence EVT-tail cases to pseudo-labels **only
under a human gate** — each round requires a Phase D PR-AUC check before its label
set is used (hard stop #5). Round-0 classifier-agreement is code-enforced OFF.
Minimum 2 signals to promote (`MIN_SIGNALS_FOR_PROMOTION`).

**What it catches.** Nothing directly — it grows the label set that the supervised
components (§7 deviation, §11 ring classifier) and fusion metadata rely on.

**Failure mode to flag in the review.** This is the component "most likely to break
first in production": a slight EVT tail misalignment seeds false positives →
semantic drift in Round 1. The gate exists for exactly this reason.

**Signal out.** `pseudo_labels_v3.json` (human-approved only).

> **SVG PROMPT — "Human-gated promotion"**
> A conveyor of candidate cases reaching a gate with a human icon and a checkmark/
> cross. Approved cases pass into a "label set" bin; a big padlock labelled
> "no auto-advance (hard stop #5)". Dark theme, one green gate light.

---

## 10. Weighted Score-Level Fusion — *the panel chair* (LOCKED)

**File:** `src/fusion_classifier_v3.py` → `risk_scores_v3.csv`

**Analogy.** A panel chair who takes each specialist's verdict at face value and
adds them with fixed, transparent weights — deliberately **not** a clever manager
who is allowed to overrule a specialist and bury their finding.

**How it works.**
```
risk = minmax( 1.0·minmax(subspace_if_score)
             + 0.5·minmax(dense_block_score_ip)
             + 0.3·minmax(hybrid_anomaly_score) )
```
Each component is min-max normalised, weighted, summed, normalised again.
**Label-independent** — no learned gate can zero-out a strong raw signal. Weights
live in `config_v3.py` (`FUSION_W_SUBSPACE=1.0`, `FUSION_W_DENSE_IP=0.5`,
`FUSION_W_HYBRID=0.3`). The *same* function serves production and the comparison
harness, so they can never drift.

**Measured (locked fusion, one run — trace to IMPLEMENTATION.md / Appendix H):**

| | AGE | INCOME | IP | MOTHER | FEE | MEAN |
|---|---|---|---|---|---|---|
| Connected | 0.576 | 0.700 | **0.538** | 0.737 | 0.643 | **0.639** |
| Held-out | 0.603 | 0.744 | 0.409 | 0.752 | 0.689 | 0.640 |

**Signal out.** `risk_score_v3` (0–1, higher = more anomalous) + `label_source`
(metadata only).

> **SVG PROMPT — "Score-level fusion"**
> Three horizontal signal bars (green "subspace ×1.0", amber "dense-IP ×0.5",
> cyan "hybrid ×0.3") flowing into a plus/sum node, then a single "risk_score"
> meter 0–1. Emphasise the fixed weight labels. Add a small crossed-out icon of a
> decision tree with text "no learned gate to bury a signal". Dark theme.

---

## 11. Ring Pipeline — *the pattern librarian* (STANDING audit, not fused)

**Files:** `src/ring_candidate_v3.py` → `ring_fingerprint_v3.py` →
`ring_classifier_v3.py` (`models/ring_classifier_v3.pkl`)

**Analogy.** A librarian who keeps a catalogue of *confirmed* fraud-ring
signatures. Shown a new suspicious cluster, they answer two things: "does this match
a known pattern?" and "or is this a **new** kind we've never catalogued?"

**How it works — three stages:**
1. **Candidate generation** proposes suspicious subgraphs (structure only).
2. **Fingerprint** encodes each candidate into a fixed structural vector.
3. **Classifier** — a **LightGBM trained on confirmed vs negative rings** — emits a
   ring probability, **plus open-set novelty** = distance to the nearest confirmed
   prototype.

**This is the ONE self-improving structural component.** Unlike dense-block (§6,
static/unsupervised), the ring classifier *learns your confirmed LOE topologies* and
gets better as they accumulate. It reads **public structure + scores only — never
`h_N`** (hard stop #2).

**Why it is "standing, not fused."** It is kept as an **independent audit signal**,
not a fusion input (most stable, best held-out MOTHER; AGENTS.md). Folding it into
the fused score is deprecated (`RING_CLASSIFIER_ENABLED` OFF as a fusion input).

**Signal out.** ring-probability + novelty score (audit lane).

> **SVG PROMPT — "Ring classifier / librarian"**
> Left: a catalogue/library of small ring "fingerprint" cards (confirmed patterns).
> Centre: a new suspicious cluster arriving. Right: two outputs — a match gauge
> "known pattern?" and a separate dial "novelty — never seen before?". A dotted
> arrow from "confirmed LOEs" INTO the catalogue labelled "learns & grows". Dark
> theme, cyan.

---

## 12. XAI Reviewer Cards — *the case file* (full catalogue)

**Files:** `src/xai_layer_v3.py` (`run_xai` → JSON) + `src/xai_card_html_v3.py` (HTML)
→ `outputs/explanation_cards_v3.json` + `outputs/cards/*.html`, served at
`GET /v3/monitoring/{app_id}/card` and `/ring`.

**Analogy.** A prosecutor's case file assembled per flagged application: the
evidence, an exact breakdown of *which detector contributed how much*, and the
declared-vs-expected comparison for every field — nothing asserted that isn't
computed. **Design rule (hard stop #2 / evidence-first):** every sentence traces to
a computed statistic or an EVT-derived threshold; no raw embedding and no canned
narrative. Same evidence always renders the same prose (deterministic, auditable
for appeals).

**Who gets a card.** `run_xai(top_n=500)` ranks by `risk_score_v3` and builds cards
for the top-N. In production only **suspicious** applications (crossed EVT / carries
a trigger / non-negative label) are surfaced to reviewers.

### 12.0 Narration policy — *what the card is allowed to say*

The detector scores every model feature, but the **prose** only speaks features
where "the model expected X" is meaningful. This is **presentation only** — it never
gates or changes a score (a feature-TYPE classification like `FEATURE_LABELS`, not a
hard-stop-#1 rule). `_feature_kind` sorts each feature into:

| Kind | Narrated? | How |
|---|---|---|
| **continuous** (income, fees, ratios, percentages, name-similarity, degree/count) | yes | full "declared vs model-expected" prose |
| **identifier** (nominal IDs/codes — `IDENTIFIER_FEATURES`) | **never** | a reconstruction error on a scaled ID is entropy, not evidence ("expected mobile_no ≈ 0.75" is nonsense) |
| **binary** (0/1 flags — `BINARY_FEATURES`) | only as a network **inconsistency** | surfaced only when the applicant's flag **disagrees** with its co-applicant majority (e.g. a male applicant whose shared-IP ring is 84% female); grounded in the neighbour distribution (`_binary_context`), not the raw reconstruction scalar |

`_binary_context` returns `None` — dropping the flag from the narrative — when the
applicant *agrees* with its neighbours (base-rate noise, not evidence) or has no
neighbours to ground the expectation. Binaries render as a "co-applicants k/M" block
(`_binary_field_body`), not the signed declared-vs-expected bars. Rationale and the
supporting ablation are in the [feature-drop record](../outputs/ablation/noid_fused_confirmation.json)
and project memory.

### 12.1 How a card is generated (the assembly line)

`run_xai` joins six inputs — `hybrid_scores_v3.csv`, `risk_scores_v3.csv`,
`subspace_if_scores_v3.csv`, `dense_block_scores_v3.csv`, `evt_thresholds_v3.json`,
`pseudo_labels_v3.json` — plus the feature CSV and identity graph. For each
application it computes population percentiles (`build_population_stats`), the
closed-form fusion split (`build_fusion_contributions`), per-feature error ranks,
graph-degree ranks, and (HAN only) attention weights, then renders a deterministic
`narrative` from that evidence (`_narrative`).

### 12.2 The card element catalogue — *what each field is and which detector produced it*

This is the answer to "which model could have flagged each explanation." Every
card element traces to exactly one detector (or to the transparent fusion of all
three):

| Card element (JSON key) | Generated by | What it represents to a reviewer | Source detector(s) |
|---|---|---|---|
| `risk_score_v3` + `risk_rank` / `risk_percentile` | `_pct_rank` vs `risk_sorted` | where this application sits in the scored population | **Fusion** (all three, weighted) |
| `triggers` | `pseudo_labels_v3.json` → `trigger_map` | which EVT signals fired hard enough to be promotion candidates | see 12.3 — one detector per trigger |
| `evt_signals` | `TRIGGER_SIGNAL_KEY` + `evt_thresholds_v3.json` | observed value vs fitted threshold for each fired trigger | the detector behind each trigger (12.3) |
| `evt_crossings` | direct threshold compare (lines 721–745) | thresholds crossed *independent of promotion* (single-signal evidence) | Hybrid (`hybrid`, `edge_pred_error`) + Subspace groups |
| `top_feature_errors` | `_top_features` on `per_feature_error_json` | the fields the model least expected, with declared-vs-expected + percentile | **Hybrid GraphMCM** (feature stream) |
| `fusion_contributions` + `provenance` | `build_fusion_contributions` | exact % each detector contributed to the fused score (closed-form, replaces TreeSHAP) | **Fusion split** across subspace / dense-IP / hybrid |
| `subspace_groups` | `subspace_map` + group EVT thresholds | per-group tabular anomaly (financial / identity / network) + crossed flag | **Subspace IF** |
| `dense_block_ip` | `dense_map` + `dense_ip_sorted` percentile | membership in a dense same-IP cluster, with concentration percentile | **Dense-block (FraudAR)** |
| `graph_connections` | `neighbor_index` + degree percentiles | who this application shares IP / mobile / parent-name / pincode with | **Graph builder** structure (read by Hybrid) |
| `attention` (β_r, top_edges) | `model.last_beta_r` / `top_alpha` — **HAN only** | which relation the model weighted most; `null` under default RGCN | **Hybrid GraphMCM** (graph stream) |
| `review_status` | `LABEL_SOURCE_DESCRIPTIONS[label_source]` | label state: pending / EVT-flagged / model-flagged / confirmed | Self-training + supervisor (metadata) |
| `narrative` | `_narrative` | the deterministic prose stitched from all of the above | composed — cites the detector per claim |

### 12.3 Trigger → detector provenance (the core map)

A `trigger` is what a self-training round *proposes*. Each maps to exactly one
detector via `TRIGGER_SIGNAL_KEY` (xai_layer_v3.py:126):

| Card trigger | `evt_thresholds` signal key | Detector that raised it | Reviewer meaning |
|---|---|---|---|
| `EVT_HYBRID` | `hybrid` | **Hybrid GraphMCM** (overall score) | whole-application anomaly |
| `EVT_FINANCIAL` | `subspace_if_financial` | **Subspace IF — financial group** | income/fee profile statistically extreme |
| `EVT_IDENTITY` | `subspace_if_identity` | **Subspace IF — identity group** | name/verifier fields cluster oddly |
| `EVT_NETWORK` | `subspace_if_network` | **Subspace IF — network group** | IP/mobile sharing extreme |
| `EVT_EDGE_RING` | `edge_pred_error` | **Hybrid GraphMCM — graph stream** | connectivity inconsistent with features |

**Honest gap to state in the review.** There is **no trigger for the dense-block
detector.** Dense-block-IP influences the fused `risk_score_v3` and appears as card
evidence (`dense_block_ip`, and inside `fusion_contributions`), but it **cannot by
itself promote a self-training label** — no `EVT_DENSE` trigger exists. So a pure
dense-IP ring shows up in the *score* and the *narrative* but not in `triggers`.
Reviewers relying on the trigger list alone would under-count IP rings; the fusion
split and `dense_block_ip` evidence are where that signal actually lives.

### 12.4 Reading a card as a reviewer

The narrative is assembled in a fixed order so the story never contradicts itself:
(1) population placement → (2) EVT flags (observed vs threshold) → (3) top feature
evidence → (3b) **score composition** (which detector drove it) → (3c) dense-IP
evidence when it fires → (4) strongest subspace group as ranking context → (5)
network links / isolation → attention (HAN) → (6) recommended action keyed off label
state and EVT crossings, **not** a hand-set score cut.

> **SVG PROMPT — "Reviewer card + provenance"**
> A two-pane card UI mockup on dark background. Left pane: small ego-network + three
> labelled signal bars (green "subspace", amber "dense-IP", cyan "hybrid") + a
> stacked "fusion split %" bar. Right pane: a risk gauge, a list of "triggers" each
> tagged with the detector that raised it (EVT_FINANCIAL → Subspace/financial,
> EVT_EDGE_RING → GraphMCM/graph), and an expandable "declared vs expected" field
> row. Add a small footnote chip "dense-IP: no trigger — score & evidence only".
> Clean, product-like, cyan accents.

---

## 13. Division of labour — who catches what (put this table in the review)

| Fraud shape | Caught by | Why the others miss it |
|---|---|---|
| Income / fee outlier on one application | **Subspace IF** | isolated tabular signal |
| Applicant with NO shared identifiers | **Subspace IF** | relational detectors need edges |
| Declared values inconsistent with neighbours | **Hybrid GraphMCM** | tabular detectors ignore graph |
| Dense IP ring (many apps, one IP) | **Dense-block (FraudAR)** | reconstruction smooths dense cliques |
| Ring padded with legit "camouflage" | **Dense-block (FraudAR)** | camouflage-resistant weighting |
| A ring matching a *known confirmed* pattern | **Ring classifier** (audit) | learns confirmed fingerprints |
| A *novel* ring shape never seen | **Ring classifier** novelty | open-set distance |
| Category fraud resembling past confirmed cases | **Deviation layer** (dormant) | needs enough confirmed labels |

> **SVG PROMPT — "Coverage matrix"**
> A matrix/heatmap: rows = fraud shapes (from the table above), columns = detectors
> (Subspace IF, Hybrid GraphMCM, Dense-block, Ring classifier, Deviation). Fill the
> owning cell solid green, partial contributors faint, blanks dark. Legend
> "primary / supporting / blind". Dark theme.

---

## 14. Signal reference — what each component emits

| Component | Output signal | Direction | Consumed by |
|---|---|---|---|
| Feature engine | 44 numeric features | — | all detectors |
| Graph builder | 5-edge graph + degree feats | — | GraphMCM, subspace |
| Hybrid GraphMCM | `hybrid_anomaly_score`, `feature_pred_error`, `edge_pred_error` | higher = worse | fusion, XAI |
| Subspace IF | `subspace_if_score` | higher = worse | fusion, XAI |
| Dense-block | `dense_block_score_ip` | higher = worse | fusion, XAI |
| Deviation (dormant) | `deviation_score`, `evidence_source` | higher = worse | (fusion, when active) |
| EVT | `evt_thresholds_v3.json` | threshold | suspicious gate |
| Self-training | `pseudo_labels_v3.json` | labels | deviation, ring, metadata |
| Fusion | `risk_score_v3` | higher = worse | EVT, cards, supervisor |
| Ring classifier | ring-prob + novelty | higher = worse | audit lane (not fused) |

---

## 15. The fusion question — weighted sum vs LightGBM (the part to argue carefully)

Your framing was: *"we were sure about LightGBM — but what if it were used
differently for feature weighting?"* The evidence flips the premise: **the system
already moved OFF LightGBM and ON to a weighted score-level sum, and that was the
right call — measured, not asserted.** Here is the honest account for the review.

### What actually happened
The V4 pipeline **replaced** a 14-positive LightGBM fusion with a locked weighted
sum. The reason is in `fusion_classifier_v3.py` and AGENTS.md H.8: the tree, given
only ~14 positive labels, **learned a gate that destroyed the raw detector signals**:

| Raw detector signal | Alone | After LightGBM fusion |
|---|---|---|
| Subspace INCOME | **0.966** | 0.315 |
| RGCN IP | **0.51** | 0.169 |

A boosted tree with almost no positives will happily learn "when subspace is high,
push risk *down*" if that fits the 14 points — inverting a detector that is right
96% of the time. The weighted sum **cannot** do this: every weight is positive and
fixed, so a strong raw signal always pushes risk up. That single property — *no
learned gate can bury a specialist* — is why the weighted sum wins here.

### But your instinct isn't wrong — it's a *labels* problem, not a *tuning* problem
LightGBM failing here is **not** because trees are worse in principle. It is because
**14 positives cannot support a learned combiner.** "Utilising it differently for
feature weighting" is a real idea, and there are two disciplined versions:

1. **Monotonic constraints.** Force the tree to be *non-decreasing* in every
   detector score (`monotone_constraints = all +1`). This structurally forbids the
   signal-burying inversion above — the tree may only *reshape* the response, never
   flip it. This is the sanctioned "use it differently" path.
2. **Tree as re-weighter, not gate.** Use it to learn *per-category* weights that a
   transparent sum then applies — keeping the label-independent floor while letting
   labels tune emphasis.

The locked doc says exactly this: LightGBM is **parked, revisit with monotonic
constraints only once labels grow.** So the correct review statement is not
"weighted beats LightGBM forever" — it is:

> **At the current label count (~14 positives), a fixed positive-weighted sum is
> provably safer than a learned tree, because it cannot invert a specialist. A
> monotonic-constrained LightGBM becomes worth re-testing only once confirmed labels
> are numerous enough to fit a combiner without over-fitting — the exact regime the
> AAD feedback loop and ring/deviation layers are designed to reach.**

### The measured trade the weighted sum makes
The locked fusion gives up a little tabular peak (subspace-only mean 0.727) to buy
**+0.21 on IP** (0.327 → 0.538) — the relational capability the whole architecture
exists to provide. A learned tree on 14 labels gave up *everything* (≈0.22 mean).

> **SVG PROMPT — "Weighted sum vs LightGBM"**
> Two side-by-side panels. LEFT "LightGBM on 14 labels": three tall raw bars
> (INCOME 0.97, IP 0.51) with red down-arrows crashing them to 0.31 / 0.17,
> labelled "learned gate buries specialists". RIGHT "Weighted sum": the same raw
> bars each multiplied by a fixed weight and added, all arrows pointing UP into a
> combined score, labelled "positive weights — a strong signal can only help".
> Footer strip: "revisit LightGBM only WITH monotonic constraints once labels grow".
> Dark theme, red = destructive, green = safe.

---

## 16. Known structural weaknesses (include for credibility)

| Component | Assumption | What breaks it |
|---|---|---|
| GraphMCM DeepSVDD centroid | normal data is clean | if fraud dominates, hypersphere inflates to include fraud |
| EVT scorer | tail fits GPD smoothly | discontinuous tail → threshold explodes/collapses |
| Self-training | EVT tail is true fraud | if tail is typos, classifier anchors on data-entry errors |
| Isolated nodes | every node has ≥1 edge | unique-mobile+unique-IP nodes get zero structural signal (rely on subspace IF) |
| Synthetic exposure | archetypes match real fraud geometry | too-narrow archetypes bias toward obvious fraud only |
| Reconstruction on dense rings | anomalies are hard to reconstruct | dense cliques reconstruct *easily* → weak signal (why dense-block exists) |

**What breaks first in production:** self-training label promotion (§9). The
human gate is the mitigation.

---

### Reproducibility caveat (state it in the review)
`RGCNConv(aggr="add")` uses CUDA scatter-add atomics that are not seed-controlled →
±0.03–0.04 run-to-run noise on detector-derived scores. Use
`torch.use_deterministic_algorithms(True)` or CPU scoring before comparing effects
smaller than that. Every number in this doc is a single-run figure traceable to
Appendix H, not an average.
