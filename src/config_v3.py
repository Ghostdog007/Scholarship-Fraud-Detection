import os  # V4_SEED env override for multi-seed ablation runs

_BFP = "S2FuaXNoayBTaGFybWEgfCBOU1VUIHwgQmF0Y2ggMjAyNyB8IHNvbGUgYXV0aG9yLCBOSUMgRnJhdWQgRGV0ZWN0aW9uIFByb2plY3Q="

# N_FEATURES is the model feature-vector width. Adopted 44 on 2026-07-15: the 24
# nominal identifier/code features in IDENTIFIER_FEATURES were dropped from the
# detector input (noid ablation — safe, no regression at detector or fused level;
# see project memory 'noisy-identifier-feature-drop'). The V4_N_FEATURES env
# override lets the ablation harness reconstruct the old 68-feature control.
N_FEATURES      = int(os.environ.get("V4_N_FEATURES", "44"))

# Training data source for main_v3.py's build_base/build_graph steps (step 4
# cut-over switch, hard stop 13): "file" is the default, unchanged path —
# raw CSV + file-based feature/graph builders. "postgres" switches to
# build_base_pg()/build_graph_pg() (src/db/features.py), reading every
# MERGED batch (primary + any admin-merged cohorts) instead of just the
# original raw CSV, writing to the SAME canonical file paths so every
# downstream step (add_degree_features, train, ...) is unaffected. Gate 4
# already demonstrated bit-exact parity on the 15k primary population — this
# flag is how an operator (or the API) opts a given run into reading Postgres
# without changing the default for anyone else. Set via NIC_DATA_SOURCE=postgres.
DATA_SOURCE = os.environ.get("NIC_DATA_SOURCE", "file")
if DATA_SOURCE not in ("file", "postgres"):
    raise ValueError(f"NIC_DATA_SOURCE must be 'file' or 'postgres', got '{DATA_SOURCE}'")

# Nominal identifier / code features dropped from the MODEL feature set (single
# source of truth — consumed by tabular_feature_engine_v3 to shrink the schema and
# by xai_layer_v3 as the never-narrated set). Their sharing signal is preserved by
# the graph edges (built from RAW columns) + degree/count features, all kept.
IDENTIFIER_FEATURES = [
    "mobile_no", "aadhaar_vault_ref_token", "permanent_pincode",
    "village_id", "sub_district_id", "permanent_district_id",
    "c_institution_id", "c_course_id", "p_university_id", "p_course_id",
    "x_university_id", "pfms_district_code", "lgd_district_code",
    "domicile_state_id", "category_id", "inst_verify_by", "state_verify_by",
    "religion", "marital_status", "parent_occupation", "disablity_type",
    "application_level", "modeofstudy", "pre_post_matric",
]
N_EDGE_TYPES    = 5
MASK_NUM        = 8
GRAPH_HIDDEN    = 128
GRAPH_EMB_DIM   = 64
MLP_HIDDEN      = 256
Z_DIM           = 64
LOE_MARGIN      = 2.0
# Persistent Stage 2 LOE weight (added 2026-07-22 — see README changelog). Stage 2
# previously had NO exposure term at all, so any separation Stage 1 bought could be
# freely re-absorbed by 120 epochs of unconstrained reconstruction — this showed up
# concretely in prototyping: dense synthetic cliques reconstruct too easily (MAR
# critique), and Stage 2 had nothing stopping the model re-learning to reconstruct
# them well. A small non-zero weight through Stage 2 keeps that separation alive
# instead of letting it decay to nothing. Paired with the hinge-margin _loe_loss
# rewrite (was exp(-sqrt(dist)), which saturated to ~0 within the first few epochs
# in BOTH a 30-epoch and a 150-epoch prototype run — an exponential decay is not a
# usable training signal once distances exceed a handful of units).
LOE_STAGE2_WEIGHT = 0.15
LAMBDA_EDGE     = 0.3   # TRAINING loss weight only (edge-prediction head) -- unchanged,
                        # untested whether removing it from training hurts embedding quality
# Inference-time score-composition weight (added 2026-07-23, redlined under
# explicit lead direction). hybrid_anomaly_score = feature_pred_error +
# LAMBDA_EDGE_SCORE * edge_pred_error -- was LAMBDA_EDGE (0.3) prior to this
# change, now decoupled from the training-loss weight above. Evidence:
# edge_pred_error showed NO usable signal on any of its 3 designed ring
# categories (IP/MOBILE/PINCODE clusters, held-out PR-AUC 0.011-0.023, at or
# below noise floor) and actively diluted feature_pred_error's real
# MOBILE_CLUSTER signal (0.268 standalone vs 0.017 combined). Dropping it
# from the score improved 6/7 categories (mean PR-AUC 0.017->0.056) with only
# a negligible regression on IP_CLUSTER (-0.0007), replicated on a second,
# independently-seeded stress population (mean 0.017->0.048, same sign on
# every category). See outputs/ablation_lambda_edge_v3_44.json and
# outputs/ablation_lambda_edge_v3_44_stress2.json for raw stdout. Does NOT
# affect training (LAMBDA_EDGE above is untouched) -- the edge-prediction
# head still trains and edge_pred_error is still computed/available for XAI,
# it is simply excluded from the ranking score. See README changelog
# 2026-07-23.
LAMBDA_EDGE_SCORE = 0.0
LAMBDA_EXPOSURE = 1.0
EPOCHS_STAGE1   = 80
EPOCHS_STAGE2   = 120
LR              = 1e-3
BATCH_SIZE      = 256
RANDOM_SEED     = int(os.environ.get("V4_SEED", "42"))  # override per run for seed replication

# Incremental fine-tune (CPU server, post-cycle update)
INCREMENTAL_EPOCHS = 10    # Stage 2 epochs only, RGCN frozen
INCREMENTAL_LR     = 1e-4  # lower than full LR to avoid overwriting graph knowledge

# Confirmed fraud sample weight in LightGBM fusion
# Confirmed = hard label from supervisor; pseudo = EVT-promoted soft label
CONFIRMED_WEIGHT = 3.0

# KS test p-value below which full GPU retrain is recommended
DRIFT_KS_THRESHOLD = 0.01

EDGE_TYPES = [
    "shares_mobile",
    "shares_ip",
    "shares_father_name",
    "shares_mother_name",
    "shares_pincode",
]

DEGREE_FEATURES = [
    "degree_shares_mobile",
    "degree_shares_ip",
    "degree_shares_father_name",
    "degree_shares_mother_name",
    "degree_shares_pincode",
]

SUBSPACE_GROUPS = {
    "financial": [
        "annual_family_income",
        "fee_income_ratio",
        "income_rank_in_district",
        "income_deviation_from_state_median",
        "admission_fee",
        "tution_fee",
        "misc_fee",
    ],
    "identity": [
        "name_similarity_score",
        "is_father_name_eq_mother",
        "is_applicant_name_eq_father",
        "is_applicant_name_eq_mother",
        "mobile_unique_names",
        "mobile_unique_fathers",
    ],
    "network": [
        "ip_application_count",
        "ip_to_mobile_ratio",
        "mobile_application_count",
        "institute_application_count",
        "degree_shares_ip",
        "degree_shares_mobile",
        "degree_shares_pincode",
    ],
}

LOG1P_COLS = [
    "annual_family_income",
    "admission_fee",
    "tution_fee",
    "misc_fee",
]

NULL_COLS_TO_DROP = [
    "updated_by", "delete_record", "deleted_by", "delete_on",
    "delete_ip_address", "deleted_by_level", "c_university_id",
    "p_institution_id", "x_institution_id", "xii_institution_id",
    "competitive_exam_score", "xii_course_id",
    "new_entitled_fee_amount_centre_share", "sub_category_id",
    "updated_by-2", "updated_on-2",
]

DUPLICATE_COLS_TO_DROP = [
    "state_id", "state_id-2", "pfms_state_code",
    "state_name-2",
    "district_id",
    "district_name-2",
    "district_short_name",
]

EXCLUDED_FROM_FEATURES = ["application_id", "sanity", "jwt"]

# Self-training: minimum number of EVT signals that must fire for Round 0 promotion.
# 1 = original OR logic (noisy); 2 = requires multi-signal agreement (recommended).
MIN_SIGNALS_FOR_PROMOTION = 2

# EVT GPD shape validity range. Fits outside this range are rejected and fall back
# to empirical quantile. Values outside [-0.5, 1.0] indicate a distribution that
# violates GPD regularity assumptions (heavy tails or discrete cluster spikes).
EVT_SHAPE_MIN = -0.5
EVT_SHAPE_MAX = 1.0

# DeepSVDD centroid: fraction of highest-norm embeddings excluded before computing
# the centroid mean. Excludes likely-contaminated nodes (potential fraud) from the
# normal centroid definition.
CENTROID_CLEAN_PERCENTILE = 95  # keep bottom 95% by embedding norm

import os as _os  # env overrides for the ablation switches (set once per run)

def _env(name, default):
    return _os.environ.get(name, default)

# ── V4: encoder switch (lets all 3 ablation configs run from one codebase) ───
# Override per run with env vars, e.g.  V4_ENCODER_ARCH=rgcn V4_TOPO_EXPOSURE=0
ENCODER_ARCH         = _env("V4_ENCODER_ARCH", "rgcn")  # "rgcn" | "han" — default rgcn: 3-seed ablation (2026-07-04) showed HAN drop-in regresses (-0.091 mean, all seeds); see AGENTS.md V4 ablation results
ARCH_VERSION         = {"rgcn": "rgcn_v1", "han": "han_v1"}[ENCODER_ARCH]  # ckpt tag (hard stop #15)

# ── V4: HAN encoder (ADR-015) ───────────────────────────────────────────────
ATTN_HEADS           = 4          # node-level GAT heads per relation
ATTN_LEAKY_SLOPE     = 0.2        # LeakyReLU slope in node-level attention
SEMANTIC_ATTN_HIDDEN = 32         # hidden dim of the semantic-attention MLP

# ── V4: topology synthetic exposure (ADR-016) ───────────────────────────────
TOPO_EXPOSURE_ENABLED   = _env("V4_TOPO_EXPOSURE", "1") == "1"  # "0" reproduces config-1
N_TOPO_CLUSTERS         = int(_env("V4_TOPO_CLUSTERS", "50"))   # synthetic connected clusters
TOPO_CLUSTER_SIZE_RANGE = (6, 40) # nodes per synthetic cluster (min, max)

# ── V4: connected-cluster evaluation (T1) ───────────────────────────────────
EVAL_CONNECTED_N_CLUSTERS   = int(_env("V4_EVAL_CLUSTERS", "30"))  # injected clusters/category
EVAL_CONNECTED_SIZE_RANGE   = (6, 40)

# ── V4 Phase 2: Tier-1 attention-summary features (T7) ───────────────────────
TIER1_ATTN_FEATURES = _env("V4_TIER1_ATTN", "0") == "1"  # default OFF; flip to adopt in fusion
ATTN_READOUT_K      = 8    # top-k neighbours summarised per node in the read-out head

# ── V4 Phase 2: subgraph ring-classifier (T8) ────────────────────────────────
RING_MIN_SIZE       = 4    # candidate rings smaller than this are ignored (structural, not policy)
RING_SPECTRAL_K     = 3    # top-k Laplacian eigenvalues kept in the fingerprint
RING_N_NEG_SAMPLES  = int(_env("V4_RING_NEG", "400"))  # random real subgraphs as classifier negatives
RING_OPEN_SET_ENABLED = True   # compute nearest-prototype distance for novel-ring flagging
RING_CLASSIFIER_ENABLED = _env("V4_RING", "0") == "1"  # default OFF; flip to adopt in pipeline

# ── V4 Phase 2: head-to-head (T9) ────────────────────────────────────────────
COMPARE_SEEDS       = (42, 43, 44)   # the three seeds the head-to-head averages over
EVAL_HELDOUT_SIZE_RANGE = (6, 40)    # T9b held-out topology size (structure differs, see T9b)

# ── V4.1: dense-block detector — relational specialist, part of the locked architecture ─
# Extended 2026-07-22 from shares_ip-only to shares_mobile + shares_ip + shares_pincode,
# per the stress_testing_1 ablation (50k synthetic cohort, ground-truth fraud rings):
# ip-only PR-AUC 0.209 -> relational (IP-priority-weighted) 0.261, with IP-ring
# detection held ~unchanged (0.2199 -> 0.2196) and mobile-ring detection going from
# near-zero (0.030) to real signal (0.149). Equal-weighting the 3 relations was
# tried first and rejected: it gained more overall (0.268) but let ordinary,
# non-fraud density in mobile/pincode outrank true IP-ring members (IP PR-AUC
# collapsed to 0.067) -- unacceptable given IP is the dominant real fraud vector.
# See README.md changelog (2026-07-22) and outputs/stress_testing_1_v2b_stats.json
# for the full sweep. Redlined under explicit lead direction (sole author).
DENSE_BLOCK_ENABLED   = _env("V4_DENSE_BLOCK", "1") == "1"   # default ON (architected component)
# Pincode dropped 2026-07-22 per lead direction: not a valid fraud signal on its
# own (shared pincode reflects legitimate geographic clustering, not collusion) —
# reverted to shares_mobile + shares_ip only.
DENSE_BLOCK_RELATIONS = [0, 1]                                # shares_mobile, shares_ip
# Priority weights applied to each relation's own min-max-normalised score BEFORE
# the max-combine (DENSE_BLOCK_RELATIONS index -> weight). IP dominant by design —
# most real fraud in this population runs through IP; mobile is a boost, not equal.
DENSE_BLOCK_RELATION_WEIGHTS = {0: 0.3, 1: 1.0}               # mobile, ip
DENSE_BLOCK_KCORE_PREFILTER = True                            # k-core narrows, then peel
DENSE_BLOCK_CAMOUFLAGE_C    = 5.0                             # w = 1/log(deg + c)

# ── V4.2: Deep SAD center-distance (2026-07-22, XAI-only — NOT in fusion) ──────
# Separate encoder + objective from Hybrid GraphMCM: pulls real nodes toward a
# learned "normal" center, pushes topology exposure's synthetic archetypes away
# via an inverted-distance term (Ruff et al., ICLR 2020). No reconstruction loss,
# so it doesn't inherit the MAR reconstruct-too-easily failure mode reconstruction
# based components have. Validated on stress_testing_1 against hybrid_reconstruction
# (0.153 overall / 0.029 mobile-ring): center_dist_score reached 0.201 overall /
# 0.093 mobile-ring, 0.050 IP-ring — the single strongest relational signal found
# this session. A companion per-cluster "nearest known archetype" prototype-match
# score was also tested and rejected (0.116, near-random on every category — no
# inter-prototype separation term, prototypes collapsed too close together).
# Deliberately kept OUT of FUSION_COMPONENTS / final_risk_score. Tested directly
# (2026-07-22): re-scored stress_testing_1 with a candidate 4-way max fusion
# (existing 3 + minmax(center_dist_score)) — overall PR-AUC 0.4182 -> 0.4181
# (noise-level, not an improvement). Deep SAD only won the argmax in 483/50,000
# nodes (<1%): the existing trio already dominates its specialty categories so
# completely (e.g. mobile-ring alone reaches 0.53 PR-AUC under the current fusion,
# far above Deep SAD's standalone 0.09-0.10 there) that a 4th max-input rarely
# gets to matter. REJECTED for fusion on this evidence; remains XAI-card-only
# (xai_layer_v3.py). See outputs/stress_testing_1_{deepsad,fusion4}_stats.json.
DEEPSAD_ENABLED  = _env("V4_DEEPSAD", "1") == "1"
DEEPSAD_HIDDEN   = 64
DEEPSAD_EMB_DIM  = 32
DEEPSAD_EPOCHS   = 150
DEEPSAD_ETA      = 1.0     # inverted-distance weight for exposure (anomaly) nodes
DEEPSAD_LR       = 1e-3
DEEPSAD_CENTER_REFRESH_EVERY = 25   # epochs between centroid recomputation

# ── V4.1: score-level fusion (LOCKED — replaces the 14-positive LightGBM) ──────
# risk = minmax( max( minmax(subspace), minmax(dense_block_relational), minmax(hybrid) ) )
# Changed 2026-07-22 from the additive weighted-sum to an unweighted max, per the
# stress_testing_1 ablation: on every category where one detector actually had
# signal, the weighted-sum scored WORSE than that detector alone (e.g. mobile-ring:
# subspace alone 0.674 vs fused 0.349) -- summing let the other two detectors'
# near-random noise dilute the one that found the fraud. Plain max: overall PR-AUC
# 0.403 -> 0.447 on the same data. Still label-independent (no learned gate) and,
# if anything, MORE protective of a strong raw signal than the sum was, since max
# preserves the single strongest detector's value exactly regardless of the other
# two. See AGENTS.md H.8 for why a LEARNED fusion (LightGBM) was rejected instead —
# that risk (overfitting on sparse labels) does not apply to switching combination
# functions, only to learning the combination from labels.
# FUSION_W_SUBSPACE/DENSE_IP/HYBRID are retired (max has no per-component weight);
# FUSION_COMPONENTS lists the three inputs for display/ordering only.
FUSION_COMPONENTS = ("subspace", "dense_relational", "hybrid")

# ── V4 revamp: deviation layer (D2/D3) ───────────────────────────────────────
DEVIATION_LAYER_ENABLED     = _env("V4_DEVIATION", "0") == "1"  # default OFF
DEV_MIN_CONFIRMED_PER_CATEGORY = int(_env("V4_DEV_MIN_CONF", "5"))  # hard stop #19
DEV_OOF_FOLDS         = 5                                     # leakage-safe stacking
DEV_HIDDEN            = 64
DEV_EPOCHS            = 50
DEV_LR                = 1e-3
DEV_CONF_MARGIN       = 5.0                                   # DevNet confidence margin a
DEV_PAIRWISE          = True                                  # PReNet pairwise augmentation

# ── V4 revamp: fusion input set (keep GNN cols until comparison says otherwise) ─
FUSION_INCLUDE_GNN_COLS = True   # baseline cols stay; comparison tests removal

