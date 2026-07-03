N_FEATURES      = 68
N_EDGE_TYPES    = 5
MASK_NUM        = 8
GRAPH_HIDDEN    = 128
GRAPH_EMB_DIM   = 64
MLP_HIDDEN      = 256
Z_DIM           = 64
LOE_MARGIN      = 2.0
LAMBDA_EDGE     = 0.3
LAMBDA_EXPOSURE = 1.0
EPOCHS_STAGE1   = 80
EPOCHS_STAGE2   = 120
LR              = 1e-3
BATCH_SIZE      = 256
RANDOM_SEED     = 42

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
ENCODER_ARCH         = _env("V4_ENCODER_ARCH", "han")   # "rgcn" | "han"
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
