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
