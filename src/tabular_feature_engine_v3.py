"""
tabular_feature_engine_v3.py

Two entry points called by main_v3.py in order:
  1. build_base()          → writes engineered_features_v3_nodeg.csv (63 features)
  2. add_degree_features() → merges degree_features_v3.csv → writes engineered_features_v3.csv (68 features)
                             and updates v3_feature_schema.json

Run standalone to execute both steps sequentially.
"""

import json
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config_v3 import (
    DEGREE_FEATURES,
    DUPLICATE_COLS_TO_DROP,
    EXCLUDED_FROM_FEATURES,
    IDENTIFIER_FEATURES,
    LOG1P_COLS,
    N_FEATURES,
    NULL_COLS_TO_DROP,
)

RAW_CSV       = Path("data/raw/data_for_ml_model.csv")
NODEG_CSV     = Path("data/processed/engineered_features_v3_nodeg.csv")
FINAL_CSV     = Path("data/processed/engineered_features_v3.csv")
DEGREE_CSV    = Path("data/processed/degree_features_v3.csv")
SCHEMA_JSON   = Path("data/processed/v3_feature_schema.json")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _name_similarity(a: str, b: str) -> float:
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, low_memory=False)

    existing_null_cols = [c for c in NULL_COLS_TO_DROP if c in df.columns]
    df.drop(columns=existing_null_cols, inplace=True)

    existing_dup_cols = [c for c in DUPLICATE_COLS_TO_DROP if c in df.columns]
    df.drop(columns=existing_dup_cols, inplace=True)

    for col in EXCLUDED_FROM_FEATURES:
        if col in df.columns and col != "application_id":
            pass  # keep application_id for row tracking; drop sanity/jwt later

    high_nullity_fill = {
        "disability_percentage": 0,
        "disablity_type": 0,
        "orphan_flag": 0,
        "gaurdian_name": "",
        "enroll_udid_no": 0,
        "ration_card_no": 0,
        "ration_card_member_no": 0,
    }
    for col, fill_val in high_nullity_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Scalar derived features ---
    df["registered_date"] = pd.to_datetime(df["registered_date"], errors="coerce")
    df["date_of_birth"]   = pd.to_datetime(df["date_of_birth"],   errors="coerce")
    df["age_at_registration"] = (
        (df["registered_date"] - df["date_of_birth"]).dt.days / 365.25
    ).clip(lower=0)

    for col in ["admission_fee", "tution_fee", "misc_fee", "annual_family_income"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    total_fee = df["admission_fee"] + df["tution_fee"] + df["misc_fee"]
    df["fee_income_ratio"] = total_fee / (df["annual_family_income"].clip(lower=1))

    df["name_similarity_score"] = df.apply(
        lambda r: _name_similarity(
            str(r.get("applicant_name", "")),
            str(r.get("father_name", ""))
        ),
        axis=1,
    )

    # --- Boolean identity matches ---
    df["is_applicant_name_eq_father"] = (
        df["applicant_name"].str.lower().str.strip()
        == df["father_name"].str.lower().str.strip()
    ).astype(int)

    df["is_applicant_name_eq_mother"] = (
        df["applicant_name"].str.lower().str.strip()
        == df["mother_name"].str.lower().str.strip()
    ).astype(int)

    df["is_father_name_eq_mother"] = (
        df["father_name"].str.lower().str.strip()
        == df["mother_name"].str.lower().str.strip()
    ).astype(int)

    # --- Cross-row aggregates ---
    df["mobile_application_count"] = df.groupby("mobile_no")["mobile_no"].transform("count")
    df["ip_application_count"]     = df.groupby("ip_address")["ip_address"].transform("count")

    df["mobile_unique_names"]   = df.groupby("mobile_no")["applicant_name"].transform("nunique")
    df["mobile_unique_fathers"] = df.groupby("mobile_no")["father_name"].transform("nunique")

    df["institute_application_count"] = df.groupby("c_institution_id")["c_institution_id"].transform("count")

    df["ip_to_mobile_ratio"] = (
        df["ip_application_count"] / (df["mobile_application_count"].clip(lower=1))
    )

    # --- District/state rank and deviation features ---
    df["income_rank_in_district"] = df.groupby("permanent_district_id")["annual_family_income"].rank(pct=True)

    state_median = df.groupby("domicile_state_id")["annual_family_income"].transform("median")
    df["income_deviation_from_state_median"] = df["annual_family_income"] - state_median

    # --- Binary-encoded categoricals ---
    if "gender" in df.columns:
        df["is_female"] = (df["gender"].str.upper().str.strip() == "F").astype(int)

    if "rural_urban" in df.columns:
        df["is_rural"] = (df["rural_urban"].str.upper().str.strip() == "R").astype(int)
        df["is_urban"] = (df["rural_urban"].str.upper().str.strip() == "U").astype(int)

    for bool_col in ["disability_flag", "orphan_flag", "hosteller", "is_singlegirlchild"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(False).astype(bool).astype(int)

    if "state_verify" in df.columns:
        df["has_state_verify"] = df["state_verify"].notna().astype(int)

    return df


def _select_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exclude = set(EXCLUDED_FROM_FEATURES) | {"application_id"}

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    exclude.update(non_numeric)

    feature_cols = [c for c in df.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if df[c].isnull().mean() < 0.5]

    feat_df = df[feature_cols].copy()

    for col in LOG1P_COLS:
        if col in feat_df.columns:
            feat_df[col] = np.log1p(feat_df[col].clip(lower=0))

    feat_df = feat_df.fillna(0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feat_df)
    feat_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)

    return feat_df, feature_cols


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_base() -> None:
    """Step 1: engineer 63 tabular features, write nodeg CSV."""
    print("[feature_engine] build_base() starting ...")
    df = _load_and_clean()
    df = _engineer_features(df)
    feat_df, feature_cols = _select_and_scale(df)

    out = feat_df.copy()
    out.insert(0, "application_id", df["application_id"].values)

    NODEG_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(NODEG_CSV, index=False)

    n_base = len(feature_cols)
    print(f"[feature_engine] build_base() done -- {n_base} base features, {len(out)} rows -> {NODEG_CSV}")
    if n_base != 63:
        print(f"[feature_engine] WARNING: expected 63 base features, got {n_base}")


def add_degree_features() -> None:
    """Step 3: merge degree CSV into nodeg CSV → write final 68-feature CSV + schema."""
    print("[feature_engine] add_degree_features() starting ...")

    if not NODEG_CSV.exists():
        raise FileNotFoundError(f"Run build_base() first — {NODEG_CSV} not found")
    if not DEGREE_CSV.exists():
        raise FileNotFoundError(f"Run graph_builder.build_graph() first — {DEGREE_CSV} not found")

    base_df   = pd.read_csv(NODEG_CSV)
    deg_df    = pd.read_csv(DEGREE_CSV)

    if "application_id" not in deg_df.columns:
        base_df = base_df.join(deg_df)
    else:
        base_df = base_df.merge(deg_df, on="application_id", how="left")

    for col in DEGREE_FEATURES:
        if col not in base_df.columns:
            base_df[col] = 0.0
        base_df[col] = base_df[col].fillna(0.0)
        scaler = MinMaxScaler()
        base_df[col] = scaler.fit_transform(base_df[[col]])

    feature_cols     = [c for c in base_df.columns if c != "application_id"]
    # V4 adoption (2026-07-15): drop the nominal identifier/code features from the
    # MODEL feature set. They carry no ordinal meaning for the reconstruction
    # detector and polluted the XAI narratives; their sharing signal survives via
    # the RAW-built graph edges + degree/count features. See noid ablation.
    dropped_ids      = [c for c in feature_cols if c in IDENTIFIER_FEATURES]
    feature_cols     = [c for c in feature_cols if c not in IDENTIFIER_FEATURES]
    base_df          = base_df[["application_id"] + feature_cols]
    base_features    = [c for c in feature_cols if c not in DEGREE_FEATURES]
    aggregation_feats = [
        "mobile_application_count", "ip_application_count",
        "mobile_unique_names", "mobile_unique_fathers",
        "institute_application_count", "ip_to_mobile_ratio",
        "income_rank_in_district", "income_deviation_from_state_median",
    ]
    aggregation_feats = [c for c in aggregation_feats if c in feature_cols]

    n_total = len(feature_cols)
    if n_total != N_FEATURES:
        print(f"[feature_engine] WARNING: expected {N_FEATURES} features, got {n_total}")

    schema = {
        "features": feature_cols,
        "base_features": base_features,
        "aggregation_features": aggregation_feats,
        "degree_features": DEGREE_FEATURES,
        "log1p_cols": LOG1P_COLS,
        "excluded": list(EXCLUDED_FROM_FEATURES),
        "dropped_from_model": sorted(dropped_ids),
        "n_features": n_total,
        "timestamp": datetime.utcnow().isoformat(),
    }

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(FINAL_CSV, index=False)
    SCHEMA_JSON.write_text(json.dumps(schema, indent=2))

    print(f"[feature_engine] add_degree_features() done -- {n_total} total features -> {FINAL_CSV}")
    print(f"[feature_engine] Schema written -> {SCHEMA_JSON}")


if __name__ == "__main__":
    build_base()
    print("[feature_engine] Waiting for graph_builder to produce degree_features_v3.csv ...")
    if DEGREE_CSV.exists():
        add_degree_features()
    else:
        print(f"[feature_engine] {DEGREE_CSV} not found -- run graph_builder_v3.py next, then call add_degree_features()")
