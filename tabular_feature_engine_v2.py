import pandas as pd
import numpy as np
import json
import os
from difflib import SequenceMatcher
from datetime import datetime

def similar(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def engineer_features():
    print("Loading data...")
    df = pd.read_csv('datasets/data_for_ml_model.csv', low_memory=False)

    # 1. Drop 100% null columns (§2.2)
    null_cols = [
        'updated_by', 'delete_record', 'deleted_by', 'delete_on', 'delete_ip_address',
        'deleted_by_level', 'c_university_id', 'p_institution_id', 'x_institution_id',
        'xii_institution_id', 'competitive_exam_score', 'xii_course_id',
        'new_entitled_fee_amount_centre_share', 'sub_category_id',
        'updated_by-2', 'updated_on-2'
    ]
    df = df.drop(columns=[col for col in null_cols if col in df.columns])

    # 2. Drop duplicate columns (§2.3)
    # Keep: domicile_state_id, state_name, permanent_district_id, district_name
    dup_cols_to_drop = [
        'state_id', 'state_id-2', 'pfms_state_code',
        'state_name-2', 'district_id', 'district_name-2'
    ]
    df = df.drop(columns=[col for col in dup_cols_to_drop if col in df.columns])

    # 3. Handle high-nullity fields (§2.4)
    if 'disability_percentage' in df.columns:
        df['disability_percentage'] = df['disability_percentage'].fillna(0.0)
    if 'disablity_type' in df.columns:
        df['disablity_type'] = df['disablity_type'].fillna(-1.0)
    if 'orphan_flag' in df.columns:
        df['orphan_flag'] = df['orphan_flag'].fillna(False)
    if 'gaurdian_name' in df.columns:
        df['gaurdian_name'] = df['gaurdian_name'].fillna('Unknown')
    if 'enroll_udid_no' in df.columns:
        df['enroll_udid_no'] = df['enroll_udid_no'].fillna('None')
    if 'ration_card_no' in df.columns:
        df['ration_card_no'] = df['ration_card_no'].fillna('None')
    if 'ration_card_member_no' in df.columns:
        df['ration_card_member_no'] = df['ration_card_member_no'].fillna('None')
    if 'district_short_name' in df.columns:
        df = df.drop(columns=['district_short_name'])

    print("Engineering scalar features...")
    # Date features
    df['registered_date'] = pd.to_datetime(df['registered_date'], errors='coerce')
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['age_at_registration'] = (df['registered_date'] - df['date_of_birth']).dt.days / 365.25

    # Fee income ratio
    income = df['annual_family_income'].replace(0, 1)
    df['fee_income_ratio'] = (df['admission_fee'] + df['tution_fee'] + df['misc_fee']) / income

    # Name similarity
    # applying sequence matcher can be slow but we have 15000 rows, should take a few seconds
    df['name_similarity_score'] = df.apply(lambda row: similar(row['applicant_name'], row['father_name']), axis=1)

    print("Engineering aggregation features...")
    # Aggregation features
    df['mobile_application_count'] = df.groupby('mobile_no')['application_id'].transform('count')
    df['ip_application_count'] = df.groupby('ip_address')['application_id'].transform('count')
    
    df['mobile_unique_names'] = df.groupby('mobile_no')['applicant_name'].transform('nunique')
    df['mobile_unique_fathers'] = df.groupby('mobile_no')['father_name'].transform('nunique')
    
    df['institute_application_count'] = df.groupby('c_institution_id')['application_id'].transform('count')
    
    df['ip_to_mobile_ratio'] = df['ip_application_count'] / (df['mobile_application_count'] + 1)

    # Boolean identity matches
    def safe_lower(x):
        return str(x).lower().strip() if not pd.isna(x) else ""
        
    app_name = df['applicant_name'].apply(safe_lower)
    fat_name = df['father_name'].apply(safe_lower)
    mot_name = df['mother_name'].apply(safe_lower)

    df['is_applicant_name_eq_father'] = (app_name == fat_name).astype(int)
    # wait, if both are empty/nan, is it eq? No, if empty it shouldn't be equal.
    df['is_applicant_name_eq_father'] = df['is_applicant_name_eq_father'] & (app_name != "")
    df['is_applicant_name_eq_mother'] = ((app_name == mot_name) & (app_name != "")).astype(int)
    df['is_father_name_eq_mother'] = ((fat_name == mot_name) & (fat_name != "")).astype(int)

    # Save outputs
    print("Saving outputs...")
    output_csv = 'engineered_features_v2.csv'
    df.to_csv(output_csv, index=False)
    
    excluded_cols = ['application_id', 'sanity', 'jwt']
    
    # We define new engineered features
    engineered_features = [
        'age_at_registration',
        'fee_income_ratio',
        'name_similarity_score',
        'is_applicant_name_eq_father',
        'is_applicant_name_eq_mother',
        'is_father_name_eq_mother'
    ]
    aggregation_features = [
        'mobile_application_count',
        'ip_application_count',
        'mobile_unique_names',
        'mobile_unique_fathers',
        'institute_application_count',
        'ip_to_mobile_ratio'
    ]
    
    # Actually all columns in df that are not excluded should be in "features"? No, the contract just gives an example. Let's just list the ones we engineered.
    # Wait, the contract says:
    # "features": ["age_at_registration", "fee_income_ratio", ...],
    # "aggregation_features": ["mobile_application_count", "ip_application_count", ...],
    # "excluded": ["application_id", "sanity", "jwt"],
    
    # Let's include all non-excluded columns in "features" list, or just numeric ones?
    # Actually, the model typically consumes numerical/categorical. Let's list all features we want VAE to potentially use, but the simplest is just putting all column names that are not excluded.
    # Let's read contract carefully:
    # {
    #   "features": ["age_at_registration", "fee_income_ratio", ...],
    #   "aggregation_features": ["mobile_application_count", "ip_application_count", ...],
    #   "excluded": ["application_id", "sanity", "jwt"],
    #   "n_features": 42,
    #   "timestamp": "..."
    # }
    # So "features" is a list of features. Let's just put the columns that are typically features.
    # I will just list all columns except excluded and aggregation_features.
    all_cols = df.columns.tolist()
    features = [c for c in all_cols if c not in excluded_cols and c not in aggregation_features]
    
    schema = {
        "features": features,
        "aggregation_features": aggregation_features,
        "excluded": [c for c in excluded_cols if c in all_cols],
        "n_features": len(features) + len(aggregation_features),
        "timestamp": datetime.now().isoformat()
    }
    
    with open('v2_feature_schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    engineer_features()
