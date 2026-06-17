import pandas as pd
import numpy as np
import json
import datetime
import os
import torch
import argparse
from sklearn.feature_selection import mutual_info_classif

# Use pytorch_gpu per user global rule
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hard Constraints from AGENTS.md
# 2. Never use sanity as a feature.
# 3. Never use application_id as a feature.
# 4. Never use jwt as a feature.

MI_KEEP_RATIO = 0.8
MI_MAX_FEATURES = 50
MRMR_MAX_FEATURES = 40

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    
    # 2.2 Drop 100% null columns
    cols_to_drop = [
        'updated_by', 'delete_record', 'deleted_by', 'delete_on', 'delete_ip_address',
        'deleted_by_level', 'c_university_id', 'p_institution_id', 'x_institution_id',
        'xii_institution_id', 'competitive_exam_score', 'xii_course_id',
        'new_entitled_fee_amount_centre_share', 'sub_category_id', 'updated_by-2', 'updated_on-2'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 2.3 Drop duplicate columns
    dup_cols_to_drop = [
        'state_id-2', 'pfms_state_code',
        'state_name-2', 
        'district_id',
        'district_name-2'
    ]
    df = df.drop(columns=[c for c in dup_cols_to_drop if c in df.columns])
    
    return df

def engineer_features(df):
    registered_date = pd.to_datetime(df['registered_date'], errors='coerce')
    dob = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['age_at_registration'] = (registered_date - dob).dt.days / 365.25
    
    admission_fee = pd.to_numeric(df['admission_fee'], errors='coerce').fillna(0)
    tution_fee = pd.to_numeric(df['tution_fee'], errors='coerce').fillna(0)
    misc_fee = pd.to_numeric(df['misc_fee'], errors='coerce').fillna(0)
    family_income = pd.to_numeric(df['annual_family_income'], errors='coerce').replace(0, np.nan)
    df['fee_income_ratio'] = (admission_fee + tution_fee + misc_fee) / family_income
    df['fee_income_ratio'] = df['fee_income_ratio'].fillna(0)
    
    # PLACEHOLDER (non-functional on current data slice):
    # Ideally this compares domicile_state_id to the institution's state column.
    # However, no distinct institution-state column (e.g. c_state_id) exists in
    # this 15,000-record CSV. The column is hardcoded to 1 (constant) so it
    # carries zero variance and will receive SHAP = 0 — this is expected and
    # documented in AGENTS.md. Do not change until AISHE/DISE location data is
    # integrated. See AGENTS.md Section 10 open question: state_match_flag.
    if 'c_state_id' in df.columns:
        df['state_match_flag'] = (df['domicile_state_id'] == df['c_state_id']).astype(int)
    else:
        df['state_match_flag'] = 1  # constant placeholder — see comment above
        
    # --- Layer 0: Text to Boolean Flags & Fuzzy Match ---
    df['is_applicant_name_eq_father']  = (df['applicant_name'] == df['father_name']).astype(int)
    df['is_applicant_name_eq_mother']  = (df['applicant_name'] == df['mother_name']).astype(int)
    df['is_father_name_eq_mother']     = (df['father_name']    == df['mother_name']).astype(int)
    
    from difflib import SequenceMatcher
    def fuzzy_match(a, b):
        if pd.isnull(a) or pd.isnull(b): return 0.0
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    df['name_similarity_score'] = df.apply(lambda r: fuzzy_match(r['applicant_name'], r['father_name']), axis=1)

    # --- Layer 1: Relational / Cross-Row Aggregation ---
    df['mobile_application_count']     = df.groupby('mobile_no')['application_id'].transform('count').fillna(0)
    df['ip_application_count']         = df.groupby('ip_address')['application_id'].transform('count').fillna(0)
    df['mobile_unique_names_count']    = df.groupby('mobile_no')['applicant_name'].transform('nunique').fillna(0)
    df['mobile_unique_fathers_count']  = df.groupby('mobile_no')['father_name'].transform('nunique').fillna(0)
    if 'c_institution_id' in df.columns:
        df['institute_application_count']  = df.groupby('c_institution_id')['application_id'].transform('count').fillna(0)
    if 'district_id' in df.columns:
        df['district_application_count']   = df.groupby('district_id')['application_id'].transform('count').fillna(0)
    df['ip_to_mobile_ratio'] = df['ip_application_count'] / (df['mobile_application_count'] + 1)

    # --- Layer 2: Policy Boundary Flags ---
    income_num = pd.to_numeric(df['annual_family_income'], errors='coerce').fillna(0)
    df['flag_income_below_20000']     = (income_num < 20000).astype(int)
    df['flag_income_below_10000']     = (income_num <= 10000).astype(int)
    df['flag_income_extreme_low']     = (income_num < 1000).astype(int)
    df['flag_prematric_age_over20']   = ((df['pre_post_matric'] == 1) & (df['age_at_registration'] > 20)).astype(int)
    df['flag_postmatric_age_over35']  = ((df['pre_post_matric'] == 2) & (df['age_at_registration'] > 35)).astype(int)
    df['flag_postmatric_age_under13'] = ((df['pre_post_matric'] == 2) & (df['age_at_registration'] < 13)).astype(int)
    df['flag_fee_exceeds_income']     = (df['fee_income_ratio'] > 1.0).astype(int)
        
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/data_for_ml_model.csv')
    parser.add_argument('--output_json', type=str, default='selected_features.json')
    args = parser.parse_args()

    print("Loading data...")
    df = load_and_clean_data(args.data_path)
    print("Engineering features...")
    df = engineer_features(df)
    
    PROTECTED_FEATURES = [
        # Layer 0 — Text-to-boolean (AGENTS.md Section 11.1)
        'is_applicant_name_eq_father', 'is_applicant_name_eq_mother',
        'is_father_name_eq_mother', 'name_similarity_score',
        # Layer 1 — Relational aggregates
        'mobile_application_count', 'ip_application_count',
        'mobile_unique_names_count', 'mobile_unique_fathers_count',
        'institute_application_count', 'district_application_count',
        'ip_to_mobile_ratio',
        # Layer 2 — Policy boundary flags
        'flag_income_below_20000', 'flag_income_below_10000',
        'flag_income_extreme_low', 'flag_prematric_age_over20',
        'flag_postmatric_age_over35', 'flag_postmatric_age_under13',
        'flag_fee_exceeds_income',
        # Derived
        'age_at_registration', 'fee_income_ratio', 'state_match_flag'
    ]
    
    # Target definition
    y = df['sanity'].notnull().astype(int)
    
    # Select only numeric features for ML
    exclude_cols = ['sanity', 'application_id', 'jwt']
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.select_dtypes(include=[np.number]).columns])
    
    # Fill remaining NaNs with median for MI and Correlation
    print("Imputing NaNs...")
    numeric_df = numeric_df.fillna(numeric_df.median())
    numeric_df = numeric_df.fillna(0) # For columns that were all NaNs

    protected_present = [f for f in PROTECTED_FEATURES if f in numeric_df.columns]
    candidate_cols    = [c for c in numeric_df.columns if c not in protected_present]
    numeric_candidates = numeric_df[candidate_cols]
    numeric_protected  = numeric_df[protected_present]
    
    # Step 2: Classwise Mutual Information filtering
    print("Computing class-weighted Mutual Information...")
    mi_scores = {}
    
    classes = np.unique(y)
    class_weights = {}
    for c in classes:
        w_c = max(1e-6, np.mean(y == c))
        class_weights[c] = 1.0 / w_c
        
    # Normalize class weights
    total_weight = sum(class_weights.values())
    class_weights = {c: w / total_weight for c, w in class_weights.items()}
    
    n_bins = 10
    sample_weights = y.map(class_weights).values
    
    for col in numeric_candidates.columns:
        X_col = numeric_candidates[col].values
        bins = np.linspace(np.min(X_col), np.max(X_col), n_bins + 1)
        X_binned = np.digitize(X_col, bins) - 1
        X_binned = np.clip(X_binned, 0, n_bins - 1)
        
        joint_prob = np.zeros((n_bins, len(classes)))
        for i, c in enumerate(classes):
            mask = (y == c)
            for b in range(n_bins):
                joint_prob[b, i] = np.sum(sample_weights[mask & (X_binned == b)])
                
        joint_prob /= np.sum(joint_prob)
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)
        
        mi = 0
        for b in range(n_bins):
            for i in range(len(classes)):
                if joint_prob[b, i] > 0:
                    mi += joint_prob[b, i] * np.log(joint_prob[b, i] / (p_x[b] * p_y[i]))
        mi_scores[col] = mi

    for col in numeric_protected.columns:
        X_col = numeric_protected[col].values
        bins = np.linspace(np.min(X_col), np.max(X_col), n_bins + 1)
        X_binned = np.digitize(X_col, bins) - 1
        X_binned = np.clip(X_binned, 0, n_bins - 1)
        
        joint_prob = np.zeros((n_bins, len(classes)))
        for i, c in enumerate(classes):
            mask = (y == c)
            for b in range(n_bins):
                joint_prob[b, i] = np.sum(sample_weights[mask & (X_binned == b)])
                
        joint_prob /= np.sum(joint_prob)
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)
        
        mi = 0
        for b in range(n_bins):
            for i in range(len(classes)):
                if joint_prob[b, i] > 0:
                    mi += joint_prob[b, i] * np.log(joint_prob[b, i] / (p_x[b] * p_y[i]))
        mi_scores[col] = mi
        
    mi_series = pd.Series(mi_scores).sort_values(ascending=False)
    
    # Retain top candidates by MI_KEEP_RATIO and MI_MAX_FEATURES
    mi_series_candidates = mi_series.loc[numeric_candidates.columns].sort_values(ascending=False)
    n_keep = int(min(MI_MAX_FEATURES, len(mi_series_candidates) * MI_KEEP_RATIO))
    top_features = mi_series_candidates.head(n_keep).index.tolist()
    X_top = numeric_candidates[top_features]
    
    # Step 3: Pearson correlation pruning (|r| >= 0.90) on candidates only
    print("Pearson correlation pruning...")
    corr_matrix = X_top.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for col in upper.columns:
        highly_correlated_with = upper.index[upper[col] >= 0.90].tolist()
        for hc_col in highly_correlated_with:
            if col not in to_drop and hc_col not in to_drop:
                if mi_series_candidates[col] < mi_series_candidates[hc_col]:
                    to_drop.append(col)
                else:
                    to_drop.append(hc_col)
                    
    pruned_features = [f for f in top_features if f not in to_drop]
    X_pruned = X_top[pruned_features]
    
    # Step 4: mRMR selection
    print("mRMR selection...")
    selected_features = []
    remaining_features = list(X_pruned.columns)
    
    if remaining_features:
        best_first = max(remaining_features, key=lambda f: mi_series_candidates[f])
        selected_features.append(best_first)
        remaining_features.remove(best_first)
        
        corr_matrix_pruned = X_pruned.corr().abs()
        
        while len(selected_features) < MRMR_MAX_FEATURES and remaining_features:
            mrmr_scores = {}
            for f in remaining_features:
                relevance = mi_series_candidates[f]
                redundancy = np.mean([corr_matrix_pruned.loc[f, s] for s in selected_features])
                mrmr_scores[f] = relevance - redundancy
                
            best_next = max(mrmr_scores, key=mrmr_scores.get)
            selected_features.append(best_next)
            remaining_features.remove(best_next)
            
    # Merge: mRMR candidates + all protected features
    if protected_present:
        prot_df = numeric_protected.copy()
        prot_corr = prot_df.corr().abs()
        prot_upper = prot_corr.where(
            np.triu(np.ones(prot_corr.shape), k=1).astype(bool)
        )
        prot_to_drop = set()
        for col in prot_upper.columns:
            for hc in prot_upper.index[prot_upper[col] >= 0.98].tolist():
                if col not in prot_to_drop and hc not in prot_to_drop:
                    if mi_series.get(col, 0) < mi_series.get(hc, 0):
                        prot_to_drop.add(col)
                    else:
                        prot_to_drop.add(hc)
        protected_final = [f for f in protected_present if f not in prot_to_drop]
    else:
        protected_final = []

    final_features = selected_features + [
        f for f in protected_final if f not in selected_features
    ]

    print(f"mRMR candidates selected: {len(selected_features)}")
    print(f"Protected features added: {len([f for f in protected_final if f not in selected_features])}")
    print(f"Total final features: {len(final_features)}")
    
    # Step 5: Save output
    output_data = {
        "selected_features": final_features,
        "feature_scores": {f: float(mi_series.get(f, 0.0)) for f in final_features},
        "protected_features": protected_final,
        "n_selected": len(final_features),
        "pipeline_run_timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved to {args.output_json}")

if __name__ == "__main__":
    main()
