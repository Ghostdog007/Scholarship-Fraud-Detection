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

MI_KEEP_RATIO = 0.5
MI_MAX_FEATURES = 50
MRMR_MAX_FEATURES = 20

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
    
    df['mobile_occurrence_count'] = df.groupby('mobile_no')['application_id'].transform('count').fillna(0)
    df['ip_occurrence_count'] = df.groupby('ip_address')['application_id'].transform('count').fillna(0)
    
    if 'state_id' in df.columns and 'domicile_state_id' in df.columns:
        df['state_match_flag'] = (df['domicile_state_id'] == df['state_id']).astype(int)
    else:
        df['state_match_flag'] = 0
        
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
    
    # Target definition
    y = df['sanity'].notnull().astype(int)
    
    # Select only numeric features for ML
    exclude_cols = ['sanity', 'application_id', 'jwt']
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.select_dtypes(include=[np.number]).columns])
    
    # Fill remaining NaNs with median for MI and Correlation
    print("Imputing NaNs...")
    numeric_df = numeric_df.fillna(numeric_df.median())
    numeric_df = numeric_df.fillna(0) # For columns that were all NaNs
    
    # Step 2: Classwise Mutual Information filtering
    print("Computing class-weighted Mutual Information...")
    mi_scores = {}
    
    # Upweight minority class using w_c = max(1e-6, mean(y == c))
    # Note: Since the instructions mention upweighting using this formula,
    # we calculate class weights as inversely proportional to w_c.
    classes = np.unique(y)
    class_weights = {}
    for c in classes:
        w_c = max(1e-6, np.mean(y == c))
        class_weights[c] = 1.0 / w_c
        
    # Normalize class weights
    total_weight = sum(class_weights.values())
    class_weights = {c: w / total_weight for c, w in class_weights.items()}
    
    # Calculate weighted MI using binning
    n_bins = 10
    sample_weights = y.map(class_weights).values
    
    for col in numeric_df.columns:
        X_col = numeric_df[col].values
        # Bin the continuous feature
        bins = np.linspace(np.min(X_col), np.max(X_col), n_bins + 1)
        # Use np.digitize, ensuring values are within 1 to n_bins
        X_binned = np.digitize(X_col, bins) - 1
        X_binned = np.clip(X_binned, 0, n_bins - 1)
        
        # Calculate joint and marginal probabilities
        joint_prob = np.zeros((n_bins, len(classes)))
        for i, c in enumerate(classes):
            mask = (y == c)
            for b in range(n_bins):
                joint_prob[b, i] = np.sum(sample_weights[mask & (X_binned == b)])
                
        joint_prob /= np.sum(joint_prob)
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)
        
        # Calculate MI
        mi = 0
        for b in range(n_bins):
            for i in range(len(classes)):
                if joint_prob[b, i] > 0:
                    mi += joint_prob[b, i] * np.log(joint_prob[b, i] / (p_x[b] * p_y[i]))
        mi_scores[col] = mi
        
    mi_series = pd.Series(mi_scores).sort_values(ascending=False)
    
    # Retain top features by MI_KEEP_RATIO and MI_MAX_FEATURES
    n_keep = int(min(MI_MAX_FEATURES, len(mi_series) * MI_KEEP_RATIO))
    top_features = mi_series.head(n_keep).index.tolist()
    X_top = numeric_df[top_features]
    
    # Step 3: Pearson correlation pruning (|r| >= 0.90)
    print("Pearson correlation pruning...")
    corr_matrix = X_top.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for col in upper.columns:
        highly_correlated_with = upper.index[upper[col] >= 0.90].tolist()
        for hc_col in highly_correlated_with:
            if col not in to_drop and hc_col not in to_drop:
                # Drop the one with lower MI score
                if mi_series[col] < mi_series[hc_col]:
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
        # Select first feature (highest MI)
        best_first = max(remaining_features, key=lambda f: mi_series[f])
        selected_features.append(best_first)
        remaining_features.remove(best_first)
        
        corr_matrix_pruned = X_pruned.corr().abs()
        
        while len(selected_features) < MRMR_MAX_FEATURES and remaining_features:
            mrmr_scores = {}
            for f in remaining_features:
                relevance = mi_series[f]
                redundancy = np.mean([corr_matrix_pruned.loc[f, s] for s in selected_features])
                mrmr_scores[f] = relevance - redundancy
                
            best_next = max(mrmr_scores, key=mrmr_scores.get)
            selected_features.append(best_next)
            remaining_features.remove(best_next)
            
    print(f"Selected {len(selected_features)} features.")
    
    # Step 5: Save output
    output_data = {
        "selected_features": selected_features,
        "feature_scores": {f: float(mi_series[f]) for f in selected_features},
        "n_selected": len(selected_features),
        "pipeline_run_timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved to {args.output_json}")

if __name__ == "__main__":
    main()
