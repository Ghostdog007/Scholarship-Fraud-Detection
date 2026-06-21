import pandas as pd
import numpy as np
import torch
import json
import os

def build_synthetic_exposure_set():
    print("Loading engineered features...")
    df = pd.read_csv('data/processed/engineered_features_v2.csv', low_memory=False)
    
    with open('data/processed/v2_feature_schema.json', 'r') as f:
        schema = json.load(f)
        
    features = schema.get('features', []) + schema.get('aggregation_features', [])
    excluded = schema.get('excluded', [])
    valid_cols = [c for c in features if c not in excluded and c in df.columns]
    
    # We only keep numerical features for the tensor
    numeric_cols = df[valid_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    categories = ['AGE_VIOLATION', 'INCOME_VIOLATION', 'IP_CONCENTRATION', 'MOTHER_NAME_COLLISION', 'FEE_INFLATION']
    N = 150
    
    synthetic_dfs = []
    
    np.random.seed(42)
    
    # 1. AGE_VIOLATION
    df_age = df.sample(n=N, replace=True, random_state=42).copy()
    for i in range(N):
        if df_age.iloc[i]['pre_post_matric'] == 1:
            df_age.iloc[i, df_age.columns.get_loc('age_at_registration')] = np.random.uniform(21, 30)
        else:
            df_age.iloc[i, df_age.columns.get_loc('age_at_registration')] = np.random.uniform(36, 50)
    synthetic_dfs.append(df_age)
    
    # 2. INCOME_VIOLATION
    df_inc = df.sample(n=N, replace=True, random_state=43).copy()
    df_inc['annual_family_income'] = np.random.uniform(0, 999, size=N)
    synthetic_dfs.append(df_inc)
    
    # 3. IP_CONCENTRATION
    df_ip = df.sample(n=N, replace=True, random_state=44).copy()
    df_ip['ip_application_count'] = np.random.randint(15, 40, size=N)
    df_ip['ip_to_mobile_ratio'] = df_ip['ip_application_count'] / 2.0
    synthetic_dfs.append(df_ip)
    
    # 4. MOTHER_NAME_COLLISION
    df_mot = df.sample(n=N, replace=True, random_state=45).copy()
    if 'is_father_name_eq_mother' in df_mot.columns:
        df_mot['is_father_name_eq_mother'] = 1
    if 'is_applicant_name_eq_father' in df_mot.columns:
        df_mot['is_applicant_name_eq_father'] = 0
    if 'is_applicant_name_eq_mother' in df_mot.columns:
        df_mot['is_applicant_name_eq_mother'] = 0
    synthetic_dfs.append(df_mot)
    
    # 5. FEE_INFLATION
    df_fee = df.sample(n=N, replace=True, random_state=46).copy()
    df_fee['annual_family_income'] = np.random.uniform(20001, 100000, size=N)
    df_fee['fee_income_ratio'] = np.random.uniform(1.1, 5.0, size=N)
    synthetic_dfs.append(df_fee)
    
    # Combine
    df_synthetic = pd.concat(synthetic_dfs, ignore_index=True)
    
    # We only keep numerical features for the tensor
    X_df = df_synthetic[numeric_cols].copy()
    from sklearn.preprocessing import LabelEncoder
    for col in X_df.columns:
        if X_df[col].dtype == 'object' or X_df[col].dtype == 'bool' or pd.api.types.is_string_dtype(X_df[col]):
            le = LabelEncoder()
            # Convert to string and handle NaN
            X_df[col] = le.fit_transform(X_df[col].astype(str).fillna('missing'))
        else:
            X_df[col] = X_df[col].fillna(0)
    
    X = X_df.copy()
    
    # Convert to PyTorch tensor
    tensor = torch.tensor(X.values, dtype=torch.float32)
    print(f"Generated synthetic exposure tensor of shape: {tensor.shape}")
    
    # User Rule: use pytorch_gpu for every task
    try:
        device = torch.device('cuda')
        tensor = tensor.to(device)
    except Exception as e:
        pass # fallback to cpu if forced cuda fails
    
    torch.save(tensor, 'data/processed/synthetic_exposure_set.pt')
    print("Saved to synthetic_exposure_set.pt")

if __name__ == '__main__':
    build_synthetic_exposure_set()
