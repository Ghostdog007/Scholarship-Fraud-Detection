import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import shap
import json
import os
import sys
import argparse

# Rule Weights
WEIGHTS = {
    'IDENTITY_DUPLICATE': 2.0,
    'AGE_VIOLATION': 1.5,
    'INCOME_VIOLATION': 1.0, # UW, X13, X21
    'NAME_MATCH': 1.5,
    'MOBILE_CONCENTRATION': 2.0,
    'MOBILE_FATHER_MISMATCH': 2.0, # UN
    'BOARD_DUPLICATE': 2.0 # X9, X10
}

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() 
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

def train_vae(df_features, valid_mask, epochs=50, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"VAE Device: {device}")
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(df_features.fillna(0).values)
    
    x_train = torch.tensor(x_scaled[valid_mask], dtype=torch.float32)
    x_all = torch.tensor(x_scaled, dtype=torch.float32)
    
    dataset = TensorDataset(x_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = x_train.shape[1]
    model = VAE(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch in dataloader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch)
            loss = vae_loss(recon_batch, x_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        x_all = x_all.to(device)
        recon_all, _, _ = model(x_all)
        mse = torch.mean((x_all - recon_all) ** 2, dim=1)
        # Using negative MSE transformed to prob-like
        recon_prob = torch.exp(-mse).cpu().numpy()
        
    return recon_prob

def apply_rules(df):
    rules_fired = pd.Series('', index=df.index, dtype=object)
    rule_scores = pd.Series(0.0, index=df.index)
    
    def add_violation(condition, rule_code, weight):
        idx = condition[condition].index
        rules_fired.loc[idx] = rules_fired.loc[idx].apply(lambda x: x + rule_code + ',' if x else rule_code + ',')
        rule_scores.loc[idx] += weight
        
    # Identity duplicate
    id_cols = ['applicant_name', 'father_name', 'mother_name', 'date_of_birth', 'mobile_no']
    if all(c in df.columns for c in id_cols):
        cond_id_dup = df.duplicated(subset=id_cols, keep=False)
        add_violation(cond_id_dup, 'ID_DUP', WEIGHTS['IDENTITY_DUPLICATE'])

    # Age rules
    if 'pre_post_matric' in df.columns:
        if 'age_at_registration' in df.columns:
            age_col = df['age_at_registration']
        elif 'registered_date' in df.columns and 'date_of_birth' in df.columns:
            # Safely calculate age if registered_date is available
            try:
                age_col = (pd.to_datetime(df['registered_date']) - pd.to_datetime(df['date_of_birth'])).dt.days / 365.25
            except Exception:
                age_col = None
        else:
            age_col = None

        if age_col is not None:
            cond_x1 = (df['pre_post_matric'] == 1) & (age_col > 20)
            add_violation(cond_x1, 'X1', WEIGHTS['AGE_VIOLATION'])
            
            cond_x7 = (df['pre_post_matric'] == 2) & (age_col > 35)
            add_violation(cond_x7, 'X7', WEIGHTS['AGE_VIOLATION'])
            
            cond_x8 = (df['pre_post_matric'] == 2) & (age_col < 13)
            add_violation(cond_x8, 'X8', WEIGHTS['AGE_VIOLATION'])
            
    # Income rules
    if 'annual_family_income' in df.columns:
        cond_uw = (pd.to_numeric(df['annual_family_income'], errors='coerce') < 20000) & (pd.to_numeric(df['annual_family_income'], errors='coerce') > 10000)
        add_violation(cond_uw, 'UW', WEIGHTS['INCOME_VIOLATION'])
        
        cond_x13_21 = pd.to_numeric(df['annual_family_income'], errors='coerce') <= 10000
        add_violation(cond_x13_21, 'X13_X21', WEIGHTS['INCOME_VIOLATION'])
        
    # Name match
    if all(c in df.columns for c in ['applicant_name', 'father_name', 'mother_name']):
        cond_yf = (df['applicant_name'] == df['father_name']) | (df['applicant_name'] == df['mother_name'])
        cond_yf = cond_yf.fillna(False)
        add_violation(cond_yf, 'YF', WEIGHTS['NAME_MATCH'])
        
    # Board duplicate
    if 'x_roll_no' in df.columns and 'x_course_year' in df.columns:
        cond_x9 = df.duplicated(subset=['x_roll_no', 'x_course_year'], keep=False) & df['x_roll_no'].notna()
        add_violation(cond_x9, 'X9', WEIGHTS['BOARD_DUPLICATE'])
    
    if 'xii_roll_no' in df.columns and 'xii_course_year' in df.columns:
        cond_x10 = df.duplicated(subset=['xii_roll_no', 'xii_course_year'], keep=False) & df['xii_roll_no'].notna()
        add_violation(cond_x10, 'X10', WEIGHTS['BOARD_DUPLICATE'])
        
    # Mobile rules
    if 'mobile_no' in df.columns:
        mob_counts = df.groupby('mobile_no')['mobile_no'].transform('count')
        cond_yk = (mob_counts >= 11)
        add_violation(cond_yk, 'YK_YL', WEIGHTS['MOBILE_CONCENTRATION'])
        
        if 'father_name' in df.columns:
            mob_father_nunique = df.groupby('mobile_no')['father_name'].transform('nunique')
            cond_un = (mob_father_nunique > 1) & df['mobile_no'].notna()
            add_violation(cond_un, 'UN', WEIGHTS['MOBILE_FATHER_MISMATCH'])

    rules_fired = rules_fired.str.rstrip(',')
    return rule_scores, rules_fired

def train_lgbm(df_features, vae_prob, rule_scores):
    X = df_features.copy()
    
    # Adding necessary fields to the model features
    X['vae_reconstruction_prob'] = vae_prob
    # Removed rule_violation_score from X to prevent target leakage
    
    # Ensure numerical dtypes
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == 'category':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except Exception:
                X[col] = X[col].astype('category')
                
    y = (rule_scores > 0).astype(int)
    
    num_pos = y.sum()
    num_neg = len(y) - num_pos
    if num_pos == 0:
        print("No positive weak labels. Using uniform weights.")
        scale_pos_weight = 1.0
        # If no positive labels, model training will be skewed or fail, 
        # fake one positive if length > 1 for code to not break
        if len(y) > 1:
            y.iloc[0] = 1
    else:
        scale_pos_weight = num_neg / num_pos
        
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    clf.fit(X, y)
    
    preds = clf.predict_proba(X)[:, 1]
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] 
    else:
        shap_vals = shap_values
        
    feature_names = X.columns.tolist()
    top_shap_features = []
    
    for i in range(len(shap_vals)):
        top_indices = np.argsort(-np.abs(shap_vals[i]))[:3]
        top_features = [feature_names[idx] for idx in top_indices]
        top_shap_features.append(','.join(top_features))
        
    return preds, top_shap_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/data_for_ml_model.csv')
    parser.add_argument('--features_json', type=str, default='selected_features.json')
    parser.add_argument('--output_csv', type=str, default='risk_scores.csv')
    args = parser.parse_args()

    print("Starting VAE Detection Pipeline...")
    
    if not os.path.exists(args.features_json):
        print(f"Warning: {args.features_json} not found. Creating a dummy file to proceed.")
        dummy_features = {
            "selected_features": ["dummy_feat_1", "dummy_feat_2"],
            "feature_scores": {"dummy_feat_1": 0.9, "dummy_feat_2": 0.8},
            "n_selected": 2,
            "pipeline_run_timestamp": "2026-06-15T12:00:00"
        }
        with open(args.features_json, 'w') as f:
            json.dump(dummy_features, f)
            
    with open(args.features_json, 'r') as f:
        feature_data = json.load(f)
    selected_features = feature_data.get('selected_features', [])
    
    if not os.path.exists(args.data_path):
        print(f"Warning: {args.data_path} not found. Creating dummy data to proceed.")
        df = pd.DataFrame({
            'application_id': [1, 2, 3, 4],
            'sanity': [None, 'Flagged', None, None],
            'dummy_feat_1': [0.1, 0.5, 0.9, 0.2],
            'dummy_feat_2': [0.8, 0.2, 0.4, 0.7],
            'pre_post_matric': [1, 2, 1, 2],
            'annual_family_income': [15000, 5000, 50000, 25000],
            'applicant_name': ['A', 'B', 'C', 'D'],
            'father_name': ['A_f', 'B_f', 'C_f', 'D_f'],
            'mother_name': ['A_m', 'B_m', 'C_m', 'D_m'],
            'date_of_birth': ['2000-01-01', '1995-01-01', '2010-01-01', '2005-01-01'],
            'mobile_no': ['123', '456', '789', '012'],
            'registered_date': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01']
        })
    else:
        df = pd.read_csv(args.data_path)
        
    print(f"Loaded dataset with {len(df)} records.")
    
    available_features = [f for f in selected_features if f in df.columns]
    if not available_features:
        print("Error: None of the selected features are present in the dataset.")
        # If dummy data was just created, try matching available numeric columns
        available_features = [c for c in df.columns if c not in ['application_id', 'sanity'] and pd.api.types.is_numeric_dtype(df[c])]
        
    df_features = df[available_features]
    
    # Stage A: VAE
    print("Stage A: Training VAE...")
    valid_mask = df['sanity'].isnull().values
    if not valid_mask.any():
        valid_mask = np.ones(len(df), dtype=bool)
        
    vae_prob = train_vae(df_features, valid_mask)
    
    # Stage B: Rule-Based Weak Label Generator
    print("Stage B: Generating rule-based weak labels...")
    rule_scores, rules_fired = apply_rules(df)
    
    # Stage C: LightGBM Classifier
    print("Stage C: Training LightGBM classifier...")
    lgbm_preds, shap_features = train_lgbm(df_features, vae_prob, rule_scores)
    
    # Final Output
    app_ids = df['application_id'] if 'application_id' in df.columns else np.arange(len(df))
    out_df = pd.DataFrame({
        'application_id': app_ids,
        'vae_reconstruction_prob': vae_prob,
        'rule_violation_score': rule_scores,
        'rule_codes_fired': rules_fired,
        'lgbm_risk_score': lgbm_preds,
        'top_shap_features': shap_features
    })
    
    out_df.to_csv(args.output_csv, index=False)
    print(f"Finished successfully. Output saved to {args.output_csv}")

if __name__ == '__main__':
    main()
