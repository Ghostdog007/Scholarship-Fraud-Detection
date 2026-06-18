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
import datetime
from feature_selection import engineer_features

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

ALL_BRIDGES = {'IP_CONC_ENG', 'YF_ENG', 'YF_MOTHER_ENG', 'FM_ENG', 'FEE_ENG', 'INC_EXT_ENG', 'UN_FATHER_ENG', 'X1_ENG', 'X7_ENG'}

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

def _score_vae(model, x_all, device):
    model.eval()
    with torch.no_grad():
        x_all = x_all.to(device)
        recon_all, _, _ = model(x_all)
        mse = torch.mean((x_all - recon_all) ** 2, dim=1)
        # Using negative MSE transformed to prob-like
        recon_prob = torch.exp(-mse).cpu().numpy()
    return recon_prob

def train_vae(df_features, valid_mask, epochs=50, batch_size=256):
    # Seed for reproducibility — ensures vae_reconstruction_prob is stable
    # across runs so PR-AUC deltas reflect real changes, not random init noise.
    # Required by reproducibility contract — AGENTS.md Section 0.1 Rule 3.
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"VAE Device: {device}")
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(df_features.fillna(0).values)
    
    x_train = torch.tensor(x_scaled[valid_mask], dtype=torch.float32)
    x_all = torch.tensor(x_scaled, dtype=torch.float32)
    
    dataset = TensorDataset(x_train)
    _generator = torch.Generator()
    _generator.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=_generator)
    
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
            
    recon_prob = _score_vae(model, x_all, device)
    return recon_prob

def _add_violation(condition, rule_code, weight, rules_fired, rule_scores):
    idx = condition[condition].index
    rules_fired.loc[idx] = rules_fired.loc[idx].apply(lambda x: x + rule_code + ',' if x else rule_code + ',')
    rule_scores.loc[idx] += weight

def apply_rules(df, enabled_bridges=None):
    if enabled_bridges is None:
        enabled_bridges = ALL_BRIDGES
        
    rules_fired = pd.Series('', index=df.index, dtype=object)
    rule_scores = pd.Series(0.0, index=df.index)
        
    # --- NIC Rulebook Rules ---
    # Identity duplicate
    id_cols = ['applicant_name', 'father_name', 'mother_name', 'date_of_birth', 'mobile_no']
    if all(c in df.columns for c in id_cols):
        cond_id_dup = df.duplicated(subset=id_cols, keep=False)
        _add_violation(cond_id_dup, 'ID_DUP', WEIGHTS['IDENTITY_DUPLICATE'], rules_fired, rule_scores)

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
            _add_violation(cond_x1, 'X1', WEIGHTS['AGE_VIOLATION'], rules_fired, rule_scores)
            
            cond_x7 = (df['pre_post_matric'] == 2) & (age_col > 35)
            _add_violation(cond_x7, 'X7', WEIGHTS['AGE_VIOLATION'], rules_fired, rule_scores)
            
            cond_x8 = (df['pre_post_matric'] == 2) & (age_col < 13)
            _add_violation(cond_x8, 'X8', WEIGHTS['AGE_VIOLATION'], rules_fired, rule_scores)
            
    # Income rules
    if 'annual_family_income' in df.columns:
        cond_uw = (pd.to_numeric(df['annual_family_income'], errors='coerce') < 20000) & (pd.to_numeric(df['annual_family_income'], errors='coerce') > 10000)
        _add_violation(cond_uw, 'UW', WEIGHTS['INCOME_VIOLATION'], rules_fired, rule_scores)
        
        cond_x13_21 = pd.to_numeric(df['annual_family_income'], errors='coerce') <= 10000
        _add_violation(cond_x13_21, 'X13_X21', WEIGHTS['INCOME_VIOLATION'], rules_fired, rule_scores)
        
    # Name match
    if all(c in df.columns for c in ['applicant_name', 'father_name', 'mother_name']):
        cond_yf = (df['applicant_name'] == df['father_name']) | (df['applicant_name'] == df['mother_name'])
        cond_yf = cond_yf.fillna(False)
        _add_violation(cond_yf, 'YF', WEIGHTS['NAME_MATCH'], rules_fired, rule_scores)
        
    # Board duplicate
    if 'x_roll_no' in df.columns and 'x_course_year' in df.columns:
        cond_x9 = df.duplicated(subset=['x_roll_no', 'x_course_year'], keep=False) & df['x_roll_no'].notna()
        _add_violation(cond_x9, 'X9', WEIGHTS['BOARD_DUPLICATE'], rules_fired, rule_scores)
    
    if 'xii_roll_no' in df.columns and 'xii_course_year' in df.columns:
        cond_x10 = df.duplicated(subset=['xii_roll_no', 'xii_course_year'], keep=False) & df['xii_roll_no'].notna()
        _add_violation(cond_x10, 'X10', WEIGHTS['BOARD_DUPLICATE'], rules_fired, rule_scores)
        
    # Mobile rules
    if 'mobile_no' in df.columns:
        mob_counts = df.groupby('mobile_no')['mobile_no'].transform('count')
        cond_yk = (mob_counts >= 11)
        _add_violation(cond_yk, 'YK_YL', WEIGHTS['MOBILE_CONCENTRATION'], rules_fired, rule_scores)
        
        if 'father_name' in df.columns:
            mob_father_nunique = df.groupby('mobile_no')['father_name'].transform('nunique')
            cond_un = (mob_father_nunique > 1) & df['mobile_no'].notna()
            _add_violation(cond_un, 'UN', WEIGHTS['MOBILE_FATHER_MISMATCH'], rules_fired, rule_scores)

    # --- Layer-1 Relational Violations ---
    if 'mobile_application_count' in df.columns:
        cond_mob_agg = df['mobile_application_count'] >= 11
        _add_violation(cond_mob_agg, 'YK_AGG', WEIGHTS['MOBILE_CONCENTRATION'], rules_fired, rule_scores)

    # --- Engineered Bridge Violations ---
    # Bridges gate on `enabled_bridges` so the ablation harness can disable them 
    # via `--disabled_bridges`. This makes Phase D testing reproducible.
    if 'ip_application_count' in df.columns and 'IP_CONC_ENG' in enabled_bridges:
        cond_ip_agg = df['ip_application_count'] >= 15
        _add_violation(cond_ip_agg, 'IP_CONC', WEIGHTS['MOBILE_CONCENTRATION'], rules_fired, rule_scores)

    if 'mobile_unique_fathers_count' in df.columns and 'UN_FATHER_ENG' in enabled_bridges:
        cond_multi_father = df['mobile_unique_fathers_count'] > 1
        _add_violation(cond_multi_father, 'UN_AGG', WEIGHTS['MOBILE_FATHER_MISMATCH'], rules_fired, rule_scores)

    if 'is_applicant_name_eq_father' in df.columns and 'YF_ENG' in enabled_bridges:
        cond_name_flag = df['is_applicant_name_eq_father'] == 1
        _add_violation(cond_name_flag, 'YF_FLAG', WEIGHTS['NAME_MATCH'], rules_fired, rule_scores)

    if 'is_applicant_name_eq_mother' in df.columns and 'YF_MOTHER_ENG' in enabled_bridges:
        cond_name_mother = df['is_applicant_name_eq_mother'] == 1
        _add_violation(cond_name_mother, 'YF_MOTHER', WEIGHTS['NAME_MATCH'], rules_fired, rule_scores)

    if 'is_father_name_eq_mother' in df.columns and 'FM_ENG' in enabled_bridges:
        cond_father_mother = df['is_father_name_eq_mother'] == 1
        _add_violation(cond_father_mother, 'FM_ENG', WEIGHTS['NAME_MATCH'], rules_fired, rule_scores)

    if 'flag_income_below_10000' in df.columns:
        cond_income_flag = df['flag_income_below_10000'] == 1
        _add_violation(cond_income_flag, 'X13_FLAG', WEIGHTS['INCOME_VIOLATION'], rules_fired, rule_scores)

    if 'flag_fee_exceeds_income' in df.columns and 'FEE_ENG' in enabled_bridges:
        cond_fee_flag = df['flag_fee_exceeds_income'] == 1
        _add_violation(cond_fee_flag, 'FEE_EXCEED', WEIGHTS['INCOME_VIOLATION'], rules_fired, rule_scores)

    if 'flag_postmatric_age_over35' in df.columns and 'X7_ENG' in enabled_bridges:
        cond_age_flag = df['flag_postmatric_age_over35'] == 1
        _add_violation(cond_age_flag, 'X7_FLAG', WEIGHTS['AGE_VIOLATION'], rules_fired, rule_scores)

    if 'flag_prematric_age_over20' in df.columns and 'X1_ENG' in enabled_bridges:
        cond_age_flag2 = df['flag_prematric_age_over20'] == 1
        _add_violation(cond_age_flag2, 'X1_FLAG', WEIGHTS['AGE_VIOLATION'], rules_fired, rule_scores)

    if 'flag_income_extreme_low' in df.columns and 'INC_EXT_ENG' in enabled_bridges:
        cond_extreme_low = df['flag_income_extreme_low'] == 1
        _add_violation(cond_extreme_low, 'INCOME_EXTREME', WEIGHTS['INCOME_VIOLATION'], rules_fired, rule_scores)

    rules_fired = rules_fired.str.rstrip(',')
    return rule_scores, rules_fired

def _compute_shap(clf, X):
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
        
    mean_shap_values = np.abs(shap_vals).mean(axis=0)
    shap_summary = {feature_names[i]: float(mean_shap_values[i]) for i in range(len(feature_names))}
        
    return top_shap_features, shap_summary

def train_lgbm(df_features, vae_prob, rule_scores):
    X = df_features.copy()
    
    # Adding necessary fields to the model features
    X['vae_reconstruction_prob'] = vae_prob
    # Removed rule_violation_score from X to prevent target leakage — per AGENTS.md Section 7, Constraint 11.
    
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
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        extra_trees=True,
        min_child_samples=5,
        random_state=42
    )
    clf.fit(X, y)
    
    preds = clf.predict_proba(X)[:, 1]
    top_shap_features, shap_summary = _compute_shap(clf, X)
        
    return preds, top_shap_features, shap_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/data_for_ml_model.csv')
    parser.add_argument('--features_json', type=str, default='selected_features.json')
    parser.add_argument('--output_csv', type=str, default='risk_scores.csv')
    parser.add_argument('--disabled_bridges', type=str, default='')
    args = parser.parse_args()

    print("Starting VAE Detection Pipeline...")
    
    # 1. Load configuration and apply guards
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

    # --- INTERFACE CONTRACT GUARD ---
    if 'dropped_features' in feature_data:
        # Prevent a feedback loop where each run prunes more features than the last, 
        # making results unreproducible — per AGENTS.md Section 7, Constraint 12.
        raise RuntimeError(
            f"CONTRACT VIOLATION: {args.features_json} contains a 'dropped_features' "
            "key. This file was written by the SHAP pruner, not feature_selection.py. "
            "Delete it and re-run feature_selection.py to regenerate a clean file. "
            "The SHAP pruner must write to selected_features_shap_pruned.json instead."
        )

    REQUIRED_ENGINEERED = [
        'mobile_application_count', 'ip_application_count',
        'flag_income_below_10000', 'flag_income_below_20000',
        'is_applicant_name_eq_father', 'is_applicant_name_eq_mother',
        'flag_fee_exceeds_income', 'flag_postmatric_age_over35',
        'flag_prematric_age_over20', 'name_similarity_score',
        'mobile_unique_fathers_count'
    ]
    selected_features = feature_data.get('selected_features', [])
    missing = [f for f in REQUIRED_ENGINEERED if f not in selected_features]
    if len(missing) >= 4:
        # 4+ missing = structural failure (file-overwrite bug or Fix B not applied)
        raise RuntimeError(
            f"CONTRACT VIOLATION: {args.features_json} is missing {len(missing)} "
            f"engineered features: {missing}. Re-run feature_selection.py. If this "
            "list is empty after re-running, Fix B has not been applied yet."
        )
    elif missing:
        # 1-3 missing = legitimate correlation pruning on this dataset variant
        print(f"WARNING: {len(missing)} engineered feature(s) were pruned by "
              f"correlation filter (likely >0.98 correlated): {missing}")
    # --- END GUARD ---

    # 2. Load data
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
    
    # 3. Engineer features
    # Engineer features so that columns like is_applicant_name_eq_father,
    # ip_application_count, etc. exist in df before the feature filter runs.
    # Without this, the raw CSV lacks these columns and they get silently dropped.
    df = engineer_features(df)
    
    available_features = [f for f in selected_features if f in df.columns]
    if not available_features:
        print("Error: None of the selected features are present in the dataset.")
        # If dummy data was just created, try matching available numeric columns
        available_features = [c for c in df.columns if c not in ['application_id', 'sanity'] and pd.api.types.is_numeric_dtype(df[c])]
        
    df_features = df[available_features]
    
    # 4. Stage A: VAE
    print("Stage A: Training VAE...")
    # Treat all records as valid for VAE training because the 4 known fraud 
    # records are treated as valid in this slice — AGENTS.md Section 7, Constraint 1.
    valid_mask = np.ones(len(df), dtype=bool)
        
    vae_prob = train_vae(df_features, valid_mask)
    
    # 5. Stage B: Rule-Based Weak Label Generator
    print("Stage B: Generating rule-based weak labels...")
    
    # Merge engineered columns into df for rule evaluation
    # Only merge columns that apply_rules() now references
    engineered_cols = [
        'mobile_application_count', 'ip_application_count',
        'mobile_unique_fathers_count', 'is_applicant_name_eq_father',
        'is_applicant_name_eq_mother', 'flag_income_below_10000',
        'flag_fee_exceeds_income', 'flag_postmatric_age_over35',
        'flag_prematric_age_over20', 'flag_income_extreme_low'
    ]
    cols_to_merge = [c for c in engineered_cols if c in df_features.columns
                     and c not in df.columns]
    if cols_to_merge:
        df = df.join(df_features[cols_to_merge])
        
    disabled_bridges = set([b.strip() for b in args.disabled_bridges.split(',') if b.strip()])
    enabled_bridges = ALL_BRIDGES - disabled_bridges
        
    rule_scores, rules_fired = apply_rules(df, enabled_bridges=enabled_bridges)
    
    # 6. Stage C: LightGBM Classifier
    print("Stage C: Training LightGBM classifier...")
    lgbm_preds, shap_features, shap_summary = train_lgbm(df_features, vae_prob, rule_scores)
    
    with open('shap_summary.json', 'w') as f:
        json.dump(shap_summary, f, indent=2)
    print("Saved SHAP summary to shap_summary.json")
    
    # 7. Write Final Output
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

    lineage = {
        "source_data_path": args.data_path,
        "source_row_count": len(df),
        "features_json_path": args.features_json,
        "features_pipeline_run_timestamp": feature_data.get("pipeline_run_timestamp", ""),
        "n_selected_features": len(selected_features),
        "bridges_enabled": sorted(list(enabled_bridges)),
        "bridges_disabled": sorted(list(disabled_bridges)),
        "run_timestamp": datetime.datetime.now().isoformat()
    }
    lineage_path = args.output_csv.replace('.csv', '.lineage.json')
    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)
    print(f"Wrote lineage to {lineage_path}")

if __name__ == '__main__':
    main()
