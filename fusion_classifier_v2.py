import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import shap

def main():
    # 1. Load schema
    with open('v2_feature_schema.json', 'r') as f:
        schema = json.load(f)
    
    tabular_features = schema.get('features', []) + schema.get('aggregation_features', [])
    
    # 2. Load engineered features
    df_eng = pd.read_csv('engineered_features_v2.csv')
    valid_features = [f for f in tabular_features if f in df_eng.columns]
    df_tabular = df_eng[['application_id'] + valid_features].copy()
    
    numeric_cols = df_tabular.select_dtypes(include=[np.number, bool]).columns.tolist()
    if 'application_id' not in numeric_cols:
        numeric_cols.insert(0, 'application_id')
    df_tabular = df_tabular[numeric_cols].copy()
            
    # 3. Load VAE scores
    df_vae = pd.read_csv('vae_v2_scores.csv')
    recon_df = pd.json_normalize(df_vae['recon_error_vector'].apply(json.loads))
    recon_df.columns = [f"recon_{c}" for c in recon_df.columns]
    df_vae = pd.concat([df_vae[['application_id', 'vae_anomaly_score']], recon_df], axis=1)
    
    # 4. Load Graph scores
    df_graph = pd.read_csv('graph_v2_scores.csv')
    # Should contain application_id, graph_anomaly_score, attr_recon_error, struct_recon_error
    
    # 5. Load Pseudo Labels
    with open('pseudo_labels_v2.json', 'r') as f:
        pseudo_labels = json.load(f)
    
    positives = pseudo_labels['positive_set']
    negatives = pseudo_labels['negative_set']
    
    label_data = []
    for p in positives:
        app_id = p['application_id']
        round_val = p['round']
        src = "evt_cold_start" if round_val == 0 else f"self_training_round_{round_val}"
        label_data.append({'application_id': app_id, 'label': 1, 'label_source': src})
    for n in negatives:
        app_id = n
        label_data.append({'application_id': app_id, 'label': 0, 'label_source': 'negative'})
        
    df_labels = pd.DataFrame(label_data)
    
    # 6. Merge
    df_merged = df_tabular.merge(df_vae, on='application_id', how='inner')
    df_merged = df_merged.merge(df_graph, on='application_id', how='inner')
    df_merged = df_merged.merge(df_labels, on='application_id', how='inner')
    
    # 7. Prepare X and y
    exclude_cols = ['application_id', 'label', 'label_source']
    features = [c for c in df_merged.columns if c not in exclude_cols]
    
    X = df_merged[features]
    y = df_merged['label']
    
    # 8. Train LightGBM
    clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(X, y)
    
    # 9. Predict
    df_merged['lgbm_risk_score_v2'] = clf.predict_proba(X)[:, 1]
    
    # 10. SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    
    # Handling different shap versions/outputs
    if isinstance(shap_values, list):
        shap_vals_1 = shap_values[1]
    elif len(shap_values.shape) == 3: # (samples, features, classes)
        shap_vals_1 = shap_values[:, :, 1]
    else:
        shap_vals_1 = shap_values
        
    top_shap_list = []
    for i in range(len(df_merged)):
        row_shap = shap_vals_1[i]
        top_indices = np.argsort(-np.abs(row_shap))[:5]
        top_feats = [features[idx] for idx in top_indices]
        top_shap_list.append(json.dumps(top_feats))
        
    df_merged['top_shap_features'] = top_shap_list
    
    # 11. Prepare Output
    output_cols = ['application_id', 'vae_anomaly_score', 'graph_anomaly_score', 'lgbm_risk_score_v2', 'label_source', 'top_shap_features']
    df_output = df_merged[output_cols]
    df_output.to_csv('risk_scores_v2.csv', index=False)
    print(f"Saved risk_scores_v2.csv with {len(df_output)} rows.")

if __name__ == "__main__":
    main()
