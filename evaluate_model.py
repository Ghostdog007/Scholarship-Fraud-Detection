import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, brier_score_loss, precision_recall_curve
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/data_for_ml_model.csv')
    parser.add_argument('--scores_csv', type=str, default='risk_scores.csv')
    args = parser.parse_args()

    if not os.path.exists(args.data_path) or not os.path.exists(args.scores_csv):
        print(f"Error: Make sure both {args.data_path} and {args.scores_csv} exist.")
        return

    print("Loading data...")
    df_raw = pd.read_csv(args.data_path)
    df_scores = pd.read_csv(args.scores_csv)

    # Merge on application_id if it exists, otherwise assume parallel indexing
    if 'application_id' in df_raw.columns and 'application_id' in df_scores.columns:
        df = pd.merge(df_scores, df_raw[['application_id', 'sanity']], on='application_id', how='left')
    else:
        df = df_scores.copy()
        df['sanity'] = df_raw['sanity']

    print(f"\n--- EVALUATION RESULTS ---")
    
    # 1. Evaluate against Weak Labels
    # Treat rule_violation_score > 0 as the "positive" weak label
    if 'rule_violation_score' in df.columns and 'lgbm_risk_score' in df.columns:
        y_true_weak = (df['rule_violation_score'] > 0).astype(int)
        y_prob = df['lgbm_risk_score']
        
        # We only evaluate if we have at least one positive weak label
        if y_true_weak.sum() > 0:
            pr_auc = average_precision_score(y_true_weak, y_prob)
            brier = brier_score_loss(y_true_weak, y_prob)
            
            # Find optimal threshold for F1
            precisions, recalls, thresholds = precision_recall_curve(y_true_weak, y_prob)
            # thresholds is len(precisions)-1
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1 = f1_scores[best_idx]
            
            y_pred_best = (y_prob >= best_thresh).astype(int)
            mcc = matthews_corrcoef(y_true_weak, y_pred_best)

            print(f"\n[1] Performance vs Rule-Based Weak Labels")
            print(f"    Positive samples (Rules broken): {y_true_weak.sum()} out of {len(y_true_weak)}")
            print(f"    PR-AUC:           {pr_auc:.4f}")
            print(f"    Brier Score:      {brier:.4f}")
            print(f"    Optimal Thresh:   {best_thresh:.4f}")
            print(f"    Best F1-Score:    {best_f1:.4f}")
            print(f"    MCC:              {mcc:.4f}")
        else:
            print("\n[1] Weak Label Evaluation: No applications broke any rules. Cannot compute PR-AUC.")

    # 2. Check the Known Fraud Cases
    print(f"\n[2] Ground-Truth Analysis (The known fraud records)")
    known_fraud_df = df[df['sanity'].notnull()].copy()
    
    if known_fraud_df.empty:
        print("    No confirmed fraud cases found in the dataset (sanity is entirely null).")
    else:
        print(f"    Found {len(known_fraud_df)} known fraud cases.")
        
        # Calculate risk percentiles for the whole dataset
        df['risk_percentile'] = df['lgbm_risk_score'].rank(pct=True) * 100
        
        for idx, row in known_fraud_df.iterrows():
            app_id = row.get('application_id', 'Unknown')
            sanity_val = row['sanity']
            lgbm_score = row.get('lgbm_risk_score', np.nan)
            percentile = row.get('risk_percentile', np.nan)
            vae_prob = row.get('vae_reconstruction_prob', np.nan)
            rules_fired = row.get('rule_codes_fired', 'None')
            shap_feats = row.get('top_shap_features', 'None')

            print(f"\n    -> Application ID: {app_id}")
            print(f"       Sanity Flag:    {sanity_val}")
            print(f"       Overall Risk:   {lgbm_score:.4f} (Top {100 - percentile:.2f}% highest risk)")
            print(f"       VAE Normality:  {vae_prob:.4f} (Lower = more anomalous)")
            print(f"       Rules Fired:    {rules_fired if pd.notnull(rules_fired) and rules_fired != '' else 'None'}")
            print(f"       Top Explanations (SHAP): {shap_feats}")

    # 3. SHAP-Based Second-Pass Feature Pruning
    shap_summary_file = 'shap_summary.json'
    features_json_file = 'selected_features.json'
    if os.path.exists(shap_summary_file) and os.path.exists(features_json_file):
        print(f"\n[3] SHAP-Based Second-Pass Feature Pruning")
        with open(shap_summary_file, 'r') as f:
            shap_summary = json.load(f)
        with open(features_json_file, 'r') as f:
            feature_data = json.load(f)
            
        original_features = feature_data.get('selected_features', [])
        
        pruned_features = []
        dropped_features = []
        for feat in original_features:
            shap_val = shap_summary.get(feat, 0.0)
            if shap_val >= 0.001:
                pruned_features.append(feat)
            else:
                dropped_features.append(feat)
                
        print(f"    Original feature count: {len(original_features)}")
        print(f"    Features dropped (mean |SHAP| < 0.001): {len(dropped_features)}")
        if dropped_features:
            for df_feat in dropped_features:
                print(f"      - {df_feat} (SHAP: {shap_summary.get(df_feat, 0.0):.6f})")
        print(f"    Pruned feature count: {len(pruned_features)}")
        
        pruned_data = {
            "selected_features": pruned_features,
            "dropped_features": dropped_features,
            "mean_shap_per_feature": shap_summary,
            "shap_prune_threshold": 0.001,
            "source_file": "selected_features.json",
            "pipeline_run_timestamp": feature_data.get("pipeline_run_timestamp", "")
        }
        with open('selected_features_shap_pruned.json', 'w') as f:
            json.dump(pruned_data, f, indent=2)
        print("    Saved pruned feature list to selected_features_shap_pruned.json")

    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
