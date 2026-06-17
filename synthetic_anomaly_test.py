import pandas as pd
import numpy as np
import subprocess
import sys
import os
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

ORIGINAL_DATA = 'datasets/data_for_ml_model.csv'
SYNTHETIC_DATA = 'datasets/synthetic_data.csv'
SYNTHETIC_FEATURES = 'synthetic_features.json'
SYNTHETIC_SCORES = 'synthetic_scores.csv'
SYNTHETIC_LABELS = 'datasets/synthetic_fraud_labels.csv'

def inject_anomalies(df):
    """
    Injects realistic fraud patterns into perfectly clean records.
    Returns the corrupted dataframe and a list of injected application IDs.
    """
    df_clean = df.copy()

    # Determine which rows to corrupt based on frozen labels, if they exist
    if os.path.exists(SYNTHETIC_LABELS):
        print(f"Loading frozen synthetic labels from {SYNTHETIC_LABELS}...")
        labels_df = pd.read_csv(SYNTHETIC_LABELS)
        
        # Merge to get the indices in the current df
        merged = df_clean[['application_id']].reset_index().merge(labels_df, on='application_id')
        
        splits = {}
        for category in labels_df['fraud_category'].unique():
            splits[category] = merged[merged['fraud_category'] == category]['index'].values
            
        corrupt_indices = merged['index'].values
        injected_app_ids = labels_df['application_id'].tolist()
    else:
        # We will inject fraud into 5% of the clean records
        n_fraud = int(len(df_clean) * 0.05)
        
        # Randomly select records to corrupt
        np.random.seed(42)
        corrupt_indices = np.random.choice(df_clean.index, size=n_fraud, replace=False)
        
        # Split into 5 types of perturbations
        split_arrays = np.array_split(corrupt_indices, 5)
        
        splits = {
            'AGE_VIOLATION': split_arrays[0],
            'INCOME_VIOLATION': split_arrays[1],
            'IP_CONCENTRATION': split_arrays[2],
            'MOTHER_NAME_COLLISION': split_arrays[3],
            'FEE_INFLATION': split_arrays[4]
        }
        
        # Save the mapping to disk
        labels_list = []
        for category, indices in splits.items():
            for idx in indices:
                labels_list.append({
                    'application_id': df_clean.loc[idx, 'application_id'],
                    'fraud_category': category
                })
                
        labels_df = pd.DataFrame(labels_list)
        labels_df.to_csv(SYNTHETIC_LABELS, index=False)
        print(f"Generated and froze synthetic labels to {SYNTHETIC_LABELS}...")
        injected_app_ids = labels_df['application_id'].tolist()

    # 1. Age Violation (Rule X7/X1 - Make them 40 years old)
    if 'AGE_VIOLATION' in splits and len(splits['AGE_VIOLATION']) > 0:
        df_clean.loc[splits['AGE_VIOLATION'], 'date_of_birth'] = '1980-01-01'
    
    # 2. Income Violation (Rule UW / X13 - Income < 10000)
    if 'INCOME_VIOLATION' in splits and len(splits['INCOME_VIOLATION']) > 0:
        df_clean.loc[splits['INCOME_VIOLATION'], 'annual_family_income'] = 5000
        
    # 3. IP Concentration (Bridge IP_CONC_ENG - same IP, distinct mobile)
    if 'IP_CONCENTRATION' in splits and len(splits['IP_CONCENTRATION']) > 0:
        df_clean.loc[splits['IP_CONCENTRATION'], 'ip_address'] = '192.168.99.99'
        df_clean.loc[splits['IP_CONCENTRATION'], 'mobile_no'] = np.arange(len(splits['IP_CONCENTRATION'])) + 8000000000
        
    # 4. Mother Name Collision (Bridge FM_ENG - father_name == mother_name, distinct applicant_name)
    if 'MOTHER_NAME_COLLISION' in splits and len(splits['MOTHER_NAME_COLLISION']) > 0:
        df_clean.loc[splits['MOTHER_NAME_COLLISION'], 'father_name'] = df_clean.loc[splits['MOTHER_NAME_COLLISION'], 'mother_name']
        df_clean.loc[splits['MOTHER_NAME_COLLISION'], 'applicant_name'] = df_clean.loc[splits['MOTHER_NAME_COLLISION'], 'applicant_name'].astype(str) + '_distinct'
        
    # 5. Fee Inflation (Bridge FEE_ENG - fee > income, income > 20000)
    if 'FEE_INFLATION' in splits and len(splits['FEE_INFLATION']) > 0:
        df_clean.loc[splits['FEE_INFLATION'], 'annual_family_income'] = 25000
        df_clean.loc[splits['FEE_INFLATION'], 'admission_fee'] = 30000
        df_clean.loc[splits['FEE_INFLATION'], 'tuition_fee'] = 0

    return df_clean, set(injected_app_ids)

def run_pipeline():
    print("\n[1] Running Feature Selection on Synthetic Data...")
    subprocess.run([
        sys.executable, 'feature_selection.py', 
        '--data_path', SYNTHETIC_DATA,
        '--output_json', SYNTHETIC_FEATURES
    ], check=True)
    
    print("\n[2] Running VAE Anomaly Detection on Synthetic Data...")
    subprocess.run([
        sys.executable, 'vae_detection.py', 
        '--data_path', SYNTHETIC_DATA,
        '--features_json', SYNTHETIC_FEATURES,
        '--output_csv', SYNTHETIC_SCORES
    ], check=True)

def evaluate_results(injected_ids):
    print("\n[3] Evaluating Synthetic Anomaly Detection...")
    df_scores = pd.read_csv(SYNTHETIC_SCORES)
    
    # Create the true label column based on our secret injected IDs
    df_scores['true_injected_fraud'] = df_scores['application_id'].isin(injected_ids).astype(int)
    
    y_true = df_scores['true_injected_fraud']
    y_prob = df_scores['lgbm_risk_score']
    
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Calculate best threshold for F1
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred = (y_prob >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print("\n==================================================")
    print("      SYNTHETIC ANOMALY EVALUATION REPORT         ")
    print("==================================================")
    print(f"Total Clean Records Tested:    {len(y_true) - len(injected_ids)}")
    print(f"Total Frauds Secretly Injected: {len(injected_ids)}")
    print(f"PR-AUC Score:                   {pr_auc:.4f} (1.0 is perfect)")
    print(f"Optimal Risk Threshold:         {best_thresh:.4f}")
    print("\nAt the optimal threshold:")
    print(f"- Correctly Caught Fraud (TP):  {tp} out of {len(injected_ids)}")
    print(f"- Missed Fraud (FN):            {fn}")
    print(f"- False Alarms (FP):            {fp} innocent apps flagged")
    print("==================================================\n")

def main():
    if not os.path.exists(ORIGINAL_DATA):
        print(f"Error: Original data not found at {ORIGINAL_DATA}")
        return
        
    print("Loading original dataset...")
    df_raw = pd.read_csv(ORIGINAL_DATA, low_memory=False)
    
    print("Injecting synthetic fraud anomalies...")
    df_synthetic, injected_ids = inject_anomalies(df_raw)
    
    # Save the synthetic dataset
    df_synthetic.to_csv(SYNTHETIC_DATA, index=False)
    print(f"Saved synthetic test set with {len(injected_ids)} hidden frauds.")
    
    # Run the model pipeline on this corrupted data
    run_pipeline()
    
    # Grade the model's ability to find the secret frauds
    evaluate_results(injected_ids)

if __name__ == '__main__':
    main()
