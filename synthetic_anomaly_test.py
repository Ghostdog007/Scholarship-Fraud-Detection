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

def inject_anomalies(df):
    """
    Injects realistic fraud patterns into perfectly clean records.
    Returns the corrupted dataframe and a list of injected application IDs.
    """
    # Treat the entire dataset as clean for anomaly injection
    df_clean = df.copy()
    
    # We will inject fraud into 5% of the clean records
    n_fraud = int(len(df_clean) * 0.05)
    
    # Randomly select records to corrupt
    np.random.seed(42)
    corrupt_indices = np.random.choice(df_clean.index, size=n_fraud, replace=False)
    
    # Split into 4 types of perturbations
    splits = np.array_split(corrupt_indices, 4)
    
    # 1. Age Violation (Rule X7/X1 - Make them 40 years old)
    # Registration date is around 2020/2021 usually, so setting DOB to 1980 makes them ~40
    df_clean.loc[splits[0], 'date_of_birth'] = '1980-01-01'
    
    # 2. Income Violation (Rule UW / X13 - Income < 10000)
    df_clean.loc[splits[1], 'annual_family_income'] = 5000
    
    # 3. Mobile Concentration (Rules YK/YL - Over 15 applicants using the same mobile)
    # Assign the same fake mobile number to all records in this split
    df_clean.loc[splits[2], 'mobile_no'] = 9999999999
    
    # 4. Identity Match (Rule YF - Applicant name equals Father name)
    df_clean.loc[splits[3], 'applicant_name'] = df_clean.loc[splits[3], 'father_name']
    
    injected_app_ids = df_clean.loc[corrupt_indices, 'application_id'].tolist()
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
