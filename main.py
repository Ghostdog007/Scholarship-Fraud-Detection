"""
NIC Fraud Detection ML Pipeline Orchestrator.

This module orchestrates the execution of the fraud detection pipeline in three stages:
1. Feature Selection (`feature_selection.py`)
2. Model Training & Scoring (`vae_detection.py`)
3. Evaluation & Pruning (`evaluate_model.py`)

The pipeline runs in this specific order because feature selection must precede VAE 
so `selected_features.json` exists as the explicit interface between the two files 
(per AGENTS.md Section 4.1).
"""

import argparse
import subprocess
import sys
import os

def run_script(script_name, args_list):
    """Run a Python script as a subprocess and exit if it fails."""
    print(f"\n{'='*50}")
    print(f"Executing {script_name}...")
    print(f"{'='*50}")
    
    cmd = [sys.executable, script_name] + args_list
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="NIC Fraud Detection ML Pipeline Orchestrator")
    parser.add_argument('--data_path', type=str, default='datasets/data_for_ml_model.csv',
                        help='Path to the input dataset CSV')
    parser.add_argument('--features_json', type=str, default='selected_features.json',
                        help='Path to the intermediate selected features JSON')
    parser.add_argument('--scores_csv', type=str, default='risk_scores.csv',
                        help='Path to the output risk scores CSV')
    
    args = parser.parse_args()

    # 1. Feature Selection
    # Must run first to generate selected_features.json, defining the schema for the ML model.
    feature_args = [
        '--data_path', args.data_path,
        '--output_json', args.features_json
    ]
    run_script('feature_selection.py', feature_args)

    # 2. VAE Detection & LightGBM
    # Runs second, reading selected_features.json as the single source of truth for features.
    vae_args = [
        '--data_path', args.data_path,
        '--features_json', args.features_json,
        '--output_csv', args.scores_csv
    ]
    run_script('vae_detection.py', vae_args)

    # 3. Model Evaluation
    # Runs last to evaluate predictions and apply SHAP-based feature pruning on the final model.
    eval_args = [
        '--data_path', args.data_path,
        '--scores_csv', args.scores_csv
    ]
    run_script('evaluate_model.py', eval_args)

    print("\n[Pipeline Complete] All stages finished successfully.")

if __name__ == '__main__':
    main()
