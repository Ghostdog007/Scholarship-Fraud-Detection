import argparse
import subprocess
import sys
import os

def run_script(script_name, args_list):
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
    feature_args = [
        '--data_path', args.data_path,
        '--output_json', args.features_json
    ]
    run_script('feature_selection.py', feature_args)

    # 2. VAE Detection & LightGBM
    vae_args = [
        '--data_path', args.data_path,
        '--features_json', args.features_json,
        '--output_csv', args.scores_csv
    ]
    run_script('vae_detection.py', vae_args)

    # 3. Model Evaluation
    eval_args = [
        '--data_path', args.data_path,
        '--scores_csv', args.scores_csv
    ]
    run_script('evaluate_model.py', eval_args)

    print("\n[Pipeline Complete] All stages finished successfully.")

if __name__ == '__main__':
    main()
