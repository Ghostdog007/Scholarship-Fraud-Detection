import subprocess
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

PYTHON_EXE = os.path.join(".venv", "Scripts", "python.exe")

def run_script(script_name):
    logging.info(f"Starting {script_name}...")
    if not os.path.exists(script_name):
        logging.warning(f"Script {script_name} not found in the current directory.")
    
    try:
        # Run with local .venv python
        result = subprocess.run(
            [PYTHON_EXE, script_name], 
            check=True, 
            text=True, 
            capture_output=True
        )
        if script_name == "evaluate_model_v2.py":
            logging.info(f"Evaluation Results:\n{result.stdout}")
        logging.info(f"Successfully completed {script_name}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}. Exit code: {e.returncode}")
        if e.stdout:
            logging.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logging.error(f"Stderr:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error(f"Could not execute python at {PYTHON_EXE}. Is the virtual environment set up?")
        sys.exit(1)

def main():
    logging.info("Starting V2 Pipeline Orchestration...")
    
    # Phase A
    logging.info("--- Phase A: Feature Engineering ---")
    run_script("tabular_feature_engine_v2.py")
    
    # Phase B
    logging.info("--- Phase B: Graph and Synthetic Data Builders ---")
    run_script("graph_builder_v2.py")
    run_script("synthetic_exposure_builder_v2.py")
    
    # Phase C
    logging.info("--- Phase C: VAE and Graph Autoencoder ---")
    run_script("tabular_vae_v2.py")
    run_script("graph_autoencoder_v2.py")
    
    # Phase D
    logging.info("--- Phase D: EVT and Self-Training Loop ---")
    run_script("evt_scorer.py")
    run_script("self_training_loop_v2.py")
    
    # Phase E
    logging.info("--- Phase E: Fusion Classifier and XAI ---")
    run_script("fusion_classifier_v2.py")
    run_script("xai_layer_v2.py")
    # Phase F
    logging.info("--- Phase F: Evaluation Harness ---")
    run_script("evaluate_model_v2.py")
    
    logging.info("V2 Pipeline Orchestration Completed Successfully!")

if __name__ == "__main__":
    main()
