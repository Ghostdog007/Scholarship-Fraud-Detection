# Supervisor Testing & Evaluation Guide

This document is designed for project leads, supervisors, or QA testers to easily execute, test, and validate the NIC Fraud Detection ML components without needing to alter the underlying code.

## 1. Environment Setup
Ensure you are running the project from within the established virtual environment where the dependencies (`torch`, `lightgbm`, `scikit-learn`, `shap`, `pandas`) are installed.

If you are using Windows PowerShell:
```powershell
.\.venv\Scripts\activate
```

---

## 2. Running the Full End-to-End Pipeline
The easiest way to test the system is to run the orchestrator script. This script automatically hands off data between the components.

```bash
python main.py
```

**What to expect:**
- It will read from `datasets/data_for_ml_model.csv`.
- It will create `selected_features.json` in your root directory.
- It will create `risk_scores.csv` containing the final grades for every applicant.
- It will print a PR-AUC evaluation report to your console.

---

## 3. Testing Components Individually

If you wish to test or debug a specific part of the pipeline, you can run the files individually. 

### Component 1: Feature Selection
**Goal:** Test if the system can isolate the best 20 features from a given dataset.
```bash
python feature_selection.py --data_path datasets/data_for_ml_model.csv --output_json my_test_features.json
```
*Validation Check:* Open `my_test_features.json` and ensure it contains exactly 20 features and their respective Mutual Information scores.

### Component 2: VAE & LightGBM Training
**Goal:** Test the anomaly detection and risk scoring logic using a custom feature list.
```bash
python vae_detection.py --data_path datasets/data_for_ml_model.csv --features_json my_test_features.json --output_csv my_test_risk_scores.csv
```
*Validation Check:* Open `my_test_risk_scores.csv`. Look at the `lgbm_risk_score` (between 0 and 1) and ensure the `top_shap_features` column is populated with comma-separated feature names.

### Component 3: Model Evaluation
**Goal:** Test the grading metrics (PR-AUC, F1) on generated scores.
```bash
python evaluate_model.py --data_path datasets/data_for_ml_model.csv --scores_csv my_test_risk_scores.csv
```

---

## 4. Testing with New "Unseen" Data
If you receive a new batch of scholarship applications (e.g., `new_applications.csv`), you can test the pipeline's robustness by overriding the default arguments:

```bash
python main.py --data_path datasets/new_applications.csv --features_json new_features.json --scores_csv new_risk_scores.csv
```

## 5. Understanding the Evaluation Metrics
When `evaluate_model.py` finishes, you will see an output report. Here is how to interpret it:

- **PR-AUC (Precision-Recall Area Under Curve):** The most important metric. Unlike accuracy, this metric tells you how good the model is at catching fraud without flagging too many innocent students. A score above `0.70` on this extremely imbalanced data is considered highly successful.
- **Ground Truth Analysis:** The script automatically isolates applications that are *known* to be fraudulent based on external NIC sources. If the pipeline is working correctly, the `Overall Risk` score for these specific Application IDs should be in the high percentiles (e.g., > 0.50), and the SHAP features should clearly explain why they were caught.
