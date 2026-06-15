# NIC Scholarship Fraud Detection System

An advanced, hybrid Machine Learning pipeline designed to detect anomalous and potentially fraudulent scholarship applications submitted to the National Informatics Centre (NIC) portal.

## 🎯 The Challenge
Standard machine learning models fail on this dataset because the confirmed fraud cases are incredibly rare (**0.027% of the dataset**). Traditional binary classifiers (like Random Forests or standard XGBoost) will simply learn to predict that *every* application is valid, achieving 99.97% accuracy while completely failing to catch fraud.

## 🧠 Our Approach: The 3-Stage Hybrid Architecture
To overcome the extreme class imbalance, this project utilizes a research-backed, 3-stage modular architecture that generates its own weak labels and focuses on underlying data anomalies:

1. **Classical Feature Engineering & Selection (`feature_selection.py`)**
   - Engineers critical domain features (e.g., `fee_income_ratio`, `age_at_registration`, IP/Mobile concentrations).
   - Uses **Classwise Mutual Information** with minority upweighting, followed by **Pearson Correlation Pruning** and **mRMR** to strictly isolate the 20 most predictive features without introducing noise.

2. **Unsupervised Anomaly Detection (`vae_detection.py` - Stage A)**
   - A **Variational Autoencoder (VAE)** built in PyTorch. 
   - It trains *exclusively* on applications known to be valid, learning the strict statistical distribution of a normal application. It then grades every application with a `vae_reconstruction_prob`. Lower scores mean the application structurally deviates from the norm.

3. **Weak Supervision & Explainable Classification (`vae_detection.py` - Stages B & C)**
   - **Rule-Based Weak Labels:** Natively evaluates 10 groups of existing NIC static rules to flag known bad behaviors, assigning a `rule_violation_score`.
   - **LightGBM Classifier:** Treats the `rule_violation_score` as a weak positive target and trains a gradient-boosted tree using the VAE probabilities and selected features.
   - **Explainability (SHAP):** Output isn't a black box. The model provides the top 3 specific reasons (SHAP features) why an application was flagged as high-risk.

## 📂 Project Structure
```text
├── datasets/
│   ├── data_for_ml_model.csv        # The raw input dataset
│   └── Revalidation.xlsx            # The static NIC rules definition
├── docs/
│   ├── AGENTS.md                    # Core architectural constraints and research backing
│   ├── SUPERVISOR_TESTING_GUIDE.md  # Step-by-step instructions for testing the pipeline
│   └── analysis_report.md           # Deep-dive into dataset features and edge cases
├── dataset_feature_analysis.ipynb   # Initial exploratory data analysis
├── feature_selection.py             # Pipeline Stage 1
├── vae_detection.py                 # Pipeline Stages 2 & 3
├── evaluate_model.py                # Scoring and validation script
└── main.py                          # The primary orchestrator script
```

## 🚀 Quick Start
To run the full end-to-end pipeline on your local machine:
```bash
python main.py
```
This will sequentially extract the features, train the VAE, train the LightGBM classifier, and output the final model metrics. For more detailed testing scenarios, refer to the `docs/SUPERVISOR_TESTING_GUIDE.md`.
