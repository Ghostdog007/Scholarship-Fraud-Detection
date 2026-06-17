# NIC Scholarship Fraud Detection: Technical Architecture & ML Report

> [!IMPORTANT]  
> This document is designed as a permanent, first-principles knowledge base for the NIC Scholarship Fraud Detection project. It provides a complete mental model of the system for new engineers, researchers, and LLM agents.

---

## 1. Problem Definition

### The Core Challenge
The National Informatics Centre (NIC) manages scholarship applications. A subset of these applications are fraudulent—submitted by coordinated networks or individuals manipulating data to steal government funds. 

The primary technical challenge is **extreme class imbalance in an effectively unsupervised setting**. Out of a historical dataset of 15,000 applications, only 4 carried fraud flags (`sanity` column) from the live system. However, because their duplicate partner records are absent from this independent slice, they do not trigger duplicate rules locally and are treated as valid/clean. Traditional supervised machine learning (like XGBoost or Random Forests) cannot be trained on these sparse samples. 

### Why It Matters
Scholarship fraud diverts critical financial aid from legitimate students. Manually reviewing 15,000+ applications is impossible, making automated, highly accurate triage a necessity for government auditors.

### Assumptions and Limitations
*   **Assumption:** Fraudsters leave structural anomalies in their data. They either break statistical distributions (e.g., extremely high fees vs. low income) or they exhibit lockstep behavior (e.g., same IP, same mobile number across applications).
*   **Limitation:** The current system does not have access to a Graph Neural Network (GNN) backend, forcing the tabular model to infer relationships via engineered cross-row aggregation features.
*   **Input/Output:** The system ingests raw tabular CSV data and outputs a `risk_scores.csv` containing a 0.0 to 1.0 probability score for every application, along with SHAP explanations detailing *why* it was flagged.

---

## 2. High-Level System Architecture

The architecture relies on a **Weakly Supervised Hybrid Pipeline**. It bridges the gap between unsupervised anomaly detection and supervised learning.

### End-to-End Data Flow
1.  **Feature Engineering (Layer 0–2):** Raw text and numbers are converted into a dense, purely numeric matrix, enriched with relational context and hard policy flags.
2.  **Feature Selection:** High-dimensional data is aggressively pruned using Mutual Information and mRMR to prevent the "curse of dimensionality," while protecting critical domain features via a bypass list.
3.  **Unsupervised VAE:** A PyTorch Variational Autoencoder learns the "normal" manifold of the entire 15,000 applications (treating the 4 flagged records as clean since their duplicate counterparts are not present in this independent slice). It scores every application with a `vae_reconstruction_prob`.
4.  **Weak Label Generation:** A deterministic rule engine checks for hard government policy violations. If a rule is broken, the application gets a positive "weak label" (`rule_violation_score > 0`).
5.  **Supervised Classifier:** A LightGBM model is trained to predict the weak labels, using the VAE's normality score as one of its primary inputs.
6.  **Explainability:** SHAP (SHapley Additive exPlanations) extracts the top 3 reasons for the LightGBM's decision.

> [!TIP]  
> The VAE acts as the "Detective" looking for weird behavior, while LightGBM acts as the "Judge," weighing the VAE's suspicion against hard policy rules to make a final ruling.

---

## 3. Codebase Structure Analysis

| File | Purpose and Responsibilities |
| :--- | :--- |
| `main.py` | The pipeline orchestrator. Runs feature selection, detection, and evaluation sequentially. It defines the exact execution path for production. |
| `feature_selection.py` | Handles all data cleaning, feature engineering (3-Layer architecture), and dimensionality reduction (MI + mRMR). Outputs `selected_features.json`. |
| `vae_detection.py` | The ML core. Contains the PyTorch VAE model, the weak-label rule engine, and the LightGBM classifier. Outputs `risk_scores.csv` and `shap_summary.json`. |
| `evaluate_model.py` | Analyzes performance against the weak labels, inspects the 4 known ground-truth frauds, and performs a second-pass SHAP-based feature pruning. |
| `synthetic_anomaly_test.py` | A critical testing harness. Automatically mutates clean data to inject hidden fraud (relational duplicates, income manipulation) to evaluate if the model can actually catch unseen tactics. |
| `docs/AGENTS.md` | The architectural source of truth. It defines hard constraints, protected logic, and recently applied research improvements (v1.1). |

---

## 4. Machine Learning Pipeline

### 4.1 Feature Engineering (The 3-Layer Architecture)
Because the VAE is row-blind, features are engineered to force relational context into single rows:
*   **Layer 0 (Text to Boolean):** Text is converted to math. Flags like `is_applicant_name_eq_father` and fuzzy string matching (`name_similarity_score`) are computed before any categorical data is dropped.
*   **Layer 1 (Relational Aggregates):** The dataset is pre-aggregated. Counts like `mobile_application_count` give the model network-level context (e.g., "Is this mobile number used by 50 other people?").
*   **Layer 2 (Policy Boundaries):** Domain knowledge is hardcoded (`flag_income_below_10000`) so the VAE doesn't have to guess if a number breaks a government rule.

### 4.2 Feature Pruning and Protected Bypass
To remove noise, features pass through:
1.  **Mutual Information (MI):** Retains the top 50% of features that correlate with the weak labels.
2.  **Pearson Correlation:** Drops highly correlated duplicates to ensure independence.
3.  **mRMR (Minimum Redundancy Maximum Relevance):** Caps the final output at 20 features.
*CRITICAL:* A "Protected Bypass List" ensures that all Layer 0, 1, and 2 engineered features skip the mRMR cap and are guaranteed to reach the VAE.

### 4.3 Training and Evaluation
*   **LightGBM Tuning:** Configured with `extra_trees=True` and `n_estimators=200` to prevent extreme conservatism caused by the 95%+ clean class imbalance.
*   **SHAP Pruning:** Post-training, `evaluate_model.py` checks mean SHAP values. If LightGBM completely ignored an engineered feature (SHAP = 0.0), it is permanently pruned from the next run.

---

## 5. Synthetic Data Generation

### Why It's Required
You cannot tune a fraud model with 4 labeled cases. If you tune hyperparameters against the 4 known frauds, you overfit to those 4 specific people and miss systematic fraud rings entirely.

### How It Works (`synthetic_anomaly_test.py`)
The harness clones clean applications and injects specific fraud vectors:
1.  **Identity Manipulation:** Copies the `father_name` into the `applicant_name`.
2.  **Policy Manipulation:** Crushes `annual_family_income` to ₹5,000.
3.  **Relational Clustering:** Forces a group of applications to share the exact same `mobile_no` and `ip_address`.

### Risks and Realism
*   **Risk:** Synthetic fraud is often "too easy" for the model to catch, creating artificially high PR-AUC scores. 
*   **Realism:** By mutating real records rather than generating data from scratch, the statistical distributions of the non-mutated columns (like geography and age) perfectly match production reality.

---

## 6. Detailed Component Walkthrough: VAE Detection

**Component:** `vae_detection.py`
**Flow:**
1.  **Ingestion:** Reads the cleaned numeric dataset and filters it to the features defined in `selected_features.json`.
2.  **Unsupervised Phase:** The PyTorch VAE compresses the data into a latent bottleneck and attempts to reconstruct it. The MSE (Mean Squared Error) of the reconstruction is flipped into a `vae_reconstruction_prob`. Normal data reconstructs well (Prob ~ 0.99); weird data reconstructs poorly (Prob ~ 0.01).
3.  **Weak Labeling:** The rule engine checks hardcoded rules (e.g., `rule_violation_score += 10`).
4.  **Supervised Phase:** LightGBM trains using the raw features AND the `vae_reconstruction_prob` as inputs, attempting to predict the weak labels.
5.  **Output:** Generates `risk_scores.csv` and uses TreeExplainer to extract top 3 SHAP features per application.

---

## 7. Research and Industry Comparison

This repository is highly aligned with 2024–2025 research in tabular anomaly detection.

*   **Ahead of Industry:** The use of a "Protected Bypass List" combined with mRMR feature selection mirrors the latest *LLM-Lasso* research from Stanford (2025), where domain knowledge protects critical features from statistical noise filters.
*   **Aligned with Standards:** Using VAE reconstruction errors as an input feature for an ensemble tree model (LightGBM) is a documented best practice (e.g., Deep Sparse Autoencoder Ensembles, PoliMi 2024).
*   **Architectural Gap:** The lack of a true Graph Neural Network (GNN). While Layer 1 relational aggregates patch this hole effectively for a tabular model, industry leaders (like Stripe or PayPal) use native GNNs to catch multi-hop money mule networks.

---

## 8. Diagnosis and Improvement Opportunities

### 1. The "Weak Label" Feedback Loop Risk (RESOLVED in v1.2)
*   **Diagnosis:** In early versions, the weak labels only checked for existing, hardcoded NIC rules, meaning LightGBM ignored newly engineered features (like name similarity) since they did not correlate with rule violations.
*   **Resolution:** Implemented in v1.2. The weak label generator in `vae_detection.py` was extended with 8 guarded rule violations matching these engineered feature bridges (IP_CONC_ENG, YF_ENG, etc.), driving their SHAP feature importance above 0.

### 2. Lack of Focal Loss
*   **Diagnosis:** LightGBM uses standard binary cross-entropy with `scale_pos_weight`. This still struggles with extreme imbalance, leading to slightly erratic precision-recall curves.
*   **Improvement:** Switch the backend from LightGBM to XGBoost and utilize native `obj='focal'` loss. *Complexity: Medium.*

### 3. Pipeline Scalability
*   **Diagnosis:** `df.groupby().transform()` runs entirely in pandas memory. If the dataset scales from 15,000 to 15,000,000 applications, the pipeline will OOM (Out of Memory) crash.
*   **Improvement:** Port `feature_selection.py` to PySpark or Polars for lazy evaluation and out-of-core memory execution. *Complexity: High.*

---

## 9. Repository Knowledge Base (For LLM Agents)

**Core Concepts:**
*   **Row-Blindness:** Tabular ML models look at one row at a time. Do not assume the model knows two applications share an IP address unless an aggregation feature explicitly tells it.
*   **Protected Bypass:** Never allow `feature_selection.py` to blindly delete text columns or engineered flags.
*   **Weak Supervision:** The model has no ground truth. It trains against its own hardcoded rule violations.

**File Responsibilities:**
*   `feature_selection.py` = Feature generation and dimensionality reduction.
*   `vae_detection.py` = The ML models (VAE + LGBM) and the rule engine.
*   `evaluate_model.py` = Metrics, PR-AUC, and SHAP pruning.
*   `synthetic_anomaly_test.py` = The only true way to measure model recall on unseen fraud.

**Key Algorithms:**
*   **mRMR:** Ensures the VAE isn't flooded with 50 features that all say the exact same thing (Redundancy reduction).
*   **VAE (Variational Autoencoder):** Maps data to a Gaussian latent space. Anomalies fall outside the normal distribution.
*   **LightGBM:** Fast, tree-based classifier. Highly sensitive to extreme class imbalance; requires `extra_trees` tuning.

> **End of Technical Report**
