# Recommended Machine Learning Modeling Approaches

This document details two distinct, research-backed machine learning modeling approaches proposed for the NIC Fraud Detection project, leveraging the feature analysis and domain rules mapping.

---

## Approach 1: Adaptation of the IoT Quantum/Classical Pipeline

This approach evaluates the applicability of the components from an IoT Quantum/Classical pipeline to the NIC Fraud Detection problem, focusing on feature selection under extreme class imbalance.

### Component Evaluation & Suitability

#### 1. Classwise Mutual Information (MI) Filtering — **Highly Recommended**
* **Mechanism**: Ranks features by class-weighted mutual information, upweighting the minority class using prevalence weights ($w_c$).
* **Fit for NIC**: Since the fraud class is extremely rare (0.027%), standard MI would ignore the fraud signal. Classwise weighted MI ensures features correlated with fraud are retained despite the severe imbalance.
* **Adaptation**: Apply classwise weighted MI directly during initial feature filtering.

#### 2. Correlation-based Redundancy Filtering — **Highly Recommended**
* **Mechanism**: Removes one of any pair of features with a Pearson correlation coefficient $\ge 0.90$, retaining the feature with the higher MI score.
* **Fit for NIC**: Directly eliminates the massive spatial redundancies (e.g. identical state IDs, district IDs) identified during our data analysis.
* **Adaptation**: Drop directly into the preprocessing pipeline.

#### 3. max-Relevance Min-Redundancy (mRMR) Selection — **Highly Recommended**
* **Mechanism**: Iteratively selects features that are maximally relevant to the target while being minimally redundant with already-selected features.
* **Fit for NIC**: Ensures that the final feature subset consists of complementary variables rather than redundant clusters (such as multiple fee or location fields).
* **Adaptation**: Use classical mRMR, which is computationally tractable and well-validated for this scale.

#### 4. Genetic Algorithm Optimization (GAOA) — **Recommended with Modification**
* **Mechanism**: Uses a population-based binary mask search to select optimal feature combinations based on a custom fitness function.
* **Modification for NIC**: The standard fitness function uses the F1 score on a validation split. For extreme class imbalance (0.027% fraud), standard F1 is misleading. The fitness metric must be replaced with **PR-AUC (Precision-Recall AUC)** or **macro F1**.

#### 5. Supervised Contrastive Learning (SCLNet) — **Conditionally Recommended**
* **Mechanism**: Trains a neural network to pull same-class samples together and push different classes apart in embedding space.
* **Fit for NIC**: While contrastive learning is effective for imbalanced tabular data, the dataset contains only **4 real fraud cases**, making it impossible to train on real labels. It is only viable if paired with **synthetic fraud labels** generated from the static revalidation rules.

#### 6. QUBO + QAOA Feature Selection Solver — **Viable Classically, Quantum Unnecessary**
* **Mechanism**: Maps feature selection to a Quadratic Unconstrained Binary Optimization (QUBO) problem.
* **Fit for NIC**: The classical QUBO solver (e.g. Adaptive Local Search) is fully sufficient. Running QAOA on quantum simulators or physical hardware is unnecessary for this scale.

---

### Recommended Pipeline Execution Order

```
[Raw Features]
      │
      ▼
[Stage 1: Classwise MI Filtering]
      │
      ▼
[Stage 2: Correlation Filtering (Pearson >= 0.90)]
      │
      ▼
[Stage 3: mRMR Selection]
      │
      ▼
[Stage 4: Synthetic Fraud Label Generation (via Static Rules)]
      │
      ▼
[Stage 5: GAOA Feature Mask Search (Optimizing PR-AUC)]
      │
      ▼
[Stage 6: Classical QUBO Solver Refinement]
      │
      ▼
[Selected Features for Model Training]
```

---

## Approach 2: 3-Stage Hybrid ML Architecture (Unsupervised + Weak Supervision)

This approach focuses on a hybrid architecture designed specifically to handle tabular anomaly detection under extreme label scarcity (99.97% valid cases) by combining unsupervised learning with rule-based weak supervision.

```
       Raw NIC Data (15,000 records, 136 features)
                           │
                           ▼
                 [Feature Engineering]
                  - age_at_registration
                  - fee_income_ratio  
                  - mobile_occurrence_count
                  - ip_occurrence_count
                  - Deduplicate spatial columns
                           │
                           ▼
       ┌───────────────────┴───────────────────┐
       │                                       │
       ▼                                       ▼
  [Stage 1: VAE Anomaly Model]            [Stage 2: Weak Label Generator]
  - Trained ONLY on valid records         - Programmatically run evaluable rules
  - Outputs reconstruction probability    - Outputs rule_violation_score
       │                                       │
       └───────────────────┬───────────────────┘
                           │
                           ▼
               [Stage 3: Feature Augmentation]
               - Concatenate: [original_features | VAE_score | rule_score]
                           │
                           ▼
         [Stage 4: Supervised Boosted Classifier]
         - Train XGBoost/LightGBM with weak labels as targets
         - Output: Anomaly Risk Score per Application
```

### Stage-by-Stage Breakdown

### Stage 1: Unsupervised Representation & Anomaly Scoring (VAE)
* **Goal**: Learn the distribution of legitimate applications and flag statistical deviations.
* **Why it works**: A **Variational Autoencoder (VAE)** is trained *only* on the 14,996 valid records. The bottleneck probabilistic latent space captures the core correlations of valid applications. 
* **Output**: For each application, the VAE outputs a continuous **reconstruction probability** score (from 0 to 1). A low reconstruction probability indicates a high likelihood of being an anomaly.

### Stage 2: Rule-Guided Weak Label Generation
* **Goal**: Incorporate domain-specific fraud indicators (the 99 static revalidation rules).
* **Why it works**: Purely unsupervised models flag statistical outliers, which might not be fraudulent. By running the evaluable static rules (e.g. age limits, income limits, duplicate identities) programmatically, we compute a deterministic **rule_violation_score** (weighted count of rules violated).
* **Output**: Soft pseudo-labels used as supervision targets for the final classifier.

### Stage 3: Supervised Boosted Classifier (XGBoost / LightGBM)
* **Goal**: Combine the unsupervised anomaly score and the deterministic rule violation signals into a robust, generalizable classifier.
* **Why it works**: Feature-level representation of both the raw data and the VAE anomaly metrics allows a gradient-boosted tree model (like XGBoost or LightGBM) to learn complex non-linear boundaries. The tree models natively handle mixed feature types and are highly robust to tabular noise.
* **Output**: A final probability score denoting the fraud risk level of the application.

---

## Comparison of Approaches

| Metric | Approach 1: Adapted IoT Pipeline | Approach 2: 3-Stage Hybrid Architecture |
| :--- | :--- | :--- |
| **Primary Focus** | Rigorous feature selection and dimension reduction | End-to-end classification and anomaly detection |
| **Handling of Imbalance** | Classwise weighted MI and genetic search tuning | VAE training on normal-only data + weak supervision |
| **Modeling Strengths** | Highly optimal feature subset selection | Combines statistical outliers with domain-rules violation |
| **Implementation Complexity** | High (involves GA optimization and QUBO matrices) | Moderate (standard VAE, rule matching, and XGBoost) |
| **Interpretability** | Moderate (based on selected feature importance) | High (fully compatible with SHAP values and rule tracing) |
