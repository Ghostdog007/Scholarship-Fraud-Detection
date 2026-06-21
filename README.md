# NIC Scholarship Fraud Detection — V2 (GPU Accelerated)

A fully unsupervised, GPU-accelerated, rule-free pipeline for flagging anomalous Pre-Matric and Post-Matric scholarship applications submitted through the NIC national portal.

Unlike the V1 architecture (which relied heavily on hardcoded rule engines and weak-supervised LightGBM), **V2 relies purely on mathematical anomaly detection** utilizing PyTorch, Extreme Value Theory (EVT), and Graph Neural Networks (GNNs).

For full architectural contracts and constraints, see **`docs/AGENTS_v2_rulefree.md`**.

---

## The GPU-Accelerated V2 Architecture

The V2 architecture operates under a strict "No Human Rules" mandate. It leverages **Latent Outlier Exposure (LOE)** via mathematically planted synthetic anomalies during a pretraining curriculum. It uses a dual-engine approach:
1. **Tabular VAE**: To map numerical policy violations (e.g., extreme age, income mismatch).
2. **Graph Autoencoder (DOMINANT + DeepSVDD)**: To map structural and relational fraud rings (e.g., identity farming, IP concentration).

### Strengths and Weaknesses

**Strengths:**
* **Zero-Day Detection (Unsupervised):** Because it does not rely on predefined rules (`flag_income_under_10000`), it is capable of discovering entirely novel fraud archetypes mathematically.
* **Relational Awareness:** The PyTorch Geometric Graph AE shattered the V1 baseline for relational fraud (delivering a **4x to 8x PR-AUC improvement** on graph-centric anomalies like IP Concentration and Name Collisions).
* **GPU Acceleration:** Pre-training, exposure curriculums, and scoring run entirely on CUDA tensors, allowing massive scalability over V1's CPU-bound rule engines.
* **Mathematical False Positive Control:** By relying on Extreme Value Theory (EVT) and fitting a Generalized Pareto Distribution, we dynamically set thresholds (`q=0.002`) instead of relying on arbitrary probability cutoffs.

**Weaknesses:**
* **Lack of Ground Truth:** The model still operates on an assumption of a largely "clean" normal hypersphere. Without true fraud labels, hyperparameter tuning (like VAE learning rate or LOE margin) remains difficult and relies heavily on proxy synthetic testing.
* **Tabular Performance Drop:** Fully stripping the hardcoded rules caused a slight drop in the PR-AUC for purely tabular univariate anomalies (like Age/Income) compared to the over-fitted V1 rule engine.

---

## Data Flow & Module Lineage (LLM Reference Guide)

The entire V2 pipeline is executed sequentially via `main_v2.py`.
The architecture is designed to strictly pass explicit outputs between phases. No module reaches into another module's internal state. 

### Phase A: Feature Engineering
* **File:** `tabular_feature_engine_v2.py`
* **Inputs:** `datasets/data_for_ml_model.csv` (Shape: 15,000 x 136)
* **Processing:** Drops 100%-null columns, drops spatial duplicates, encodes booleans, and drops all raw string columns. Retains exactly **63 numerical dimensions**.
* **Outputs:** 
  * `engineered_features_v2.csv` (15,000 x 63)
  * `v2_feature_schema.json` (Record of column names and excluded features)

### Phase B: Graph & Synthetic Exposure
* **Files:** `graph_builder_v2.py`, `synthetic_exposure_builder_v2.py`
* **Inputs:** `engineered_features_v2.csv` (15,000 x 63)
* **Processing:** 
  * *Graph Builder*: Converts 5 relational attributes into a PyTorch Geometric `HeteroData` object with 5 typed edges (`shares_mobile`, `shares_ip`, `shares_father_name`, `shares_mother_name`, `shares_pincode`). The node feature vector length is `63`.
  * *Exposure Builder*: Plants 750 mathematically manipulated anomalies (150 per archetype) across the 63 dimensions to serve as a repulsive force during training.
* **Outputs:** 
  * `identity_graph.pt` (PyTorch Geometric Graph)
  * `synthetic_exposure_set.pt` (Tensor Shape: 750 x 63)

### Phase C: Core Autoencoders
* **Files:** `tabular_vae_v2.py`, `graph_autoencoder_v2.py`
* **Inputs:** `engineered_features_v2.csv` (15,000 x 63), `identity_graph.pt`, `synthetic_exposure_set.pt`
* **Processing:** 
  * *Tabular VAE*: Input Dim (63) -> Encoder (32, 16) -> Latent Mu/Sigma (8) -> Decoder (16, 32) -> Output Dim (63). Uses Sigmoid activation.
  * *Graph AE (DOMINANT)*: Input Dim (63) -> RGCNConv Encoder (64, 32) -> Latent Node Embeddings (32). Outputs an Attribute Reconstruction and a Structural Reconstruction (Dot-product Adjacency). Embeddings are penalized via DeepSVDD.
  * *Curriculum*: Stage 1 uses `synthetic_exposure_set.pt` to push anomalies away from the hypersphere centroid. Stage 2 is free reconstruction.
* **Outputs:** 
  * `vae_v2_scores.csv` (`vae_anomaly_score` and `recon_error_vector` MSE per feature)
  * `graph_v2_scores.csv` (`graph_anomaly_score`, `attr_recon_error`, `struct_recon_error`)
  * `tabular_vae_v2.pth` & `graph_autoencoder_v2.pth` (Saved Model Checkpoints)

### Phase D: EVT Scoring & Self-Training
* **Files:** `evt_scorer.py`, `self_training_loop_v2.py`
* **Inputs:** `vae_v2_scores.csv`, `graph_v2_scores.csv`
* **Processing:** Fits a Generalized Pareto Distribution (GPD) to the extreme right-tail of the combined anomaly scores (quantile `q=0.002`). Applications exceeding the EVT threshold are promoted to true positives.
* **Outputs:** 
  * `evt_thresholds_v2.json`
  * `pseudo_labels_v2.json` (List of confirmed `application_id`s flagged as anomalous)

### Phase E: Fusion Classifier & XAI
* **Files:** `fusion_classifier_v2.py`, `xai_layer_v2.py`
* **Inputs:** `engineered_features_v2.csv`, `pseudo_labels_v2.json`
* **Processing:** Trains a LightGBM strictly on the pseudo-labels to smooth decision boundaries across the 63 numerical features. Uses `SHAP` and `PGExplainer` to write human-readable explanations detailing *why* the application was flagged.
* **Outputs:** 
  * `risk_scores_v2.csv` (Final `lgbm_risk_score_v2`)
  * `explanation_cards_v2.json` (SHAP-backed localized feature importance)

### Phase F: Evaluation Harness
* **File:** `evaluate_model_v2.py`
* **Inputs:** `engineered_features_v2.csv`, `.pth` model checkpoints.
* **Processing:** Generates 750 entirely *unseen* synthetic test rows (no leakage from Phase B). Passes them through the saved Tabular and Graph autoencoders to calculate PR-AUC against the normal data.
* **Outputs:** Console standard output detailing V1 vs V2 PR-AUC per category.

---

## Execution
Run the full V2 pipeline:
```bash
python main_v2.py
```
