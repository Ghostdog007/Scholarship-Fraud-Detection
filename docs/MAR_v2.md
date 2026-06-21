# Model and Architecture Review (MAR)
**System Name:** NIC Scholarship Fraud Detection (V2 Rule-Free Architecture)
**Reviewer:** Automated Review
**Date:** June 2026

This is an internal technical document written for an audience of ML engineers and senior technical reviewers. It assesses the theoretical grounding, implementation anatomy, structural weaknesses, and unresolved architecture surface of the NIC V2 system.

---

## LAYER 1 — THEORETICAL GROUNDING

### 1. Tabular Feature Engine (No Rules)
* **a) Class of problem:** Feature extraction and representation.
* **b) Why appropriate:** The raw NIC data contains complex administrative rules and high-nullity text. Reducing it to continuous variables and simple structural identities without enforcing hardcoded policy boundaries enables unsupervised latent discovery.
* **c) Research basis:** Foundational ML engineering (preventing hardcoded threshold leakage into continuous embedding spaces).
* **d) Assumptions:** The available continuous and boolean features contain enough intrinsic signal variance to isolate fraud without human-defined boundaries.

### 2. Graph Identity Builder
* **a) Class of problem:** Relational and structural representation learning.
* **b) Why appropriate:** Fraud rings inherently share attributes (mobile numbers, IPs, parent names). Graph representations natively capture these multi-hop relationships rather than relying on flattened aggregate counts.
* **c) Research basis:** Schlichtkrull et al. (R-GCN, 2018). Typed-edge encoders allow heterogeneous graphs to retain specific interaction semantics (e.g., distinguishing an IP share from a Father Name share).
* **d) Assumptions:** Shared fields (like IP or mobile) are accurately recorded in the database and logically imply human association.

### 3. Core Autoencoders (Tabular VAE & DOMINANT Graph AE)
* **a) Class of problem:** Unsupervised Anomaly Detection.
* **b) Why appropriate:** Without accurate ground truth labels (only 4 confirmed frauds in 15,000), autoencoders can learn the "normal" manifold. Applications that reconstruct poorly or embed far from the cluster centroid are mathematically anomalous.
* **c) Research basis:** Ding et al. (DOMINANT, SDM 2019) for dual-decoder graph AE; Ruff et al. (DeepSVDD, ICML 2018) for hypersphere boundary mapping.
* **d) Assumptions:** The vast majority of the 15,000 applications are genuinely "normal" and uncorrupted, creating a stable, dense hypersphere centroid.

### 4. Latent Outlier Exposure (LOE) Pretraining
* **a) Class of problem:** Cold-start anomaly detection with contaminated data.
* **b) Why appropriate:** Pure reconstruction often fails to distinguish normal from subtle fraud. Injecting mathematically derived synthetic archetypes pushes the decision boundary inward, tightly around normal behavior.
* **c) Research basis:** Hendrycks et al. (Outlier Exposure, ICLR 2019) and Qiu et al. (LOE, ICML 2022).
* **d) Assumptions:** Programmatically generated synthetic anomalies accurately mirror the latent geometry of real-world unknown fraud.

### 5. Extreme Value Theory (EVT) Scoring
* **a) Class of problem:** Statistical threshold derivation.
* **b) Why appropriate:** Hardcoded probability cutoffs fail when distributions shift. Fitting a Generalized Pareto Distribution (GPD) on the extreme right tail of scores sets a mathematically principled, false-positive-controlled threshold.
* **c) Research basis:** Siffer et al. (SPOT, KDD 2017).
* **d) Assumptions:** The anomaly scores' right tail actually follows a continuous extreme value distribution.

### 6. Self-Training Loop & Fusion Classifier
* **a) Class of problem:** Weak supervision and score fusion.
* **b) Why appropriate:** EVT yields high-precision/low-recall tails. Self-training iteratively expands the boundary by finding cases a LightGBM classifier confidently agrees look like the EVT tail.
* **c) Research basis:** Karamanolakis et al. (ASTRA, NAACL 2021).
* **d) Assumptions:** The initial EVT mutual tail contains true positives, and the classifier's inductive bias doesn't cause semantic drift in subsequent rounds.

---

## LAYER 2 — IMPLEMENTATION ANATOMY

### 1. `tabular_feature_engine_v2.py`
* **a) Responsibility:** Engineers tabular node features strictly without rules.
* **b) Input contract:** `data/raw/data_for_ml_model.csv` (CSV).
* **c) Internal:** Drops 100% null columns, computes continuous metrics (age, fee_income_ratio), computes identity aggregations (ip_application_count), and evaluates name similarities via `SequenceMatcher`.
* **d) Output contract:** `data/processed/engineered_features_v2.csv` and `data/processed/v2_feature_schema.json` (JSON).

### 2. `graph_builder_v2.py`
* **a) Responsibility:** Constructs the typed identity graph from application records.
* **b) Input contract:** `engineered_features_v2.csv` (CSV).
* **c) Internal:** Connects rows using shared attributes (`shares_mobile`, `shares_ip`) via `networkx` and converts them into a PyG `HeteroData` object with 63 tabular dimensions as node features.
* **d) Output contract:** `data/processed/identity_graph.pt` (PyG Tensor).

### 3. `tabular_vae_v2.py` & `graph_autoencoder_v2.py`
* **a) Responsibility:** Learn the manifold of "normal" applications via tabular and relational topologies.
* **b) Input contract:** `engineered_features_v2.csv`, `identity_graph.pt`, and `synthetic_exposure_set.pt`.
* **c) Internal:** Stage 1 pretrains the network using an auxiliary loss to push synthetic anomalies away from the normal centroid (LOE). Stage 2 fine-tunes on pure reconstruction MSE. Graph AE specifically uses RGCNConv and DeepSVDD.
* **d) Output contract:** `outputs/vae_v2_scores.csv` and `outputs/graph_v2_scores.csv`.

### 4. `evt_scorer.py`
* **a) Responsibility:** Derives data-driven anomaly thresholds without domain rules.
* **b) Input contract:** `vae_v2_scores.csv`, `graph_v2_scores.csv`.
* **c) Internal:** Uses Peak Over Threshold (POT) to isolate the 95th percentile, then applies Maximum Likelihood Estimation (MLE) to fit a GPD, exporting the final threshold at quantile `q=0.002`.
* **d) Output contract:** `outputs/evt_thresholds_v2.json`.

### 5. `self_training_loop_v2.py` & `fusion_classifier_v2.py`
* **a) Responsibility:** Expand the positive label set and train the final risk classifier.
* **b) Input contract:** Score CSVs, `evt_thresholds_v2.json`, `engineered_features_v2.csv`.
* **c) Internal:** Round 0 accepts only applications breaching *both* EVT thresholds. Subsequent rounds add applications where the trained LightGBM highly agrees. The final LightGBM model is trained purely on these pseudo-labels.
* **d) Output contract:** `outputs/pseudo_labels_v2.json` and `outputs/risk_scores_v2.csv`.

### 6. `xai_layer_v2.py`
* **a) Responsibility:** Produce a human-readable explanation card for every flagged application.
* **b) Input contract:** Trained LightGBM classifier, trained graph AE, `engineered_features_v2.csv`.
* **c) Internal:** Uses SHAP for the LightGBM classifier and PGExplainer/GNNExplainer to pinpoint the exact subgraph that triggered the anomaly, templating a narrative string.
* **d) Output contract:** `outputs/explanation_cards_v2.json`.

### Data Flow Diagram

```text
data_for_ml_model.csv [CSV]
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
tabular_feature_engine_v2.py                      graph_builder_v2.py
        │                                                 │
        │ engineered_features_v2.csv [CSV]                │ identity_graph.pt [Tensor]
        │ v2_feature_schema.json [JSON]                   │
        ▼                                                 ▼
tabular_vae_v2.py                                 graph_autoencoder_v2.py
        │                                                 │
        │ vae_v2_scores.csv [CSV]                         │ graph_v2_scores.csv [CSV]
        └────────────────────────┬────────────────────────┘
                                 ▼
                         evt_scorer.py
                                 │
                                 │ evt_thresholds_v2.json [JSON]
                                 ▼
                     self_training_loop_v2.py
                                 │
                                 │ pseudo_labels_v2.json [JSON]
                                 ▼
                     fusion_classifier_v2.py
                                 │
                                 │ risk_scores_v2.csv [CSV]
                                 ▼
                          xai_layer_v2.py
                                 │
                                 │ explanation_cards_v2.json [JSON]
                                 ▼
                              [END]
```

### Output Guarantees & Fragilities
* **`tabular_feature_engine_v2.py`:**
  * **Guarantee:** Downstream modules can assume all outputs are strictly continuous/boolean with 0 nulls.
  * **Fragility:** Implicitly relies on the input CSV maintaining exactly the same 136 column schema.
* **`graph_builder_v2.py`:**
  * **Guarantee:** Outputs a PyG `HeteroData` tensor where every node index directly maps to the `engineered_features_v2.csv` row index.
  * **Fragility:** Fragile to disconnected nodes. Nodes with zero shared features will exist purely as isolated components in the graph.
* **`tabular_vae_v2.py` & `graph_autoencoder_v2.py`:**
  * **Guarantee:** Downstream modules are guaranteed that `vae_anomaly_score` and `graph_anomaly_score` are strictly scaled such that **higher = more anomalous**.
  * **Fragility:** Implicitly assumes that `synthetic_exposure_set.pt` accurately reflects real fraud topology. If not, Stage 1 actively harms the latent space.
* **`evt_scorer.py`:**
  * **Guarantee:** Thresholds are statistically derived without human-set boundaries.
  * **Fragility:** Downstream modules implicitly rely on `q=0.002` remaining stable. If the EVT scorer encounters a bizarre distribution where the GPD fails to fit cleanly, it will output a wildly fluctuating threshold without a hard error, skewing the self-training round 0 seed.
* **`self_training_loop_v2.py` & `fusion_classifier_v2.py`:**
  * **Guarantee:** The final `risk_scores_v2.csv` score bounded [0,1].
  * **Fragility:** The fusion classifier acts blindly on its pseudo-labels. If Round 0 feeds it bad EVT labels, the classifier will smoothly and silently fit to the noise.
* **`xai_layer_v2.py`:**
  * **Guarantee:** Outputs valid JSON explanations directly tied to the fusion classifier's logic.
  * **Fragility:** Assumes GNNExplainer/PGExplainer subgraphs are faithful to the true fraud mechanism without active validation.

---

## LAYER 3 — UNBIASED CRITIQUE

**a) STRENGTHS:**
* The EVT threshold is genuinely data-derived rather than hand-set, which means it adapts automatically when the scoring distribution shifts or a new state's data is introduced.
* The dual-engine LOE pretraining forces the latent space to understand specific relational boundaries (IP concentration) without relying on fragile human-authored logic or static counts (e.g. `count > 15`).
* Graph Autoencoders strictly separate attribute and structural reconstruction. This means organized identity farming is caught based on the *shape* of the network, even if individual application attributes look completely normal.

**b) STRUCTURAL WEAKNESSES:**
* **Assumption 1:** The normal hypersphere is mostly clean.
  * *Violation condition:* A significant percentage of actual applications are organized fraud that was missed by v1 rules.
  * *Failure Mode:* The VAE and DOMINANT encoder map the organized fraud as "normal" because it dominates the density. The hypersphere inflates to include fraud.
  * *Detectability:* Silent. EVT tails simply shift right, accepting mass fraud as normal behavior.
* **Assumption 2:** The EVT mutual tail contains true positives at Round 0.
  * *Violation condition:* High EVT scores are driven by severe data entry typos (e.g., income = 5 INR) rather than malicious fraud.
  * *Failure Mode:* The self-training loop anchors on data entry errors and promotes similar errors, entirely missing sophisticated fraud.
  * *Detectability:* Detectable via manual audit of Round 0 pseudo-labels or massive Phase D PR-AUC drops.

**c) KNOWN FAILURE MODES:**
* GAN-generated synthetic anomalies degrade composite relational signals, rendering Stage 1 LOE pretraining useless or actively harmful if programmatic construction isn't used.
* Nodes isolated in the identity graph (no shared IP or mobile) default to solely attribute reconstruction, completely bypassing the GNN structural capabilities and severely dampening their `graph_anomaly_score`.

**d) WHAT THE METRICS DON'T MEASURE:**
* The Phase D Synthetic Harness PR-AUC proves the system detects injected archetypes accurately. It *does not* prove the system can detect zero-day real-world fraud patterns outside those specific synthetic topologies.

**e) WHAT WOULD BREAK FIRST:**
* The **Self-Training Label Promotion**. Without robust ground truth, relying on EVT mutual agreement to seed a LightGBM is highly fragile. A slight misalignment in score distribution tails could seed the classifier with false positives, triggering a catastrophic semantic drift loop in Round 1.

### Critique Summary

| Component | Core Assumption | Failure Condition | Failure Mode | Detectable? |
|---|---|---|---|---|
| DeepSVDD Graph AE | Normal data density is clean | Fraud dominates the dataset | Hypersphere inflates to accept fraud as normal | Silent |
| EVT Scorer | Tail fits GPD smoothly | Data has extreme discontinuities | Threshold explodes or drops to 0 | Partially |
| Self-Training Loop | EVT tail is true fraud | EVT tail is mostly data typos | Model learns to flag typos instead of fraud | Yes (Manual) |
| Graph AE (isolated nodes) | Every node has at least one typed edge | Applicant uses unique mobile, unique IP, common name — no shared attributes | Node defaults to attribute reconstruction only; structural anomaly signal is zero regardless of true fraud | Silent — `struct_recon_error` will be low for isolated fraudulent nodes and nothing will surface it |
| Stage 1 Synthetic Exposure | Programmatic archetypes are representative of real fraud geometry | Archetypes are too narrow or too obvious — only exact IP-cluster or name-collision patterns | Stage 2 latent space is biased toward obvious fraud; subtle novel patterns are treated as normal because they don't resemble Stage 1 exposure | Partially — Phase D will pass (it tests exact archetypes) but real novel fraud will not be caught |

---

## LAYER 4 — OPEN QUESTIONS AND IMPROVEMENT SURFACE

**a) Open Questions:**
* *What is the optimal LOE annealing schedule (λ(t))?* Options: linear, step, cosine. Evidence: ablation on Phase D. Call: Technical.
* *How do we initialize the DeepSVDD centroid?* Options: mean of all Stage 1 embeddings vs mean of only normal embeddings. Evidence: clustering density variance. Call: Technical.
* *How are threshold appeals handled in production?* Options: reject without rule text, or surface the XAI narrative. Call: Policy.

**b) Next Honest Evaluation Step:**
* Deploy a shadow run on the next 15,000 fresh applications and have NIC human investigators perform a blind review of the top 50 EVT-flagged applications to confirm the cold-start precision rate *before* initiating Round 1 of self-training.

**c) Redesign with Hindsight:**
* The self-training loop. ASTRA-style promotion without any human-in-the-loop validation for Round 0 is incredibly risky. I would replace the fully automated Round 0 with an active learning interface where an investigator explicitly reviews the EVT tail before the LightGBM is ever allowed to train.

**d) External Needs:**
* **Domain Expertise:** Manual review of the Round 0 pseudo-label set to guarantee the cold-start anchor is pure.
* **Data Integration:** AISHE/DISE institutional geo-location data to structurally ground the `institute_application_count` aggregation into a true physical feature.
