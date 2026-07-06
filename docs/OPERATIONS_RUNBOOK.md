# NIC Fraud Detection V3 — Operations Runbook
<!-- VERSION: 1.0 | OWNER: Project Lead | DATE: 2026-06-29 -->
<!-- Audience: Deployment team, supervisor, system administrator -->
<!-- Companion documents: AGENTS.md (architecture), docs/AGENTS.md (ADRs) -->

---

## 0. Purpose

This document answers the question: **"What do I actually do, and when?"**

It does not explain why the model works. For the architecture, read `docs/AGENTS.md`.
This document covers the yearly operational cycle from data arrival to output
delivery, including what must happen before each inference batch, how
confirmed fraud is fed back into the model, and how to handle common failure
modes.

---

## 1. System Overview (One Page)

```
YEARLY CYCLE
─────────────────────────────────────────────────────────────────
    [New scholarship applications arrive as CSV]
              │
    PRE-INFERENCE PHASE (this document, §3)
    ├── 1. Submit confirmed fraud from last year  ← human action
    ├── 2. Drift check (KS test)
    ├── 3. Incremental model update               ← ~15 min CPU
    └── 4. Verify model checkpoint version

    INFERENCE PHASE (§4)
    ├── 5. Rebuild identity graph (new batch)
    ├── 6. Score all 15,000 applications          ← ~2–4 hr CPU
    ├── 7. EVT thresholds → pseudo-labels (Round 0)
    ├── 8. LightGBM fusion → risk_scores_v3.csv
    └── 9. XAI cards → explanation_cards_v3.json

    POST-INFERENCE PHASE (§5)
    ├── 10. Human review of EVT tail              ← human action
    ├── 11. Round 1 self-training (if authorised) ← human gate
    └── 12. Investigators confirm fraud / FP      ← human action
         (feeds into next year's Step 1)
─────────────────────────────────────────────────────────────────
```

**Hard constraint:** The model is a batch system. All 15,000 applications
must be loaded before scoring begins. Real-time single-application scoring
requires the ego-graph inference server (ADR-012 in AGENTS.md — not yet
implemented).

---

## 2. Who Does What

| Role | Responsibilities |
|---|---|
| **System Administrator** | Server maintenance, checkpoint backups, Docker/Kubernetes, log monitoring |
| **Project Lead (ML)** | Authorises self-training round advancement, interprets drift alerts, decides full vs incremental retrain |
| **Investigator / Supervisor** | Reviews flagged applications, submits confirmed fraud and false positives, authorises Round 1 |
| **Developer** | Implements new ADRs, hotfixes, dependency updates — never during an active inference run |

---

## 3. Pre-Inference Phase

**Do this before running the scoring pipeline on each new yearly batch.**

### 3.1 Submit confirmed fraud from the previous cycle

This is the most important operational action each year. Every application
that investigators confirmed as fraud during the year must be submitted
before the model is updated. The model uses these as real training signal
(replacing or supplementing synthetic examples).

**Who does this:** Investigator / Supervisor  
**When:** After investigators have completed their review of last year's
flagged applications, before the new batch arrives.

**Command (Python):**
```python
from src.confirmed_fraud_store import add_confirmed, add_false_positive, summary

# For each confirmed fraud case:
add_confirmed(
    app_id="APP_2024_00123",          # application_id from the CSV
    fraud_type="IP_CLUSTER",          # see valid types below
    confirmed_by="investigator_name",
    notes="25 applicants sharing one IP at same school",
    cycle="2024-25",
)

# For each case incorrectly flagged (false positive):
add_false_positive(
    app_id="APP_2024_00456",
    confirmed_by="investigator_name",
    notes="Legitimate rural school with shared broadband",
)

# Verify the store:
summary()
```

**Valid fraud types:**
- `IP_CLUSTER` — multiple applicants sharing an IP address ring
- `FEE_INFLATION` — inflated or fabricated fee amounts
- `INCOME_VIOLATION` — income inconsistent with other features
- `NAME_COLLISION` — duplicate father/mother name rings
- `CROSS_CHANNEL` — cross-channel identity reuse
- `OTHER` — confirmed fraud not fitting above categories

**What happens automatically:**
- Feature vector is pulled from `engineered_features_v3.csv` and stored
- Next incremental update uses these as real LOE examples (3× sample weight in LightGBM)
- Self-training loop includes these as hard labels (bypass EVT threshold)
- If ≥ 5 confirmed cases exist, real feature vectors replace synthetic fallback in LOE

**What does NOT happen automatically:**
- The model does not retrain immediately — retraining is triggered separately (§3.3)
- No notification is sent — the system administrator must be informed manually
  (until the FastAPI server is deployed — see ADR-005 in AGENTS.md)

---

### 3.2 Check for distribution drift

Run the KS drift check before every update. This compares the current cycle's
anomaly score distribution against last year's baseline.

**Command:**
```bash
python -m src.retraining_orchestrator --cycle 2025-26 --check-drift
```

**Interpreting the result:**

| Output | Meaning | Action |
|---|---|---|
| `Drift OK (p=0.XXXX)` | Score distribution stable | Proceed with incremental update (§3.3) |
| `DRIFT DETECTED (p=0.XXXX < 0.01)` | Significant distribution shift | See below |
| `No previous cycle scores found — first run` | No baseline yet | Proceed normally |

**If drift is detected:**
1. Check `outputs/feature_drift_v3.json` (once ADR-009 is implemented) for which
   input features shifted — this narrows down the cause.
2. If the shift is due to a genuine policy change (e.g., new income bands, new fee
   structure): the model needs a full GPU retrain on the project lead's laptop
   before the incremental update can proceed.
3. If the shift appears to be a data quality issue (e.g., systematic encoding change):
   fix the data source, do not retrain.
4. Project lead makes the call. Do not proceed with incremental update until
   the drift cause is understood.

**Full GPU retrain (on the laptop — only when authorised):**
```bash
# On the GPU laptop, from project root:
python main_v3.py
# Transfers the new checkpoint to the server via DVC:
dvc push
# On the server:
dvc pull
```

---

### 3.3 Run the incremental model update

This fine-tunes the prediction head on the confirmed fraud examples from §3.1.
Safe to run on the CPU server. Takes ~15 minutes.

**Command:**
```bash
python -m src.retraining_orchestrator --cycle 2025-26
```

**What this does (in order):**
1. Loads confirmed fraud from `data/processed/confirmed_fraud.json`
2. Builds exposure tensor (real confirmed + synthetic fallback)
3. Fine-tunes MLP prediction head only (10 epochs, LR=1e-4)
   - If confirmed fraud ≥ 50: also unfreezes RGCN encoder
4. Refits Subspace IF on current cycle data
5. Refits EVT thresholds on updated scores
6. Runs self-training Round 0 (confirmed fraud bypasses EVT, becomes hard labels)
7. Retrains LightGBM fusion with confirmed fraud at 3× sample weight
8. Saves current scores as KS baseline for next year

**Verification after update:**
```bash
# Check MLflow run (once ADR-002 is implemented):
mlflow ui  # open http://localhost:5000, verify PR-AUC did not regress

# Manual check (before MLflow is set up):
python src/evaluate_model_v3.py
# PR-AUC should not drop more than 0.05 on any category vs the prior run
```

**If PR-AUC regresses after incremental update:**
```bash
# Roll back to the previous checkpoint:
# (once ADR-008 is implemented)
python -m src.retraining_orchestrator --rollback <mlflow_run_id>

# Manual rollback (before rollback function is implemented):
cp models/hybrid_graphmcm_v3.pth.bak models/hybrid_graphmcm_v3.pth
```

---

### 3.4 Verify model checkpoint

Confirm the correct checkpoint is loaded before inference.

```bash
python - <<'EOF'
import torch
from pathlib import Path
ckpt = Path("models/hybrid_graphmcm_v3.pth")
assert ckpt.exists(), "Checkpoint missing"
d = torch.load(ckpt, weights_only=True, map_location="cpu")
print("Checkpoint keys:", list(d.keys()))
print("Checkpoint size:", f"{ckpt.stat().st_size / 1e6:.1f} MB")
EOF
```

---

## 4. Inference Phase

Run the full scoring pipeline on the new batch. This processes all 15,000
applications from the new cycle.

### 4.1 Place the new batch CSV

Copy the new scholarship application CSV to:
```
data/raw/data_for_ml_model.csv
```

The file must have 136 columns. The `sanity` column is dropped automatically
at load time and must never be used as a feature or label (AGENTS.md hard stop #4).

**Verify the file:**
```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/raw/data_for_ml_model.csv")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
assert len(df.columns) == 136, f"Expected 136, got {len(df.columns)}"
assert "application_id" in df.columns, "application_id column missing"
assert "sanity" in df.columns, "sanity column missing (will be dropped at load)"
print("File OK")
EOF
```

### 4.2 Run the full pipeline

```bash
# Full inference pipeline (from project root):
python main_v3.py

# With smoke test first (2 epochs — verifies nothing crashes before committing hours):
python main_v3.py --smoke
python main_v3.py  # run for real only after smoke test passes
```

**Pipeline steps and expected runtime (CPU server, 16 vCPU):**

| Step | Module | Expected time |
|---|---|---|
| Feature engineering (base) | `tabular_feature_engine_v3.py` | 2–5 min |
| Graph construction | `graph_builder_v3.py` | 5–10 min |
| Degree feature merge | `tabular_feature_engine_v3.py` | 1 min |
| Synthetic exposure build | `synthetic_exposure_builder_v3.py` | 1 min |
| Hybrid GraphMCM scoring | `hybrid_graphmcm_v3.py` | 60–120 min |
| Subspace IF scoring | `subspace_if_v3.py` | 5 min |
| EVT threshold fitting | `evt_scorer_v3.py` | 2 min |
| Self-training Round 0 | `self_training_loop_v3.py` | 1 min |
| Score-level fusion (V4) | `fusion_classifier_v3.py` | <1 min |
| XAI cards (JSON) | `xai_layer_v3.py` | 10–20 min |
| Reviewer cards (suspicious, HTML) | `xai_card_html_v3.py` | <1 min |
| Evaluation (synthetic harness) | `evaluate_model_v3.py` | 5 min |
| **Total** | | **~2–4 hours** |

> The fusion is the locked V4 **weighted score-level combine**
> (`risk = minmax(1.0·subspace + 0.5·dense_block_ip + 0.3·hybrid)`) — not a
> LightGBM tree. The reviewer-card step renders interactive HTML only for the
> flagged (suspicious) applications into `outputs/cards/`.

### 4.3 Verify outputs

After the pipeline completes, confirm all output files exist and have the
expected row count.

```bash
python - <<'EOF'
import pandas as pd, json
from pathlib import Path

files = {
    "outputs/hybrid_scores_v3.csv":        ("hybrid_anomaly_score",),
    "outputs/subspace_if_scores_v3.csv":   ("subspace_if_score",),
    "outputs/risk_scores_v3.csv":          ("risk_score_v3",),
}
for path, cols in files.items():
    df = pd.read_csv(path)
    print(f"{path}: {len(df)} rows, score range [{df[cols[0]].min():.4f}, {df[cols[0]].max():.4f}]")

labels = json.loads(Path("outputs/pseudo_labels_v3.json").read_text())
print(f"pseudo_labels_v3.json: {labels['n_positive']} positives | {labels['n_negative']} negatives")

cards = json.loads(Path("outputs/explanation_cards_v3.json").read_text())
print(f"explanation_cards_v3.json: {len(cards)} cards")
EOF
```

**Score direction check (AGENTS.md hard stop #3):**
Higher score = more anomalous. If `risk_score_v3.max()` < 0.1, the model may
have failed to learn any signal. Stop and investigate before delivering outputs.

---

### 4.4 MLflow run — metrics + artifacts (incl. the reviewer cards)

Every `main_v3.py` run opens an MLflow run under experiment
`nic-fraud-detection-v3` (tracking store: `mlflow.db`; artifact files under
`mlruns/`). Browse it with:

```bash
mlflow ui            # then open http://localhost:5000
# or, from the project root, use the helper:
# ./mlflow_ui.bat    (Windows)
```

Pick the latest run under `nic-fraud-detection-v3`. The **Metrics** tab shows the
per-category PR-AUC and `pipeline_duration_seconds`; the **Artifacts** tab holds:

| Artifact path | Contents |
|---|---|
| `checkpoints/` | `hybrid_graphmcm_v3.pth` (the scored model) |
| `scores/` | `risk_scores_v3.csv` |
| `thresholds/` | `evt_thresholds_v3.json` |
| `labels/` | `pseudo_labels_v3.json` |
| `xai/` | `explanation_cards_v3.json` (raw evidence, all top cards) |
| **`cards/`** | **the interactive reviewer-card gallery** — `index.html` + one `card_NNN_<app_id>.html` per flagged application |

**View the cards from MLflow:** in the Artifacts tab, open `cards/index.html` — the
MLflow UI renders HTML artifacts inline. `index.html` lists every suspicious
application ranked by risk; each row opens its evidence card. The cards are the
**simplistic** representation (inline ego-graph + explanation); their *"Examine
full ring in 3D"* links resolve against the **live API** (`/v3/monitoring/{app_id}/ring`),
so the heavy Plotly view is still computed lazily and is only reachable when the
API is up. Nothing large is stored in MLflow — no Plotly bundle is logged, just the
few-KB card HTML.

**Verify the cards were logged** (from raw stdout of the run):
```
[main] Rendered N suspicious reviewer card(s) -> outputs/cards/
```
If `N` is 0, no application crossed an EVT threshold this run — expected on a clean
cohort, not an error. Regenerate the local set at any time without a full run:
```bash
python -m src.xai_card_html_v3                 # suspicious only, lazy 3D links (default)
python -m src.xai_card_html_v3 --ring-mode file # also pre-render offline Plotly rings
```

---

## 5. Post-Inference Phase

### 5.1 Deliver outputs to investigators

Primary output files for investigators:

| File | Contents | Use |
|---|---|---|
| `outputs/risk_scores_v3.csv` | One row per application, `risk_score_v3` in [0,1] | Sort descending, prioritise top N for review |
| `outputs/explanation_cards_v3.json` | Per-application: top feature errors (declared vs model-expected), triggered signals, EVT crossings, closed-form fusion split (subspace/dense-IP/hybrid shares), graph links | Machine-readable decision support |
| `outputs/cards/index.html` + `card_*.html` | Interactive reviewer cards for the flagged applications (rendered from the JSON above); also served live at `GET /v3/monitoring/{app_id}/card` | Human review — the primary "why is this suspicious" view |

Sort for investigator delivery:
```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("outputs/risk_scores_v3.csv")
df.sort_values("risk_score_v3", ascending=False).to_csv(
    "outputs/risk_scores_v3_sorted.csv", index=False
)
top100 = df.nlargest(100, "risk_score_v3")
print(f"Top 100 risk scores: {top100['risk_score_v3'].min():.4f} to {top100['risk_score_v3'].max():.4f}")
EOF
```

### 5.2 Human review gate — EVT tail (MANDATORY before Round 1)

The self-training Round 0 labels are based on EVT tail agreement only — no
classifier has been trained yet. Before allowing the LightGBM to train on
these pseudo-labels (Round 1), investigators must review a sample of the
EVT-flagged applications to confirm that the tail contains genuine fraud
patterns rather than data entry errors.

**Who:** Investigator + Project Lead  
**When:** After Round 0 outputs are delivered, before Round 1 is authorised  

```bash
# Pull EVT-flagged applications for human review:
python - <<'EOF'
import pandas as pd, json
from pathlib import Path

labels = json.loads(Path("outputs/pseudo_labels_v3.json").read_text())
scores = pd.read_csv("outputs/risk_scores_v3.csv")

evt_ids = [r["application_id"] for r in labels["positive_set"] if r["source"] == "evt_pseudo"]
print(f"EVT-flagged (Round 0): {len(evt_ids)} applications")

raw = pd.read_csv("data/raw/data_for_ml_model.csv")
review_set = raw[raw["application_id"].isin(evt_ids[:50])]  # first 50 for blind review
review_set.to_csv("outputs/evt_tail_review_sample.csv", index=False)
print("Saved 50-application review sample -> outputs/evt_tail_review_sample.csv")
EOF
```

**Investigators review this sample and answer:**
1. Are the flagged patterns consistent with known fraud archetypes?
2. Are there obvious data entry errors (e.g., income = 5 INR) in the top flags?
3. Is the precision acceptable? (Target: ≥ 50% of flagged cases are genuine fraud
   or strongly suspicious)

If precision is unacceptable: do not advance to Round 1. Adjust EVT q-parameter
(currently 0.002 in `src/config_v3.py`) upward to tighten the threshold, then
rerun from Step 7.

### 5.3 Round 1 self-training (project lead must authorise)

Round 1 is never triggered automatically. The project lead authorises it after
reviewing the EVT tail (§5.2). This is a hard stop in `main_v3.py`:
the orchestrator always runs `current_round=0`.

**To run Round 1 (after explicit authorisation):**
```bash
python - <<'EOF'
from src.self_training_loop_v3 import run_self_training
from src.fusion_classifier_v3 import run_fusion

# ONLY RUN AFTER HUMAN EVT-TAIL REVIEW AND PROJECT LEAD SIGN-OFF
run_self_training(current_round=1)  # requires risk_scores_v3.csv from Round 0 fusion
run_fusion()
EOF
```

Check PR-AUC after Round 1. If any category regresses below the Round 0
baseline, do not deliver Round 1 scores.

### 5.4 End-of-cycle: submit confirmed fraud

After investigators complete their review of the full output, submit all
confirmed cases using the procedure in §3.1. These feed into next year's
pre-inference update.

**Target:** investigators should submit confirmed fraud to the system within
30 days of receiving the output — before the next yearly batch arrives.

---

## 6. Yearly Operational Calendar

```
CYCLE TIMELINE (example: 2025-26 application batch)
────────────────────────────────────────────────────────────
Month 0   Applications close → CSV exported from NIC portal
          └── Investigator submits all 2024-25 confirmed fraud  ← §3.1

Month 0   Pre-inference phase
          ├── KS drift check                                     ← §3.2
          ├── Incremental model update (CPU, ~15 min)            ← §3.3
          └── Model checkpoint verified                          ← §3.4

Month 0   Full inference pipeline (CPU server, ~2–4 hr)          ← §4.2
Month 0   Outputs delivered to investigators                     ← §5.1

Month 1   EVT tail human review (project lead gate)             ← §5.2
Month 1   Round 1 self-training (if authorised)                 ← §5.3

Month 1–6 Investigators work through the flagged list
          └── As confirmations are made:
              add_confirmed() / add_false_positive()             ← §3.1

Month 11  Investigators submit all remaining confirmations       ← §3.1
          (deadline: before next year's batch arrives)

NEXT CYCLE repeats from Month 0
────────────────────────────────────────────────────────────
```

---

## 7. Year-by-Year Model Evolution

| Year | What changes in the model |
|---|---|
| **Year 0 (current)** | Full GPU training on historical data. Synthetic LOE only (no confirmed fraud). Round 0 only. |
| **Year 1** | First real confirmed fraud examples added. Incremental MLP fine-tune. Real LOE replaces synthetic if ≥ 5 cases. LightGBM sees confirmed fraud at 3× weight. |
| **Year 2** | Confirmed fraud store grows. If ≥ 50 cases: RGCN encoder also unfrozen on incremental update. KS drift check has a 2-year baseline. |
| **Year 3+** | Stable incremental cycle. Full GPU retrain only if KS drift p < 0.01 or a major data schema change. IP fraud clusters and relational patterns accumulate across cycles. |

---

## 8. Common Failure Modes and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `FileNotFoundError: identity_graph_v3.pt` | Pipeline ran out of order | Rerun from `build_base` step via `main_v3.py` |
| `risk_score_v3.max() < 0.1` | LightGBM had < 5 positives; fallback to hybrid score | Check `pseudo_labels_v3.json` — n_positive too low. Check EVT thresholds. |
| `WARNING: fewer than 10 pseudo-positives` | EVT thresholds too tight | Check score distributions; consider lowering `u_percentile` from 95 to 90 as a temporary measure. Flag to project lead. |
| `DRIFT DETECTED (p < 0.01)` | Score distribution shifted vs last year | See §3.2. Do not proceed until cause is understood. |
| `KeyError: hybrid_anomaly_score` in EVT | `hybrid_scores_v3.csv` not found | Run `hybrid_graphmcm_v3.py` first |
| PR-AUC regresses after incremental update | Poor confirmed fraud signal, or too few cases | Roll back checkpoint (§3.3), investigate confirmed fraud quality |
| `confirmed_fraud.json` has 0 entries | Investigators did not submit this year | Treat as Year 0 — synthetic LOE only. Flag to supervisor. |
| Isolated node warning (1,663 nodes) | Expected — 11.1% of dataset has degree=0 | These nodes use the trainable `isolated_embedding` parameter. Not an error. |

---

## 9. What Must Never Be Done

These are hard architectural stops (defined in `docs/AGENTS.md`). Operational
staff should know them to avoid accidentally triggering a bad state:

1. **Never advance self-training rounds without project lead sign-off.** The
   pipeline always runs Round 0. Round 1 requires the EVT tail review in §5.2.
2. **Never use the `sanity` column.** It is dropped at load time in every module.
3. **Never run two inference jobs concurrently.** The model writes intermediate
   files to fixed paths. Concurrent runs overwrite each other's outputs.
4. **Never mix V1 and V3 outputs.** `lgbm_risk_score` (V1) and `risk_scores.csv`
   (V1) must not be combined with V3 outputs in any reporting.
5. **Never trigger a full retrain without the project lead's explicit instruction.**
   Full retrain requires GPU hardware and takes 2–4 hours. Incremental updates
   cover normal yearly drift.
6. **Never feed the model CTGAN or any GAN-generated synthetic data.** The
   synthetic exposure set uses only composite degradation. See AGENTS.md Appendix B.

---

## 10. Key File Reference

| File | Updated when | Contains |
|---|---|---|
| `data/raw/data_for_ml_model.csv` | Each cycle | Raw application data (136 columns, ~15K rows) |
| `data/processed/confirmed_fraud.json` | When investigator submits confirmation | All confirmed fraud and false positive records |
| `data/processed/engineered_features_v3.csv` | Each cycle after build_base | 68 numeric features per application |
| `data/processed/identity_graph_v3.pt` | Each cycle after build_graph | PyG HeteroData graph (5 typed edges) |
| `models/hybrid_graphmcm_v3.pth` | After each training run | RGCN encoder + MLP head weights + isolated_embedding |
| `outputs/hybrid_scores_v3.csv` | Each cycle | `hybrid_anomaly_score`, `feature_pred_error`, `edge_pred_error` |
| `outputs/risk_scores_v3.csv` | Each cycle | Final `risk_score_v3` per application — primary deliverable |
| `outputs/explanation_cards_v3.json` | Each cycle | Per-application XAI (SHAP values, top features, triggered signals) |
| `outputs/pseudo_labels_v3.json` | Each cycle (Round 0) | EVT-promoted pseudo-positives + confirmed hard labels |
| `outputs/prev_cycle_scores_ks.json` | After each cycle completes | Score distribution baseline for next year's KS test |
| `outputs/evt_thresholds_v3.json` | Each cycle | Per-signal EVT thresholds (GPD-derived) |
