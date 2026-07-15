# NIC Fraud Detection V3 — Operations Runbook
<!-- VERSION: 2.0 | OWNER: Project Lead | DATE: 2026-07-10 -->
<!-- Audience: Supervisor / reviewer operating the console -->
<!-- Companion: deploy/README.md (full local+k8s deploy), docs/AGENTS.md (architecture) -->

This runbook covers only three things: **starting the stack in Docker**,
**opening the console**, and **what each screen does**. For the full
deployment story (Kubernetes, storage, the four portability gotchas) see
`deploy/README.md`.

---

## 1. Start the stack (Docker)

From the project root:

```bash
docker compose up --build
```

This starts four services: `redis` (broker), `nic-api` (FastAPI), `nic-worker`
(Celery, does the training), and `nginx` (the front door). Wait until the logs
settle — `nic-api` is ready when you see the health check passing.

To stop:

```bash
docker compose down          # stop; keep data
```

Data lives in the mounted `./data`, `./models`, `./outputs` folders, so it
survives restarts. The console only shows content once a pipeline run has
produced scores and cards in `outputs/` (a fresh checkout with empty `outputs/`
shows empty queues — that's expected, not an error).

---

## 2. Open the console

In a browser, go to:

```
http://localhost:8080/
```

Everything is served from this one address — the UI and the API share the
origin, so nothing else needs configuring. (On a server, replace `localhost`
with the server address / the port your deploy exposes.)

There are three tabs across the top: **Review queue**, **Pattern queue (LOE)**,
and **Model audit & deploy (admin)**.

---

## 3. Review queue — triage flagged applications

This is the reviewer's main screen.

- **Status tiles** (top): confirmed-fraud count, false-positive count, live
  checkpoint size, and the current drift recommendation.
- **Top suspicious applications**: the ranked queue from the last pipeline run.
  Click any row to open it. Each row carries a colored **risk badge**
  (High / Medium / Low).
- **Triage toolbar** (above the queue): tick rows (or **Select all**), then
  filter by application-ID or risk level. With rows selected you can:
  - **⚑ Label / retrain selected** — opens a batch dialog where you tag each
    application confirmed-fraud (with type) or false-positive, enter your name,
    then either **Record labels only** or **Record + retrain (incremental)**.
    The retrain is the human gate — recording labels alone changes nothing.
    Tick **smoke test** for a fast dry run. The job id + live status appear in
    the dialog.
  - **⤓ Export selected** — downloads one zip of the chosen applications
    (scorecard CSV + reviewer card + 3D identity ring + evidence, per app,
    plus a combined `manifest.csv`).
  - **✕ Remove selected** / **↺ Restore removed** — hide triaged rows for this
    session only; server data is untouched.
- **Reviewer card** (opens below the row): the full evidence card for that
  application — a risk gauge, the ranked reason codes, per-field
  *declared-vs-model-expected* comparison bars, and an interactive identity
  network. It has its own built-in buttons to **Confirm fraud**, **Mark false
  positive**, or **Undo label** — those write straight to the confirmed-fraud
  store. (After submitting, hit **Refresh** on the queue to update the tiles.)
- **Topology detail** — the buttons above the card:
  - **◎ 3D identity ring** — a rotatable 3-D view of the application and everyone
    it shares an IP / mobile / name / pincode with.
  - **⌗ Ego-graph** — a flat neighbourhood graph of the same.
  Both open in a large pop-up with a Ring ⇄ Ego toggle, an **↗ Open in new tab**
  button, and close on **Esc** or clicking outside.
- **⚑ Flag for LOE** — records the application (and the ring of IDs you name) as
  a candidate fraud *pattern*, sending it to the Pattern queue.

---

## 4. Pattern queue (LOE) — confirm and promote fraud patterns

Candidate patterns flagged from reviewer cards land here.

- Each pending pattern shows its id, fraud type, and the sub-graph you flagged.
- Tick the ones you want to act on and click **Promote selected patterns**.
  Promotion appends each pattern's ring to the model's **topology-exposure set**
  (extracting the members' real shared-attribute edges, or a clique on the
  relation you asserted) and dispatches an **incremental retrain** so the model
  learns them. Tick **smoke test** first for a fast, no-real-training dry run.
- The job id and live status appear beneath the button.

---

## 5. Model audit & deploy (admin) — the deployment loop + model stats

> ⚠ This tab triggers real retraining and file changes on the server and has
> **no authentication** in this build. Keep the console behind a VPN / trusted
> network.

This tab replaces the old MLflow dashboard — all model state lives here.

**Running model — status** (top strip): the live checkpoint (size, feature and
edge counts), the scored-population size, confirmed / false-positive counts, the
drift recommendation, the last evaluation's PR-AUC numbers, and the last run.
Hit **↻ Refresh** to re-read.

**Drift explanation — should you full-retrain?**: plain-English rationale for the
drift decision, built only from numbers the pipeline already computed — the
overall score-distribution KS p-value vs the alert threshold, and a table of the
model features that shifted most. Counts cover the **44 model features** (the 24
dropped identifier columns are excluded and noted). A red verdict means a full
retrain is recommended before the next incremental update.

**Deployment loop** — four numbered steps, top to bottom:

1. **Intake** — first pick **what the data is for**:
   - **New cohort to score** (default) — drag a cohort **CSV** onto the drop-zone
     (or *browse*). It's saved server-side and checked against the raw schema; you
     get a row count and a pass/fail. A schema mismatch blocks the next steps.
   - **New fraud pattern (relational LOE)** — for a brand-new fraud **ring** a
     supervisor found that the model has never seen. Upload the ring (full
     raw-schema rows, fresh IDs), then use the pattern box that appears:
     **Test detection** scores the ring read-only (rebuilds features + the graph,
     so the members' shared IP/name edges are real) to show whether the current
     model catches it; **Ingest as pattern + retrain** permanently adds the ring,
     appends its real subgraph as a topology-exposure cluster, records it in the
     confirmed stores, and dispatches the human-gated fine-tune. **Re-test after
     retrain** shows detection once the model has learned it. (Steps 2–3 below are
     for cohorts and are hidden in this mode; the retrain job still shows in step 4.)
2. **Evaluate** — scores the uploaded cohort read-only (no model change) and
   reports a drift p-value. The dataset path is filled in for you from step 1.
   *This rebuilds features + graph synchronously and can take a few minutes.*
3. **Decide** (human-gated) — after reviewing the evaluation, choose:
   **Log "no action"**, **Merge + incremental update**, or **Merge + full
   retrain**. The merge/retrain options permanently add the cohort to the data
   and dispatch a job (you must confirm the warning). A separate row lets you
   **retrain on current data only** (no new cohort) — incremental or full.
4. **Watch** — the dispatched job's id auto-fills here; status polls
   automatically, or paste any job id and **Poll status**.

**Run history**: every training run and checkpoint swap (newest first) — when,
type, cycle, metrics, checkpoint size. This is the audit trail.

**Rollback checkpoint**: paste a versioned checkpoint path (shown in run
history) and roll the live model back to it.

---

## 6. Quick reference

| I want to… | Where |
|---|---|
| Start everything | `docker compose up --build` |
| Open the console | `http://localhost:8080/` |
| Triage a flagged application | Review queue → click a row |
| See an application's network in 3-D | Review queue → **◎ 3D identity ring** |
| Export one application (CSV + card + evidence) | Review queue → open a row → **⤓ Export** |
| Export all flagged applications | Review queue → **⤓ Export all flagged** (zip: `manifest.csv` + cards + evidence) |
| Confirm / clear a label | Buttons inside the reviewer card |
| Turn a pattern into training signal | Pattern queue → **Promote** |
| Score a new cohort | Admin → Intake → Evaluate |
| Retrain the model | Admin → Decide |
| See model metrics / history | Admin → status strip + Run history |
| Undo a bad checkpoint | Admin → Rollback |
| Stop everything | `docker compose down` |
