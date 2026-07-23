# NIC Fraud Detection V3 — Operations Runbook
<!-- VERSION: 2.4 | OWNER: Project Lead | DATE: 2026-07-23 -->
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

This starts six services: `postgres` (system of record, V4-Scale), `db-init`
(one-shot — applies the schema and ingests the primary dataset + confirmed-fraud/
pattern/run-history stores into Postgres, then exits), `redis` (broker), `nic-api`
(FastAPI), `nic-worker` (Celery, does the training), and `nginx` (the front door).
`nic-api`/`nic-worker` wait for `db-init` to finish successfully before starting, so
Postgres is schema-current and populated the moment the API accepts a request — not
just an optional fallback. `db-init` re-runs (idempotently) on every `docker compose
up`, so Postgres stays in sync with whatever is currently in `data/`/`outputs/`. Wait
until the logs settle — `nic-api` is ready when you see the health check passing.

Postgres is reachable from the host at `localhost:5433` (not 5432 — chosen to avoid
colliding with a locally-installed PostgreSQL); connection settings come from a
git-ignored `.env` at the project root (`NIC_DB_HOST/PORT/NAME/USER/PASSWORD`).

To stop:

```bash
docker compose down          # stop; keep data (including the Postgres volume)
```

**Gotcha:** if you rebuild just `nic-api`/`nic-worker` (e.g. `docker compose up
--build nic-api nic-worker`) without recreating `nginx`, nginx keeps the old
container's cached IP and every request 502s. Restart it: `docker compose restart
nginx`.

Data lives in the mounted `./data`, `./models`, `./outputs` folders (files) and the
`postgres-data` volume (Postgres), so it survives restarts. The console only shows
content once a pipeline run has produced scores and cards in `outputs/` (a fresh
checkout with empty `outputs/` shows empty queues — that's expected, not an error).

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

- **Dataset switcher** (top-left of the queue): **Primary dataset · 15k scored
  applications** or any **evaluated cohort** you uploaded (admin → Intake →
  Evaluate). The primary dataset is the population the unsupervised detector is
  fit on *and* scores (not a held-out test set — genuinely unseen data is scored
  via an evaluated cohort). Pick a cohort to review how the model scored that
  ingested data *read-only*, before committing it. In cohort mode a cyan banner
  reminds you the scores are **pre-fusion** (`hybrid_anomaly_score`, bucketed by
  within-cohort percentile). The review tools have **full visual parity** —
  the reviewer card uses the same identity-network tab, ranked reason codes,
  and expandable declared-vs-expected fields as the primary card, with a real
  network preview drawn from the cohort's own graph. **3D ring**, **ego-graph**,
  and **export** (single / all / selected) all work on cohort apps too. The
  **Signal drivers** tab also works pre-commit (added 2026-07-22): subspace IF,
  dense-block, and a preview fusion score are computed read-only over the
  cohort's own merged population (`POST /evaluate-dataset`), so the same bars
  and fusion-composition footer render as on a committed card. What's
  genuinely different pre-commit (labeled "PREVIEW · pre-fusion" on the card,
  never silently hidden): no EVT-threshold reason codes or model-traceability
  margins (EVT thresholds are fitted against the canonical population, not
  this cohort's preview — comparing a preview score to them would be
  misleading), and
  no Confirm-fraud / Mark-false-positive buttons (that write to the committed
  features file, which doesn't have this app yet). Flag-for-LOE and
  label/retrain stay gated the same way — commit the cohort first (Decide →
  Merge). Switch back to the **Primary dataset** for those.
  A **✕ Remove cohort** button (cohort mode only) drops that cohort from the
  console. It first shows a warning that **all of the cohort's outputs are
  discarded on the server** — its explanation/reviewer cards, 3D rings,
  ego-graphs, pre-fusion scores, evidence, and the uploaded CSV — and deletes
  them only after you accept. The base data and the downloadable sample CSV are
  untouched. Re-upload + re-evaluate the CSV to bring the cohort back. (Good
  for a demo: add a dataset, show it, remove it.)
- **Status tiles** (top): confirmed-fraud count, false-positive count, live
  checkpoint size, and the current drift recommendation.
- **Top suspicious applications**: the ranked queue from the last pipeline run,
  paged **50 per page** with **← Prev / Next →** at the bottom (it covers the full
  flagged set — ~500 carded applications — not just the top 50). The pager shows
  "Showing a–b of N flagged · Page x / y". Click any row to open it. Each row
  carries a colored **risk badge** (High / Medium / Low).
- **Triage toolbar** (above the queue): tick rows (or **Select all** — which
  selects the current page), then filter by application-ID or risk level.
  **Selection persists across pages**, so you can gather members of one ring from
  several pages before acting. With rows selected you can:
  - **⚑ Label / retrain selected** — opens a batch dialog where you tag each
    application confirmed-fraud (with type) or false-positive, enter your name,
    then either **Record labels only** or **Record + retrain (incremental)**.
    The retrain is the human gate — recording labels alone changes nothing.
    Tick **smoke test** for a fast dry run. The job id + live status appear in
    the dialog.
  - **◈ Flag for LOE (selected)** — sends every selected application to the
    Pattern queue together as one candidate ring. Opens the same Flag-for-LOE
    dialog (below) pre-filled with all the IDs; set the fraud type, shared link,
    and your name, then **Record pattern**. The console then jumps to the Pattern
    queue where the new candidate shows as *pending*. (Use this to flag a ring in
    bulk; use the per-card **⚑ Flag for LOE** to flag a single open application.)
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
- **"Already flagged?" banner** — when you open a flagged application, the console
  checks whether its **IP cluster** has already been flagged in a previous session
  (a soft match on the shared-IP link). If so, an amber banner names the earlier
  pattern(s) and whether they're already in LOE exposure. It's a **heuristic, not a
  block** — open the **◎ 3D identity ring** to confirm it's the same ring before
  re-flagging, so the same cluster isn't added twice. Cross-check under **Pattern
  queue → Flagged history**.

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
- **Flagged history** (bottom panel) — the persistent record of **every** ring
  flagged for LOE across all sessions (survives restarts), with a state badge
  (pending / promoted / rejected), an **"in LOE exposure"** tag + cluster id once
  promoted, its members, and who flagged it when. This is the store the
  "already flagged?" banner matches against — use it to verify before re-adding.
  - **Delete** — tick rows (or **Select all**) and click **✕ Delete selected** to
    remove flagged patterns from the history (cleanup of mistaken or test flags).
    This deletes the **record only**: if a pattern was already **promoted**, its
    ring may already be in the topology-exposure set and the current checkpoint —
    deleting the record does **not** un-train the model or remove the exposure
    cluster (that needs a rebuild/retrain). The confirm dialog warns you when any
    selected pattern is promoted.

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

   A blue **"What your CSV needs"** note at the top of Intake lists the required
   raw columns (incl. the identity fields — `ip_address`, `mobile_no`,
   `father_name`, `mother_name`, `permanent_pincode` — that build the 3D ring) and
   reassures that **the system does its own feature engineering** (raw → 44 model
   features); you supply raw columns only. **Download sample CSV** there gives a
   ready-to-fill file (`frontend/sample_cohort.csv`) with fresh IDs and a planted
   shared-IP ring.

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
   It also **persists a cohort bundle** so the cohort becomes reviewable in the
   Review queue's **Dataset** dropdown (with working 3D rings) — see §3.
3. **Decide** (human-gated) — after reviewing the evaluation, choose:
   **Log "no action"**, **Merge + incremental update**, or **Merge + full
   retrain**. The merge/retrain options permanently add the cohort to the data
   and dispatch a job (you must confirm the warning). A separate row lets you
   **retrain on current data only** (no new cohort) — incremental or full.
4. **Watch** — the dispatched job's id auto-fills here; status polls
   automatically, or paste any job id and **Poll status**.

### 5a. Committing new data to the model — step by step

Use this whenever you have a new batch of applications (a cohort CSV) that
should actually become part of what the model is trained/scored on, not just
previewed. This is the **Merge** path through the deployment loop above.

1. **Intake** — Admin → Intake → pick **"New cohort to score"** → drop the CSV
   (full 136-column raw schema, fresh `application_id`s). Fix any schema
   mismatch before continuing.
2. **Evaluate** — click **Evaluate**. This scores the cohort read-only
   (pre-fusion `hybrid_anomaly_score`) and reports a drift p-value; it also
   persists a cohort bundle so you can review it in the Review queue's
   **Dataset** switcher (§3) before deciding anything.
3. **Review the cohort (recommended)** — switch to the cohort in the Review
   queue, sanity-check the flagged applications, rings, and signal drivers.
   Nothing is committed yet — this step is still read-only.
4. **Decide** — Admin → Decide, then pick one:
   - **Log "no action"** — discard; nothing is committed.
   - **Merge + incremental update** — permanently adds the cohort's rows to
     the base data and dispatches a human-gated incremental fine-tune
     (10 epochs, RGCN frozen). Use for routine batches.
   - **Merge + full retrain** — permanently adds the cohort and dispatches a
     full retrain. Use when the drift explanation (§5) recommends it, or
     after several incremental merges have accumulated.
   Either merge option requires confirming the on-screen warning — this is
   the point of no return; the rows become part of the scored population and
   the model's training data.
5. **Watch** — the job id auto-fills in step 4 of the deployment loop; status
   polls automatically. Wait for it to finish before treating the cohort as
   live.
6. **Verify** — Admin → status strip → **↻ Refresh** to confirm the scored-
   population count grew and a new entry appears in **Run history** with the
   checkpoint size/metrics for this merge. The merged applications now also
   show up in the primary Review queue (no longer under the cohort dataset
   switcher, since they're part of the primary population).

To commit a brand-new **fraud ring** (not a scoring cohort) instead, use
Intake → **"New fraud pattern (relational LOE)"** — see step 1's second bullet
above — or promote an already-flagged ring via **Pattern queue → Promote
selected patterns** (§4); both of those commit directly without a separate
Decide step.

### 5b. Triggering a full retrain straight from Postgres (no CSV) — added 2026-07-23

Once data is merged (§5a steps 1–4 above, by any route — cohort Merge, LOE
pattern Ingest/Promote), a full retrain can be dispatched via the API so it
reads every merged batch **directly from Postgres**, with no CSV file
involved at all — useful for scripted/portal-triggered retrains that don't
go through the console's Decide button.

There is no console button for this yet — it's API-only:

```bash
curl -X POST "http://localhost:8080/v3/training/full?data_source=postgres"
```

(swap `localhost:8080` for wherever the console is reachable; add
`&smoke_test=true` for a fast dry run first). This returns a `job_id` —
poll it the same way as any other training job (`GET
/v3/training/jobs/{job_id}`, or Admin → Watch, pasting the job id in).

**What this does and doesn't do:**
- It only changes **where the retrain reads its raw data from** — Postgres
  instead of `data/raw/data_for_ml_model.csv`. It does **not** merge,
  preprocess, or stage anything by itself. If nothing has been merged since
  the last file-based retrain, a `postgres`-sourced retrain sees the same
  population as the primary 15k (plus whatever cohorts/patterns are already
  merged) — it is not a way to skip Evaluate/Decide.
- Omit `data_source` (or pass `data_source=file`, the default) to keep
  today's behavior — reads the CSV, exactly as before. Nothing changes for
  existing callers.
- Every downstream pipeline step (graph build, training, EVT, fusion, XAI)
  is unaffected either way — the switch only touches the first two pipeline
  steps (feature engineering + graph build), and both paths write to the
  same canonical files those downstream steps already read.

### 5c. Getting data in without the console at all — CSV-free portal/ETL push (added 2026-07-23)

For a portal or ETL job that needs to hand off a batch of applications
without a person clicking through Intake, `POST /v3/monitoring/push-dataset`
does the Intake + Evaluate half of §5a automatically — **Decide/Merge/retrain
still require a separate, human-gated call**, same as every other path in
this runbook.

1. **The sender writes a raw-schema CSV** (same full 136-column contract as
   §6) to a path both `nic-api` and `nic-worker` can read — the `data/`
   folder already mounted into both containers (e.g. `data/uploads/`, same
   place the console's browser upload lands). This is a file-system drop,
   not an HTTP upload — no inline JSON row payload, since a multi-million-row
   JSON body doesn't hold up at scale; a file path does.
2. **Call the push endpoint:**
   ```bash
   curl -X POST "http://localhost:8080/v3/monitoring/push-dataset" \
     -H "Content-Type: application/json" \
     -d '{"dataset_path": "data/uploads/portal_batch.csv", "name": "portal_batch_2026_07_23"}'
   ```
   This runs the same schema check as a console upload (missing/extra
   columns, duplicate IDs). If it fails, the response says so and nothing is
   staged — fix the file and re-push. If it passes, the batch is staged in
   Postgres immediately and a `job_id` comes back right away — the
   preprocessing (feature engineering + graph rebuild + scoring) runs in the
   background, it does not block the request.
3. **Poll the job** the same way as any other training job:
   `GET /v3/training/jobs/{job_id}`.
4. **Review it like any other cohort** once the job completes — it shows up
   in the Review queue's **Dataset** switcher (§3) exactly as if someone had
   uploaded it through Intake and clicked Evaluate.
5. **Decide is still separate and still manual** — nothing here merges the
   batch into the base data or retrains anything. Use Admin → Decide (§5a
   step 4) or `POST /v3/training/decision` to Merge + incremental/full
   retrain when you're ready, exactly as with a console-uploaded cohort.

**Run history**: every training run and checkpoint swap (newest first) — when,
type, cycle, metrics, checkpoint size. This is the audit trail.

**Install pretrained checkpoint (.pth)**: trained the model elsewhere (e.g. a
full GPU retrain on the laptop)? Upload the `.pth` here with a cycle label and
a source note. The server validates it **before anything changes** — the file
must contain `{model_state_dict, centroid, config}` with this deployment's
exact feature/edge dimensions, or it is rejected and the live model stays as
it was. On success the server backs up the current checkpoint, keeps a
versioned copy, and hot-swaps atomically; the job status shows in the panel.
(Same mechanism scripted: `POST /v3/training/upload-checkpoint`.)

**Rollback checkpoint**: paste a versioned checkpoint path (shown in run
history) and roll the live model back to it — the undo for a bad install.

---

## 6. What columns each data case needs

Every CSV that enters the system — whatever the case — must carry the **full
raw schema: all 136 columns of `data/raw/data_for_ml_model.csv`**, same
names. The intake check hard-fails and blocks the next steps if any column is
missing (extra columns are reported but tolerated). **Never pre-engineer
anything** — you supply raw columns; the system does its own feature
engineering (raw → 44 model features) and identity extraction. The
downloadable **sample CSV** (admin → Intake) is a ready-to-fill template with
every required column.

| Case | How it enters | Column requirements on top of the full raw schema |
|---|---|---|
| **Valid applications / new cohort to score** | Admin → Intake → "New cohort to score" → Evaluate | Fresh, unique `application_id`s (not already in the base data). Populate the identity fields honestly — they build the graph and 3D rings. |
| **Fraudulent ring / new LOE pattern** | Admin → Intake → "New fraud pattern (relational LOE)" | Fresh `application_id`s, and the ring members must actually **share the linking value** (same `ip_address`, or same `mobile_no`, etc.) so their real edges materialise; otherwise the system falls back to the relation you assert in the dialog. |
| **Confirmed fraud / false positives (individual labels)** | No CSV — reviewer card buttons or the batch Label/retrain dialog | Only an `application_id` that already exists in the scored population. Feature vectors are pulled automatically. |
| **Bulk / portal ingestion (V4-Scale, in progress)** | Direct to PostgreSQL staging | Same contract: raw-schema rows only into the staging batch; nothing derived until an admin triggers Evaluate/Merge. |

Columns that actually drive detection (get these right first):

- **Identity / graph (the 5 relations + rings):** `ip_address`, `mobile_no`,
  `father_name`, `mother_name`, `permanent_pincode` — plus `applicant_name`
  for the name-similarity signals.
- **Financial:** `annual_family_income`, `admission_fee`, `tution_fee`,
  `misc_fee`.
- **Temporal:** `date_of_birth`, `registered_date` (→ age at registration).
- **Context:** `permanent_district_id`, `domicile_state_id` (income
  rank/deviation), `c_institution_id` (institute concentration), `gender`,
  `rural_urban`, and the boolean flags (`disability_flag`, `orphan_flag`,
  `hosteller`, `is_singlegirlchild`).
- **Ignored by the model but still required by the schema check:** the
  null/duplicate/audit columns (`updated_by`, `delete_*`, `state_id-2`, …)
  and `sanity` / `jwt` (never used — hard stop). Fill them with empty/0
  values if you have nothing; they just have to exist.

---

## 7. Quick reference

| I want to… | Where |
|---|---|
| Start everything | `docker compose up --build` |
| Open the console | `http://localhost:8080/` |
| Triage a flagged application | Review queue → click a row |
| Page through all flagged apps | Review queue → **← Prev / Next →** (50 per page) |
| Flag several apps as one ring | Review queue → tick rows (across pages) → **◈ Flag for LOE (selected)** |
| See an application's network in 3-D | Review queue → **◎ 3D identity ring** |
| Export one application (CSV + card + evidence) | Review queue → open a row → **⤓ Export** |
| Export all flagged applications | Review queue → **⤓ Export all flagged** (zip: `manifest.csv` + cards + evidence) |
| Confirm / clear a label | Buttons inside the reviewer card |
| Turn a pattern into training signal | Pattern queue → **Promote** |
| Score a new cohort | Admin → Intake → Evaluate |
| Commit new data to the model | Admin → Intake → Evaluate → Decide → Merge + incremental/full retrain (§5a) |
| Retrain the model | Admin → Decide |
| Full retrain reading straight from Postgres (no CSV) | `POST /v3/training/full?data_source=postgres` (§5b) |
| Push data in without the console (portal/ETL, no browser upload) | `POST /v3/monitoring/push-dataset` (§5c) |
| See model metrics / history | Admin → status strip + Run history |
| Install a GPU-laptop-trained model | Admin → Install pretrained checkpoint |
| Undo a bad checkpoint | Admin → Rollback |
| Stop everything | `docker compose down` |
