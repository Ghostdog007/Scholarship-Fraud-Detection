# Deploying the NIC Fraud Review Console

One stack, two targets, **one nginx front door**. The browser only ever talks
to nginx; nginx serves the `frontend/` UI and reverse-proxies the API. Same
image, same nginx config, local and on the server — so "works on my machine"
and "works on the server" stop diverging.

```
        browser (local or remote PC)
                │  http
                ▼
        ┌───────────────┐   /            → static frontend/
        │     nginx      │   /v3, /health,/docs → proxy
        └───────┬────────┘
                │ (service name: nic-api)
                ▼
        ┌───────────────┐        ┌──────────────┐
        │   nic-api      │◄──────►│    redis     │◄──── broker/results
        │  (FastAPI)     │        └──────┬───────┘
        └───┬───────┬────┘               │
            │       │ shared data/models/outputs
            │  ┌────▼───────────┐        │
            │  │   nic-worker   │◄───────┘  (Celery, concurrency=1, singleton)
            │  └────────────────┘
            ▼
    ┌────────────────┐
    │    postgres    │◄── populated on every startup by db-init (one-shot,
    │ (system of      │    runs before nic-api/nic-worker are allowed to start)
    │  record)        │
    └────────────────┘
```

Why nginx and not "FastAPI serves the static files": it keeps the API a pure
API, collapses everything to one origin (no CORS to reason about, remote access
just works), and gives one clean place to add auth later (OQ-2) without touching
Python. The draft that proposed `app.frontend()` was based on a FastAPI method
that does not exist — nginx is the real, portable equivalent.

---

## 1. Local — Docker Compose

```bash
# From the repo root:
docker compose up --build
```

- Console:  **http://localhost:8080/**
- Swagger:  http://localhost:8080/docs  (proxied through nginx)
- API direct (debugging, bypasses nginx): http://localhost:8000/health
- Postgres direct (debugging, e.g. `psql`): `localhost:5433` — **not 5432**,
  chosen to avoid colliding with a locally-installed PostgreSQL.

This brings up **six** containers: `postgres`, `db-init` (one-shot — applies
the schema and ingests/replays data into Postgres, then exits; `nic-api`/
`nic-worker` wait for it via `depends_on: condition: service_completed_
successfully`), `redis`, `nic-api`, `nic-worker`, `nginx`. `db-init` reruns
(idempotently) on every `up`, so Postgres stays in sync with whatever is
currently in `data/`/`outputs/` — you don't need to seed it separately from
the file-based data described below.

`docker-compose.override.yml` is auto-merged and bind-mounts `./frontend` into
nginx, so editing `index.html` / `app.js` / `style.css` shows up on refresh with
no rebuild. **Python (`src/`) changes are NOT bind-mounted** — they're baked into
the image, so after editing backend code run
`docker compose up -d --build nic-api nic-worker`.

> **502 after rebuilding `nic-api`?** `up --build` gives the API container a new
> IP, but the already-running nginx has the old one cached (it resolves the
> upstream at config-load). Fix: `docker compose restart nginx`. (Health via
> `:8000` stays green while `:8080` 502s — that's the tell.) This bit us during
> V4-Scale live testing — always restart nginx after rebuilding `nic-api`/
> `nic-worker` alone.

Prod-equivalent run (no override, frontend baked into the image — this is what
the server runs):

```bash
docker compose -f docker-compose.yml up --build
```

**Seeding data:** the console reads a completed pipeline's outputs
(`outputs/risk_scores_v3.csv`, `outputs/top_suspicious_v3.tsv`,
`outputs/explanation_cards_v3.json`, `models/hybrid_graphmcm_v3.pth`). The
`./data`, `./models`, `./outputs` host folders are mounted in, so a prior local
run populates the UI. Empty folders → empty queue (endpoints 404 cleanly, the UI
shows "No suspicious applications"). **Postgres seeding is automatic** —
`db-init` reads these same host-mounted folders on every startup and mirrors
them into the database (primary batch + confirmed-fraud/pattern/run-history
stores), so an empty `outputs/` also means an empty Postgres, not a stale one.

---

## 2. Server — Kubernetes (16 vCPU / 64 GB / Ubuntu 22.04 / no GPU)

Everything is in `deploy/k8s/nic-fraud.yaml`, a 1:1 mirror of the compose stack.

```bash
# 1. Build both images and make them visible to the cluster.
docker build -f Dockerfile               -t nic-fraud-api:latest   .
docker build -f deploy/nginx/Dockerfile  -t nic-fraud-nginx:latest .
# k3s:      sudo k3s ctr images import <(docker save nic-fraud-api:latest); (repeat for nginx)
# microk8s: docker save nic-fraud-api:latest | microk8s ctr image import - ; (repeat)
# or docker push both to a registry and edit the image: fields.

# 2. Create the Postgres credentials Secret FIRST — the postgres Deployment
#    in nic-fraud.yaml reads it on startup and will CrashLoop without it:
kubectl -n nic-fraud create secret generic nic-db \
  --from-literal=NIC_DB_NAME=nic_fraud \
  --from-literal=NIC_DB_USER=nic_app \
  --from-literal=NIC_DB_PASSWORD='<a strong password>'

# 3. Provide RWX storage for data/models/outputs (see §3), then:
kubectl apply -f deploy/k8s/nic-fraud.yaml

# 4. Apply the Postgres schema (one-off, or after any schema.sql change):
kubectl -n nic-fraud exec deploy/nic-api -- python -m src.db.migrate

# 5. Console is at NodePort 30080:
#    http://<server-ip>:30080/
```

Seed the shared volume the same way as local: copy a completed run's
`data/`, `models/`, `outputs/` into the PVC (e.g. `kubectl cp` into the
`nic-api` pod, or pre-populate the hostPath dir), or trigger a full pipeline
from the admin tab once (slow on CPU — see §4). **Postgres is not
auto-seeded in k8s** the way `db-init` seeds it in Compose — after step 4,
run `kubectl -n nic-api exec deploy/nic-api -- python -m src.db.ingest` once
to mirror the seeded files into Postgres (or wire a k8s Job equivalent to
`db-init` if you want this automatic — not yet done for the k8s manifest).

---

## 3. The five things that make this "portable to a server", not just local

1. **Shared RWX storage.** `nic-api` reads `data/models/outputs`; `nic-worker`
   **writes** them. Locally that's a host bind mount (trivially shared). In k8s
   it must be a **ReadWriteMany** PersistentVolume so both pods see the same
   files. The manifest requests RWX; supply an NFS/CephFS StorageClass, or use
   the commented single-node **hostPath PV** block at the bottom of the manifest
   (`/srv/nic-fraud`) if you have no RWX provisioner. This is the single biggest
   port gotcha — get it wrong and the worker's retrain output never reaches the
   API.

2. **The worker is a singleton and must stay one.** `replicas: 1` **plus**
   `strategy: Recreate`. Training writes fixed output paths; a RollingUpdate
   would momentarily run two worker pods and corrupt intermediates (AGENTS.md
   hard stop #16). `Recreate` kills the old pod before starting the new one.

3. **CPU-only, no GPU.** Both images are CPU-only already, so they run as-is on
   this server. But that means the admin tab's **"Trigger full pipeline" /
   incremental training runs on CPU and is slow.** The intended production path
   is: train on a GPU box elsewhere, ship the `.pth` in via
   **upload-checkpoint** or **dvc pull** (both already wired). Treat the
   server-side training buttons as a fallback, not the norm.

4. **Same service name both places.** nginx proxies to `http://nic-api:8000`.
   Keep the k8s Service named `nic-api` (it is) and the nginx config never
   changes between compose and k8s.

5. **Postgres is a separate volume from `data/models/outputs`.** The shared
   RWX volume in point 1 is for files; Postgres gets its own **RWO**
   (ReadWriteOnce) PersistentVolumeClaim (`postgres-data`, 100Gi) since only
   one pod ever needs it. Don't conflate the two when sizing storage. Unlike
   Compose (where `db-init` auto-seeds Postgres on every `up`), the k8s
   manifest has **no auto-seed step** — `nic-api`/`nic-worker` will start
   fine without it (the app falls back to files if Postgres has no data or
   is unreachable — the same graceful-degradation path proven during local
   testing), but you must run `src.db.migrate` + `src.db.ingest` manually
   (§2 steps 4–5) to get Postgres-backed reads actually serving data.

---

## 4. Open items — deliberately NOT resolved here

- **OQ-2 (auth).** This build ships with **no authentication** and CORS is still
  `allow_origins=["*"]` in `src/api/main.py`. The admin tab triggers real
  retraining and file mutation and is only *visually* separated. Because the
  server is **accessed remotely**, do not expose NodePort 30080 to an untrusted
  network — keep it behind the VPN / a firewall / an nginx `auth_basic` block
  until the project lead decides OQ-2. nginx is the right place to add that gate
  (no Python change): add `auth_basic` + an `htpasswd` file to the `/v3` location,
  or front it with an auth proxy.
- **`patterns/confirm` subgraph shape.** The "Flag for LOE" button asks the
  reviewer for node IDs + edge type via a prompt rather than deriving them from
  the rendered ring view. That's a stopgap UI, not a resolution of the subgraph
  schema question — a real subgraph builder is still an open design task.

Do not resolve either autonomously; flag and stop if a task seems to require it.

---

## 5. Verification checklist

```bash
# Local
curl -s http://localhost:8080/            | head -5           # index.html
curl -s http://localhost:8080/health                          # {"status":"ok",...} via nginx
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/nope.js   # 404 (missing asset)

# job_id fix — promote a real pending pattern, then poll the RETURNED id:
curl -s -X POST http://localhost:8080/v3/supervisor/patterns/promote \
  -H "Content-Type: application/json" \
  -d '{"pattern_ids":["<a real pending pattern id>"],"smoke_test":true}'
# → note job_id, then:
curl -s http://localhost:8080/v3/training/jobs/<that job_id>
# must return a real status (pending/running/complete), NOT a perpetual "pending"
# for an id Celery never heard of.

# Server
kubectl -n nic-fraud get pods            # all Running; nic-worker exactly 1
curl -s http://<server-ip>:30080/health  # via the nginx NodePort

# Postgres — confirm reads are actually PG-backed, not silently on the file fallback:
docker compose logs nic-api | grep '"source": "pg"'   # should appear after hitting fraud-store-summary
# or in k8s:
kubectl -n nic-fraud logs deploy/nic-api | grep '"source": "pg"'
```

---

## 6. PostgreSQL — the system of record (V4-Scale, all 5 migration steps done)

Full architecture, the real schema, and the scaling story:
`docs/TECHNICAL_REFERENCE_AND_SCALING.md` §11. This section is the
deploy-facing summary.

**Local dev without Docker** (running `src/` directly against a native
PostgreSQL install): create the `nic_fraud` database + `nic_app` role, put
credentials in a git-ignored `.env` at the project root
(`NIC_DB_HOST/PORT/NAME/USER/PASSWORD`), then:

```bash
python -m src.db.migrate     # idempotent; applies deploy/postgres/schema.sql + migrations/
python -m src.db.ingest      # mirrors the primary batch's files into Postgres
```

**Compose:** `postgres` (postgres:18, named volume `postgres-data`, host port
**5433** — chosen to avoid colliding with a locally-installed PostgreSQL) plus
a one-shot `db-init` service that runs `python -m src.db.bootstrap` (schema +
ingest + JSON-store replay) on every `docker compose up`, before `nic-api`/
`nic-worker` are allowed to start. You don't need to run `migrate`/`ingest`
by hand in Compose — `db-init` does it automatically, every time.

> **postgres:18 gotcha, hit during live testing:** the image changed its
> expected volume-mount convention in v18 — mount at `/var/lib/postgresql`,
> **not** `/var/lib/postgresql/data` (the old convention). The wrong mount
> crashes the container on startup with a clear error naming the mismatch;
> `docker-compose.yml` already has this right, but if you're hand-writing a
> similar manifest elsewhere, don't copy the pre-v18 convention.

**K8s:** PVC (`postgres-data`, 100Gi, RWO) + Deployment + Service in
`deploy/k8s/nic-fraud.yaml`. Create the `nic-db` Secret **before** applying
the manifest (§2 step 2) — `postgres` reads it on startup and CrashLoops
without it; `nic-api`/`nic-worker` also read it (added alongside this doc
update — they previously had no `NIC_DB_*` env vars wired at all, which
would have made every Postgres-backed read silently fail over to files with
no indication anything was wrong). Unlike Compose, there is **no `db-init`
equivalent in k8s yet** — run `migrate` + `ingest` manually after first
apply (§2 steps 4–5).

**All SQL access goes through `src/db/`** — see `docs/AGENTS.md` hard stop 14.
No inline SQL anywhere else in the codebase.

**Read path:** the review queue, status tiles, 3D rings, and ego-graphs are
served from Postgres by default (`NIC_READS_FROM_PG=1`); set it to `0` to
force the file path, or rely on the automatic fallback — any Postgres query
failure degrades to files with a logged warning, never a hard error.

**Write path:** confirmed-fraud/pattern/run-history writes go to both the
JSON file (authoritative) and Postgres (best-effort) during the migration.
CSV intake lands in a Postgres staging batch (raw rows only) — Evaluate
populates derived tables, Decide→Merge makes it permanent. Full lifecycle:
`docs/TECHNICAL_REFERENCE_AND_SCALING.md` §11.4.

**External GPU-trained checkpoints** can be installed without any in-cluster
training — see the admin console's "Install pretrained checkpoint" widget,
or `docs/TECHNICAL_REFERENCE_AND_SCALING.md` §11.5 for the full validation
mechanism.

**K_CAP profiling** (the hub-cap threshold for the identity graph at scale)
has a reusable query: `python -m scripts.profile_group_sizes` — rerun it
against the real ingest before choosing a production value (open decision,
`TECHNICAL_REFERENCE_AND_SCALING.md` §15 #1).
