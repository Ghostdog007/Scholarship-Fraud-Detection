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
        └───────┬────────┘               │
                │ shared data/models/outputs
        ┌───────▼────────┐               │
        │   nic-worker   │◄──────────────┘  (Celery, concurrency=1, singleton)
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

`docker-compose.override.yml` is auto-merged and bind-mounts `./frontend` into
nginx, so editing `index.html` / `app.js` / `style.css` shows up on refresh with
no rebuild.

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
shows "No suspicious applications").

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

# 2. Provide RWX storage (see §3), then:
kubectl apply -f deploy/k8s/nic-fraud.yaml

# 3. Console is at NodePort 30080:
#    http://<server-ip>:30080/
```

Seed the shared volume the same way as local: copy a completed run's
`data/`, `models/`, `outputs/` into the PVC (e.g. `kubectl cp` into the
`nic-api` pod, or pre-populate the hostPath dir), or trigger a full pipeline
from the admin tab once (slow on CPU — see §4).

---

## 3. The four things that make this "portable to a server", not just local

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
```
