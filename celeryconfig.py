"""
Celery configuration — ADR-006.

Broker: Redis. URL read from env so the same config works locally
(redis://localhost:6379/0) and inside Docker (redis://redis:6379/0).

Worker concurrency is fixed at 1. Training jobs write to fixed output
paths (outputs/*.csv, models/*.pth). Scaling nic-worker above 1 replica
causes concurrent runs to overwrite each other's intermediates.
See AGENTS.md hard stop #16.
"""
import os

broker_url      = os.environ.get("CELERY_BROKER_URL",    "redis://localhost:6379/0")
result_backend  = os.environ.get("CELERY_RESULT_BACKEND", broker_url)

# Hard stop #16: concurrency=1 — training must not run concurrently
worker_concurrency        = 1
worker_prefetch_multiplier = 1

task_serializer   = "json"
result_serializer = "json"
accept_content    = ["json"]
timezone          = "UTC"
enable_utc        = True

# Keep results 24 h so GET /v3/training/jobs/{job_id} can be polled after a long run
result_expires = 86400

task_default_queue = "nic_fraud"
