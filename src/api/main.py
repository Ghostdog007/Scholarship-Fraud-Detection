"""
FastAPI inference and supervisor server — ADR-005.

20 endpoints across 6 groups:
  /health, /ready                        health
  /v3/supervisor/*                       supervisor feedback
  /v3/training/*                         async training jobs (ADR-006)
  /v3/monitoring/*                       drift + fraud store
  /v3/model/*                            checkpoint info + rollback (ADR-008)

Structured logging via structlog — ADR-007.

Authentication: OQ-2 is UNRESOLVED. No auth layer is implemented.
Deploy behind VPN until the project lead decides on internal-only vs
public-facing (see AGENTS.md Appendix F, OQ-2).
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.handlers import model, monitoring, supervisor, training

# ── Structured logging (ADR-007) ──────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt_exists = Path("models/hybrid_graphmcm_v3.pth").exists()
    log.info("api.startup", checkpoint_exists=ckpt_exists, version="3.0.0")
    yield
    log.info("api.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NIC Fraud Detection V3 — Supervisor & Inference API",
    description=(
        "REST API for supervisor feedback, async training jobs, "
        "model monitoring, and checkpoint management. "
        "See AGENTS.md ADR-005 for endpoint contract."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# Restrict origins once OQ-2 (auth / VPN decision) is resolved
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"], summary="Liveness — is the server up?")
def health():
    ckpt = Path("models/hybrid_graphmcm_v3.pth")
    return {"status": "ok", "checkpoint_exists": ckpt.exists()}


@app.get("/ready", tags=["health"], summary="Readiness — do scored outputs exist?")
def ready():
    scores = Path("outputs/risk_scores_v3.csv")
    ready  = scores.exists()
    return {"status": "ready" if ready else "not_ready", "scores_exist": ready}


# ── Route registration ────────────────────────────────────────────────────────
app.include_router(supervisor.router,  prefix="/v3/supervisor",  tags=["supervisor"])
app.include_router(training.router,    prefix="/v3/training",    tags=["training"])
app.include_router(monitoring.router,  prefix="/v3/monitoring",  tags=["monitoring"])
app.include_router(model.router,       prefix="/v3/model",       tags=["model"])
