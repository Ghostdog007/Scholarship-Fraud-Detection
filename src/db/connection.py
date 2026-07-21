"""Connection management for the NIC Postgres system of record.

Env vars (loaded from project-root .env if present):
    NIC_DB_HOST / NIC_DB_PORT / NIC_DB_NAME / NIC_DB_USER / NIC_DB_PASSWORD
"""

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_pool: ConnectionPool | None = None


def _load_env_file() -> None:
    """Populate os.environ from .env (existing env vars win)."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def conninfo() -> str:
    _load_env_file()
    host = os.environ.get("NIC_DB_HOST", "localhost")
    port = os.environ.get("NIC_DB_PORT", "5432")
    name = os.environ.get("NIC_DB_NAME", "nic_fraud")
    user = os.environ.get("NIC_DB_USER", "nic_app")
    password = os.environ.get("NIC_DB_PASSWORD", "")
    return f"host={host} port={port} dbname={name} user={user} password={password}"


def get_pool() -> ConnectionPool:
    """Process-wide pool. API replicas use this; batch jobs may prefer
    a single get_connection() instead."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo(), min_size=1, max_size=8, open=True)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_connection():
    """One transaction per with-block: commits on success, rolls back on error."""
    with psycopg.connect(conninfo()) as conn:
        yield conn
