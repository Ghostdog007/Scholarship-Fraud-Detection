"""
src/db — the ONLY module that talks SQL (AGENTS.md hard stop 14).

Everything else (API handlers, stores, model modules in later migration
steps) calls typed functions from this package. No inline SQL anywhere else.

Connection parameters come from the environment (NIC_DB_*), loaded from a
git-ignored .env at the project root if present. Never hardcode credentials.
"""

from src.db.connection import get_connection, get_pool, close_pool

__all__ = ["get_connection", "get_pool", "close_pool"]
