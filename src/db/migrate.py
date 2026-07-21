"""Schema migration runner.

Applies deploy/postgres/schema.sql (idempotent base schema), then any
deploy/postgres/migrations/NNN_*.sql not yet recorded in schema_migrations,
in ascending order. Run as:

    .\\.venv\\Scripts\\python.exe -m src.db.migrate
"""

import re
from pathlib import Path

from src.db.connection import get_connection

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_SQL = ROOT / "deploy" / "postgres" / "schema.sql"
MIGRATIONS_DIR = ROOT / "deploy" / "postgres" / "migrations"
_MIGRATION_RE = re.compile(r"^(\d{3})_.*\.sql$")


def apply_schema() -> None:
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    with get_connection() as conn:
        conn.execute(sql)
    print(f"[db.migrate] base schema applied: {SCHEMA_SQL}")


def apply_migrations() -> None:
    if not MIGRATIONS_DIR.exists():
        print("[db.migrate] no migrations directory; base schema only")
        return
    files = sorted(
        p for p in MIGRATIONS_DIR.iterdir() if _MIGRATION_RE.match(p.name)
    )
    with get_connection() as conn:
        applied = {
            row[0]
            for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        for path in files:
            version = int(_MIGRATION_RE.match(path.name).group(1))
            if version in applied:
                continue
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version, filename) VALUES (%s, %s)",
                (version, path.name),
            )
            print(f"[db.migrate] applied migration {path.name}")
    print("[db.migrate] migrations up to date")


if __name__ == "__main__":
    apply_schema()
    apply_migrations()
