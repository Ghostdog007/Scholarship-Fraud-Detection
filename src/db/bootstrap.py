"""One-shot startup bootstrap for the docker-compose / k8s stack.

Runs schema migration, primary-batch ingest, and store-replay in order so
Postgres is populated and authoritative-for-reads the moment the API comes
up — not just tolerated as an optional fallback target. Idempotent: safe to
run on every container start (migrate is IF NOT EXISTS; ingest wipes and
reloads the primary batch; replay upserts every JSON-store record).

    python -m src.db.bootstrap
"""

from src.db.migrate import apply_migrations, apply_schema


def main() -> None:
    print("[db.bootstrap] applying schema ...")
    apply_schema()
    apply_migrations()

    print("[db.bootstrap] ingesting primary batch ...")
    from src.db.ingest import ingest_primary
    try:
        ingest_primary()
    except FileNotFoundError as e:
        print(f"[db.bootstrap] WARNING: primary ingest skipped — {e}")

    print("[db.bootstrap] replaying JSON stores (confirmed fraud, patterns, runs) ...")
    from src.db.stores import replay_all
    counts = replay_all()
    print(f"[db.bootstrap] store replay: {counts}")

    print("[db.bootstrap] done — Postgres is schema-current and data-populated.")


if __name__ == "__main__":
    main()
