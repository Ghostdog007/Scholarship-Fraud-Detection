"""
Append-only audit trail for drift-review decisions (evaluate -> human decision -> action).
"""
import json
import time
from pathlib import Path
from typing import Any

AUDIT_LOG = Path("outputs/drift_audit_log.json")


def append_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(AUDIT_LOG.read_text()) if AUDIT_LOG.exists() else []
    full_record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    records.append(full_record)
    AUDIT_LOG.write_text(json.dumps(records, indent=2))
    return full_record
