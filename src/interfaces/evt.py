"""
Stable entry point for EVT/GPD tail thresholds — the only thresholds allowed
in this system (hard stop 1).
Concrete implementation: src/evt_scorer_v3.py — import from here, not the
_v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.evt_scorer_v3 import (
    run_evt,
)
