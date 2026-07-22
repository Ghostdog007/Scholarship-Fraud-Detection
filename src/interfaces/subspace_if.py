"""
Stable entry point for the per-group Isolation Forest (financial/identity/
network subspaces) — the locked fusion's backbone detector.
Concrete implementation: src/subspace_if_v3.py — import from here, not the
_v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.subspace_if_v3 import (
    compute_subspace_if_scores,
    run_subspace_if,
)
