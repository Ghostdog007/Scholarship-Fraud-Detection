"""
Stable entry point for the FRAUDAR-style dense-block detector, gated to
shares_mobile + shares_ip (IP-priority-weighted max; pincode dropped
2026-07-22 — see docs/AGENTS.md §1).
Concrete implementation: src/dense_block_detector_v3.py — import from here,
not the _v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.dense_block_detector_v3 import (
    dense_block_scores,
    run_dense_block,
)
