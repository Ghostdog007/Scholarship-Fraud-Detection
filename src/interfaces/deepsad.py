"""
Stable entry point for Deep SAD (center-distance, XAI-only — NOT a fusion
input; see docs/AGENTS.md §1 for the 2026-07-22 rejection of a 4th fusion
input). Separate encoder/checkpoint from the hybrid detector.
Concrete implementation: src/deepsad_detector_v3.py — import from here, not
the _v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.deepsad_detector_v3 import (
    train,
    run_deepsad,
)
