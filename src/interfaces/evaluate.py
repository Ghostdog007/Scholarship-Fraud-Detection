"""
Stable entry point for the synthetic evaluation harness (v2 floors = pass bar).
Concrete implementation: src/evaluate_model_v3.py — import from here, not the
_v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.evaluate_model_v3 import (
    evaluate,
    evaluate_connected,
)
