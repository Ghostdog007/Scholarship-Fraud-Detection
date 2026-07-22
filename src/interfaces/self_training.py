"""
Stable entry point for the human-gated self-training loop (hard stop 5:
Round 0 classifier-agreement is code-enforced OFF; no round advances
automatically).
Concrete implementation: src/self_training_loop_v3.py — import from here,
not the _v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.self_training_loop_v3 import (
    run_self_training,
)
