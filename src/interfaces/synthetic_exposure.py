"""
Stable entry point for programmatic LOE (level-of-exposure) synthetic exposure
building (hard stop 7: never CTGAN/TVAE/copula).
Concrete implementation: src/synthetic_exposure_builder_v3.py — import from
here, not the _v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.synthetic_exposure_builder_v3 import (
    build_exposure_set,
    build_topology_exposure,
)
