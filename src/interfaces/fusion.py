"""
Stable entry point for the LOCKED score-level fusion:
final_risk = minmax(max(minmax(subspace), minmax(dense_relational), minmax(hybrid)))
(unweighted max since 2026-07-22 — see docs/AGENTS.md §1). LightGBM and the
prior weighted-sum fusion are retired; do not reintroduce either.
Concrete implementation: src/fusion_classifier_v3.py — import from here, not
the _v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.fusion_classifier_v3 import (
    score_level_fusion,
    run_fusion,
)
