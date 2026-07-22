"""
Stable entry point for feature engineering (44 numeric features, MinMax-scaled).
Concrete implementation: src/tabular_feature_engine_v3.py — import from here,
not the _v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.tabular_feature_engine_v3 import (
    apply_stored_scaling,
    build_base,
    build_base_pg,
    add_degree_features,
)
