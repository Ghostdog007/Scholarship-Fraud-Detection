"""
Stable entry point for the 5-relation identity graph (shares_mobile, shares_ip,
shares_father_name, shares_mother_name, shares_pincode).
Concrete implementation: src/graph_builder_v3.py — import from here, not the
_v3 module directly. See docs/AGENTS.md §3 (module ownership).
"""
from src.graph_builder_v3 import (
    build_graph,
    build_graph_pg,
    derive_group_ceiling,
)
