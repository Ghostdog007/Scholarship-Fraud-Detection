"""
Stable entry point for XAI evidence cards (JSON + HTML). Spans two concrete
files — src/xai_layer_v3.py (JSON evidence: fusion contributions, population
stats, per-app cards) and src/xai_card_html_v3.py (HTML rendering: cards,
3D identity rings) — both re-exported here since docs/AGENTS.md §3 lists them
as one "XAI" row.
Import from here, not the _v3 modules directly.
"""
from src.xai_layer_v3 import (
    build_fusion_contributions,
    build_population_stats,
    get_xai_context,
    build_card_for_app,
    run_xai,
)
from src.xai_card_html_v3 import (
    build_ring_html,
    build_staged_card_html,
    generate_ego_rings,
    build_card_html,
    render_cards,
)
