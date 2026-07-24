"""
src/interfaces/ — stable per-layer entry points.

Each module here re-exports the public functions of exactly one concrete
implementation file (the "_v3" modules, or their unversioned equivalents),
matching the module-ownership table in docs/AGENTS.md §3.

Why this exists: consumers (main_v3.py, src/api/*, other detectors) should
import from src.interfaces.<layer> rather than reaching into the concrete
module directly. If a concrete file is ever renamed or split, only that one
import line here changes — not every call site across the codebase.

This package is purely additive: it does not replace or modify the
concrete _v3 modules, and existing call sites into them keep working
unchanged. New code should prefer these interfaces; migrating old call
sites over is optional and can happen gradually.

Layers without an interface module here (src/db/, src/api/, checkpoint_manager.py,
retraining_orchestrator.py, model_registry.py) already have unversioned names
and don't need one.
"""
