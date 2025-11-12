"""HeRoN Crafter Project package.

This module intentionally keeps imports lightweight so that importing
``src`` does not trigger heavy dependencies (torch, transformers, etc.).

Access submodules directly (e.g. ``from src.environment import make_crafter_env``)
to avoid loading large ML libraries at package import time.
"""

__version__ = "0.1.0"

# Note: Do NOT import heavy submodules here. Import them explicitly where needed.

__all__ = ["__version__"]
