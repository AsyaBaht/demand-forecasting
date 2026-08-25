"""Weekly retail demand forecasting on Iowa liquor sales.

Author: Anastasiia Bakhtoiarova
"""
import sys as _sys
from pathlib import Path as _Path

# config/ lives at the repo root, outside the installed package (it's meant
# to be edited directly, not shipped) — make it importable as `config.settings`
# regardless of the caller's working directory.
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
