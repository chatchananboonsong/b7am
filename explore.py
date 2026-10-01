#!/usr/bin/env python3
"""Shim for explore.py forwarding to src/explorer.py."""
import importlib.util
from pathlib import Path
import sys

_src_file = Path(__file__).resolve().parent / "src" / "explorer.py"
_spec = importlib.util.spec_from_file_location("src.explorer", str(_src_file))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["src.explorer"] = _mod
_spec.loader.exec_module(_mod)

# Export all symbols from src.explorer
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("__")})

if __name__ == "__main__":
    sys.exit(_mod.main())
