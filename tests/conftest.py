"""
Mock gliner and torch before any app module is imported.
Prevents torch's BLAS FPE check from crashing on Windows during unit tests.
"""
import sys
from unittest.mock import MagicMock

for mod in ("torch", "gliner", "gliner.model"):
    sys.modules.setdefault(mod, MagicMock())

_gliner_stub = sys.modules["gliner"]
_gliner_stub.GLiNER = MagicMock()
