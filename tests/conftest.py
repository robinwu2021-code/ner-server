"""
Mock gliner and torch before any app module is imported.
This prevents torch's BLAS FPE check from crashing on Windows during tests.
"""
import sys
from unittest.mock import MagicMock

# Stub out gliner and its torch dependency so the app can be imported safely
for mod in ("torch", "gliner", "gliner.model"):
    sys.modules.setdefault(mod, MagicMock())

_gliner_stub = sys.modules["gliner"]
_gliner_stub.GLiNER = MagicMock()
