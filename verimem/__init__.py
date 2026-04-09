"""VeriMem — Verified AI memory. No API key required."""

from .cli import main
from .memory import DEFAULT_RETRIEVAL_MODE, Memory, RETRIEVAL_MODES
from .version import __version__

__all__ = [
    "DEFAULT_RETRIEVAL_MODE",
    "Memory",
    "RETRIEVAL_MODES",
    "__version__",
    "main",
]
