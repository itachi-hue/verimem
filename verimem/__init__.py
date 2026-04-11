"""VeriMem — Verified AI memory. No API key required."""

from .cli import main
from .memory import DEFAULT_RETRIEVAL_MODE, Memory, RETRIEVAL_MODES
from .recall import RetrievalUncertainty, compute_retrieval_uncertainty
from .version import __version__

__all__ = [
    "DEFAULT_RETRIEVAL_MODE",
    "Memory",
    "RETRIEVAL_MODES",
    "RetrievalUncertainty",
    "compute_retrieval_uncertainty",
    "__version__",
    "main",
]
