"""
Backward-compatible entry point — delegates to ``memory_perf_bench.py``.

Prefer: ``python benchmarks/memory_perf_bench.py`` (ingest + recall, miss + hit).
"""

from __future__ import annotations

import sys
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
_root = _bench_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_bench_dir) not in sys.path:
    sys.path.insert(0, str(_bench_dir))

from memory_perf_bench import main  # noqa: E402

if __name__ == "__main__":
    main()
