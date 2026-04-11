"""
End-to-end: Memory.recall with vs without sync_contradictions (batched NLI on hit pairs).

Requires sentence-transformers and models on disk (same as normal recall).
Run from repo root: python benchmarks/recall_sync_contradictions_latency.py
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

from verimem import Memory


def main() -> None:
    texts = [
        f"Policy snippet {i}: SLA is {99.0 + (i % 5) * 0.1}% for service region {i % 7}."
        for i in range(12)
    ]

    # ignore_cleanup_errors: Windows may hold contradiction_cache.db briefly after close.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = str(Path(td) / "store")
        mem = Memory(path=path)
        mem.warm_retrieval_models()
        for t in texts:
            mem.remember(t, source="bench")

        q = "What is the SLA policy?"
        top_k = 10
        reps = 5

        def run(sync: bool) -> list[float]:
            ms: list[float] = []
            for _ in range(reps):
                t0 = time.perf_counter()
                mem.recall(
                    q,
                    top_k=top_k,
                    mode="raw",
                    sync_contradictions=sync,
                )
                ms.append((time.perf_counter() - t0) * 1000.0)
            return ms

        # Warmup
        mem.recall(q, top_k=top_k, mode="raw", sync_contradictions=False)
        mem.recall(q, top_k=top_k, mode="raw", sync_contradictions=True)

        base = run(False)
        sync = run(True)
        overhead = [s - b for s, b in zip(sync, base)]

        def _line(name: str, xs: list[float]) -> str:
            return (
                f"{name}: mean={statistics.mean(xs):.1f}ms  "
                f"median={statistics.median(xs):.1f}ms  "
                f"min={min(xs):.1f}ms  max={max(xs):.1f}ms"
            )

        print(f"store={path}  top_k={top_k}  mode=raw  reps={reps}")
        print(_line("recall (async NLI path, default)", base))
        print(_line("recall + sync_contradictions=True", sync))
        print(_line("estimated sync NLI overhead (sync - base)", overhead))
        print(f"mean extra latency from sync contradictions: {statistics.mean(overhead):.1f}ms")


if __name__ == "__main__":
    main()
