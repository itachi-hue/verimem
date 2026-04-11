"""
VeriMem ingest + recall performance (default recall mode follows ``Memory()``; use ``--both`` for raw+rerank).

Sections
--------
1. **Ingest** — per ``remember()`` on **unique** text only (embedding **miss** path each time).
2. **Recall** — after filling a corpus: each timed call uses a **new** query string (query-embedding **miss**).

Uses ``time.perf_counter``. NLI is primed before recall timing so lazy model load does not
dominate the numbers.

Huge corpora: raise ``--chunks`` for index size; raise ``--ingest-n`` for more cold writes.

Example (PowerShell)::

  python benchmarks/memory_perf_bench.py --chunks 2000 --ingest-n 100 --queries 50 --both
  python benchmarks/memory_perf_bench.py --device cuda --chunks 500 --queries 40 --both

``--device`` is ``auto`` (CUDA if PyTorch sees a GPU), ``cpu``, or ``cuda``/``gpu``.
For GPU cross-encoder ONNX, use a CUDA build of PyTorch and ``pip install onnxruntime-gpu``
(avoid installing both ``onnxruntime`` and ``onnxruntime-gpu`` in the same env).
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verimem import Memory  # noqa: E402
from verimem.memory import DEFAULT_RETRIEVAL_MODE  # noqa: E402


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _make_corpus(n: int, seed: int) -> List[str]:
    rng = __import__("random").Random(seed)
    words = (
        "alpha beta gamma delta service database cache auth token session "
        "user request response latency graph memory hybrid retrieval chunk"
    ).split()
    out: List[str] = []
    for i in range(n):
        parts = [words[rng.randrange(len(words))] for _ in range(12 + rng.randrange(8))]
        parts.append(f"docid{i}")
        out.append(" ".join(parts))
    return out


def _make_queries(n: int, seed: int, corpus: List[str]) -> List[str]:
    rng = __import__("random").Random(seed)
    vocab = corpus[seed % len(corpus)].split()
    if len(vocab) < 6:
        vocab = corpus[0].split()
    out: List[str] = []
    for _ in range(n):
        out.append(" ".join(vocab[rng.randrange(len(vocab))] for _ in range(6)))
    return out


def _make_unique_ingest_texts(n: int, seed: int) -> List[str]:
    """One chunk per string — guaranteed unique for cold embed path."""
    rng = __import__("random").Random(seed)
    words = (
        "ingest unique payload stream segment block volume shard replica "
        "partition node cluster warehouse lake bronze silver gold"
    ).split()
    out: List[str] = []
    for i in range(n):
        parts = [words[rng.randrange(len(words))] for _ in range(14 + rng.randrange(6))]
        parts.append(f"uid-{seed}-{i}")
        out.append(" ".join(parts))
    return out


def _summarize(label: str, ms: List[float]) -> None:
    s = sorted(ms)
    mean = statistics.mean(ms)
    med = statistics.median(ms)
    p95 = _percentile(s, 95)
    print(f"  {label:32} mean={mean:8.2f} ms  median={med:8.2f} ms  p95={p95:8.2f} ms  (n={len(ms)})")


def _run_recall_timed(mem: Memory, mode: str, queries: List[str], warmup: int) -> List[float]:
    w = min(warmup, len(queries))
    for q in queries[:w]:
        mem.recall(q, top_k=5, mode=mode, rerank_pool=20)
    times: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        mem.recall(q, top_k=5, mode=mode, rerank_pool=20)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_ingest(ingest_n: int, seed: int, device: str) -> None:
    texts = _make_unique_ingest_texts(ingest_n, seed + 1000)
    tmp = tempfile.mkdtemp(prefix="verimem_ingest_bench_")
    try:
        mem = Memory(path=tmp, device=device)
        miss_ms: List[float] = []
        for i, t in enumerate(texts):
            t0 = time.perf_counter()
            mem.remember(t, source=f"ingest_{i // 25}", topic="general")
            miss_ms.append((time.perf_counter() - t0) * 1000.0)

        total_s = sum(miss_ms) / 1000.0
        thr = len(texts) / total_s if total_s > 0 else 0.0
        print(f"\n  store: {tmp}")
        print(f"  compute device: {mem._compute_device}")
        print(f"  chunks after ingest: {mem.count()}")
        _summarize("ingest remember (embed MISS)", miss_ms)
        print(f"  ingest MISS throughput: {thr:.1f} chunks/s (mean per call)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench_recall(
    chunks: int,
    queries: int,
    warmup: int,
    prime: int,
    seed: int,
    both_modes: bool,
    mode: str,
    device: str,
) -> None:
    corpus = _make_corpus(chunks, seed)
    queries_raw = _make_queries(queries, seed + 11, corpus)
    queries_rerank = _make_queries(queries, seed + 97, corpus)
    queries_hybrid = _make_queries(queries, seed + 88, corpus)

    tmp = tempfile.mkdtemp(prefix="verimem_recall_bench_")
    try:
        mem = Memory(path=tmp, device=device)
        mem.warm_retrieval_models()
        t_build = time.perf_counter()
        for i, text in enumerate(corpus):
            mem.remember(text, source=f"s{i // 50}", topic="general")
        build_s = time.perf_counter() - t_build

        print(f"\n  store: {tmp}")
        print(f"  compute device: {mem._compute_device}")
        print(f"  corpus chunks: {mem.count()}  (bulk build {build_s:.2f}s, not per-op timed)")
        print(f"  queries: {queries}  warmup/mode: {warmup}  prime: {prime}")

        q_ce, q_r = queries_rerank[0], queries_raw[0]
        for _ in range(max(0, prime)):
            mem.recall(q_ce, top_k=5, mode="rerank", rerank_pool=20)
        for _ in range(max(0, prime)):
            mem.recall(q_r, top_k=5, mode="raw")

        modes = ("raw", "rerank") if both_modes else (mode,)
        print(f"\n  --- recall: QUERY EMBEDDING MISS (unique query each timed call) ---")
        means_miss: dict[str, float] = {}
        for m in modes:
            if m == "raw":
                qset = queries_raw
            elif m == "rerank":
                qset = queries_rerank
            else:
                qset = queries_hybrid
            ms = _run_recall_timed(mem, m, qset, min(warmup, len(qset)))
            _summarize(f"{m} recall (query MISS)", ms)
            means_miss[m] = statistics.mean(ms)
        if "raw" in means_miss and "rerank" in means_miss:
            r, ce = means_miss["raw"], means_miss["rerank"]
            print(f"\n  rerank/raw mean ratio (query miss): {ce / r if r > 0 else 0:.2f}x")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark VeriMem ingest + recall (miss paths only)")
    p.add_argument("--no-ingest", action="store_true", help="Skip ingest section")
    p.add_argument("--no-recall", action="store_true", help="Skip recall section")
    p.add_argument(
        "--ingest-n",
        type=int,
        default=60,
        help="Unique texts for timed remember() embed-miss path (0 = skip ingest)",
    )
    p.add_argument("--chunks", type=int, default=500, help="Corpus size for recall tests")
    p.add_argument("--queries", type=int, default=60, help="Timed recall calls (unique queries, miss path)")
    p.add_argument("--warmup", type=int, default=8, help="Warm-up recalls per mode before timed miss loop")
    p.add_argument(
        "--prime",
        type=int,
        default=20,
        help="Pre-roll hybrid+raw recalls to load NLI before recall timing",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        choices=("raw", "hybrid", "rerank"),
        default=DEFAULT_RETRIEVAL_MODE,
        help="Single recall mode when not using --both",
    )
    p.add_argument(
        "--both",
        action="store_true",
        help="Run raw and rerank (default stack) for recall section",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "gpu"),
        default="auto",
        help="Embedding + cross-encoder device: auto (cuda if available), cpu, or cuda/gpu",
    )
    args = p.parse_args()

    dev = "cuda" if args.device == "gpu" else args.device

    print(f"\n{'=' * 60}")
    print("  VeriMem memory_perf_bench  (ingest + recall)")
    print(f"  --device {args.device}")
    print(f"{'=' * 60}")

    if not args.no_ingest and args.ingest_n > 0:
        print(f"\n{'-' * 60}")
        print("  INGEST")
        print(f"{'-' * 60}")
        bench_ingest(args.ingest_n, args.seed, dev)

    if not args.no_recall:
        print(f"\n{'-' * 60}")
        print("  RECALL")
        print(f"{'-' * 60}")
        bench_recall(
            chunks=args.chunks,
            queries=args.queries,
            warmup=args.warmup,
            prime=args.prime,
            seed=args.seed,
            both_modes=args.both,
            mode=args.mode,
            device=dev,
        )

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
