"""One-off: sequential vs batched CrossEncoder NLI latency (same model as BackgroundNLI)."""
from __future__ import annotations

import statistics
import time
from typing import List, Tuple

MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NUM_PAIRS = 45  # k=10 -> C(10,2)


def _pairs() -> List[Tuple[str, str]]:
    """Synthetic pairs ~ chunk-sized (vary slightly to avoid pathological cache effects)."""
    a = [
        f"The SLA target is 99.{i}% for region {i % 7}."
        for i in range(10)
    ]
    b = [
        f"Service level was revised to 99.{(i + 3) % 10}% last quarter (item {i})."
        for i in range(10)
    ]
    out: List[Tuple[str, str]] = []
    for i in range(10):
        for j in range(i + 1, 10):
            out.append((a[i] + f" [h{i}]", b[j] + f" [t{j}]"))
    assert len(out) == NUM_PAIRS, len(out)
    return out


def main() -> None:
    from sentence_transformers import CrossEncoder

    pairs = _pairs()
    model = CrossEncoder(MODEL)

    # Warmup (load + compile-ish)
    model.predict([pairs[0]], apply_softmax=True, show_progress_bar=False)
    model.predict(pairs[:8], apply_softmax=True, show_progress_bar=False)

    reps = 5
    seq_ms: list[float] = []
    bat_ms: list[float] = []

    for _ in range(reps):
        t0 = time.perf_counter()
        for p in pairs:
            model.predict([p], apply_softmax=True, show_progress_bar=False)
        seq_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        model.predict(pairs, apply_softmax=True, show_progress_bar=False, batch_size=len(pairs))
        bat_ms.append((time.perf_counter() - t0) * 1000.0)

    def _summ(xs: list[float]) -> str:
        return f"mean={statistics.mean(xs):.1f}ms  p50={statistics.median(xs):.1f}ms  min={min(xs):.1f}ms  max={max(xs):.1f}ms"

    print(f"model={MODEL}")
    print(f"pairs={NUM_PAIRS} (same as top_k=10 hit pairs)")
    print(f"sequential (one predict per pair, {reps} reps): {_summ(seq_ms)}")
    print(f"batched   (single predict, batch_size={NUM_PAIRS}, {reps} reps): {_summ(bat_ms)}")
    ratio = statistics.mean(seq_ms) / statistics.mean(bat_ms)
    print(f"speedup (mean sequential / mean batched): {ratio:.2f}x")
    print(f"extra cost if you added sync sequential CE for all pairs: ~{statistics.mean(seq_ms):.0f}ms mean on this machine")
    print(f"extra cost if you added sync batched CE once:           ~{statistics.mean(bat_ms):.0f}ms mean on this machine")


if __name__ == "__main__":
    main()
