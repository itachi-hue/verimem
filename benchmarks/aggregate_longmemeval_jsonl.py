#!/usr/bin/env python3
"""Mean session metrics from longmemeval_bench.py JSONL (ConvoMem-aligned keys)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_KS = (1, 3, 5, 10, 20, 50)
KEYS = ("recall_any", "recall", "ndcg", "all_gold_hit")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    args = p.parse_args()
    buckets: dict[str, list[float]] = {f"{n}@{k}": [] for k in METRIC_KS for n in KEYS}
    n = 0
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sess = row["retrieval_results"]["metrics"]["session"]
            for k in METRIC_KS:
                for name in KEYS:
                    key = f"{name}@{k}"
                    buckets[key].append(float(sess[key]))
            n += 1

    def mean(key: str) -> float:
        xs = buckets[key]
        return sum(xs) / len(xs) if xs else 0.0

    print(f"# n={n}  file={args.jsonl.name}")
    print("k\trecall_any\trecall\tndcg\tall_gold_hit")
    for k in METRIC_KS:
        print(
            f"@{k}\t{mean(f'recall_any@{k}'):.6f}\t{mean(f'recall@{k}'):.6f}\t"
            f"{mean(f'ndcg@{k}'):.6f}\t{mean(f'all_gold_hit@{k}'):.6f}"
        )


if __name__ == "__main__":
    main()
