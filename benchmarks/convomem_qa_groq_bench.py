#!/usr/bin/env python3
"""
ConvoMem Q&A Benchmark — Groq (Llama 4 Scout) + VeriMem hybrid retrieval
==========================================================================

1. Ingest messages with ``Memory.remember`` (ephemeral store).
2. Retrieve with ``Memory.recall(..., mode="hybrid")`` (dense + BM25 fusion).
3. Answer + binary judge via **Llama 4 Scout** on **Groq** (same model for both).

Requires: ``pip install groq`` and ``GROQ_API_KEY`` (or ``--groq-key``).

Retries on 429 / 5xx with exponential backoff (and ``Retry-After`` when present); no client-side RPM cap.

Usage::

    set GROQ_API_KEY=sk-...   # cmd
    $env:GROQ_API_KEY='sk-...'   # PowerShell
    python benchmarks/convomem_qa_groq_bench.py --total 100
    python benchmarks/convomem_qa_groq_bench.py --total 100 --packet-mode extras

``--packet-mode`` (default ``hits-only``) appends ``ContextPacket.to_simple()`` data for A/B
comparing answer quality with retrieval uncertainty / contradiction metadata (``extras`` or ``full``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from convomem_bench import CATEGORIES, gold_corpus_indices, load_evidence_items
from groq_bench_core import (
    DEFAULT_GROQ_MODEL,
    GroqRunner,
    aggregate_results,
    generate_answer,
    get_groq_client,
    judge_answer,
)
from verimem import Memory
from verimem.recall import ContextPacket


def retrieve_context_hybrid_verimem(
    conversations: list,
    question: str,
    top_k: int = 10,
    rerank_pool: int = 50,
    device: str = "auto",
    *,
    sync_contradictions: bool = False,
    include_uncertainty: bool = True,
) -> tuple[list[str], list[int], Optional[ContextPacket]]:
    """Retrieve using ``Memory.recall(..., mode='hybrid')``.

    Returns (context texts, message-index ranking, raw ``ContextPacket`` for ``to_simple()``).
    Use ``sync_contradictions=True`` so NLI runs on ephemeral stores (otherwise contradiction list is empty).
    """
    corpus: list[str] = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            corpus.append(msg["text"])

    if not corpus:
        return [], [], None

    dev = "cuda" if device == "gpu" else device
    mem = Memory(path=":memory:", device=dev)
    mem.warm_retrieval_models()

    drawer_to_msg: dict[str, int] = {}
    for i, text in enumerate(corpus):
        for cid in mem.remember(text, source=f"turn{i}", topic="general"):
            drawer_to_msg[cid] = i

    pool = max(rerank_pool, top_k, 20)
    packet = mem.recall(
        question,
        top_k=top_k,
        mode="hybrid",
        rerank_pool=pool,
        sync_contradictions=sync_contradictions,
        include_uncertainty=include_uncertainty,
    )
    hits = packet.hits
    contexts = [h.text for h in hits]

    ranked_msg: list[int] = []
    for h in hits:
        mid = drawer_to_msg.get(h.drawer_id or "")
        if mid is not None:
            ranked_msg.append(mid)

    unique_ranked: list[int] = []
    seen: set[int] = set()
    for mid in ranked_msg:
        if mid not in seen:
            seen.add(mid)
            unique_ranked.append(mid)

    rest = [i for i in range(len(corpus)) if i not in seen]
    ranked_indices = unique_ranked + rest

    return contexts, ranked_indices, packet


def evaluate_item(
    runner: GroqRunner,
    item: dict,
    top_k: int = 10,
    rerank_pool: int = 50,
    device: str = "auto",
    packet_mode: str = "hits-only",
    sync_contradictions: bool = False,
    include_uncertainty: bool = True,
) -> dict:
    question = item["question"]
    ground_truth = item["answer"]
    conversations = item.get("conversations", [])
    evidence_messages = item.get("message_evidences", [])

    contexts, ranked_indices, packet = retrieve_context_hybrid_verimem(
        conversations,
        question,
        top_k=top_k,
        rerank_pool=rerank_pool,
        device=device,
        sync_contradictions=sync_contradictions,
        include_uncertainty=include_uncertainty,
    )

    corpus: list[str] = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            corpus.append(msg["text"])

    if not contexts:
        return {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": "",
            "correct": 0,
            "reasoning": "No context retrieved",
            "category": item.get("_category_key", "unknown"),
            "retrieval_successful": False,
        }

    gold = gold_corpus_indices(corpus, evidence_messages)
    top_k_set = set(ranked_indices[:top_k]) if ranked_indices else set()
    retrieval_recall = len(gold & top_k_set) / len(gold) if gold else 1.0

    packet_simple = packet.to_simple() if packet else None
    generated = generate_answer(
        runner,
        question,
        contexts,
        packet_mode=packet_mode,
        packet_simple=packet_simple,
    )
    judge_result = judge_answer(runner, question, ground_truth, generated)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated,
        "correct": judge_result.get("correct", 0),
        "reasoning": judge_result.get("reasoning", ""),
        "category": item.get("_category_key", "unknown"),
        "retrieval_successful": True,
        "retrieval_recall_at_k": retrieval_recall,
        "top_contexts": contexts[:3],
        "packet_mode": packet_mode,
        "sync_contradictions": sync_contradictions,
        "include_uncertainty": include_uncertainty,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="ConvoMem Q&A with Groq Llama 4 Scout + VeriMem hybrid")
    p.add_argument("--total", type=int, default=100, help="Total items across all categories")
    p.add_argument("--category", type=str, help="Single category key (e.g. abstention_evidence)")
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument(
        "--rerank-pool", type=int, default=50, help="Hybrid candidate pool (dense fetch cap)"
    )
    p.add_argument(
        "--cache-dir", default=None, help="ConvoMem HF cache (default: system temp dir subfolder)"
    )
    p.add_argument("--out", help="Output JSON path")
    p.add_argument(
        "--groq-key",
        default="",
        help="Groq API key (default: GROQ_API_KEY env)",
    )
    p.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model id")
    p.add_argument(
        "--groq-max-retries",
        type=int,
        default=8,
        help="Retries per completion on 429 / 5xx (exponential backoff; respects Retry-After when present).",
    )
    p.add_argument(
        "--groq-retry-base-sec",
        type=float,
        default=1.0,
        help="Base delay for exponential backoff when Retry-After is missing.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "gpu"),
        default="auto",
        help="Embedding device for VeriMem",
    )
    p.add_argument(
        "--packet-mode",
        choices=("hits-only", "extras", "full"),
        default="hits-only",
        help=(
            "What to pass to the answer model beyond numbered context strings: "
            "'hits-only' (baseline), 'extras' (contradictions + retrieval uncertainty + notes, "
            "no duplicate hit bodies), or 'full' (entire ContextPacket.to_simple() JSON)."
        ),
    )
    p.add_argument(
        "--sync-contradictions",
        action="store_true",
        help="Run NLI on hit pairs before return (needed for contradiction flags on :memory: stores).",
    )
    p.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Omit retrieval_uncertainty / retrieval block from ContextPacket.",
    )
    args = p.parse_args()

    api_key = args.groq_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY or pass --groq-key")
        sys.exit(1)

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = str(Path(tempfile.gettempdir()) / "convomem_cache")

    if args.category:
        categories = [args.category]
        per_category_limit = args.total
    else:
        categories = list(CATEGORIES.keys())
        per_category_limit = max(1, args.total // len(categories))

    dev = "cuda" if args.device == "gpu" else args.device

    print("=" * 80)
    print("ConvoMem Q&A — Groq Llama 4 Scout + VeriMem hybrid retrieval")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(
        f"Retrieval: Memory.recall(mode='hybrid'), top_k={args.top_k}, rerank_pool={args.rerank_pool}"
    )
    print(f"Packet prompt mode: {args.packet_mode}")
    print(f"Sync contradictions (NLI): {args.sync_contradictions}")
    print(f"Include retrieval uncertainty: {not args.no_uncertainty}")
    print(f"Device: {args.device}")
    print(
        f"Groq: max_retries={args.groq_max_retries}, retry_base={args.groq_retry_base_sec}s (no RPM cap)"
    )
    if args.category:
        print(f"Category: {CATEGORIES.get(args.category, args.category)} | total={args.total}")
    else:
        print(f"Total items: {args.total} (~{per_category_limit} per category)")
    print()

    runner = GroqRunner(
        client=get_groq_client(api_key),
        model=args.model,
        max_retries=args.groq_max_retries,
        retry_base_sec=args.groq_retry_base_sec,
    )
    print("Loading ConvoMem evidence items...")
    items = load_evidence_items(categories, per_category_limit, cache_dir)
    print(f"Loaded {len(items)} items\n")

    results: list[dict] = []
    for i, item in enumerate(items):
        try:
            cat = item.get("_category_key", "unknown")
            print(f"[{i + 1}/{len(items)}] {cat} | {item['question'][:80]}...")
            result = evaluate_item(
                runner,
                item,
                top_k=args.top_k,
                rerank_pool=args.rerank_pool,
                device=dev,
                packet_mode=args.packet_mode,
                sync_contradictions=args.sync_contradictions,
                include_uncertainty=not args.no_uncertainty,
            )
            results.append(result)
            if result["retrieval_successful"]:
                ok = result.get("correct", 0) == 1
                tag = "CORRECT" if ok else "INCORRECT"
                print(f"  -> {tag} | {result['generated_answer'][:100]}...")
            else:
                print(f"  -> FAIL: {result.get('reasoning', '')}")
            print()
        except Exception as e:
            print(f"  -> Error: {e}")
            traceback.print_exc()
            results.append(
                {
                    "question": item.get("question", ""),
                    "ground_truth": item.get("answer", ""),
                    "generated_answer": "",
                    "correct": 0,
                    "reasoning": str(e),
                    "category": item.get("_category_key", "unknown"),
                    "retrieval_successful": False,
                }
            )

    summary = aggregate_results(results)
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    if "error" in summary:
        print(summary["error"])
    else:
        print(f"Accuracy: {summary['overall']['accuracy'] * 100:.1f}%")
        print(f"Retrieval recall@k (mean): {summary['overall']['retrieval_recall_at_k']:.3f}")
        for cat, st in summary["by_category"].items():
            print(f"  {cat}: {st['accuracy'] * 100:.1f}% (n={st['count']})")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = args.out or str(_BENCH_DIR / f"results_convomem_qa_groq_{ts}.json")
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "timestamp": ts,
            "retrieval_mode": "hybrid",
            "groq_model": args.model,
            "total_requested": args.total,
            "per_category_limit": per_category_limit,
            "top_k": args.top_k,
            "rerank_pool": args.rerank_pool,
            "scoring": "binary",
            "groq_max_retries": args.groq_max_retries,
            "groq_retry_base_sec": args.groq_retry_base_sec,
            "packet_mode": args.packet_mode,
            "sync_contradictions": args.sync_contradictions,
            "include_uncertainty": not args.no_uncertainty,
        },
        "summary": summary,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
