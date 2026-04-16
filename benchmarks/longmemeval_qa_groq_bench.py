#!/usr/bin/env python3
"""
LongMemEval-S Q&A — Groq + VeriMem hybrid (ConvoMem-style E2E)
===============================================================

For each benchmark row: ingest haystack (user turns) with ``Memory.remember``,
``Memory.recall(..., mode='hybrid')``, then answer + binary judge on Groq — same
pattern as ``convomem_qa_groq_bench.py``.

Gold retrieval metric: fraction of gold **corpus row** indices (messages or
sessions in ``answer_session_ids``) covered by the top-``k`` ranked rows.

Usage::

    $env:GROQ_API_KEY='sk-...'
    python benchmarks/longmemeval_qa_groq_bench.py data/longmemeval_s_cleaned.json
    python benchmarks/longmemeval_qa_groq_bench.py data/longmemeval_s_cleaned.json --limit 50
    python benchmarks/longmemeval_qa_groq_bench.py data/longmemeval_s_cleaned.json --limit 100 --balance-categories
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

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


def balanced_sample_by_question_type(
    data: list[dict],
    total: int,
) -> tuple[list[dict], dict[str, int]]:
    """
    Take ``total`` items split across ``question_type`` as evenly as possible
    (remainder goes to the first types in sorted order). Preserves within-type order.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        by_type[row.get("question_type", "unknown")].append(row)

    types_sorted = sorted(by_type.keys())
    n_types = len(types_sorted)
    if n_types == 0:
        return [], {}

    base, extra = divmod(total, n_types)
    out: list[dict] = []
    counts: dict[str, int] = {}
    for i, t in enumerate(types_sorted):
        take = base + (1 if i < extra else 0)
        pool = by_type[t]
        chunk = pool[:take]
        counts[t] = len(chunk)
        out.extend(chunk)
    return out, counts


def build_corpus_from_entry(entry: dict, granularity: str) -> tuple[list[str], list[str]]:
    """
    Build parallel corpus + haystack session id per row.
    User turns only (aligned with ``longmemeval_bench`` default corpus).
    """
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    corpus: list[str] = []
    sid_row: list[str] = []

    for session, sid in zip(sessions, session_ids):
        if granularity == "session":
            user_turns = [t["content"] for t in session if t.get("role") == "user"]
            if user_turns:
                corpus.append("\n".join(user_turns))
                sid_row.append(sid)
        else:
            for turn in session:
                if turn.get("role") == "user":
                    corpus.append(turn["content"])
                    sid_row.append(sid)

    return corpus, sid_row


def retrieve_corpus_hybrid_verimem(
    corpus: list[str],
    question: str,
    top_k: int = 10,
    rerank_pool: int = 50,
    device: str = "auto",
    *,
    sync_contradictions: bool = False,
    include_uncertainty: bool = True,
) -> tuple[list[str], list[int], Optional[ContextPacket]]:
    """Same as ConvoMem path: ephemeral Memory, hybrid recall."""
    if not corpus:
        return [], [], None

    dev = "cuda" if device == "gpu" else device
    mem = Memory(path=":memory:", device=dev)
    mem.warm_retrieval_models()

    drawer_to_idx: dict[str, int] = {}
    for i, text in enumerate(corpus):
        for cid in mem.remember(text, source=f"row{i}", topic="general"):
            drawer_to_idx[cid] = i

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
        mid = drawer_to_idx.get(h.drawer_id or "")
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
    entry: dict,
    top_k: int,
    rerank_pool: int,
    device: str,
    packet_mode: str,
    granularity: str,
    sync_contradictions: bool = False,
    include_uncertainty: bool = True,
) -> dict:
    question = entry["question"]
    ground_truth = entry["answer"]
    qtype = entry.get("question_type", "unknown")
    qid = entry.get("question_id", "")
    answer_sids = set(entry["answer_session_ids"])

    corpus, sid_row = build_corpus_from_entry(entry, granularity)
    if not corpus:
        return {
            "question_id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": "",
            "correct": 0,
            "reasoning": "Empty corpus (no user turns)",
            "category": qtype,
            "retrieval_successful": False,
        }

    gold = {i for i, sid in enumerate(sid_row) if sid in answer_sids}

    contexts, ranked_indices, packet = retrieve_corpus_hybrid_verimem(
        corpus,
        question,
        top_k=top_k,
        rerank_pool=rerank_pool,
        device=device,
        sync_contradictions=sync_contradictions,
        include_uncertainty=include_uncertainty,
    )

    if not contexts:
        return {
            "question_id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": "",
            "correct": 0,
            "reasoning": "No context retrieved",
            "category": qtype,
            "retrieval_successful": False,
        }

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
        "question_id": qid,
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated,
        "correct": judge_result.get("correct", 0),
        "reasoning": judge_result.get("reasoning", ""),
        "category": qtype,
        "retrieval_successful": True,
        "retrieval_recall_at_k": retrieval_recall,
        "top_contexts": contexts[:3],
        "packet_mode": packet_mode,
        "granularity": granularity,
        "sync_contradictions": sync_contradictions,
        "include_uncertainty": include_uncertainty,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="LongMemEval-S Q&A with Groq + VeriMem hybrid (ConvoMem-style E2E)"
    )
    p.add_argument(
        "data_file",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "data" / "longmemeval_s_cleaned.json"),
        help="Path to longmemeval_s_cleaned.json",
    )
    p.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    p.add_argument(
        "--balance-categories",
        action="store_true",
        help="With --limit, sample evenly across question_type (remainder to first types alphabetically).",
    )
    p.add_argument("--skip", type=int, default=0, help="Skip first N questions")
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument("--rerank-pool", type=int, default=50, dest="rerank_pool")
    p.add_argument("--out", help="Output JSON path")
    p.add_argument("--groq-key", default="", help="Groq API key (default: GROQ_API_KEY)")
    p.add_argument("--model", default=DEFAULT_GROQ_MODEL)
    p.add_argument("--groq-max-retries", type=int, default=8)
    p.add_argument("--groq-retry-base-sec", type=float, default=1.0)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "gpu"),
        default="auto",
    )
    p.add_argument(
        "--packet-mode",
        choices=("hits-only", "extras", "full"),
        default="hits-only",
    )
    p.add_argument(
        "--granularity",
        choices=("turn", "session"),
        default="turn",
        help="Corpus unit: one row per user message (turn) or one row per session (joined user turns).",
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

    data_path = Path(args.data_file)
    if not data_path.is_file():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    with open(data_path, encoding="utf-8") as f:
        data: list = json.load(f)

    per_type_counts: dict[str, int] = {}
    if args.skip > 0:
        data = data[args.skip :]
    if args.limit > 0:
        if args.balance_categories:
            data, per_type_counts = balanced_sample_by_question_type(data, args.limit)
        else:
            data = data[: args.limit]

    dev = "cuda" if args.device == "gpu" else args.device

    print("=" * 80)
    print("LongMemEval-S Q&A — Groq + VeriMem hybrid (E2E, ConvoMem-style)")
    print("=" * 80)
    print(f"Data: {data_path.name} | questions: {len(data)}")
    if per_type_counts:
        print("  Balanced per question_type:", per_type_counts)
    print(f"Granularity: {args.granularity}")
    print(f"Model: {args.model}")
    print(
        f"Retrieval: Memory.recall(mode='hybrid'), top_k={args.top_k}, rerank_pool={args.rerank_pool}"
    )
    print(f"Packet prompt mode: {args.packet_mode}")
    print(f"Sync contradictions (NLI): {args.sync_contradictions}")
    print(f"Include retrieval uncertainty: {not args.no_uncertainty}")
    print(f"Device: {args.device}")
    print()

    runner = GroqRunner(
        client=get_groq_client(api_key),
        model=args.model,
        max_retries=args.groq_max_retries,
        retry_base_sec=args.groq_retry_base_sec,
    )

    results: list[dict] = []
    for i, entry in enumerate(data):
        qid = entry.get("question_id", "")[:8]
        qtype = entry.get("question_type", "?")
        try:
            print(f"[{i + 1}/{len(data)}] {qtype} | {qid} | {entry['question'][:72]}...")
            result = evaluate_item(
                runner,
                entry,
                top_k=args.top_k,
                rerank_pool=args.rerank_pool,
                device=dev,
                packet_mode=args.packet_mode,
                granularity=args.granularity,
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
                    "question_id": entry.get("question_id", ""),
                    "question": entry.get("question", ""),
                    "ground_truth": entry.get("answer", ""),
                    "generated_answer": "",
                    "correct": 0,
                    "reasoning": str(e),
                    "category": entry.get("question_type", "unknown"),
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
        for cat, st in sorted(summary["by_category"].items()):
            print(f"  {cat}: {st['accuracy'] * 100:.1f}% (n={st['count']})")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = args.out or str(_BENCH_DIR / f"results_longmemeval_qa_groq_{ts}.json")
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "timestamp": ts,
            "dataset": str(data_path),
            "granularity": args.granularity,
            "retrieval_mode": "hybrid",
            "groq_model": args.model,
            "n_questions": len(data),
            "balance_categories": bool(args.balance_categories and args.limit > 0),
            "per_type_sample_counts": per_type_counts if per_type_counts else None,
            "skip": args.skip,
            "top_k": args.top_k,
            "rerank_pool": args.rerank_pool,
            "packet_mode": args.packet_mode,
            "sync_contradictions": args.sync_contradictions,
            "include_uncertainty": not args.no_uncertainty,
            "scoring": "binary_llm_judge",
        },
        "summary": summary,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
