#!/usr/bin/env python3
"""
LongMemEval-S Q&A — Groq (e.g. GPT OSS 120B) + Hindsight retain/recall
======================================================================

Uses the `hindsight-client` against a running **Hindsight API** (see
https://github.com/vectorize-io/hindsight — Docker: API on port 8888).

For each benchmark row:

1. Create an ephemeral memory **bank**, **retain** each corpus row (user turns)
   with ``document_id=str(i)`` for gold alignment.
2. **Recall** with the benchmark question; take up to ``top_k`` result texts as
   context for the answer model.
3. **Answer** + binary **judge** on Groq (same pattern as ``longmemeval_qa_groq_bench.py``).

The Hindsight server performs its own LLM-backed extraction on retain; configure
``HINDSIGHT_API_LLM_*`` in Docker (e.g. ``HINDSIGHT_API_LLM_PROVIDER=groq``) per
upstream docs. This script only needs ``GROQ_API_KEY`` for answer + judge.

Requires::

    pip install hindsight-client groq

Usage::

    # Terminal 1: Hindsight (example from upstream README)
    docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 ^
      -e HINDSIGHT_API_LLM_API_KEY=%OPENAI_OR_GROQ_KEY% ^
      -e HINDSIGHT_API_LLM_PROVIDER=groq ^
      -v %USERPROFILE%\\.hindsight-docker:/home/hindsight/.pg0 ^
      ghcr.io/vectorize-io/hindsight:latest

    # Terminal 2: benchmark
    set GROQ_API_KEY=sk-...
    python benchmarks/longmemeval_qa_hindsight_groq_bench.py data/longmemeval_s_cleaned.json ^
      --limit 100 --balance-categories --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from groq_bench_core import (
    GroqRunner,
    aggregate_results,
    generate_answer,
    get_groq_client,
    judge_answer,
)

if TYPE_CHECKING:
    from hindsight_client import Hindsight


def _make_hindsight_client(base_url: str, api_key: Optional[str], timeout: float) -> "Hindsight":
    try:
        from hindsight_client import Hindsight
    except ImportError as e:
        print(
            "ERROR: Install hindsight-client: pip install hindsight-client",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return Hindsight(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)


def balanced_sample_by_question_type(
    data: list[dict],
    total: int,
) -> tuple[list[dict], dict[str, int]]:
    """Even split across ``question_type`` (same as ``longmemeval_qa_groq_bench``)."""
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
    """User turns only; parallel corpus row index -> haystack session id."""
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


def _result_row_index(
    text: str,
    document_id: Optional[str],
    corpus: list[str],
) -> Optional[int]:
    if document_id:
        try:
            return int(document_id)
        except ValueError:
            pass
    t = (text or "").strip()
    if not t:
        return None
    for i, c in enumerate(corpus):
        cs = c.strip()
        if not cs:
            continue
        if cs in t or t in cs:
            return i
    return None


def retrieve_corpus_hindsight(
    client: "Hindsight",
    corpus: list[str],
    question: str,
    *,
    bank_id: str,
    top_k: int,
    recall_max_tokens: int,
    recall_budget: str,
) -> tuple[list[str], list[int]]:
    """
    Retain all corpus rows into ``bank_id``, recall for ``question``.

    Returns parallel (contexts_for_prompt, ranked_corpus_indices) where
    ranked indices list gold-alignment order (unique recall order + remainder).
    """
    if not corpus:
        return [], []

    client.create_bank(bank_id=bank_id)
    try:
        for i, text in enumerate(corpus):
            client.retain(bank_id=bank_id, content=text, document_id=str(i))

        resp = client.recall(
            bank_id=bank_id,
            query=question,
            max_tokens=recall_max_tokens,
            budget=recall_budget,
        )
        raw = list(resp.results or [])
        contexts: list[str] = []
        ranked_msg: list[int] = []
        for r in raw[:top_k]:
            if r.text:
                contexts.append(r.text)
            mid = _result_row_index(r.text, r.document_id, corpus)
            if mid is not None:
                ranked_msg.append(mid)

        seen: set[int] = set()
        unique_ranked: list[int] = []
        for mid in ranked_msg:
            if mid not in seen:
                seen.add(mid)
                unique_ranked.append(mid)
        rest = [i for i in range(len(corpus)) if i not in seen]
        ranked_indices = unique_ranked + rest

        return contexts, ranked_indices
    finally:
        try:
            client.delete_bank(bank_id=bank_id)
        except Exception:
            pass


def evaluate_item(
    client: "Hindsight",
    runner: GroqRunner,
    entry: dict,
    top_k: int,
    granularity: str,
    recall_max_tokens: int,
    recall_budget: str,
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
    bank_id = f"lm_{uuid.uuid4().hex}"

    try:
        contexts, ranked_indices = retrieve_corpus_hindsight(
            client,
            corpus,
            question,
            bank_id=bank_id,
            top_k=top_k,
            recall_max_tokens=recall_max_tokens,
            recall_budget=recall_budget,
        )
    except Exception as e:
        return {
            "question_id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": "",
            "correct": 0,
            "reasoning": f"Hindsight error: {e}",
            "category": qtype,
            "retrieval_successful": False,
        }

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

    generated = generate_answer(
        runner,
        question,
        contexts,
        packet_mode="hits-only",
        packet_simple=None,
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
        "packet_mode": "hits-only",
        "granularity": granularity,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="LongMemEval-S Q&A with Groq + Hindsight retain/recall (HTTP API)"
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
        help="With --limit, sample evenly across question_type.",
    )
    p.add_argument("--skip", type=int, default=0, help="Skip first N questions")
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument("--out", help="Output JSON path")
    p.add_argument("--groq-key", default="", help="Groq API key (default: GROQ_API_KEY)")
    p.add_argument("--model", default="openai/gpt-oss-120b", help="Groq model for answer + judge")
    p.add_argument("--groq-max-retries", type=int, default=8)
    p.add_argument("--groq-retry-base-sec", type=float, default=1.0)
    p.add_argument(
        "--hindsight-url",
        default=os.environ.get("HINDSIGHT_URL", "http://127.0.0.1:8888"),
        help="Hindsight API base URL (default: http://127.0.0.1:8888 or HINDSIGHT_URL)",
    )
    p.add_argument(
        "--hindsight-api-key",
        default="",
        help="Optional API key for Hindsight (default: HINDSIGHT_API_KEY env)",
    )
    p.add_argument(
        "--hindsight-timeout",
        type=float,
        default=600.0,
        help="HTTP timeout seconds for Hindsight client (retain/recall can be slow)",
    )
    p.add_argument(
        "--recall-max-tokens",
        type=int,
        default=4096,
        help="recall(max_tokens=...) passed to Hindsight",
    )
    p.add_argument(
        "--recall-budget",
        choices=("low", "mid", "high"),
        default="mid",
        help="recall(budget=...) passed to Hindsight",
    )
    p.add_argument(
        "--granularity",
        choices=("turn", "session"),
        default="turn",
        help="Corpus unit: one row per user message or per session.",
    )
    args = p.parse_args()

    hs_host = urlparse(args.hindsight_url.rstrip("/")).hostname or ""
    if hs_host in ("127.0.0.1", "localhost", "::1"):
        for key in ("NO_PROXY", "no_proxy"):
            extra = "127.0.0.1,localhost"
            cur = (os.environ.get(key) or "").strip()
            if extra not in cur:
                os.environ[key] = f"{extra},{cur}".strip(",")

    api_key = args.groq_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY or pass --groq-key")
        sys.exit(1)

    hs_key = args.hindsight_api_key or os.environ.get("HINDSIGHT_API_KEY") or None

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

    print("=" * 80)
    print("LongMemEval-S Q&A — Groq + Hindsight retain/recall (HTTP API)")
    print("=" * 80)
    print(f"Data: {data_path.name} | questions: {len(data)}")
    if per_type_counts:
        print("  Balanced per question_type:", per_type_counts)
    print(f"Granularity: {args.granularity}")
    print(f"Hindsight URL: {args.hindsight_url}")
    print(f"Groq model (answer + judge): {args.model}")
    print(
        f"Recall: max_tokens={args.recall_max_tokens}, budget={args.recall_budget}, top_k={args.top_k}"
    )
    print()

    client = _make_hindsight_client(args.hindsight_url, hs_key, args.hindsight_timeout)
    runner = GroqRunner(
        client=get_groq_client(api_key),
        model=args.model,
        max_retries=args.groq_max_retries,
        retry_base_sec=args.groq_retry_base_sec,
    )

    results: list[dict] = []
    for i, entry in enumerate(data):
        qid = str(entry.get("question_id", ""))[:8]
        qtype = entry.get("question_type", "?")
        try:
            print(f"[{i + 1}/{len(data)}] {qtype} | {qid} | {entry['question'][:72]}...")
            result = evaluate_item(
                client,
                runner,
                entry,
                top_k=args.top_k,
                granularity=args.granularity,
                recall_max_tokens=args.recall_max_tokens,
                recall_budget=args.recall_budget,
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
    out_file = args.out or str(_BENCH_DIR / f"results_longmemeval_qa_hindsight_groq_{ts}.json")
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "timestamp": ts,
            "dataset": str(data_path),
            "granularity": args.granularity,
            "retrieval": "hindsight",
            "hindsight_url": args.hindsight_url,
            "groq_model": args.model,
            "n_questions": len(data),
            "balance_categories": bool(args.balance_categories and args.limit > 0),
            "per_type_sample_counts": per_type_counts if per_type_counts else None,
            "skip": args.skip,
            "top_k": args.top_k,
            "recall_max_tokens": args.recall_max_tokens,
            "recall_budget": args.recall_budget,
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
