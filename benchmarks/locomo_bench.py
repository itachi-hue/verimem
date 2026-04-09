#!/usr/bin/env python3
"""
LoCoMo retrieval benchmark
==========================

Evaluates retrieval against the LoCoMo benchmark.
10 conversations, ~200 QA pairs across 5 categories.

For each conversation:
1. Ingest all sessions into a fresh Chroma collection
2. For each QA pair, query the collection
3. Score retrieval recall (did we find the evidence dialog?)
4. Score F1 (optional, if --llm is provided)

Usage:
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json --top-k 10
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json --mode hybrid
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json --mode hybrid --llm-rerank
"""

import os
import sys
import json
import re
import string
import shutil
import tempfile
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from verimem.hybrid_retrieval import full_ranking_after_fusion

# ── Optional bge-large embeddings ────────────────────────────────────────────
_fastembed_model = None


def _get_embedder(model_name: str):
    """Lazy-load a fastembed model. Cached globally after first load."""
    global _fastembed_model
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding

            print(f"  Loading embedding model: {model_name} (first run may download ~1.3GB)")
            _fastembed_model = TextEmbedding(model_name=model_name)
            print("  Embedding model loaded.")
        except ImportError:
            print("  fastembed not installed — pip3 install fastembed")
            sys.exit(1)
    return _fastembed_model


def _embed(texts: list, embed_model: str) -> list:
    """Embed a list of texts. Returns list of float lists, or None for default."""
    if not embed_model or embed_model == "default":
        return None
    embedder = _get_embedder(embed_model)
    return [vec.tolist() for vec in embedder.embed(texts)]


def _query(collection, question: str, n_results: int, embed_model: str, include=None, where=None):
    """Query collection with either query_texts or query_embeddings."""
    if include is None:
        include = ["distances", "metadatas", "documents"]
    q_emb = _embed([question], embed_model)
    kwargs = dict(n_results=n_results, include=include)
    if where:
        kwargs["where"] = where
    if q_emb is not None:
        kwargs["query_embeddings"] = q_emb
    else:
        kwargs["query_texts"] = [question]
    return collection.query(**kwargs)


CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}


# =============================================================================
# METRICS (from LoCoMo's evaluation.py)
# =============================================================================


def normalize_answer(s):
    """Normalize answer for F1 comparison."""
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return s.lower().strip()


def f1_score(prediction, ground_truth):
    """Token-level F1 with normalization."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_conversation_sessions(conversation, session_summaries=None):
    """Extract sessions from a LoCoMo conversation dict."""
    sessions = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conversation:
            break
        dialogs = conversation[key]
        date = conversation.get(date_key, "")
        summary = ""
        if session_summaries:
            summary = session_summaries.get(f"session_{session_num}_summary", "")
        sessions.append(
            {
                "session_num": session_num,
                "date": date,
                "dialogs": dialogs,
                "summary": summary,
            }
        )
        session_num += 1
    return sessions


def build_corpus_from_sessions(sessions, granularity="dialog"):
    """
    Build retrieval corpus from conversation sessions.

    granularity:
        'dialog'  — one doc per dialog turn (matches evidence format D1:3)
        'session' — one doc per session (all dialog text joined)
        'rooms'   — one doc per session using pre-computed summary (room routing label)
    """
    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    for sess in sessions:
        if granularity in ("session", "rooms"):
            if granularity == "rooms" and sess.get("summary"):
                doc = sess["summary"]
            else:
                texts = []
                for d in sess["dialogs"]:
                    speaker = d.get("speaker", "?")
                    text = d.get("text", "")
                    texts.append(f'{speaker} said, "{text}"')
                doc = "\n".join(texts)
            corpus.append(doc)
            corpus_ids.append(f"session_{sess['session_num']}")
            corpus_timestamps.append(sess["date"])
        else:
            for d in sess["dialogs"]:
                dia_id = d.get("dia_id", f"D{sess['session_num']}:?")
                speaker = d.get("speaker", "?")
                text = d.get("text", "")
                doc = f'{speaker} said, "{text}"'
                corpus.append(doc)
                corpus_ids.append(dia_id)
                corpus_timestamps.append(sess["date"])

    return corpus, corpus_ids, corpus_timestamps


# =============================================================================
# ROOMS MODE — keyword routing via session summaries (LoCoMo)
# =============================================================================

STOP_WORDS = {
    "what",
    "when",
    "where",
    "who",
    "how",
    "which",
    "did",
    "do",
    "was",
    "were",
    "have",
    "has",
    "had",
    "is",
    "are",
    "the",
    "a",
    "an",
    "my",
    "me",
    "i",
    "you",
    "your",
    "their",
    "it",
    "its",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "ago",
    "last",
    "that",
    "this",
    "there",
    "about",
    "get",
    "got",
    "give",
    "gave",
    "buy",
    "bought",
    "made",
    "make",
    "said",
}


def _kw(text):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def _kw_overlap(query_kws, doc_text):
    doc_lower = doc_text.lower()
    if not query_kws:
        return 0.0
    hits = sum(1 for kw in query_kws if kw in doc_lower)
    return hits / len(query_kws)


# =============================================================================
# LLM RERANK
# =============================================================================


def llm_rerank_locomo(
    question, retrieved_ids, retrieved_docs, api_key, top_k=10, model="claude-sonnet-4-6"
):
    """
    Ask LLM to pick the single most relevant document for this question.
    Returns reordered retrieved_ids with the best candidate first.
    """
    candidates = retrieved_ids[:top_k]
    candidate_docs = retrieved_docs[:top_k]

    if len(candidates) <= 1:
        return retrieved_ids

    # Build numbered list of candidates
    lines = []
    for i, (cid, doc) in enumerate(zip(candidates, candidate_docs), 1):
        snippet = doc[:300].replace("\n", " ")
        lines.append(f"{i}. [{cid}] {snippet}")

    prompt = (
        f"Question: {question}\n\n"
        f"Which of the following passages most directly answers this question? "
        f"Reply with just the number (1-{len(candidates)}).\n\n" + "\n".join(lines)
    )

    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    import socket as _socket

    for _attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            raw = result["content"][0]["text"].strip()
            m = re.search(r"\b(\d+)\b", raw)
            if m:
                pick = int(m.group(1))
                if 1 <= pick <= len(candidates):
                    chosen_id = candidates[pick - 1]
                    reordered = [chosen_id] + [cid for cid in retrieved_ids if cid != chosen_id]
                    return reordered
            break
        except (_socket.timeout, TimeoutError):
            if _attempt < 2:
                import time as _time

                _time.sleep(3)
        except (urllib.error.URLError, KeyError, ValueError, IndexError, OSError):
            break

    return retrieved_ids


def _load_api_key(key_arg):
    if key_arg:
        return key_arg
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        return env_key
    keys_path = os.path.expanduser("~/.config/lu/keys.json")
    if os.path.exists(keys_path):
        try:
            with open(keys_path) as f:
                keys = json.load(f)
            for name in ("lu_key", "anthropic_milla", "anthropic_claude_code_main"):
                val = keys.get(name, "")
                if isinstance(val, str) and val.startswith("sk-ant-"):
                    return val
            for section in ("anthropic", "anthropic_milla", "anthropic_claude_code_main"):
                sec = keys.get(section, {})
                if isinstance(sec, dict):
                    for subkey in ("lu_key", "key", "api_key"):
                        val = sec.get(subkey, "")
                        if isinstance(val, str) and val.startswith("sk-ant-"):
                            return val
        except Exception:
            pass
    return ""


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def run_benchmark(
    data_file,
    top_k=10,
    mode="raw",
    limit=0,
    granularity="dialog",
    out_file=None,
    llm_rerank_enabled=False,
    llm_key="",
    llm_model="claude-sonnet-4-6",
    hybrid_weight=0.35,
    embed_model="default",
):
    """Run LoCoMo retrieval benchmark."""
    with open(data_file) as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    api_key = ""
    if llm_rerank_enabled:
        api_key = _load_api_key(llm_key)
        if not api_key:
            print("ERROR: --llm-rerank requires an API key (--llm-key or ANTHROPIC_API_KEY).")
            sys.exit(1)

    rerank_label = f" + LLM re-rank ({llm_model.split('-')[1]})" if llm_rerank_enabled else ""

    print(f"\n{'=' * 60}")
    print("  LoCoMo retrieval benchmark")
    print(f"{'=' * 60}")
    print(f"  Data:        {Path(data_file).name}")
    print(f"  Conversations: {len(data)}")
    print(f"  Top-k:       {top_k}")
    print(f"  Mode:        {mode}{rerank_label}")
    print(f"  Granularity: {granularity}")
    print(f"{'─' * 60}\n")

    all_recall = []
    per_category = defaultdict(list)
    results_log = []
    total_qa = 0

    start_time = datetime.now()

    for conv_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"conv-{conv_idx}")
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]

        session_summaries = sample.get("session_summary", {})
        sessions = load_conversation_sessions(conversation, session_summaries)
        corpus, corpus_ids, corpus_timestamps = build_corpus_from_sessions(
            sessions, granularity=granularity
        )

        print(
            f"  [{conv_idx + 1}/{len(data)}] {sample_id}: "
            f"{len(sessions)} sessions, {len(corpus)} docs, {len(qa_pairs)} questions"
        )

        tmpdir = tempfile.mkdtemp(prefix="verimem_locomo_")
        chroma_dir = os.path.join(tmpdir, "chroma_store")

        try:
            client = chromadb.PersistentClient(path=chroma_dir)
            collection = client.create_collection("verimem_chunks")

            docs_to_ingest = corpus

            corpus_embeddings = _embed(docs_to_ingest, embed_model)
            add_kwargs = dict(
                documents=docs_to_ingest,
                ids=[f"doc_{i}" for i in range(len(corpus))],
                metadatas=[
                    {"corpus_id": cid, "timestamp": ts}
                    for cid, ts in zip(corpus_ids, corpus_timestamps)
                ],
            )
            if corpus_embeddings is not None:
                add_kwargs["embeddings"] = corpus_embeddings
            collection.add(**add_kwargs)

            for qa in qa_pairs:
                question = qa["question"]
                answer = qa.get("answer", qa.get("adversarial_answer", ""))
                category = qa["category"]
                evidence = qa.get("evidence", [])

                predicate_kws = _kw(question) if mode == "rooms" else []

                if mode == "rooms":
                    # ── Two-stage room-scoped retrieval ──────────────────────────
                    # Stage 1: route via session summaries to find relevant rooms.
                    #   Score each session's summary by predicate keyword overlap.
                    #   Keep top third of sessions (or at least top_k sessions).
                    n_rooms = max(top_k, len(sessions) // 3)
                    room_scores = []
                    for sess in sessions:
                        summary = sess.get("summary", "")
                        overlap = (
                            _kw_overlap(predicate_kws, summary)
                            if (summary and predicate_kws)
                            else 0.0
                        )
                        room_scores.append((overlap, f"session_{sess['session_num']}"))
                    room_scores.sort(reverse=True)
                    top_room_ids = [sid for _, sid in room_scores[:n_rooms]]

                    # Stage 2: embedding query filtered to those rooms, then hybrid rerank
                    n_in_rooms = min(top_k * 2, len(top_room_ids))
                    where_filter = (
                        {"corpus_id": {"$in": top_room_ids}} if len(top_room_ids) > 1 else None
                    )
                    results_r = _query(
                        collection, question, n_in_rooms, embed_model, where=where_filter
                    )
                    raw_ids = [m["corpus_id"] for m in results_r["metadatas"][0]]
                    raw_distances = results_r["distances"][0]
                    raw_docs = results_r["documents"][0]

                    scored = []
                    for cid, dist, doc in zip(raw_ids, raw_distances, raw_docs):
                        pred_overlap = _kw_overlap(predicate_kws, doc)
                        fused = dist * (1.0 - 0.50 * pred_overlap)
                        scored.append((cid, dist, doc, fused))
                    scored.sort(key=lambda x: x[3])
                    retrieved_ids = [x[0] for x in scored[:top_k]]
                    retrieved_docs = [x[2] for x in scored[:top_k]]

                else:
                    n_retrieve = min(top_k * 3 if mode == "hybrid" else top_k, len(corpus))
                    results = _query(collection, question, n_retrieve, embed_model)
                    raw_ids = [m["corpus_id"] for m in results["metadatas"][0]]
                    raw_distances = results["distances"][0]
                    raw_docs = results["documents"][0]

                    if mode == "hybrid":
                        idx_by_cid = {cid: i for i, cid in enumerate(corpus_ids)}
                        cand_idx = [idx_by_cid[c] for c in raw_ids]
                        ranked_idx = full_ranking_after_fusion(
                            cand_idx,
                            list(raw_distances),
                            corpus,
                            question,
                            lexical_weight=hybrid_weight,
                            phrase_boost=False,
                            entity_boost=False,
                        )
                        retrieved_ids = [corpus_ids[i] for i in ranked_idx[:top_k]]
                        retrieved_docs = [corpus[i] for i in ranked_idx[:top_k]]
                    else:
                        retrieved_ids = raw_ids[:top_k]
                        retrieved_docs = raw_docs[:top_k]

                # LLM rerank
                if llm_rerank_enabled and api_key:
                    rerank_pool = min(10, len(retrieved_ids))
                    retrieved_ids = llm_rerank_locomo(
                        question,
                        retrieved_ids,
                        retrieved_docs,
                        api_key,
                        top_k=rerank_pool,
                        model=llm_model,
                    )

                # Compute recall
                if granularity == "dialog":
                    evidence_set = evidence_to_dialog_ids(evidence)
                else:
                    evidence_set = evidence_to_session_ids(evidence)

                recall = compute_retrieval_recall(retrieved_ids, evidence_set)
                all_recall.append(recall)
                per_category[category].append(recall)
                total_qa += 1

                results_log.append(
                    {
                        "sample_id": sample_id,
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "evidence": evidence,
                        "retrieved_ids": retrieved_ids,
                        "recall": recall,
                    }
                )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = (datetime.now() - start_time).total_seconds()

    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — {mode}{rerank_label}, {granularity}, top-{top_k}")
    print(f"{'=' * 60}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / max(total_qa, 1):.2f}s per question)")
    print(f"  Questions:   {total_qa}")
    print(f"  Avg Recall:  {avg_recall:.3f}")

    print("\n  PER-CATEGORY RECALL:")
    for cat in sorted(per_category.keys()):
        vals = per_category[cat]
        avg = sum(vals) / len(vals)
        name = CATEGORIES.get(cat, f"Cat-{cat}")
        print(f"    {name:25} R={avg:.3f}  (n={len(vals)})")

    perfect = sum(1 for r in all_recall if r >= 1.0)
    partial = sum(1 for r in all_recall if 0 < r < 1.0)
    zero = sum(1 for r in all_recall if r == 0)
    print("\n  RECALL DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect:4} ({perfect / len(all_recall) * 100:.1f}%)")
    print(f"    Partial (0-1):  {partial:4} ({partial / len(all_recall) * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero:4} ({zero / len(all_recall) * 100:.1f}%)")

    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"  Results saved to: {out_file}")


# =============================================================================
# RETRIEVAL HELPERS (used by run_benchmark)
# =============================================================================


def compute_retrieval_recall(retrieved_ids, evidence_ids):
    """What fraction of evidence dialog IDs were retrieved?"""
    if not evidence_ids:
        return 1.0
    found = sum(1 for eid in evidence_ids if eid in retrieved_ids)
    return found / len(evidence_ids)


def evidence_to_dialog_ids(evidence):
    return set(evidence)


def evidence_to_session_ids(evidence):
    sessions = set()
    for eid in evidence:
        match = re.match(r"D(\d+):", eid)
        if match:
            sessions.add(f"session_{match.group(1)}")
    return sessions


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoCoMo retrieval benchmark")
    parser.add_argument("data_file", help="Path to locomo10.json")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k retrieval (default: 50)")
    parser.add_argument(
        "--mode",
        choices=["raw", "hybrid", "rooms"],
        default="raw",
        help="raw=dense; hybrid=dense+BM25 fusion; rooms=summary keyword routing then dense+lexical",
    )
    parser.add_argument(
        "--granularity",
        choices=["dialog", "session"],
        default="session",
        help="Corpus granularity: dialog (per turn) or session (per session)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N conversations")
    parser.add_argument("--out", default=None, help="Output JSON file path")
    parser.add_argument("--llm-rerank", action="store_true", help="Use LLM to rerank top results")
    parser.add_argument(
        "--llm-model",
        default="claude-sonnet-4-6",
        help="Model for LLM rerank (default: claude-sonnet-4-6)",
    )
    parser.add_argument("--llm-key", default="", help="API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.35,
        help="BM25 weight in hybrid mode [0,1] (default: 0.35)",
    )
    parser.add_argument(
        "--embed-model",
        default="default",
        help="Embedding model: 'default' (ChromaDB built-in) or "
        "'BAAI/bge-large-en-v1.5' (requires fastembed)",
    )
    args = parser.parse_args()

    if not args.out:
        rerank_tag = "_llmrerank" if args.llm_rerank else ""
        args.out = (
            f"benchmarks/results_locomo_{args.mode}{rerank_tag}"
            f"_{args.granularity}_top{args.top_k}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )

    run_benchmark(
        args.data_file,
        args.top_k,
        args.mode,
        args.limit,
        args.granularity,
        args.out,
        args.llm_rerank,
        args.llm_key,
        args.llm_model,
        args.hybrid_weight,
        embed_model=args.embed_model,
    )
