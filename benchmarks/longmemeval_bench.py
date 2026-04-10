#!/usr/bin/env python3
"""
VeriMem / LongMemEval retrieval benchmark
=========================================

For each question: build a fresh Chroma collection from haystack sessions,
query with the question, score recall@k vs answer_session_ids.

Retrieval modes (``--mode``):

    raw | hybrid | rerank | hybrid_rerank - same as ``Memory()`` (MiniLM CE when reranking).
    rerank_bge_v2 | hybrid_rerank_bge_v2 - BAAI/bge-reranker-v2-m3 instead of MiniLM.
    hybrid_rrf_bge_v2 - same RRF(hybrid, BGE CE) fusion as ``convomem_bench`` ``hybrid_rrf_bge_v2``.

Per-question ``metrics.session`` / ``metrics.turn`` use ConvoMem-aligned keys:
``recall_any@k``, ``recall@k`` (gold-set coverage), ``ndcg@k``, ``all_gold_hit@k``
for *k* in {1, 3, 5, 10, 20, 50}. ``ndcg_any@k`` is kept as an alias of ``ndcg@k``.

Optional ``--llm-rerank`` stacks Claude reranking on top of any mode (API key).

Usage:
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode raw
    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode hybrid_rerank

Experimental local reranker (does not change Memory() defaults):

    python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json \\
        --mode hybrid_rerank --limit 100 \\
        --local-ce-model jinaai/jina-reranker-v3 \\
        --out benchmarks/results_jina_v3_100q.jsonl

jina-reranker-v3 is CC BY-NC 4.0; requires transformers + trust_remote_code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from listwise_rerank import apply_local_rerank_indices
from verimem.hybrid_retrieval import full_ranking_after_fusion
from verimem.memory import DEFAULT_RETRIEVAL_MODE, RETRIEVAL_MODES

# Same cutoffs as ``convomem_bench.METRIC_KS`` (no @30 - use @20 for mid-rank reporting).
METRIC_KS = (1, 3, 5, 10, 20, 50)

# BGE reranker for ``rerank_bge_v2`` / ``hybrid_rerank_bge_v2`` (MiniLM unchanged for ``rerank`` / ``hybrid_rerank``).
DEFAULT_BGE_RERANKER_V2 = "BAAI/bge-reranker-v2-m3"

# Memory modes + BGE CE + RRF BGE (aligned with ``convomem_bench`` naming).
LME_BENCH_MODES = (
    "raw",
    "hybrid",
    "rerank",
    "hybrid_rerank",
    "rerank_bge_v2",
    "hybrid_rerank_bge_v2",
    "hybrid_rrf_bge_v2",
)

# =============================================================================
# METRICS
# =============================================================================


def dcg(relevances, k):
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(rankings, correct_ids, corpus_ids, k):
    relevances = [1.0 if corpus_ids[idx] in correct_ids else 0.0 for idx in rankings[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def evaluate_retrieval(rankings, correct_ids, corpus_ids, k):
    """Legacy tuple API; prefer ``convo_style_metrics`` for ConvoMem-aligned keys."""
    top_k_ids = set(corpus_ids[idx] for idx in rankings[: min(k, len(rankings))])
    recall_any = float(any(cid in top_k_ids for cid in correct_ids))
    recall_all = float(all(cid in top_k_ids for cid in correct_ids))
    ndcg_score = ndcg(rankings, correct_ids, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


def convo_style_metrics(
    rankings: list[int],
    gold: set,
    id_at_rank: list,
    k: int,
) -> dict[str, float]:
    """
    ConvoMem-aligned metrics for one cutoff *k*.

    ``gold`` = set of relevant ids (session ids for session-level, chunk ids for turn-level).
    ``id_at_rank[i]`` = id at corpus row index *i* (same indexing as ``rankings``).
    """
    kk = min(k, len(rankings)) if rankings else 0
    top_ids = {id_at_rank[i] for i in rankings[:kk]}
    if not gold:
        recall_any = 1.0
        recall_cov = 1.0
        all_gold_hit = 1.0
    else:
        inter = gold & top_ids
        recall_any = 1.0 if inter else 0.0
        recall_cov = len(inter) / len(gold)
        all_gold_hit = 1.0 if gold <= top_ids else 0.0
    ndcg_score = ndcg(rankings, gold, id_at_rank, k)
    return {
        "recall_any": recall_any,
        "recall": recall_cov,
        "ndcg": ndcg_score,
        "all_gold_hit": all_gold_hit,
    }


def rrf_fuse_two_rankings(
    order_a: list[int],
    order_b: list[int],
    *,
    rrf_k: float = 60.0,
) -> list[int]:
    """RRF over two orderings of the same corpus indices (matches ``convomem_bench``)."""
    sa, sb = set(order_a), set(order_b)
    if sa != sb:
        raise ValueError(f"RRF requires identical sets; symmetric diff size {len(sa ^ sb)}")
    scores: dict[int, float] = {i: 0.0 for i in sa}
    for pos, doc_id in enumerate(order_a):
        scores[doc_id] += 1.0 / (rrf_k + pos + 1)
    for pos, doc_id in enumerate(order_b):
        scores[doc_id] += 1.0 / (rrf_k + pos + 1)
    return sorted(sa, key=lambda x: (-scores[x], x))


def session_id_from_corpus_id(corpus_id):
    if "_turn_" in corpus_id:
        return corpus_id.rsplit("_turn_", 1)[0]
    return corpus_id


# =============================================================================
# CHROMA (shared ephemeral client)
# =============================================================================

_bench_client = chromadb.EphemeralClient()
_bench_embed_fn = None


def _make_embed_fn(model_name: str):
    if model_name == "default" or not model_name:
        return None

    MODEL_MAP = {
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "nomic": "nomic-ai/nomic-embed-text-v1.5",
        "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    }
    hf_name = MODEL_MAP.get(model_name, model_name)

    try:
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
        from fastembed import TextEmbedding

        class _FastEmbedFn(EmbeddingFunction):
            def __init__(self, name):
                print(f"  Loading embedding model: {name} (first run downloads ~300-1300MB)...")
                self._model = TextEmbedding(name)
                print("  Model ready.")

            def __call__(self, input: Documents) -> Embeddings:
                return [list(vec) for vec in self._model.embed(input)]

        return _FastEmbedFn(hf_name)
    except ImportError:
        print("ERROR: fastembed not installed. Run: pip install fastembed")
        print("       Falling back to default embedding model.")
        return None


def _fresh_collection(name="verimem_chunks"):
    global _bench_embed_fn
    try:
        _bench_client.delete_collection(name)
    except Exception:
        pass
    if _bench_embed_fn is not None:
        return _bench_client.create_collection(name, embedding_function=_bench_embed_fn)
    return _bench_client.create_collection(name)


# =============================================================================
# RETRIEVAL
# =============================================================================


def build_corpus_and_retrieve(entry, granularity="session", n_results=50):
    """Baseline: user turns only, dense retrieval."""
    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    for session, sess_id, date in zip(sessions, session_ids, dates):
        if granularity == "session":
            user_turns = [t["content"] for t in session if t["role"] == "user"]
            if user_turns:
                corpus.append("\n".join(user_turns))
                corpus_ids.append(sess_id)
                corpus_timestamps.append(date)
        else:
            turn_num = 0
            for turn in session:
                if turn["role"] == "user":
                    corpus.append(turn["content"])
                    corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                    corpus_timestamps.append(date)
                    turn_num += 1

    if not corpus:
        return [], corpus, corpus_ids, corpus_timestamps

    collection = _fresh_collection()
    collection.add(
        documents=corpus,
        ids=[f"doc_{i}" for i in range(len(corpus))],
        metadatas=[
            {"corpus_id": cid, "timestamp": ts} for cid, ts in zip(corpus_ids, corpus_timestamps)
        ],
    )

    query = entry["question"]
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, len(corpus)),
        include=["distances", "metadatas"],
    )

    doc_id_to_idx = {f"doc_{i}": i for i in range(len(corpus))}
    ranked_indices = [doc_id_to_idx[rid] for rid in results["ids"][0]]

    seen = set(ranked_indices)
    for i in range(len(corpus)):
        if i not in seen:
            ranked_indices.append(i)

    return ranked_indices, corpus, corpus_ids, corpus_timestamps


def build_hybrid_retrieve(
    entry,
    granularity="session",
    n_results=50,
    lexical_weight=0.35,
    phrase_boost=False,
    entity_boost=False,
):
    """User-turn corpus: Chroma dense first stage, then BM25+dense fusion (+ optional thin boosts)."""
    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    for session, sess_id, date in zip(sessions, session_ids, dates):
        if granularity == "session":
            user_turns = [t["content"] for t in session if t["role"] == "user"]
            if user_turns:
                corpus.append("\n".join(user_turns))
                corpus_ids.append(sess_id)
                corpus_timestamps.append(date)
        else:
            turn_num = 0
            for turn in session:
                if turn["role"] == "user":
                    corpus.append(turn["content"])
                    corpus_ids.append(f"{sess_id}_turn_{turn_num}")
                    corpus_timestamps.append(date)
                    turn_num += 1

    if not corpus:
        return [], corpus, corpus_ids, corpus_timestamps

    collection = _fresh_collection()
    collection.add(
        documents=corpus,
        ids=[f"doc_{i}" for i in range(len(corpus))],
        metadatas=[
            {"corpus_id": cid, "timestamp": ts} for cid, ts in zip(corpus_ids, corpus_timestamps)
        ],
    )

    query = entry["question"]
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, len(corpus)),
        include=["distances", "metadatas", "documents"],
    )

    doc_id_to_idx = {f"doc_{i}": i for i in range(len(corpus))}
    cand_idx = [doc_id_to_idx[rid] for rid in results["ids"][0]]
    dists = results["distances"][0]

    ranked_indices = full_ranking_after_fusion(
        cand_idx,
        list(dists),
        corpus,
        query,
        lexical_weight=lexical_weight,
        phrase_boost=phrase_boost,
        entity_boost=entity_boost,
    )
    return ranked_indices, corpus, corpus_ids, corpus_timestamps


# =============================================================================
# LLM RERANK (optional)
# =============================================================================


def llm_rerank(
    question, rankings, corpus, corpus_ids, api_key, top_k=10, model="claude-haiku-4-5-20251001"
):
    import urllib.error
    import urllib.request

    candidates = rankings[:top_k]
    if not candidates:
        return rankings

    session_blocks = []
    for rank, idx in enumerate(candidates):
        text = corpus[idx][:500].replace("\n", " ").strip()
        session_blocks.append(f"Session {rank + 1}:\n{text}")

    sessions_text = "\n\n".join(session_blocks)

    prompt = (
        f"Question: {question}\n\n"
        f"Below are {len(candidates)} conversation sessions from someone's memory. "
        f"Which single session is most likely to contain the answer to the question above? "
        f"Reply with ONLY a number between 1 and {len(candidates)}. Nothing else.\n\n"
        f"{sessions_text}\n\n"
        f"Most relevant session number:"
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
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read())
            raw = result["content"][0]["text"].strip()
            m = re.search(r"\b(\d+)\b", raw)
            if m:
                pick = int(m.group(1))
                if 1 <= pick <= len(candidates):
                    chosen_idx = candidates[pick - 1]
                    reordered = [chosen_idx] + [i for i in rankings if i != chosen_idx]
                    return reordered
            break
        except (_socket.timeout, TimeoutError):
            if _attempt < 2:
                import time as _time

                _time.sleep(3)
        except (urllib.error.URLError, KeyError, ValueError, IndexError, OSError):
            break

    return rankings


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


def _load_or_create_split(split_file: str, data: list, dev_size: int = 50, seed: int = 42) -> dict:
    import random

    split_path = Path(split_file)
    if split_path.exists():
        with open(split_path) as f:
            return json.load(f)

    all_ids = [entry["question_id"] for entry in data]
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    dev_ids = all_ids[:dev_size]
    held_out_ids = all_ids[dev_size:]
    split = {"dev": dev_ids, "held_out": held_out_ids, "seed": seed, "dev_size": dev_size}
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"  Created new split: {len(dev_ids)} dev / {len(held_out_ids)} held-out → {split_path}")
    return split


def _mode_to_base_and_local(mode: str) -> tuple[str, bool]:
    """Map CLI mode to (dense|hybrid base, use_local_cross_encoder_rerank)."""
    if mode == "raw":
        return "raw", False
    if mode == "hybrid":
        return "hybrid", False
    if mode == "rerank":
        return "raw", True
    if mode == "hybrid_rerank":
        return "hybrid", True
    if mode == "rerank_bge_v2":
        return "raw", True
    if mode == "hybrid_rerank_bge_v2":
        return "hybrid", True
    if mode == "hybrid_rrf_bge_v2":
        return "hybrid_rrf_bge_v2", False
    raise ValueError(f"unknown mode {mode!r}; expected one of {LME_BENCH_MODES}")


def run_benchmark(
    data_file,
    granularity="session",
    limit=0,
    out_file=None,
    mode=DEFAULT_RETRIEVAL_MODE,
    skip=0,
    hybrid_weight=0.35,
    llm_rerank_enabled=False,
    llm_key="",
    llm_model="claude-haiku-4-5-20251001",
    split_file=None,
    split_subset=None,
    local_rerank_pool=20,
    local_ce_model=None,
    phrase_boost=False,
    entity_boost=False,
    rrf_k: float = 60.0,
):
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    if split_file and split_subset:
        split = _load_or_create_split(split_file, data)
        subset_ids = set(split[split_subset])
        before = len(data)
        data = [entry for entry in data if entry["question_id"] in subset_ids]
        print(f"  Split filter ({split_subset}): {before} → {len(data)} questions")

    if limit > 0:
        data = data[:limit]

    if skip > 0:
        print(f"  Skipping first {skip} questions (resume mode)")
        data = data[skip:]

    api_key = ""
    if llm_rerank_enabled:
        api_key = _load_api_key(llm_key)
        if not api_key:
            print(
                "ERROR: --llm-rerank requires an API key. "
                "Set ANTHROPIC_API_KEY, use --llm-key, "
                "or store in ~/.config/lu/keys.json as 'lu_key'."
            )
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  LongMemEval retrieval benchmark")
    print(f"{'=' * 60}")
    print(f"  Data:        {Path(data_file).name}")
    print(f"  Questions:   {len(data)}")
    print(f"  Granularity: {granularity}")
    base, local_rerank_enabled = _mode_to_base_and_local(mode)

    model_short = llm_model.split("-")[1] if "-" in llm_model else llm_model
    rerank_label = f" + LLM re-rank ({model_short})" if llm_rerank_enabled else ""
    local_rerank_label = (
        f" + local cross-encoder re-rank (pool={local_rerank_pool})" if local_rerank_enabled else ""
    )
    rrf_bge_label = ""
    if mode == "hybrid_rrf_bge_v2":
        rrf_bge_label = f" + RRF(hybrid order, BGE CE order) rrf_k={rrf_k} pool={local_rerank_pool}"
    print(f"  Mode:        {mode}{rerank_label}{local_rerank_label}{rrf_bge_label}")
    if mode == "hybrid_rrf_bge_v2" and local_ce_model:
        print(f"  BGE CE:      {local_ce_model} (bench-only; Memory default unchanged)")
    elif local_rerank_enabled and local_ce_model:
        print(f"  Local CE:    {local_ce_model} (bench-only; Memory default unchanged)")
    elif local_rerank_enabled:
        print("  Local CE:    cross-encoder/ms-marco-MiniLM-L-6-v2 (default)")
    print(f"{'-' * 60}\n")

    ks = list(METRIC_KS)
    keys = (
        "recall_any",
        "recall",
        "ndcg",
        "all_gold_hit",
        "ndcg_any",  # alias of ndcg for older readers
    )
    metrics_session = {f"{name}@{k}": [] for k in ks for name in keys}

    metrics_turn = {f"{name}@{k}": [] for k in ks for name in keys}

    per_type = defaultdict(lambda: defaultdict(list))

    results_log = []
    start_time = datetime.now()

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

        if base == "hybrid_rrf_bge_v2":
            rankings, corpus, corpus_ids, corpus_timestamps = build_hybrid_retrieve(
                entry,
                granularity=granularity,
                lexical_weight=hybrid_weight,
                phrase_boost=phrase_boost,
                entity_boost=entity_boost,
            )
            if rankings:
                pool = min(local_rerank_pool, len(rankings))
                ranked_h = rankings
                hybrid_head = ranked_h[:pool]
                try:
                    ce_full = apply_local_rerank_indices(
                        question,
                        ranked_h,
                        corpus,
                        local_rerank_pool,
                        local_ce_model,
                    )
                    ce_head = ce_full[:pool]
                    rankings = (
                        rrf_fuse_two_rankings(hybrid_head, ce_head, rrf_k=rrf_k) + ranked_h[pool:]
                    )
                except Exception as exc:
                    print(f"  WARN hybrid_rrf_bge_v2 rerank failed for {qid[:8]}: {exc}")
        elif base == "hybrid":
            rankings, corpus, corpus_ids, corpus_timestamps = build_hybrid_retrieve(
                entry,
                granularity=granularity,
                lexical_weight=hybrid_weight,
                phrase_boost=phrase_boost,
                entity_boost=entity_boost,
            )
        else:
            rankings, corpus, corpus_ids, corpus_timestamps = build_corpus_and_retrieve(
                entry, granularity=granularity
            )

        if not rankings:
            print(f"  [{i + 1:4}/{len(data)}] {qid[:30]:30} SKIP (empty corpus)")
            continue

        if llm_rerank_enabled:
            rerank_pool = 20 if base in ("hybrid", "hybrid_rrf_bge_v2") else 10
            rankings = llm_rerank(
                question, rankings, corpus, corpus_ids, api_key, top_k=rerank_pool, model=llm_model
            )

        if local_rerank_enabled:
            try:
                rankings = apply_local_rerank_indices(
                    question,
                    rankings,
                    corpus,
                    local_rerank_pool,
                    local_ce_model,
                )
            except Exception as exc:
                print(f"  WARN local rerank failed for {qid[:8]}: {exc}")

        session_level_ids = [session_id_from_corpus_id(cid) for cid in corpus_ids]
        session_correct = answer_sids

        turn_correct = set()
        for cid in corpus_ids:
            sid = session_id_from_corpus_id(cid)
            if sid in answer_sids:
                turn_correct.add(cid)

        entry_metrics = {"session": {}, "turn": {}}

        for k in ks:
            ms = convo_style_metrics(rankings, session_correct, session_level_ids, k)
            metrics_session[f"recall_any@{k}"].append(ms["recall_any"])
            metrics_session[f"recall@{k}"].append(ms["recall"])
            metrics_session[f"ndcg@{k}"].append(ms["ndcg"])
            metrics_session[f"all_gold_hit@{k}"].append(ms["all_gold_hit"])
            metrics_session[f"ndcg_any@{k}"].append(ms["ndcg"])
            entry_metrics["session"][f"recall_any@{k}"] = ms["recall_any"]
            entry_metrics["session"][f"recall@{k}"] = ms["recall"]
            entry_metrics["session"][f"ndcg@{k}"] = ms["ndcg"]
            entry_metrics["session"][f"all_gold_hit@{k}"] = ms["all_gold_hit"]
            entry_metrics["session"][f"ndcg_any@{k}"] = ms["ndcg"]

            mt = convo_style_metrics(rankings, turn_correct, corpus_ids, k)
            metrics_turn[f"recall_any@{k}"].append(mt["recall_any"])
            metrics_turn[f"recall@{k}"].append(mt["recall"])
            metrics_turn[f"ndcg@{k}"].append(mt["ndcg"])
            metrics_turn[f"all_gold_hit@{k}"].append(mt["all_gold_hit"])
            metrics_turn[f"ndcg_any@{k}"].append(mt["ndcg"])
            entry_metrics["turn"][f"recall_any@{k}"] = mt["recall_any"]
            entry_metrics["turn"][f"recall@{k}"] = mt["recall"]
            entry_metrics["turn"][f"ndcg@{k}"] = mt["ndcg"]
            entry_metrics["turn"][f"all_gold_hit@{k}"] = mt["all_gold_hit"]
            entry_metrics["turn"][f"ndcg_any@{k}"] = mt["ndcg"]

        per_type[qtype]["recall_any@5"].append(metrics_session["recall_any@5"][-1])
        per_type[qtype]["recall_any@10"].append(metrics_session["recall_any@10"][-1])
        per_type[qtype]["ndcg_any@10"].append(metrics_session["ndcg_any@10"][-1])

        ranked_items = []
        for idx in rankings[:50]:
            ranked_items.append(
                {
                    "corpus_id": corpus_ids[idx],
                    "text": corpus[idx][:500],
                    "timestamp": corpus_timestamps[idx],
                }
            )

        results_log.append(
            {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "answer": entry["answer"],
                "retrieval_results": {
                    "query": question,
                    "ranked_items": ranked_items,
                    "metrics": entry_metrics,
                },
            }
        )

        r5 = metrics_session["recall_any@5"][-1]
        r10 = metrics_session["recall_any@10"][-1]
        status = "HIT" if r5 > 0 else "miss"
        print(f"  [{i + 1:4}/{len(data)}] {qid[:30]:30} R@5={r5:.0f} R@10={r10:.0f}  {status}")

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({mode} mode, {granularity} granularity)")
    print(f"{'=' * 60}")
    per_q = elapsed / len(data) if data else 0.0
    print(f"  Time: {elapsed:.1f}s ({per_q:.2f}s per question)\n")

    def _mean(key: str, bucket: dict) -> float:
        xs = bucket[key]
        return sum(xs) / len(xs) if xs else 0.0

    print("  SESSION-LEVEL METRICS (ConvoMem-aligned keys; gold = answer_session_ids):")
    print("    k   recall_any  recall(cov)   ndcg    all_gold_hit")
    for k in ks:
        ra = _mean(f"recall_any@{k}", metrics_session)
        rc = _mean(f"recall@{k}", metrics_session)
        nd = _mean(f"ndcg@{k}", metrics_session)
        ag = _mean(f"all_gold_hit@{k}", metrics_session)
        print(f"    @{k:<2}   {ra:.3f}       {rc:.3f}      {nd:.3f}     {ag:.3f}")

    print("\n  TURN-LEVEL METRICS (gold = chunks in answer sessions):")
    print("    k   recall_any  recall(cov)   ndcg    all_gold_hit")
    for k in ks:
        ra = _mean(f"recall_any@{k}", metrics_turn)
        rc = _mean(f"recall@{k}", metrics_turn)
        nd = _mean(f"ndcg@{k}", metrics_turn)
        ag = _mean(f"all_gold_hit@{k}", metrics_turn)
        print(f"    @{k:<2}   {ra:.3f}       {rc:.3f}      {nd:.3f}     {ag:.3f}")

    print("\n  PER-TYPE BREAKDOWN (session recall_any@10):")
    for qtype, vals in sorted(per_type.items()):
        r10 = sum(vals["recall_any@10"]) / len(vals["recall_any@10"])
        n = len(vals["recall_any@10"])
        print(f"    {qtype:35} R@10={r10:.3f}  (n={n})")

    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            for row in results_log:
                f.write(json.dumps(row) + "\n")
        print(f"  Results saved to: {out_file}")


if __name__ == "__main__":
    # Avoid UnicodeEncodeError on Windows consoles (cp1252) when printing separators or tracebacks.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LongMemEval retrieval benchmark")
    parser.add_argument("data_file", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument(
        "--granularity",
        choices=["session", "turn"],
        default="session",
        help="Retrieval granularity (default: session)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0 = all)")
    parser.add_argument(
        "--mode",
        choices=list(LME_BENCH_MODES),
        default=DEFAULT_RETRIEVAL_MODE,
        help=(
            "raw | hybrid | rerank | hybrid_rerank (MiniLM CE); "
            "rerank_bge_v2 | hybrid_rerank_bge_v2 | hybrid_rrf_bge_v2 (BGE; RRF matches convomem). "
            "Default hybrid."
        ),
    )
    parser.add_argument("--out", default=None, help="Output JSONL file path")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N questions (resume)")
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.35,
        help="BM25 weight in [0,1] for hybrid mode; dense weight is 1 minus this (default: 0.35).",
    )
    parser.add_argument(
        "--hybrid-phrase-boost",
        action="store_true",
        help="Hybrid only: small boost when quoted query phrases appear in a candidate.",
    )
    parser.add_argument(
        "--hybrid-entity-boost",
        action="store_true",
        help="Hybrid only: small boost when capitalized query tokens appear in a candidate.",
    )
    parser.add_argument(
        "--llm-rerank",
        action="store_true",
        default=False,
        help="Claude rerank over top pool (requires API key).",
    )
    parser.add_argument(
        "--local-rerank",
        action="store_true",
        default=False,
        help="Deprecated: use --mode rerank or --mode hybrid_rerank (same as raw/hybrid + local rerank).",
    )
    parser.add_argument(
        "--local-rerank-pool",
        type=int,
        default=20,
        help="Candidates for local reranking / RRF pool (default: 20).",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=60.0,
        metavar="K",
        help="RRF smoothing constant for hybrid_rrf_bge_v2 (default: 60, same as convomem_bench).",
    )
    parser.add_argument(
        "--local-ce-model",
        default=None,
        metavar="HF_ID",
        help=(
            "Benchmark-only Hugging Face reranker id. "
            "jinaai/jina-reranker-v3: listwise transformers (0.6B, CC BY-NC). "
            "Other ids: sentence_transformers.CrossEncoder. "
            "Omit for default MS MARCO MiniLM (unchanged for Memory API)."
        ),
    )
    parser.add_argument(
        "--llm-key",
        default="",
        help="Anthropic API key (or env / ~/.config/lu/keys.json).",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-haiku-4-5-20251001",
        help="Model for LLM reranking.",
    )
    parser.add_argument(
        "--embed-model",
        choices=["default", "bge-base", "bge-large", "nomic", "mxbai"],
        default="default",
        help="Embedding model for Chroma (non-default needs fastembed).",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="JSON train/dev split file; use with --dev-only or --held-out.",
    )
    parser.add_argument(
        "--create-split",
        action="store_true",
        default=False,
        help="Create 50/450 dev/held-out split and exit.",
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        default=False,
        help="Run 50 dev questions only (requires --split-file).",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        default=False,
        help="Run 450 held-out questions (requires --split-file).",
    )
    args = parser.parse_args()

    if args.local_rerank:
        import warnings

        warnings.warn(
            "--local-rerank is deprecated; use --mode rerank (from raw) or "
            "--mode hybrid_rerank (from hybrid).",
            DeprecationWarning,
            stacklevel=1,
        )
        if args.mode == "raw":
            args.mode = "rerank"
        elif args.mode == "hybrid":
            args.mode = "hybrid_rerank"

    if args.create_split:
        if not args.split_file:
            args.split_file = "benchmarks/lme_split_50_450.json"
        with open(args.data_file) as f:
            _all_data = json.load(f)
        _load_or_create_split(args.split_file, _all_data)
        sys.exit(0)

    if (args.dev_only or args.held_out) and not args.split_file:
        parser.error("--dev-only / --held-out require --split-file.")

    if args.dev_only and args.held_out:
        parser.error("--dev-only and --held-out are mutually exclusive.")

    split_subset = "dev" if args.dev_only else ("held_out" if args.held_out else None)

    ce_resolved = args.local_ce_model
    if args.mode in ("rerank_bge_v2", "hybrid_rerank_bge_v2", "hybrid_rrf_bge_v2"):
        ce_resolved = ce_resolved or DEFAULT_BGE_RERANKER_V2

    if not args.out:
        embed_tag = f"_{args.embed_model}" if args.embed_model != "default" else ""
        suffix = "_llmrerank" if args.llm_rerank else ""
        subset_tag = f"_{split_subset}" if split_subset else ""
        ce_tag = ""
        if ce_resolved:
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", ce_resolved).strip("_")[:48]
            ce_tag = f"_{slug}"
        args.out = (
            f"benchmarks/results_verimem_{args.mode}{embed_tag}{suffix}{ce_tag}{subset_tag}_"
            f"{args.granularity}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        )

    if args.embed_model != "default":
        _mod = sys.modules[__name__]
        _mod._bench_embed_fn = _make_embed_fn(args.embed_model)

    run_benchmark(
        args.data_file,
        args.granularity,
        args.limit,
        args.out,
        args.mode,
        args.skip,
        args.hybrid_weight,
        args.llm_rerank,
        args.llm_key,
        args.llm_model,
        split_file=args.split_file,
        split_subset=split_subset,
        local_rerank_pool=args.local_rerank_pool,
        local_ce_model=ce_resolved,
        phrase_boost=args.hybrid_phrase_boost,
        entity_boost=args.hybrid_entity_boost,
        rrf_k=args.rrf_k,
    )
