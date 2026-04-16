#!/usr/bin/env python3
"""
VeriMem × ConvoMem Benchmark
==============================

Evaluates VeriMem retrieval against the ConvoMem benchmark.
75,336 QA pairs across 6 evidence categories.

For each evidence item:
1. Ingest all conversations into a fresh Chroma store (one collection item per message)
2. Query with the question
3. Check if any retrieved message matches the evidence messages

Since ConvoMem has 75K items across many files, we sample a subset for benchmarking.
Downloads evidence files from HuggingFace on first run.

Usage:
    python benchmarks/convomem_bench.py                          # sample 100 items
    python benchmarks/convomem_bench.py --limit 500              # sample 500 items
    python benchmarks/convomem_bench.py --category user_evidence  # one category only
"""

import math
import os
import sys
import json
import shutil
import ssl
import tempfile
import argparse
import urllib.request
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import datetime

import chromadb

# Bypass SSL for restricted environments
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, str(Path(__file__).parent.parent))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from listwise_rerank import apply_local_rerank_indices
from verimem.hybrid_retrieval import full_ranking_after_fusion

HF_BASE = "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/core_benchmark/evidence_questions"

CATEGORIES = {
    "user_evidence": "User Facts",
    "assistant_facts_evidence": "Assistant Facts",
    "changing_evidence": "Changing Facts",
    "abstention_evidence": "Abstention",
    "preference_evidence": "Preferences",
    "implicit_connection_evidence": "Implicit Connections",
}

# Sample files per category (1_evidence = single-message evidence, simplest)
SAMPLE_FILES = {
    "user_evidence": "1_evidence/0050e213-5032-42a0-8041-b5eef2f8ab91_Telemarketer.json",
    "assistant_facts_evidence": None,  # will discover
    "changing_evidence": None,
    "abstention_evidence": None,
    "preference_evidence": None,
    "implicit_connection_evidence": None,
}

# ConvoMem uses 1_evidence/ for most categories; changing_evidence only has 2_evidence+ on HF
DEFAULT_EVIDENCE_SUBDIR = "1_evidence"
CATEGORY_EVIDENCE_SUBDIR = {
    "changing_evidence": "2_evidence",
}

# Cutoffs for recall@K / NDCG@K (golden = corpus row ids matching evidence turns)
METRIC_KS = (1, 3, 5, 10, 20, 50)

# BGE reranker v2 — used only for ``rerank_bge_v2`` / ``hybrid_rrf_bge_v2`` (MiniLM unchanged elsewhere).
DEFAULT_BGE_RERANKER_V2 = "BAAI/bge-reranker-v2-m3"


def ce_model_for_convo_mode(mode: str, bge_model_id: str) -> Optional[str]:
    """``None`` → default ``CrossEncoderReranker`` (ms-marco MiniLM). Else HF id for ``listwise_rerank``."""
    if mode in ("rerank_bge_v2", "hybrid_rrf_bge_v2"):
        return bge_model_id
    return None


# =============================================================================
# METRICS (golden corpus indices vs ranked list)
# =============================================================================


def gold_corpus_indices(corpus: list[str], evidence_messages: list) -> set[int]:
    """Corpus row ids whose text matches any evidence message (same overlap rules as before)."""
    gold: set[int] = set()
    evidence_texts = [e["text"].strip().lower() for e in evidence_messages]
    for i, doc in enumerate(corpus):
        t = doc.strip().lower()
        for ev in evidence_texts:
            if ev in t or t in ev:
                gold.add(i)
                break
    return gold


def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """Fraction of golden corpus ids that appear in the first k ranks (per item; mean over items = recall)."""
    if not gold:
        return 1.0
    top = set(ranked[: min(k, len(ranked))])
    return len(gold & top) / len(gold)


def recall_any_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """1.0 if at least one golden corpus id appears in top-k (LongMemEval-style recall_any)."""
    if not gold:
        return 1.0
    top = set(ranked[: min(k, len(ranked))])
    return 1.0 if (gold & top) else 0.0


def dcg_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    dcg = 0.0
    for pos, idx in enumerate(ranked[:k]):
        rel = 1.0 if idx in gold else 0.0
        dcg += rel / math.log2(pos + 2)
    return dcg


def ndcg_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return 1.0
    dcg = dcg_at_k(ranked, gold, k)
    num_rel = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_rel))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def metrics_row(ranked: list[int], gold: set[int]) -> dict:
    """Per item: recall@k (coverage), recall_any@k, NDCG@k, all_gold_hit@k for each k in METRIC_KS."""
    row = {}
    for k in METRIC_KS:
        row[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        row[f"recall_any@{k}"] = recall_any_at_k(ranked, gold, k)
        row[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        if gold:
            topk = set(ranked[: min(k, len(ranked))])
            row[f"all_gold_hit@{k}"] = 1.0 if gold <= topk else 0.0
        else:
            row[f"all_gold_hit@{k}"] = 1.0
    return row


def rrf_fuse_two_rankings(
    order_a: list[int],
    order_b: list[int],
    *,
    rrf_k: float = 60.0,
) -> list[int]:
    """
    Reciprocal Rank Fusion over two orderings of the **same** set of document ids.

    score(d) += 1 / (rrf_k + rank_pos) for each list; rank_pos is 1-based position.
    """
    sa, sb = set(order_a), set(order_b)
    if sa != sb:
        raise ValueError(f"RRF requires identical sets; symmetric diff size {len(sa ^ sb)}")
    scores: dict[int, float] = {i: 0.0 for i in sa}
    for pos, doc_id in enumerate(order_a):
        scores[doc_id] += 1.0 / (rrf_k + pos + 1)
    for pos, doc_id in enumerate(order_b):
        scores[doc_id] += 1.0 / (rrf_k + pos + 1)
    return sorted(sa, key=lambda x: (-scores[x], x))


# =============================================================================
# DATA LOADING
# =============================================================================


def download_evidence_file(category, subpath, cache_dir):
    """Download a single evidence file from HuggingFace."""
    url = f"{HF_BASE}/{category}/{subpath}"
    cache_path = os.path.join(cache_dir, category, subpath.replace("/", "_"))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    print(f"    Downloading: {category}/{subpath}...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return None


def discover_files(category, cache_dir):
    """Discover available files for a category via HuggingFace API."""
    sub = CATEGORY_EVIDENCE_SUBDIR.get(category, DEFAULT_EVIDENCE_SUBDIR)
    api_url = f"https://huggingface.co/api/datasets/Salesforce/ConvoMem/tree/main/core_benchmark/evidence_questions/{category}/{sub}"
    cache_path = os.path.join(cache_dir, f"{category}_{sub}_filelist.json")

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            files = json.loads(resp.read().decode("utf-8"))
            paths = [
                f["path"].split(f"{category}/")[1] for f in files if f["path"].endswith(".json")
            ]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(paths, f)
            return paths
    except Exception as e:
        print(f"    Failed to list files for {category}: {e}")
        return []


def load_evidence_items(categories, limit, cache_dir):
    """Load evidence items from specified categories."""
    all_items = []

    for category in categories:
        # Discover files
        files = discover_files(category, cache_dir)
        if not files:
            # Fallback to known file
            known = SAMPLE_FILES.get(category)
            if known:
                files = [known]
            else:
                print(f"  Skipping {category} — no files found")
                continue

        # Download files until we have enough items
        items_for_cat = []
        for fpath in files:
            if len(items_for_cat) >= limit:
                break
            data = download_evidence_file(category, fpath, cache_dir)
            if data and "evidence_items" in data:
                for item in data["evidence_items"]:
                    item["_category_key"] = category
                    items_for_cat.append(item)

        all_items.extend(items_for_cat[:limit])
        print(f"  {CATEGORIES.get(category, category)}: {len(items_for_cat[:limit])} items loaded")

    return all_items


# =============================================================================
# RETRIEVAL
# =============================================================================


def retrieve_for_item(
    item,
    top_k=10,
    mode="raw",
    *,
    hybrid_weight=0.35,
    rerank_pool=20,
    rrf_k: float = 60.0,
    bge_reranker_model: str = DEFAULT_BGE_RERANKER_V2,
):
    """
    Ingest conversations, query, rank corpus rows.

    Modes (aligned with ``longmemeval_bench``):
    - raw: Chroma dense only
    - hybrid: dense + BM25 fusion over candidates
    - rerank: dense pool + local cross-encoder (ms-marco MiniLM)
    - hybrid_rerank: hybrid then CE fully replaces order in the rerank pool
    - hybrid_rrf: RRF fuses hybrid order vs MiniLM CE order on the same top-``rerank_pool`` ids
    - rerank_bge_v2: dense pool + BGE reranker v2 (``bge_reranker_model``), not MiniLM
    - hybrid_rrf_bge_v2: RRF(hybrid, BGE v2 CE order); MiniLM unused

    Golden ids = corpus row indices matching ``message_evidences`` (substring overlap).
    Reports recall@K / NDCG@K for K in METRIC_KS over the full ranked list.

    Returns:
        overlap_recall: legacy metric — fraction of evidence strings hit in top-``top_k``
        details: dict with per-K metrics and gold size
    """
    conversations = item.get("conversations", [])
    question = item["question"]
    evidence_messages = item.get("message_evidences", [])
    evidence_texts = set(e["text"].strip().lower() for e in evidence_messages)

    # Build corpus: one doc per message
    corpus = []
    corpus_speakers = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            corpus.append(msg["text"])
            corpus_speakers.append(msg["speaker"])

    if not corpus:
        return 0.0, {"error": "empty corpus"}

    gold = gold_corpus_indices(corpus, evidence_messages)

    tmpdir = tempfile.mkdtemp(prefix="verimem_convomem_")
    chroma_dir = os.path.join(tmpdir, "chroma_store")

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.create_collection("verimem_chunks")

        collection.add(
            documents=corpus,
            ids=[f"msg_{i}" for i in range(len(corpus))],
            metadatas=[{"speaker": s, "idx": i} for i, s in enumerate(corpus_speakers)],
        )

        doc_id_to_idx = {f"msg_{i}": i for i in range(len(corpus))}
        n_corpus = len(corpus)
        max_metric_k = max(METRIC_KS)
        ce_bge = ce_model_for_convo_mode(mode, bge_reranker_model)

        if mode == "raw":
            n_q = min(n_corpus, max(max_metric_k, top_k, 20))
            results = collection.query(
                query_texts=[question],
                n_results=n_q,
                include=["documents", "metadatas", "distances"],
            )
            cand_idx = [doc_id_to_idx[rid] for rid in results["ids"][0]]
            seen_dense = set(cand_idx)
            ranked = cand_idx + [i for i in range(n_corpus) if i not in seen_dense]

        else:
            n_stage = min(max(rerank_pool, top_k * 3, max_metric_k, 20), n_corpus)
            results = collection.query(
                query_texts=[question],
                n_results=n_stage,
                include=["documents", "metadatas", "distances"],
            )
            cand_idx = [doc_id_to_idx[rid] for rid in results["ids"][0]]
            dists = list(results["distances"][0])
            seen_dense = set(cand_idx)
            ranked = cand_idx + [i for i in range(n_corpus) if i not in seen_dense]

            if mode == "hybrid":
                ranked = full_ranking_after_fusion(
                    cand_idx,
                    dists,
                    corpus,
                    question,
                    lexical_weight=hybrid_weight,
                    phrase_boost=False,
                    entity_boost=False,
                )
            elif mode == "rerank":
                ranked = apply_local_rerank_indices(
                    question,
                    ranked,
                    corpus,
                    rerank_pool,
                    None,
                )
            elif mode == "rerank_bge_v2":
                ranked = apply_local_rerank_indices(
                    question,
                    ranked,
                    corpus,
                    rerank_pool,
                    ce_bge,
                )
            elif mode == "hybrid_rerank":
                ranked = full_ranking_after_fusion(
                    cand_idx,
                    dists,
                    corpus,
                    question,
                    lexical_weight=hybrid_weight,
                    phrase_boost=False,
                    entity_boost=False,
                )
                ranked = apply_local_rerank_indices(
                    question,
                    ranked,
                    corpus,
                    rerank_pool,
                    None,
                )
            elif mode == "hybrid_rrf":
                ranked_h = full_ranking_after_fusion(
                    cand_idx,
                    dists,
                    corpus,
                    question,
                    lexical_weight=hybrid_weight,
                    phrase_boost=False,
                    entity_boost=False,
                )
                pool = min(rerank_pool, n_corpus)
                hybrid_head = ranked_h[:pool]
                ce_full = apply_local_rerank_indices(
                    question,
                    ranked_h,
                    corpus,
                    rerank_pool,
                    None,
                )
                ce_head = ce_full[:pool]
                ranked = rrf_fuse_two_rankings(hybrid_head, ce_head, rrf_k=rrf_k) + ranked_h[pool:]
            elif mode == "hybrid_rrf_bge_v2":
                ranked_h = full_ranking_after_fusion(
                    cand_idx,
                    dists,
                    corpus,
                    question,
                    lexical_weight=hybrid_weight,
                    phrase_boost=False,
                    entity_boost=False,
                )
                pool = min(rerank_pool, n_corpus)
                hybrid_head = ranked_h[:pool]
                ce_full = apply_local_rerank_indices(
                    question,
                    ranked_h,
                    corpus,
                    rerank_pool,
                    ce_bge,
                )
                ce_head = ce_full[:pool]
                ranked = rrf_fuse_two_rankings(hybrid_head, ce_head, rrf_k=rrf_k) + ranked_h[pool:]
            else:
                raise ValueError(f"unknown mode: {mode!r}")

        m = metrics_row(ranked, gold)

        # Legacy: substring overlap of evidence texts in top-``top_k`` ranks
        retrieved_indices = ranked[: min(top_k, len(ranked))]
        retrieved_texts = [corpus[i].strip().lower() for i in retrieved_indices]

        found = 0
        for ev_text in evidence_texts:
            for ret_text in retrieved_texts:
                if ev_text in ret_text or ret_text in ev_text:
                    found += 1
                    break

        overlap_recall = found / len(evidence_texts) if evidence_texts else 1.0

        return overlap_recall, {
            "retrieved_count": len(retrieved_indices),
            "evidence_count": len(evidence_texts),
            "gold_size": len(gold),
            "found": found,
            "metrics": m,
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def run_benchmark(
    categories,
    limit_per_cat,
    top_k,
    mode,
    cache_dir,
    out_file,
    *,
    hybrid_weight=0.35,
    rerank_pool=20,
    rrf_k: float = 60.0,
    bge_reranker_model: str = DEFAULT_BGE_RERANKER_V2,
):
    """Run the ConvoMem retrieval benchmark."""

    print(f"\n{'=' * 60}")
    print("  VeriMem × ConvoMem Benchmark")
    print(f"{'=' * 60}")
    print(f"  Categories:  {len(categories)}")
    print(f"  Limit/cat:   {limit_per_cat}")
    print(f"  Top-k:       {top_k}")
    print(f"  Mode:        {mode}")
    if mode in ("hybrid", "hybrid_rerank", "hybrid_rrf", "hybrid_rrf_bge_v2"):
        print(f"  Hybrid w:    {hybrid_weight}")
    if mode in ("rerank", "rerank_bge_v2", "hybrid_rerank", "hybrid_rrf", "hybrid_rrf_bge_v2"):
        print(f"  Rerank pool: {rerank_pool}")
    if mode in ("hybrid_rrf", "hybrid_rrf_bge_v2"):
        print(f"  RRF k:       {rrf_k}")
    if mode in ("rerank_bge_v2", "hybrid_rrf_bge_v2"):
        print(f"  BGE rerank:  {bge_reranker_model}")
    print(f"{'─' * 60}")
    print("\n  Loading data from HuggingFace...\n")

    items = load_evidence_items(categories, limit_per_cat, cache_dir)

    print(f"\n  Total items: {len(items)}")
    print(f"{'─' * 60}\n")

    all_recall = []
    per_category = defaultdict(list)
    results_log = []
    start_time = datetime.now()

    for i, item in enumerate(items):
        question = item["question"]
        answer = item.get("answer", "")
        cat_key = item.get("_category_key", "unknown")
        CATEGORIES.get(cat_key, cat_key)

        recall, details = retrieve_for_item(
            item,
            top_k=top_k,
            mode=mode,
            hybrid_weight=hybrid_weight,
            rerank_pool=rerank_pool,
            rrf_k=rrf_k,
            bge_reranker_model=bge_reranker_model,
        )
        all_recall.append(recall)
        per_category[cat_key].append(recall)

        results_log.append(
            {
                "question": question,
                "answer": answer,
                "category": cat_key,
                "recall": recall,
                "details": details,
            }
        )

        status = "HIT" if recall >= 1.0 else ("part" if recall > 0 else "miss")
        if (i + 1) % 20 == 0 or i == len(items) - 1:
            print(
                f"  [{i + 1:4}/{len(items)}] avg_recall={sum(all_recall) / len(all_recall):.3f}  last={status}"
            )

    elapsed = (datetime.now() - start_time).total_seconds()
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — VeriMem ({mode} mode, top-{top_k})")
    print(f"{'=' * 60}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / max(len(items), 1):.2f}s per item)")
    print(f"  Items:       {len(items)}")
    print(f"  Avg Recall:  {avg_recall:.3f}")

    print("\n  PER-CATEGORY RECALL:")
    for cat_key in sorted(per_category.keys()):
        vals = per_category[cat_key]
        avg = sum(vals) / len(vals)
        name = CATEGORIES.get(cat_key, cat_key)
        perfect = sum(1 for v in vals if v >= 1.0)
        print(f"    {name:25} R={avg:.3f}  perfect={perfect}/{len(vals)}")

    perfect_total = sum(1 for r in all_recall if r >= 1.0)
    zero_total = sum(1 for r in all_recall if r == 0)
    print("\n  DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect_total:4} ({perfect_total / len(all_recall) * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero_total:4} ({zero_total / len(all_recall) * 100:.1f}%)")

    # Aggregate golden-id metrics (same keys as details["metrics"])
    agg: dict[str, list[float]] = defaultdict(list)
    for row in results_log:
        d = row.get("details") or {}
        m = d.get("metrics")
        if not m:
            continue
        for key, val in m.items():
            if isinstance(val, (int, float)):
                agg[key].append(float(val))

    if agg:
        print("\n  GOLDEN CORPUS IDS (mean over items; recall_any = ≥1 gold id in top-K):")
        print("    K    recall_any@K    recall@K      NDCG@K    all_gold@K")
        print("             (≥1 gold)   (coverage)")
        for k in METRIC_KS:
            ra = agg.get(f"recall_any@{k}", [])
            rc = agg.get(f"recall@{k}", [])
            nd = agg.get(f"ndcg@{k}", [])
            ag = agg.get(f"all_gold_hit@{k}", [])
            print(
                f"    @{k:<4}"
                f"{100.0 * sum(ra) / len(ra) if ra else 0:7.1f}%     "
                f"{100.0 * sum(rc) / len(rc) if rc else 0:7.1f}%     "
                f"{sum(nd) / len(nd) if nd else 0:7.3f}   "
                f"{100.0 * sum(ag) / len(ag) if ag else 0:7.1f}%"
            )

    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=2)
        print(f"  Results saved to: {out_file}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriMem × ConvoMem Benchmark")
    parser.add_argument("--limit", type=int, default=100, help="Items per category (default: 100)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval (default: 10)")
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES.keys()) + ["all"],
        default="all",
        help="Category to test (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "raw",
            "hybrid",
            "rerank",
            "rerank_bge_v2",
            "hybrid_rerank",
            "hybrid_rrf",
            "hybrid_rrf_bge_v2",
        ],
        default="raw",
        help=(
            "raw=dense; hybrid=dense+BM25; rerank=MiniLM CE; rerank_bge_v2=BGE v2 CE; "
            "hybrid_rerank=hybrid+MiniLM CE; hybrid_rrf=RRF(hybrid,MiniLM); "
            "hybrid_rrf_bge_v2=RRF(hybrid,BGE v2)"
        ),
    )
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.35,
        help="BM25 weight in hybrid / hybrid_rerank (default: 0.35)",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=20,
        help="Cross-encoder candidate pool for rerank / hybrid_rerank / hybrid_rrf (default: 20)",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=60.0,
        help="RRF smoothing constant k (default: 60) for hybrid_rrf / hybrid_rrf_bge_v2",
    )
    parser.add_argument(
        "--bge-reranker-model",
        default=DEFAULT_BGE_RERANKER_V2,
        help="HF id for rerank_bge_v2 / hybrid_rrf_bge_v2 (default: BAAI/bge-reranker-v2-m3)",
    )
    parser.add_argument("--cache-dir", default="/tmp/convomem_cache", help="Cache directory")
    parser.add_argument("--out", default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.category == "all":
        categories = list(CATEGORIES.keys())
    else:
        categories = [args.category]

    if not args.out:
        args.out = f"benchmarks/results_convomem_{args.mode}_top{args.top_k}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    run_benchmark(
        categories,
        args.limit,
        args.top_k,
        args.mode,
        args.cache_dir,
        args.out,
        hybrid_weight=args.hybrid_weight,
        rerank_pool=args.rerank_pool,
        rrf_k=args.rrf_k,
        bge_reranker_model=args.bge_reranker_model,
    )
