#!/usr/bin/env python3
"""
ConvoMem Q&A Benchmark with Claude
===================================

End-to-end Q&A evaluation:
1. Retrieve context using VeriMem (hybrid_rrf_bge_v2)
2. Generate answer using Claude Sonnet 4.5
3. Judge answer quality using Claude Sonnet 4.5

Usage:
    python benchmarks/convomem_qa_bench.py --total 100  # 100 total questions across all categories
"""

import json
import os
import sys
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
import traceback

import boto3
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from verimem.hybrid_retrieval import full_ranking_after_fusion

# Import from existing convomem_bench
from convomem_bench import (
    CATEGORIES,
    load_evidence_items,
    gold_corpus_indices,
    DEFAULT_BGE_RERANKER_V2,
)

# AWS/Claude settings
AWS_REGION = "us-east-1"
CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Judge scoring rubric - BINARY (1 or 0)
JUDGE_PROMPT = """You are evaluating an AI assistant's answer to a question.

Question: {question}

Ground Truth Answer: {ground_truth}

Assistant's Answer: {generated_answer}

Evaluate: Does the assistant's answer correctly answer the question based on the ground truth?

Score 1 if: The answer is factually correct and matches the ground truth (minor wording differences are OK).
Score 0 if: The answer is incorrect, contradicts the ground truth, or says "I don't know" when information was available.

Respond in this exact JSON format:
{{
  "correct": <0 or 1>,
  "reasoning": "<brief explanation>"
}}

Output only the JSON, nothing else."""


def get_bedrock_client():
    """Initialize AWS Bedrock client."""
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def call_claude(client, prompt: str, max_tokens: int = 1000) -> str:
    """Call Claude via Bedrock."""
    try:
        response = client.invoke_model(
            modelId=CLAUDE_MODEL,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }
            ),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        print(f"Claude API error: {e}")
        return ""


def retrieve_context_hybrid_rrf_bge_v2(
    conversations: list,
    question: str,
    top_k: int = 10,
    rerank_pool: int = 20,
    rrf_k: float = 60.0,
    bge_model: str = DEFAULT_BGE_RERANKER_V2,
) -> tuple[list[str], list[int]]:
    """
    Retrieve top-k context chunks using hybrid RRF with BGE v2 reranker.

    Returns:
        (context_docs, ranked_indices)
    """
    # Build corpus
    corpus = []
    corpus_speakers = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            corpus.append(msg["text"])
            corpus_speakers.append(msg["speaker"])

    if not corpus:
        return [], []

    tmpdir = tempfile.mkdtemp(prefix="verimem_qa_")
    chroma_dir = os.path.join(tmpdir, "chroma_store")

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.create_collection("verimem_qa")

        collection.add(
            documents=corpus,
            ids=[f"msg_{i}" for i in range(len(corpus))],
            metadatas=[{"speaker": s, "idx": i} for i, s in enumerate(corpus_speakers)],
        )

        doc_id_to_idx = {f"msg_{i}": i for i in range(len(corpus))}
        n_corpus = len(corpus)
        n_stage = min(max(rerank_pool, top_k * 3, 20), n_corpus)

        # Dense retrieval
        results = collection.query(
            query_texts=[question],
            n_results=n_stage,
            include=["documents", "metadatas", "distances"],
        )
        cand_idx = [doc_id_to_idx[rid] for rid in results["ids"][0]]
        dists = list(results["distances"][0])

        # Hybrid fusion
        ranked = full_ranking_after_fusion(
            cand_idx,
            dists,
            corpus,
            question,
            lexical_weight=0.35,
            phrase_boost=False,
            entity_boost=False,
        )

        # RRF with BGE v2 reranker
        # Apply RRF fusion with BGE reranker
        # For hybrid_rrf_bge_v2, we need to do RRF between hybrid and BGE reranker
        ranked_hybrid = ranked[:rerank_pool]

        # Get BGE reranker scores
        try:
            from sentence_transformers import CrossEncoder

            ce = CrossEncoder(bge_model)
            pairs = [[question, corpus[i]] for i in ranked_hybrid]
            ce_scores = ce.predict(pairs)

            # Sort by CE scores to get CE ranking
            ce_ranking = sorted(enumerate(ce_scores), key=lambda x: -x[1])
            ce_ranked_idx = [ranked_hybrid[i] for i, _ in ce_ranking]

            # RRF fusion
            rrf_scores = {}
            for rank, idx in enumerate(ranked_hybrid):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
            for rank, idx in enumerate(ce_ranked_idx):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

            final_ranked = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
            # Add remaining corpus indices
            final_ranked.extend([i for i in range(n_corpus) if i not in set(final_ranked)])
            ranked = final_ranked

        except Exception as e:
            print(f"BGE reranker error: {e}, falling back to hybrid only")
            pass

        # Get top-k contexts
        top_indices = ranked[:top_k]
        top_contexts = [corpus[i] for i in top_indices]

        return top_contexts, ranked

    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_answer(client, question: str, contexts: list[str]) -> str:
    """Generate answer using Claude given retrieved context."""
    context_str = "\n\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""Answer the following question based on the conversation context provided.

Context:
{context_str}

Question: {question}

Provide a helpful answer based on the context. If the context doesn't contain enough information to answer, say "I don't know."

Answer:"""

    return call_claude(client, prompt, max_tokens=500)


def judge_answer(client, question: str, ground_truth: str, generated: str) -> dict:
    """Judge the generated answer using Claude."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated,
    )

    response = call_claude(client, prompt, max_tokens=1000)

    try:
        # Parse JSON response
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        result = json.loads(json_str)
        return result
    except Exception as e:
        print(f"Failed to parse judge response: {e}")
        print(f"Response was: {response}")
        return {
            "correct": 0,
            "reasoning": f"Failed to parse: {str(e)}",
        }


def evaluate_item(client, item, top_k=10) -> dict:
    """Run full pipeline for one item: retrieve, answer, judge."""
    question = item["question"]
    ground_truth = item["answer"]
    conversations = item.get("conversations", [])
    evidence_messages = item.get("message_evidences", [])

    # 1. Retrieve context
    contexts, ranked_indices = retrieve_context_hybrid_rrf_bge_v2(
        conversations, question, top_k=top_k
    )

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

    # Calculate retrieval metrics
    corpus = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            corpus.append(msg["text"])

    gold = gold_corpus_indices(corpus, evidence_messages)
    top_k_set = set(ranked_indices[:top_k]) if ranked_indices else set()
    retrieval_recall = len(gold & top_k_set) / len(gold) if gold else 1.0

    # 2. Generate answer
    generated = generate_answer(client, question, contexts)

    # 3. Judge answer
    judge_result = judge_answer(client, question, ground_truth, generated)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated,
        "correct": judge_result.get("correct", 0),
        "reasoning": judge_result.get("reasoning", ""),
        "category": item.get("_category_key", "unknown"),
        "retrieval_successful": True,
        "retrieval_recall_at_k": retrieval_recall,
        "top_contexts": contexts[:3],  # Save top 3 for debugging
    }


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate metrics across all results - BINARY SCORING."""
    total = len(results)
    successful = sum(1 for r in results if r["retrieval_successful"])

    if successful == 0:
        return {"error": "No successful evaluations"}

    # Overall accuracy (binary)
    correct_count = sum(r.get("correct", 0) for r in results if r["retrieval_successful"])
    accuracy = correct_count / successful if successful > 0 else 0.0

    avg_retrieval_recall = (
        sum(r["retrieval_recall_at_k"] for r in results if r["retrieval_successful"]) / successful
    )

    # Per-category breakdown
    by_category = {}
    for result in results:
        if not result["retrieval_successful"]:
            continue
        cat = result["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result)

    category_stats = {}
    for cat, cat_results in by_category.items():
        n = len(cat_results)
        cat_correct = sum(r.get("correct", 0) for r in cat_results)
        category_stats[cat] = {
            "count": n,
            "correct": cat_correct,
            "accuracy": round(cat_correct / n if n > 0 else 0.0, 3),
            "retrieval_recall": round(sum(r["retrieval_recall_at_k"] for r in cat_results) / n, 3),
        }

    return {
        "total_items": total,
        "successful_items": successful,
        "overall": {
            "correct": correct_count,
            "accuracy": round(accuracy, 3),
            "retrieval_recall_at_k": round(avg_retrieval_recall, 3),
        },
        "by_category": category_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="ConvoMem Q&A Benchmark with Claude")
    parser.add_argument("--total", type=int, default=100, help="Total items across all categories")
    parser.add_argument(
        "--category", type=str, help="Run only specific category (e.g., abstention_evidence)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k context chunks")
    parser.add_argument("--cache-dir", default="/tmp/convomem_cache", help="Cache directory")
    parser.add_argument("--out", help="Output JSON file")

    args = parser.parse_args()

    # Calculate per-category limit
    if args.category:
        categories = [args.category]
        per_category_limit = args.total
    else:
        categories = list(CATEGORIES.keys())
        per_category_limit = max(1, args.total // len(categories))

    print("=" * 80)
    print("ConvoMem Q&A Benchmark with Claude Sonnet 4.5")
    print("=" * 80)
    print("Mode: hybrid_rrf_bge_v2")
    if args.category:
        print(f"Category: {CATEGORIES.get(args.category, args.category)}")
        print(f"Total items: {args.total}")
    else:
        print(f"Total items: {args.total} (~{per_category_limit} per category)")
    print(f"Top-k: {args.top_k}")
    print(f"Model: {CLAUDE_MODEL}")
    print(f"Region: {AWS_REGION}")
    print("Scoring: Binary (1=correct, 0=incorrect)")
    print("Improved Prompt: YES (balanced abstention + inference)")
    print()

    # Initialize Bedrock client
    print("Initializing AWS Bedrock client...")
    client = get_bedrock_client()

    # Load evidence items
    print("Loading ConvoMem evidence items...")
    items = load_evidence_items(categories, per_category_limit, args.cache_dir)
    print(f"Loaded {len(items)} items total\n")

    # Evaluate each item
    results = []
    for i, item in enumerate(items):
        try:
            cat = item.get("_category_key", "unknown")
            print(f"[{i + 1}/{len(items)}] Category: {cat} | Question: {item['question'][:80]}...")

            result = evaluate_item(client, item, top_k=args.top_k)
            results.append(result)

            # Print score
            if result["retrieval_successful"]:
                correct = result.get("correct", 0)
                status = "✓ CORRECT" if correct == 1 else "✗ INCORRECT"
                print(f"  → {status} | Generated: {result['generated_answer'][:80]}...")
                print(f"  → Retrieval Recall@{args.top_k}: {result['retrieval_recall_at_k']:.3f}")
            else:
                print(f"  → Failed: {result.get('reasoning', 'Unknown error')}")
            print()

        except Exception as e:
            print(f"  → Error: {e}")
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
            print()

    # Aggregate results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    summary = aggregate_results(results)

    print("\n📊 OVERALL PERFORMANCE")
    print(f"  Total Items: {summary['total_items']}")
    print(f"  Successful: {summary['successful_items']}")
    print(f"  Correct: {summary['overall']['correct']}")
    print(f"  Accuracy: {summary['overall']['accuracy'] * 100:.1f}%")
    print(f"  Retrieval Recall@{args.top_k}: {summary['overall']['retrieval_recall_at_k']:.3f}")

    print("\n📋 PER-CATEGORY PERFORMANCE")
    for cat, stats in summary["by_category"].items():
        print(f"  {CATEGORIES.get(cat, cat)}:")
        print(f"    Items: {stats['count']}")
        print(f"    Correct: {stats['correct']}")
        print(f"    Accuracy: {stats['accuracy'] * 100:.1f}%")
        print(f"    Retrieval Recall: {stats['retrieval_recall']:.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = args.out or f"results_convomem_qa_claude_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "mode": "hybrid_rrf_bge_v2",
            "model": CLAUDE_MODEL,
            "region": AWS_REGION,
            "total_items": args.total,
            "per_category_limit": per_category_limit,
            "top_k": args.top_k,
            "scoring": "binary",
        },
        "summary": summary,
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
