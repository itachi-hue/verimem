# VeriMem benchmarks — reproduction guide

Run the benchmarks in this repo. Clone, install, run.

## Setup

```bash
git clone https://github.com/itachi-hue/verimem.git
cd verimem
pip install -e "."
```

## Benchmark 1: LongMemEval (500 questions)

Tests retrieval across ~53 conversation sessions per question. The standard benchmark for AI memory.

```bash
# Download data
mkdir -p /tmp/longmemeval-data
curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# Run (default: hybrid = dense + BM25 fusion)
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json

# Other retrieval modes: raw, rerank (raw + local cross-encoder), hybrid_rerank
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode raw
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode hybrid_rerank

# Quick test on 20 questions first
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --limit 20

# Turn-level granularity
python benchmarks/longmemeval_bench.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --granularity turn
```

**Expected output (hybrid mode, full 500, approximate):**
```
Recall@5:  0.978
Recall@10: 0.992
NDCG@10:   0.935
Time:      ~5 minutes on Apple Silicon
```

See `VERIMEM_BENCHMARKS.md` for all four modes (raw / hybrid / rerank / hybrid_rerank).

## Benchmark 2: ConvoMem (Salesforce, 75K+ QA pairs)

Tests six categories of conversational memory. Downloads from HuggingFace automatically.

```bash
# Run all categories, 50 items each (our 92.9% result)
python benchmarks/convomem_bench.py --category all --limit 50

# Single category
python benchmarks/convomem_bench.py --category user_evidence --limit 100

# Quick test
python benchmarks/convomem_bench.py --category user_evidence --limit 10
```

**Recorded results (300-item slice, BGE rerank + hybrid RRF):** [`convomem_results.md`](convomem_results.md)

**Categories available:** `user_evidence`, `assistant_facts_evidence`, `changing_evidence`, `abstention_evidence`, `preference_evidence`, `implicit_connection_evidence`

**Expected output (all categories, 50 each):**
```
Avg Recall: 0.929
Assistant Facts: 1.000
User Facts:      0.980
Time:            ~2 minutes
```

## What Each Benchmark Tests

| Benchmark | What it measures | Why it matters |
|---|---|---|
| **LongMemEval** | Can you find a fact buried in 53 sessions? | Tests basic retrieval quality — the "needle in a haystack" |
| **ConvoMem** | Does your memory system work at scale? | Tests all memory types: facts, preferences, changes, abstention |

## Results Files

Raw results are in `benchmarks/results_*.jsonl` and `benchmarks/results_*.json`. Each file contains every question, every retrieved document, and every score — fully auditable.

## Requirements

- Python 3.9+
- `chromadb` (the only dependency)
- ~300MB disk for LongMemEval data
- ~5 minutes for each full benchmark run
- No API key. No internet during benchmark (after data download). No GPU.

## Next Benchmarks (Planned)

- **Scale testing** — ConvoMem at 50/100/300 conversations per item
- **End-to-end QA** — retrieve + generate answer + measure F1 (needs LLM API key)
