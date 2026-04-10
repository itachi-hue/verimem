# VeriMem — LongMemEval benchmark report

**Dataset:** LongMemEval-style evaluation, **500 questions** per mode.  
**Metrics (session-level):** Mean over rows of `retrieval_results.metrics.session` — **recall_any**, **recall** (gold-set coverage), **NDCG**, **all_gold_hit** at *k* ∈ {1, 3, 5, 10, 20, 50} (ConvoMem-aligned; no @30).

**Source JSONLs (this repo):**

| Mode | File |
|------|------|
| `raw` | `results_longmemeval_raw_session_20260409_2322.jsonl` |
| `hybrid` | `results_longmemeval_hybrid_session_20260409_2340.jsonl` |
| `rerank` | `results_longmemeval_rerank_session_20260409_2357.jsonl` |
| `hybrid_rerank` | `results_longmemeval_hybrid_rerank_session_20260410_0034.jsonl` |

Re-aggregate: `python benchmarks/aggregate_longmemeval_jsonl.py <path.jsonl>`

---

## Session-level means

| k | Mode | recall_any | recall | NDCG | all_gold_hit |
|---|------|------------|--------|------|--------------|
| @1 | raw | 0.8060 | 0.5108 | 0.8060 | 0.2700 |
| @1 | hybrid | 0.8880 | 0.5634 | 0.8880 | 0.2980 |
| @1 | rerank | 0.9200 | 0.5837 | 0.9200 | 0.3060 |
| @1 | hybrid_rerank | 0.9240 | 0.5867 | 0.9240 | 0.3080 |
| @3 | raw | 0.9260 | 0.8389 | 0.8741 | 0.7340 |
| @3 | hybrid | 0.9640 | 0.8928 | 0.9319 | 0.7980 |
| @3 | rerank | 0.9760 | 0.9205 | 0.9528 | 0.8460 |
| @3 | hybrid_rerank | 0.9760 | 0.9258 | 0.9537 | 0.8580 |
| @5 | raw | 0.9660 | 0.9171 | 0.8877 | 0.8500 |
| @5 | hybrid | 0.9780 | 0.9404 | 0.9337 | 0.8900 |
| @5 | rerank | 0.9780 | 0.9522 | 0.9513 | 0.9140 |
| @5 | hybrid_rerank | 0.9780 | 0.9528 | 0.9530 | 0.9160 |
| @10 | raw | 0.9820 | 0.9617 | 0.8888 | 0.9320 |
| @10 | hybrid | 0.9920 | 0.9760 | 0.9354 | 0.9480 |
| @10 | rerank | 0.9900 | 0.9758 | 0.9532 | 0.9560 |
| @10 | hybrid_rerank | 0.9860 | 0.9765 | 0.9526 | 0.9640 |
| @20 | raw | 0.9960 | 0.9846 | 0.8909 | 0.9680 |
| @20 | hybrid | 0.9960 | 0.9883 | 0.9346 | 0.9780 |
| @20 | rerank | 0.9960 | 0.9846 | 0.9543 | 0.9680 |
| @20 | hybrid_rerank | 0.9960 | 0.9883 | 0.9554 | 0.9780 |
| @50 | raw | 1.0000 | 1.0000 | 0.8898 | 1.0000 |
| @50 | hybrid | 1.0000 | 1.0000 | 0.9335 | 1.0000 |
| @50 | rerank | 1.0000 | 1.0000 | 0.9511 | 1.0000 |
| @50 | hybrid_rerank | 1.0000 | 1.0000 | 0.9537 | 1.0000 |

### Notes

- **Hybrid** lifts **R@1** and mid-cutoff NDCG vs **raw** dense retrieval.
- **Local cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`) improves top-of-list quality vs raw; vs hybrid-only, rerank still helps **@1** (e.g. hybrid 0.888 → hybrid_rerank 0.924 ndcg@1 proxy via recall_any@1).
- **`longmemeval_bench.py`** also supports **`rerank_bge_v2`**, **`hybrid_rerank_bge_v2`**, **`hybrid_rrf_bge_v2`**; add JSONLs here when full 500-Q runs are available.

---

## Per question_type — session R@10 / NDCG@10

| Type | n | Raw | Hybrid | + rerank | + hybrid_rerank |
|------|---|-----|--------|----------|-----------------|
| | | R@10 / N@10 | R@10 / N@10 | R@10 / N@10 | R@10 / N@10 |
| knowledge-update | 78 | 1.00 / 0.944 | 1.00 / 0.986 | 1.00 / 0.996 | 1.00 / 0.995 |
| multi-session | 133 | 1.00 / 0.914 | 1.00 / 0.947 | 1.00 / 0.975 | 1.00 / 0.975 |
| single-session-assistant | 56 | 0.964 / 0.953 | 0.964 / 0.952 | 0.964 / 0.955 | 0.964 / 0.964 |
| single-session-preference | 30 | 0.967 / 0.833 | 0.967 / 0.823 | 0.933 / 0.777 | 0.900 / 0.767 |
| single-session-user | 70 | 0.971 / 0.847 | 1.00 / 0.951 | 1.00 / 0.979 | 1.00 / 0.979 |
| temporal-reasoning | 133 | 0.970 / 0.839 | 0.993 / 0.904 | 0.993 / 0.931 | 0.985 / 0.929 |

---

## Reproducing

```bash
python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode raw
python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode hybrid_rerank
```

See [`benchmarks/README.md`](README.md) for options.
