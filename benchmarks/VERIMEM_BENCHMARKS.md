# VeriMem — LongMemEval benchmark report

**Dataset:** LongMemEval-style evaluation, **500 questions** per mode.  
**Metrics:** Mean over rows of `retrieval_results.metrics.session` (and turn recall where stored).  
**Aggregated:** 2026-04-07 from local benchmark runs (result JSONLs since removed; this file is the canonical record).

**Bench retrieval modes** (only these; `--mode` on `longmemeval_bench.py`, default **hybrid**):

| Mode | Meaning |
|------|--------|
| `hybrid` | Dense + BM25 fusion (default) |
| `raw` | Dense / embedding search only |
| `rerank` | Raw + local cross-encoder rerank |
| `hybrid_rerank` | Hybrid + local cross-encoder rerank |

The table below maps to: Raw, Hybrid, `rerank`, `hybrid_rerank` respectively.

---

## Session-level — recall_any@k and ndcg_any@k

| Metric | Raw | Hybrid | Raw + local rerank | Hybrid + local rerank |
|--------|-----|--------|--------------------|------------------------|
| recall@1 | 0.8060 | 0.8880 | 0.9200 | 0.9240 |
| ndcg@1 | 0.8060 | 0.8880 | 0.9200 | 0.9240 |
| recall@3 | 0.9260 | 0.9640 | 0.9760 | 0.9760 |
| ndcg@3 | 0.8741 | 0.9319 | 0.9528 | 0.9537 |
| recall@5 | 0.9660 | 0.9780 | 0.9780 | 0.9780 |
| ndcg@5 | 0.8877 | 0.9337 | 0.9513 | 0.9530 |
| recall@10 | 0.9820 | 0.9920 | 0.9900 | 0.9860 |
| ndcg@10 | 0.8888 | 0.9354 | 0.9532 | 0.9526 |
| recall@30 | 0.9960 | 0.9980 | 0.9960 | 0.9980 |
| ndcg@30 | 0.8890 | 0.9332 | 0.9505 | 0.9535 |
| recall@50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ndcg@50 | 0.8898 | 0.9335 | 0.9511 | 0.9537 |

### Notes

- **Hybrid** (dense + BM25) improves **R@1** and **NDCG@k** through mid cutoffs vs raw dense-only retrieval.
- **Local rerank** strongly improves **R@1 / NDCG@1–10** vs raw; vs hybrid, rerank still lifts top-of-list quality (e.g. ndcg@1: 0.888 → 0.924 hybrid path).
- **R@10:** hybrid + rerank (0.9860) is slightly below hybrid-only (0.9920) while **NDCG@10** remains higher than hybrid-only — consistent with rerank improving top ranks while occasionally demoting relevant items deeper in the list.

---

## Turn-level

Stored `metrics.turn` in these runs included **recall_any@**{1, 3, 5, 10, 30, 50} only — **no NDCG** at turn level in the serialized output.

| recall_any@k | Raw | Hybrid | Raw + rerank | Hybrid + rerank |
|----------------|-----|--------|--------------|-----------------|
| @1 | 0.8060 | 0.8880 | 0.9200 | 0.9240 |
| @3 | 0.9260 | 0.9640 | 0.9760 | 0.9760 |
| @5 | 0.9660 | 0.9780 | 0.9780 | 0.9780 |
| @10 | 0.9820 | 0.9920 | 0.9900 | 0.9860 |
| @30 | 0.9960 | 0.9980 | 0.9960 | 0.9980 |
| @50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## Per question_type — session R@10 and NDCG@10

| Type | n | Raw | Hybrid | Raw + rerank | Hybrid + rerank |
|------|---|-----|--------|--------------|-----------------|
| | | R@10 / N@10 | R@10 / N@10 | R@10 / N@10 | R@10 / N@10 |
| knowledge-update | 78 | 1.0000 / 0.9440 | 1.0000 / 0.9864 | 1.0000 / 0.9964 | 1.0000 / 0.9953 |
| multi-session | 133 | 1.0000 / 0.9139 | 1.0000 / 0.9474 | 1.0000 / 0.9749 | 1.0000 / 0.9746 |
| single-session-assistant | 56 | 0.9643 / 0.9533 | 0.9643 / 0.9516 | 0.9643 / 0.9554 | 0.9643 / 0.9643 |
| single-session-preference | 30 | 0.9667 / 0.8328 | 0.9667 / 0.8234 | 0.9333 / 0.7768 | 0.9000 / 0.7672 |
| single-session-user | 70 | 0.9714 / 0.8472 | 1.0000 / 0.9510 | 1.0000 / 0.9789 | 1.0000 / 0.9789 |
| temporal-reasoning | 133 | 0.9699 / 0.8386 | 0.9925 / 0.9038 | 0.9925 / 0.9314 | 0.9850 / 0.9287 |

**single-session-preference** shows the largest **R@10 / NDCG@10** drop when reranking vs non-rerank baselines — worth investigating reordering behavior on that slice.

---

## Reproducing

Run the LongMemEval harness in this repo (see `benchmarks/longmemeval_bench.py` and `benchmarks/README.md`) for each mode, then aggregate means from the emitted JSONL `retrieval_results.metrics.session` (and `turn` for recall).
