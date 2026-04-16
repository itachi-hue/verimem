# ConvoMem benchmark results (VeriMem harness)

Fixed slice: **6 categories × 50 items = 300** questions (`--category all --limit 50`).  
**Default `--top-k 10`** for the legacy substring-overlap “avg recall” metric.  
**Golden corpus metrics** use `METRIC_KS = (1, 3, 5, 10, 20, 50)` over the full ranked list (see `convomem_bench.py`).

| Setting | Value |
| --- | --- |
| Dense / hybrid embedding | `all-MiniLM-L6-v2` (see harness) |
| MiniLM cross-encoder | `ms-marco-MiniLM-L-6-v2` (`rerank`, `hybrid_rerank`, `hybrid_rrf`) |
| BGE reranker | `BAAI/bge-reranker-v2-m3` (`rerank_bge_v2`, `hybrid_rrf_bge_v2`) |
| Rerank pool | 20 |
| Hybrid weight | 0.35 (`hybrid`, `hybrid_rerank`, `hybrid_rrf`, `hybrid_rrf_bge_v2`) |
| RRF smoothing `k` | 60 (`hybrid_rrf`, `hybrid_rrf_bge_v2`) |
| Run dates (local) | 2026-04-09 (all JSONs below) |

## Source JSON (300 items each)

| Mode | Results file |
| --- | --- |
| `raw` | `results_convomem_raw_top10_20260409_1609.json` |
| `hybrid` | `results_convomem_hybrid_top10_20260409_1621.json` |
| `rerank` | `results_convomem_rerank_top10_20260409_1632.json` |
| `hybrid_rerank` | `results_convomem_hybrid_rerank_top10_20260409_1644.json` |
| `hybrid_rrf` | `results_convomem_hybrid_rrf_top10_20260409_2231.json` |
| `rerank_bge_v2` | `results_convomem_rerank_bge_v2_top10_20260409_2124.json` |
| `hybrid_rrf_bge_v2` | `results_convomem_hybrid_rrf_bge_v2_top10_20260409_2140.json` |

---

## Commands

```bash
python benchmarks/convomem_bench.py --category all --limit 50 --mode raw
python benchmarks/convomem_bench.py --category all --limit 50 --mode hybrid
python benchmarks/convomem_bench.py --category all --limit 50 --mode rerank
python benchmarks/convomem_bench.py --category all --limit 50 --mode hybrid_rerank
python benchmarks/convomem_bench.py --category all --limit 50 --mode hybrid_rrf
python benchmarks/convomem_bench.py --category all --limit 50 --mode rerank_bge_v2
python benchmarks/convomem_bench.py --category all --limit 50 --mode hybrid_rrf_bge_v2
```

On Windows, set `--cache-dir` to a writable folder (e.g. `%TEMP%\convomem_cache`) if `/tmp` is not used.

---

## Timing

| Mode | Wall time | Per item |
| --- | ---: | ---: |
| `hybrid_rrf` (MiniLM RRF) | 721.2 s (~12.0 min) | 2.40 s |
| `rerank_bge_v2` | 974.7 s (~16.3 min) | 3.25 s |
| `hybrid_rrf_bge_v2` | 834.7 s (~13.9 min) | 2.78 s |

---

## Legacy avg recall (substring overlap in top‑10)

Higher is better. “Perfect” = all evidence substrings matched in the retrieved top‑10.

| Mode | Avg recall | Perfect (1.0) | Zero (0.0) |
| --- | ---: | ---: | ---: |
| `raw` | 0.927 | 273 / 300 (91.0%) | 17 (5.7%) |
| `hybrid` | **0.937** | **277 / 300 (92.3%)** | 15 (5.0%) |
| `rerank` | 0.887 | 262 / 300 (87.3%) | 30 (10.0%) |
| `hybrid_rerank` | 0.883 | 261 / 300 (87.0%) | 31 (10.3%) |
| `hybrid_rrf` | 0.928 | 274 / 300 (91.3%) | 17 (5.7%) |
| `rerank_bge_v2` | 0.920 | 273 / 300 (91.0%) | 21 (7.0%) |
| `hybrid_rrf_bge_v2` | **0.937** | **279 / 300 (93.0%)** | 17 (5.7%) |

---

## Golden corpus IDs (mean over items)

**recall_any@K** — share of items with ≥1 gold corpus row in the top‑K.  
**recall@K** — mean fraction of gold ids covered in top‑K.  
**NDCG@K** — mean normalized DCG.  
**all_gold@K** — share of items where every gold id appears in top‑K.

Saved JSON from the first four non-RRF modes was produced before per-item metrics included **@1** and **@3**; re-run with current `convomem_bench.py` to refresh. **BGE** runs and the latest **`hybrid_rrf`** (MiniLM) JSON include full `METRIC_KS`.

### @1 and @3 (BGE rerankers + latest `hybrid_rrf`)

| Mode | @1 recall_any | @1 recall | @1 NDCG | @1 all_gold | @3 recall_any | @3 recall | @3 NDCG | @3 all_gold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rerank_bge_v2` | 59.3% | 54.3% | 0.593 | 49.3% | 81.3% | 76.9% | 0.699 | 72.7% |
| `hybrid_rrf_bge_v2` | 57.3% | 52.6% | 0.573 | 48.0% | 80.7% | 76.2% | 0.684 | 71.7% |
| `hybrid_rrf` (MiniLM) | 58.0% | 53.6% | 0.580 | 49.3% | 78.3% | 73.5% | 0.671 | 68.7% |

### @5

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 83.3% | 80.5% | 0.652 | 77.7% |
| `hybrid` | 87.0% | 83.3% | 0.687 | 79.7% |
| `rerank` | 83.3% | 80.4% | 0.715 | 77.3% |
| `hybrid_rerank` | 83.3% | 80.4% | 0.714 | 77.3% |
| `hybrid_rrf` | 86.0% | 82.9% | 0.714 | 79.7% |
| `rerank_bge_v2` | 86.7% | 84.8% | 0.735 | 83.0% |
| `hybrid_rrf_bge_v2` | **90.7%** | **87.8%** | **0.735** | **85.0%** |

### @10

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 96.0% | 94.4% | 0.699 | 92.7% |
| `hybrid` | 96.7% | 95.4% | 0.730 | 94.0% |
| `rerank` | 91.7% | 90.4% | 0.748 | 89.0% |
| `hybrid_rerank` | 91.3% | 90.1% | 0.746 | 88.7% |
| `hybrid_rrf` | 96.0% | 94.5% | 0.753 | 93.0% |
| `rerank_bge_v2` | 94.7% | 93.7% | 0.765 | 92.7% |
| `hybrid_rrf_bge_v2` | **96.0%** | **95.4%** | 0.762 | **94.7%** |

### @20

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 100.0% | 99.5% | 0.713 | 99.0% |
| `hybrid` | 99.3% | 99.0% | 0.741 | 98.7% |
| `rerank` | 100.0% | 99.5% | 0.773 | 99.0% |
| `hybrid_rerank` | 99.3% | 99.0% | 0.770 | 98.7% |
| `hybrid_rrf` | 99.3% | 99.0% | 0.766 | 98.7% |
| `rerank_bge_v2` | **100.0%** | **99.5%** | **0.780** | **99.0%** |
| `hybrid_rrf_bge_v2` | 99.3% | 99.0% | 0.772 | 98.7% |

### @50

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 100.0% | 100.0% | 0.714 | 100.0% |
| `hybrid` | 100.0% | 100.0% | 0.743 | 100.0% |
| `rerank` | 100.0% | 100.0% | 0.774 | 100.0% |
| `hybrid_rerank` | 100.0% | 100.0% | 0.772 | 100.0% |
| `hybrid_rrf` | 100.0% | 100.0% | 0.768 | 100.0% |
| `rerank_bge_v2` | 100.0% | 100.0% | **0.782** | 100.0% |
| `hybrid_rrf_bge_v2` | 100.0% | 100.0% | 0.774 | 100.0% |

---

## Per-category legacy recall (top‑10 overlap)

Category order: Assistant Facts, User Facts, Changing Facts, Abstention, Implicit Connections, Preferences.

### `raw`

| Category | Mean recall | Perfect |
| --- | ---: | --- |
| Assistant Facts | 1.000 | 50/50 |
| User Facts | 0.980 | 49/50 |
| Changing Facts | 0.920 | 43/50 |
| Abstention | 0.910 | 45/50 |
| Implicit Connections | 0.893 | 43/50 |
| Preferences | 0.860 | 43/50 |

### `hybrid`

| Category | Mean recall | Perfect |
| --- | ---: | --- |
| Assistant Facts | 1.000 | 50/50 |
| User Facts | 0.980 | 49/50 |
| Changing Facts | 0.940 | 45/50 |
| Abstention | 0.910 | 45/50 |
| Implicit Connections | 0.913 | 44/50 |
| Preferences | 0.880 | 44/50 |

### `rerank`

| Category | Mean recall | Perfect |
| --- | ---: | --- |
| Assistant Facts | 1.000 | 50/50 |
| User Facts | 0.980 | 49/50 |
| Changing Facts | 0.950 | 45/50 |
| Abstention | 0.910 | 45/50 |
| Implicit Connections | 0.680 | 33/50 |
| Preferences | 0.800 | 40/50 |

### `hybrid_rerank`

| Category | Mean recall | Perfect |
| --- | ---: | --- |
| Assistant Facts | 1.000 | 50/50 |
| User Facts | 0.980 | 49/50 |
| Changing Facts | 0.950 | 45/50 |
| Abstention | 0.910 | 45/50 |
| Implicit Connections | 0.660 | 32/50 |
| Preferences | 0.800 | 40/50 |

### `hybrid_rrf`

| Category | Mean recall | Perfect |
| --- | ---: | --- |
| Assistant Facts | 1.000 | 50/50 |
| User Facts | 0.980 | 49/50 |
| Changing Facts | 0.950 | 45/50 |
| Abstention | 0.910 | 45/50 |
| Implicit Connections | 0.847 | 41/50 |
| Preferences | 0.880 | 44/50 |

### `rerank_bge_v2`

| Category | Mean recall | Perfect |
| --- | ---: | --- |
| Assistant Facts | 1.000 | 50/50 |
| User Facts | 0.980 | 49/50 |
| Changing Facts | 0.980 | 48/50 |
| Abstention | 0.910 | 45/50 |
| Implicit Connections | 0.830 | 40/50 |
| Preferences | 0.820 | 41/50 |

### `hybrid_rrf_bge_v2`

| Category | Mean recall | Perfect |
| --- | ---: | --- |
| Assistant Facts | 1.000 | 50/50 |
| User Facts | 0.980 | 49/50 |
| Changing Facts | 0.990 | 49/50 |
| Abstention | 0.910 | 45/50 |
| Implicit Connections | 0.860 | 42/50 |
| Preferences | 0.880 | 44/50 |

---

## End-to-end Q&A (Claude Sonnet 4.5)

[`convomem_qa_bench.py`](convomem_qa_bench.py) — retrieval **`hybrid_rrf_bge_v2`**, then answer generation and binary judge with **Claude Sonnet 4.5**.

**Saved run:** [`results_convomem_qa_claude_20260410_2241.json`](results_convomem_qa_claude_20260410_2241.json) (2026-04-10, `top_k=10`, ~33 items/category, binary scoring).

| | |
| --- | ---: |
| Items (successful) | 198 |
| Judge accuracy | 90.4% (179 / 198) |
| Mean retrieval recall@k | 0.97 |

| Category | Accuracy | Retrieval recall |
| --- | ---: | ---: |
| user_evidence | 90.9% | 1.00 |
| assistant_facts_evidence | 97.0% | 0.985 |
| changing_evidence | 97.0% | 1.00 |
| abstention_evidence | 78.8% | 0.97 |
| preference_evidence | 87.9% | 0.909 |
| implicit_connection_evidence | 90.9% | 0.955 |

### Groq Llama 4 Scout (300 items, full slice)

[`convomem_qa_groq_bench.py`](convomem_qa_groq_bench.py) — same ConvoMem Q&A as above, but retrieval uses **`Memory.recall(..., mode="hybrid")`** (dense + BM25), not BGE RRF. Answer + binary judge with **Llama 4 Scout** via Groq (`meta-llama/llama-4-scout-17b-16e-instruct`). **Full benchmark slice:** 6 categories × 50 = **300** items, `top_k=10`, **`packet_mode=full`** (entire `ContextPacket.to_simple()` JSON in the prompt).

**Saved run:** [`results_convomem_qa_groq_packet_full_300.json`](results_convomem_qa_groq_packet_full_300.json) (2026-04-11, run id `20260411_0030`).

| | |
| --- | ---: |
| Items (successful) | 300 |
| Judge accuracy | 90.3% (271 / 300) |
| Mean retrieval recall@k | 0.956 |

| Category | Accuracy | Retrieval recall |
| --- | ---: | ---: |
| user_evidence | 96.0% | 1.00 |
| assistant_facts_evidence | 100.0% | 0.99 |
| changing_evidence | 86.0% | 0.953 |
| abstention_evidence | 92.0% | 0.98 |
| preference_evidence | 86.0% | 0.88 |
| implicit_connection_evidence | 82.0% | 0.933 |

Install: `pip install -e ".[groq]"` (or `pip install groq`); set **`GROQ_API_KEY`**. **Retries with backoff** on 429/5xx (no client-side RPM cap). Reproduce: `python benchmarks/convomem_qa_groq_bench.py --total 300 --packet-mode full`. Other runs: `benchmarks/results_convomem_qa_groq_<timestamp>.json`.

---

## Notes

- **@1 / @3** in JSON: only the two BGE result files include those keys; the five MiniLM-path runs predate `METRIC_KS` including 1 and 3. Metrics at **@5–@50** are taken from stored `details.metrics` in each JSON.
- **MiniLM** = `ms-marco-MiniLM-L-6-v2` cross-encoder where used; **BGE** modes use `BAAI/bge-reranker-v2-m3` instead for `rerank` / RRF’s CE leg.
- Hardware affects wall time; only BGE runs logged timing above.
- For external **MemPal 92.9%** comparisons, see [`BENCHMARKS.md`](BENCHMARKS.md) — metric definitions differ slightly from this harness (legacy overlap vs golden-id table).
