<div align="center">

# VeriMem

**Local agent memory:** Chroma or FastStore, dense + BM25 hybrid, cross-encoder rerank, optional NLI and entity graph. No API keys on the core path.

[![][version-shield]][release-link]
[![][python-shield]][python-link]
[![][license-shield]][license-link]

[Quick start](#quick-start) · [Modes](#recall-modes) · [Benchmarks](#benchmarks) · [Performance](#performance)

</div>

---
## What it is

Chunks text with `SentenceTransformer`; stores in **ChromaDB** or **FastStore**. **`recall()`**: `raw`, **BM25 hybrid**, **cross-encoder rerank**, **hybrid_rerank**, optional NLI / GLiNER + spaCy graph. Design lineage: [MemPalace](https://github.com/milla-jovovich/mempalace); this repo is independent.

---

## Quick start

```bash
pip install git+https://github.com/itachi-hue/verimem.git
pip install "verimem[nli]"     # ONNX CE (~3× CPU rerank)
pip install usearch            # optional FastStore
pip install gliner             # optional entity graph
```

```python
from verimem import Memory

mem = Memory()
mem.remember("Alice owns the auth service.")
mem.remember("We moved MySQL to Postgres in Q1 2026.", topic="infra")
print(mem.recall("what database?").to_simple())
```

**Dev:** `pip install -e ".[dev]"` from repo root. `recall()` returns a **`ContextPacket`** — use **`.to_simple()`** for LLM text and **`.to_dict()`** for provenance, freshness, completeness, policy, and store revision.

---

## Recall modes

| `mode` | Behaviour |
|--------|-----------|
| **`rerank`** (default) | ANN → cross-encoder reorder (`ms-marco-MiniLM-L-6-v2`). |
| **`raw`** | Dense retrieval only. |
| **`hybrid`** | BM25 + dense fusion (reads store for BM25 stats). |
| **`hybrid_rerank`** | Hybrid fusion, then cross-encoder. |

**Benchmark-only modes (LongMemEval harness):** `rerank_bge_v2`, `hybrid_rerank_bge_v2`, `hybrid_rrf_bge_v2`.

Extras: `verimem[nli]` for ONNX rerank; freshness decay; optional NLI.

---

## Benchmarks

Metrics are **session-level means** over all questions (ConvoMem-style): **recall_any@k** (≥1 gold in top-*k*), **recall@k** (gold-set coverage), **NDCG@k**, **all_gold_hit@k**. Full detail and repro: [`benchmarks/VERIMEM_BENCHMARKS.md`](benchmarks/VERIMEM_BENCHMARKS.md). Third-party numbers: [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).

### LongMemEval — 500 questions

**Setup:** `data/longmemeval_s_cleaned.json`, **embeddings** `all-MiniLM-L6-v2`, **rerank** `ms-marco-MiniLM-L-6-v2` where applicable. **Runs:** 2026-04-09–10.

**recall_any**

| @k | `raw` | `hybrid` | `rerank` | `hybrid_rerank` |
|:--:|:-----:|:--------:|:--------:|:---------------:|
| 1 | 0.806 | 0.888 | 0.920 | **0.924** |
| 3 | 0.926 | 0.964 | 0.976 | 0.976 |
| 5 | 0.966 | 0.978 | 0.978 | 0.978 |
| 10 | 0.982 | 0.992 | 0.990 | 0.986 |
| 20 | 0.996 | 0.996 | 0.996 | 0.996 |
| 50 | 1.000 | 1.000 | 1.000 | 1.000 |

**recall** (gold-set coverage)

| @k | `raw` | `hybrid` | `rerank` | `hybrid_rerank` |
|:--:|:-----:|:--------:|:--------:|:---------------:|
| 1 | 0.511 | 0.563 | 0.584 | **0.587** |
| 3 | 0.839 | 0.893 | 0.920 | 0.926 |
| 5 | 0.917 | 0.940 | 0.952 | 0.953 |
| 10 | 0.962 | 0.976 | 0.976 | 0.977 |
| 20 | 0.985 | 0.988 | 0.985 | 0.988 |
| 50 | 1.000 | 1.000 | 1.000 | 1.000 |

**NDCG**

| @k | `raw` | `hybrid` | `rerank` | `hybrid_rerank` |
|:--:|:-----:|:--------:|:--------:|:---------------:|
| 1 | 0.806 | 0.888 | 0.920 | **0.924** |
| 3 | 0.874 | 0.932 | 0.953 | 0.954 |
| 5 | 0.888 | 0.934 | 0.951 | 0.953 |
| 10 | 0.889 | 0.935 | 0.953 | 0.953 |
| 20 | 0.891 | 0.935 | 0.954 | **0.955** |
| 50 | 0.890 | 0.934 | 0.951 | 0.954 |

**all_gold_hit**

| @k | `raw` | `hybrid` | `rerank` | `hybrid_rerank` |
|:--:|:-----:|:--------:|:--------:|:---------------:|
| 1 | 0.270 | 0.298 | 0.306 | **0.308** |
| 3 | 0.734 | 0.798 | 0.846 | 0.858 |
| 5 | 0.850 | 0.890 | 0.914 | **0.916** |
| 10 | 0.932 | 0.948 | 0.956 | **0.964** |
| 20 | 0.968 | 0.978 | 0.968 | **0.978** |
| 50 | 1.000 | 1.000 | 1.000 | 1.000 |

Reproduce aggregates from full-run JSONL: `python benchmarks/aggregate_longmemeval_jsonl.py benchmarks/results_longmemeval_<...>.jsonl`

```bash
python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode hybrid_rerank
```

External stack comparisons (different protocols): [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md)

### ConvoMem — 300 items (6×50)

Full per-mode tables (**@1–@50**): [`benchmarks/convomem_results.md`](benchmarks/convomem_results.md).

#### @5

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 83.3% | 80.5% | 0.652 | 77.7% |
| `hybrid` | 87.0% | 83.3% | 0.687 | 79.7% |
| `rerank` | 83.3% | 80.4% | 0.715 | 77.3% |
| `hybrid_rerank` | 83.3% | 80.4% | 0.714 | 77.3% |
| `hybrid_rrf` | 86.0% | 82.9% | 0.714 | 79.7% |
| `rerank_bge_v2` | 86.7% | 84.8% | 0.735 | 83.0% |
| `hybrid_rrf_bge_v2` | **90.7%** | **87.8%** | **0.735** | **85.0%** |

#### @10

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 96.0% | 94.4% | 0.699 | 92.7% |
| `hybrid` | 96.7% | 95.4% | 0.730 | 94.0% |
| `rerank` | 91.7% | 90.4% | 0.748 | 89.0% |
| `hybrid_rerank` | 91.3% | 90.1% | 0.746 | 88.7% |
| `hybrid_rrf` | 96.0% | 94.5% | 0.753 | 93.0% |
| `rerank_bge_v2` | 94.7% | 93.7% | 0.765 | 92.7% |
| `hybrid_rrf_bge_v2` | **96.0%** | **95.4%** | 0.762 | **94.7%** |

#### @20

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 100.0% | 99.5% | 0.713 | 99.0% |
| `hybrid` | 99.3% | 99.0% | 0.741 | 98.7% |
| `rerank` | 100.0% | 99.5% | 0.773 | 99.0% |
| `hybrid_rerank` | 99.3% | 99.0% | 0.770 | 98.7% |
| `hybrid_rrf` | 99.3% | 99.0% | 0.766 | 98.7% |
| `rerank_bge_v2` | **100.0%** | **99.5%** | **0.780** | **99.0%** |
| `hybrid_rrf_bge_v2` | 99.3% | 99.0% | 0.772 | 98.7% |

**ConvoMem wall time**

| Mode | Wall time | Per item |
| --- | ---: | ---: |
| `hybrid_rrf` | ~721 s | ~2.4 s |
| `rerank_bge_v2` | ~975 s | ~3.25 s |
| `hybrid_rrf_bge_v2` | ~835 s | ~2.78 s |

### LoCoMo

See [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) (hybrid + rerank baselines vs session baseline).

---

## Performance

Warm in-process cache, typical CPU, persistent store. Prefer **ONNX** rerank (`verimem[nli]`).

| Call | Warm | Cold / first use |
|------|------|------------------|
| `remember()` | ~23 ms | ~120 ms (model load) |
| `recall` (`raw`) | ~2–3 ms | ~15–25 ms |
| `recall` (`rerank`) | ~3 ms | ~40–50 ms |

GPU: `Memory(..., device="cuda")` and `onnxruntime-gpu`. `python benchmarks/memory_perf_bench.py --device cuda`.

---

## Models (first download)

| Model | ~Size | Role |
|-------|------:|------|
| `all-MiniLM-L6-v2` | 90 MB | Embeddings |
| `ms-marco-MiniLM-L-6-v2` | 22 MB | Rerank |
| `BAAI/bge-reranker-v2-m3` | ~1.2 GB | BGE rerank (`rerank_bge_v2`, `hybrid_rerank_bge_v2`, `hybrid_rrf_bge_v2`) |
| `cross-encoder/nli-MiniLM2-L6-H768` | 90 MB | NLI |
| `urchade/gliner_small-v2.1` | 67 MB | Entities |

---

## Requirements

[`pyproject.toml`](pyproject.toml): Python 3.9–3.13, Chroma, sentence-transformers. Extras: `verimem[nli]`, `verimem[fast]`, `verimem[all]`, `verimem[dev]`.

---

## Layout

```
verimem/     memory.py  recall.py  fast_store.py  reranker.py  background_nli.py  graph.py
benchmarks/  longmemeval_bench.py  convomem_bench.py  …
tests/
```

---

## Limits

Light graph expansion; NLI/graph are async (first recall may show completeness flags). Not multi-tenant HA out of the box.

---

## License

[MIT](LICENSE). Copyright 2026 Vivek Rao.

<!-- Link Definitions -->
[version-shield]: https://img.shields.io/badge/version-5.0.0-4dc9f6?style=flat-square&labelColor=0a0e14
[release-link]: https://github.com/itachi-hue/verimem/releases
[python-shield]: https://img.shields.io/badge/python-3.9+-7dd8f8?style=flat-square&labelColor=0a0e14&logo=python&logoColor=7dd8f8
[python-link]: https://www.python.org/
[license-shield]: https://img.shields.io/badge/license-MIT-b0e8ff?style=flat-square&labelColor=0a0e14
[license-link]: https://github.com/itachi-hue/verimem/blob/main/LICENSE
