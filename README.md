<div align="center">

# VeriMem

Local semantic memory: **Chroma** or **FastStore**, hybrid BM25 + dense, cross-encoder rerank, optional NLI / entity graph. No API keys on the core path.

[![][version-shield]][release-link]
[![][python-shield]][python-link]
[![][license-shield]][license-link]

[Quick start](#quick-start) · [Modes](#recall-modes) · [Benchmarks](#benchmarks) · [Performance](#performance)

</div>

---

## What it is

`SentenceTransformer` embeddings; storage in **ChromaDB** or **FastStore** (usearch + SQLite). Related ideas: [MemPalace](https://github.com/milla-jovovich/mempalace); this repo is separate.

---

## Quick start

```bash
pip install git+https://github.com/itachi-hue/verimem.git
pip install "verimem[nli]"     # ONNX rerank (faster on CPU)
pip install usearch            # FastStore
pip install gliner             # optional entities
```

```python
from verimem import Memory

mem = Memory()
mem.remember("Alice owns the auth service.")
print(mem.recall("who owns auth?").to_simple())
```

`pip install -e ".[dev]"` for dev. `recall()` returns **`ContextPacket`** (`.to_simple()` / `.to_dict()`).

---

## Recall modes

| `mode` | Behaviour |
|--------|-----------|
| `rerank` (default) | ANN → `ms-marco-MiniLM-L-6-v2` cross-encoder |
| `raw` | Dense only |
| `hybrid` | BM25 + dense fusion |
| `hybrid_rerank` | Hybrid then cross-encoder |

Install `verimem[nli]` for ONNX-backed rerank on CPU.

---

## Benchmarks

Metrics are **session-level means** (ConvoMem-style): **recall_any@k**, **recall@k** (gold-set coverage), **NDCG@k**, **all_gold_hit@k**. Extra detail and full @1–@50 LongMemEval breakdown: [`benchmarks/VERIMEM_BENCHMARKS.md`](benchmarks/VERIMEM_BENCHMARKS.md).

### LongMemEval — 500 questions

**Setup:** `data/longmemeval_s_cleaned.json`, embeddings `all-MiniLM-L6-v2`, rerank `ms-marco-MiniLM-L-6-v2` where applicable. **Runs:** 2026-04-09–10.

#### @5

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 96.6% | 91.7% | 0.888 | 85.0% |
| `hybrid` | 97.8% | 94.0% | 0.934 | 89.0% |
| `rerank` | 97.8% | 95.2% | 0.951 | 91.4% |
| `hybrid_rerank` | 97.8% | **95.3%** | **0.953** | **91.6%** |

#### @10

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 98.2% | 96.2% | 0.889 | 93.2% |
| `hybrid` | 99.2% | 97.6% | 0.935 | 94.8% |
| `rerank` | 99.0% | 97.6% | 0.953 | 95.6% |
| `hybrid_rerank` | 98.6% | 97.7% | 0.953 | **96.4%** |

#### @20

| Mode | recall_any | recall | NDCG | all_gold |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 99.6% | 98.5% | 0.891 | 96.8% |
| `hybrid` | 99.6% | 98.8% | 0.935 | 97.8% |
| `rerank` | 99.6% | 98.5% | 0.954 | 96.8% |
| `hybrid_rerank` | 99.6% | **98.8%** | **0.955** | **97.8%** |

```bash
python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode hybrid_rerank
python benchmarks/aggregate_longmemeval_jsonl.py benchmarks/results_longmemeval_<mode>_session_<run>.jsonl
```

### ConvoMem — 300 items (6×50)

Full per-mode tables (**@1–@50**, legacy overlap, per-category): [`benchmarks/convomem_results.md`](benchmarks/convomem_results.md).

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

```bash
python benchmarks/convomem_bench.py --category all --limit 50 --mode hybrid_rerank
```

---

## Performance

Typical CPU, warm cache. ONNX rerank via `verimem[nli]`.

| Call | Warm | Cold |
|------|------|------|
| `remember()` | ~23 ms | ~120 ms (model load) |
| `recall` (`raw`) | ~2–3 ms | ~15–25 ms |
| `recall` (`rerank`) | ~3 ms | ~40–50 ms |

GPU: `Memory(..., device="cuda")` + `onnxruntime-gpu`; `python benchmarks/memory_perf_bench.py --device cuda`.

---

## Models

| Model | Role |
|-------|------|
| `all-MiniLM-L6-v2` | Embeddings |
| `ms-marco-MiniLM-L-6-v2` | Rerank |
| `BAAI/bge-reranker-v2-m3` | BGE modes (see `convomem_results.md`) |
| `cross-encoder/nli-MiniLM2-L6-H768` | Optional NLI |
| `urchade/gliner_small-v2.1` | Optional entities |

---

## Requirements

Python 3.9–3.13. See [`pyproject.toml`](pyproject.toml): `verimem[nli]`, `verimem[fast]`, `verimem[dev]`, etc.

Not a full graph database; NLI / graph are async (first recall may show completeness flags).

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
