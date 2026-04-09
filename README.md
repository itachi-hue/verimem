<div align="center">

# VeriMem

### Local memory for AI agents — structured recall, optional rerank, NLI, and a small entity graph.

<br>

[![][version-shield]][release-link]
[![][python-shield]][python-link]
[![][license-shield]][license-link]

<br>

[Quick start](#quick-start) · [Context packet](#what-you-get-back) · [Pipeline](#recall-pipeline) · [Benchmarks](#benchmarks) · [Performance](#performance) · [Limits](#what-it-does-not-do)

</div>

---

## What it is

A Python library: chunk text, embed with `SentenceTransformer`, store in **ChromaDB** or (with `usearch`) a **SQLite + in-RAM HNSW** backend, retrieve with semantic search. On top of that: **local cross-encoder reranking**, **freshness decay**, **background NLI** for contradictions (cached in SQLite), and an optional **GLiNER + spaCy** entity graph. No API keys for the core path.

**Install:** `pip install verimem` (or from source). Import the `verimem` package:

```python
from verimem import Memory

mem = Memory()
mem.remember("Alice manages the auth service. JWT expires in 24h.")
mem.remember("We migrated from Postgres to MySQL in Q1 2026.", topic="infra")

result = mem.recall("what database do we use?")
print(result.to_simple())
```

### Positioning

VeriMem is meant for **structured, trustworthy recall** for agents **without** sending your transcripts to a hosted memory API. The direction is **inspired by** the push for local, long-horizon agent memory—including work and ideas in the ecosystem such as **[MemPalace](https://github.com/milla-jovovich/mempalace)** and the **MemPal** LongMemEval lines—but this repository is an **independent implementation**: its own `Memory` API, retrieval modes (dense, hybrid, rerank), NLI, entity graph, and benchmark harness. We **benchmark against** MemPal and others; that is **not** the same as VeriMem being a drop-in fork of MemPalace.

---

## Quick start

```bash
pip install git+https://github.com/itachi-hue/verimem.git
pip install "verimem[nli]"    # optional: faster cross-encoder via ONNX (~3× CPU)
pip install gliner            # entity graph (~67MB model)
pip install usearch           # optional: faster persistent backend
```

```python
from verimem import Memory

mem = Memory()                 # default ~/.verimem
mem = Memory("/path/to/store")
mem = Memory(":memory:")       # ephemeral (tests / notebooks)

mem.remember("The payment service went down at 3pm.")
mem.remember("Payment restored at 3:47pm.", topic="incidents")

result = mem.recall("payment service status")
print(result.to_simple())
result.to_dict()               # full provenance + flags

mem.recall_related("Alice", hops=1)   # entity graph (after background extraction)
mem.forget(chunk_id)
mem.count()
mem.revision()                 # monotonic write counter (e.g. cache invalidation)
mem.graph_stats()
```

Console entry point: `verimem` prints a short usage hint (there is no interactive CLI).

### Install from this repo (editable)

Use a virtual environment, then from the repository root:

```bash
pip install -e .
pip install -e ".[nli]"       # recommended: ONNX for cross-encoder + NLI
pip install -e ".[fast]"      # optional: usearch + SQLite FastStore
pip install -e ".[all]"       # nli + fast
pip install -e ".[dev]"       # pytest + ruff
```

Smoke test: `python -c "from verimem import Memory; m=Memory(':memory:'); m.remember('hi'); print(m.recall('hi').to_simple())"`  
CLI: `verimem` or `python -m verimem`.

On **Windows**, if `pip install` fails building `chroma-hnswlib`, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or use a Python version where Chroma publishes a prebuilt wheel for your platform.

---

## What you get back

`recall()` returns a **`ContextPacket`**. Use **`.to_simple()`** for LLM-facing output, **`.to_dict()`** for debugging.

### `.to_simple()`

High-signal fields only: hit text, topic, similarity, human-readable age, contradiction strings when available, and `entities` if you used `include_graph=True`.

### `.to_dict()`

Includes `filed_at`, `freshness_score`, chunk id, `completeness` flags, `policy_version` (`"default"` / `"rerank"`), and `store_revision` (monotonic write counter for cache invalidation; this field was named `palace_revision` before v5.0).

### `recall()` parameters

```python
mem.recall(
    query,
    top_k=5,
    topic=None,
    rerank=True,
    rerank_pool=20,
    min_similarity=0.0,
    decay_days=30.0,      # 0 disables freshness tie-break
    include_graph=False,
)
```

---

## Recall pipeline

`Memory.recall(..., mode=...)` picks how results are ranked (same four modes as `longmemeval_bench.py`):

| `mode` | Behaviour |
|--------|-----------|
| **`rerank`** (default) | Dense search (ChromaDB or `FastStore` + embedding cache), then **cross-encoder** reorder (`ms-marco-MiniLM-L-6-v2`; core dependency). |
| **`raw`** | Dense search only — no BM25, no cross-encoder. |
| **`hybrid`** | Dense search, then **BM25 + dense fusion** over all stored chunks (higher latency at large N). |
| **`hybrid_rerank`** | Hybrid fusion, then cross-encoder. |

**Rerank speed (same model, same scores):** without Optimum, the cross-encoder runs in PyTorch on CPU and can take hundreds of ms per query on a cold (query, chunk) cache. Install **`pip install verimem[nli]`** (pulls **Optimum ≥2.1** + ONNX Runtime; required for **PyTorch 2.9+**) so rerank uses **ONNX** — typically **~3× faster** on CPU, **no change** to weights or ranking intent.

Then: **freshness decay** re-orders by similarity × age (tie-break). Contradictions: **BackgroundNLI** scores pairs asynchronously (SQLite-cached); first run may set `contradiction_check_pending`.

```python
from verimem import Memory, RETRIEVAL_MODES, DEFAULT_RETRIEVAL_MODE

mem = Memory()
mem.recall("query", top_k=5, rerank_pool=20)  # default mode: rerank
mem.recall("query", mode="hybrid", top_k=5, hybrid_lexical_weight=0.35)  # optional BM25 fusion
```

`recall(rerank=True/False)` still works but is **deprecated** in favour of `mode`.

Hybrid modes scan **all** chunks in the store (for BM25 statistics) on each recall — fine for typical agent corpora; for very large indexes consider `raw` or `rerank` if you need to avoid full-store reads.

---

## Entity graph

Background worker: **GLiNER** (`urchade/gliner_small-v2.1`) + **spaCy** `en_core_web_sm` for light relation structure. Query with `recall_related()` or pass `include_graph=True` on `recall()`.

---

## Benchmarks

### At a glance — LongMemEval (500 questions, session-level)

Canonical numbers from this repo’s own runs are summarized in [`benchmarks/VERIMEM_BENCHMARKS.md`](benchmarks/VERIMEM_BENCHMARKS.md) (`longmemeval_bench.py`). **Recall** = gold session in top-*k*; **NDCG** rewards putting it higher in the list.

| Mode | R@1 | R@3 | R@5 | R@10 | NDCG@5 (≈) |
|------|:---:|:---:|:---:|:----:|:----------:|
| **Raw** (dense only) | 80.6% | 92.6% | 96.6% | 98.2% | 0.89 |
| **Hybrid** (dense + BM25) | 88.8% | 96.4% | 97.8% | 99.2% | 0.93 |
| **Raw + local rerank** | **92.0%** | **97.6%** | **97.8%** | **99.0%** | **0.95** |
| **Hybrid + local rerank** | **92.4%** | **97.6%** | **97.8%** | 98.6% | **0.95** |

**Takeaway:** local cross-encoder reranking lifts **R@1** by double digits vs raw dense retrieval (e.g. **80.6% → 92.0%**) while keeping **zero cloud LLM** calls. Hybrid helps the mid-funnel; rerank sharpens **top-of-list** quality (see NDCG).

Figures below are **as reported** in this repo’s `benchmarks/BENCHMARKS.md` (MemPal and other third-party claims) and from VeriMem’s own `longmemeval_bench.py` runs. **Protocols differ** (e.g. LLM rerank vs local cross-encoder, hybrid modes vs raw). Use them as orientation, not apples-to-apples without reading the scripts.

**LongMemEval R@5 — what “rerank” means here**

- **VeriMem 97.8% R@5** — yes, this is **with reranking**: a **local** cross-encoder (`ms-marco-MiniLM-L-6-v2`). No cloud LLM. Metric = gold session in the top-5 retrieved sessions (`--mode rerank` or `--mode hybrid_rerank` in `longmemeval_bench.py`).
- **MemPal 96.6% R@5** — **no** rerank: raw Chroma embedding search only (the famous zero-API baseline).
- **MemPal 99.4–100% R@5** — **yes, reranked**, but with **Claude Haiku / Sonnet** over the candidate list (`--llm-rerank` + hybrid/palace modes in their pipeline). Higher score, different cost and latency than VeriMem’s local rerank.

So MemPal’s best LongMemEval numbers **do** use reranking — **LLM** reranking. VeriMem sits between: **better than raw (96.6%)**, **below full LLM rerank (≈100%)**, without an API.

### LongMemEval — R@5 (500 questions, session-level retrieval)

| System | R@5 | LLM / API | Notes |
|--------|:---:|:---:|---|
| MemPal hybrid v4 + Haiku rerank | **100%** | Yes | First reported 500/500; `longmemeval_bench.py --llm-rerank` |
| MemPal hybrid v3 + Haiku rerank | 99.4% | Yes | — |
| MemPal palace + Haiku rerank | 99.4% | Yes | — |
| Supermemory ASMR (experimental) | ~99% | Yes | Not identical to production track |
| **VeriMem + local rerank** | **97.8%** | **No** | `ms-marco-MiniLM` cross-encoder, this repo |
| MemPal **raw** (Chroma, no rerank) | **96.6%** | **No** | Highest cited zero-LLM baseline in that doc |
| Mastra Observational Memory | 94.87% | Yes | e.g. GPT-5-mini in published leaderboard |
| Hindsight | 91.4% | Yes | Gemini-class judge / pipeline |
| Supermemory (production) | ~85% | Yes | — |
| Stella (dense retriever) | ~85% | No | Academic baseline |
| Contriever | ~78% | No | — |
| BM25 | ~70% | No | Sparse baseline |

### LongMemEval — VeriMem vs MemPal raw (all R@k)

Same benchmark family; VeriMem numbers from `longmemeval_bench.py --mode rerank` (and related modes) vs `--mode raw` (see `benchmarks/VERIMEM_BENCHMARKS.md`).

| Setup | R@1 | R@3 | R@5 | R@10 |
|------|:---:|:---:|:---:|:---:|
| **VeriMem + local rerank** | **92.0%** | **97.6%** | **97.8%** | **99.0%** |
| Raw retrieval, no rerank (same bench; **R@5 = 96.6%** matches published MemPal raw) | 80.6% | 92.6% | 96.6% | 98.2% |

### LongMemEval — per-type R@10 (500q)

| Question type | MemPal raw | VeriMem + local rerank |
|----------------|:----------:|:----------------------:|
| knowledge-update (n=78) | 100.0% | 100.0% |
| multi-session (n=133) | 100.0% | 100.0% |
| temporal-reasoning (n=133) | 97.0% | **99.2%** |
| single-session-user (n=70) | 97.1% | **100.0%** |
| single-session-preference (n=30) | 96.7% | 93.3% |
| single-session-assistant (n=56) | 96.4% | 96.4% |

```bash
# Default retrieval matches Memory(): rerank (dense + local CE). Other modes: raw, hybrid, hybrid_rerank.
python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode hybrid_rerank
```

### LoCoMo (1,986 QA pairs, MemPal family — see `locomo_bench.py`)

| Mode | R@5 | R@10 | LLM | Notes |
|------|:---:|:---:|:---:|---|
| Hybrid v5 + Sonnet rerank, top-50 | **100%** | **100%** | Yes | — |
| bge-large + Haiku rerank, top-15 | — | 96.3% | Yes | — |
| bge-large hybrid, top-10 | — | 92.4% | No | — |
| Hybrid v5, top-10 | 83.7% | 88.9% | No | — |
| Session baseline, top-10, no rerank | — | 60.3% | No | — |
| Dialog baseline, top-10 | — | 48.0% | No | Harder granularity |

### ConvoMem (Salesforce; sampled runs in `convomem_bench.py`)

| System | Score | Notes |
|--------|------:|---|
| MemPal (reported) | **92.9%** | Verbatim + semantic search, multi-category sample |
| Gemini long context | 70–82% | Full history in window |
| Block extraction | 57–71% | LLM-processed blocks |
| Mem0 (RAG-style) | 30–45% | LLM-extracted memories (per this repo’s benchmark doc) |

**Per-category (MemPal, ConvoMem sample):** Assistant facts 100%, user facts 98%, abstention 91%, implicit connections 89.3%, preferences 86%.

---

**More detail, caveats, and reproducibility:** [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) · full mode-by-mode metrics: [`benchmarks/VERIMEM_BENCHMARKS.md`](benchmarks/VERIMEM_BENCHMARKS.md).

---

## Performance

**Why it matters:** strong **R@1 / NDCG** only help agents if recall is also **fast enough** to sit in the tool loop. Below are representative **single-process** latencies on typical CPU hardware with a **warm** in-process cache (second and later queries in a session are often much cheaper).

Typical CPU, persistent store, warm models. **Cache hit** = same query (embedding cache) or same `(query, chunk)` (rerank cache) in-process.

| Call | Cache hit | Cache miss |
|------|-----------|------------|
| `remember()` | ~23 ms | ~120 ms first load |
| `recall(rerank=False)` | ~2–3 ms | ~15–25 ms |
| `recall(rerank=True)` | ~3 ms | ~40–50 ms |

With **`usearch`** + `FastStore`, persistent ANN is in-RAM; without it, ChromaDB persists on disk (somewhat higher latency). Caches reset when the process exits; vectors and text stay on disk.

**GPU:** pass `Memory(..., device="cuda")` (or `"auto"` when PyTorch is built with CUDA). Install a **CUDA** `torch` wheel (not `+cpu`) and `onnxruntime-gpu` (uninstall CPU `onnxruntime` first). With `onnxruntime-gpu`, ORT often lists TensorRT first; VeriMem forces **`CUDAExecutionProvider`** for the cross-encoder when `device=cuda` so rerank uses the GPU without a full TensorRT install. `python benchmarks/memory_perf_bench.py --device cuda` prints the resolved compute device.

---

## Models (local, after download)

| Model | ~Size | Role |
|-------|------:|------|
| `all-MiniLM-L6-v2` | 90 MB | Embeddings (cached per text) |
| `ms-marco-MiniLM-L-6-v2` | 22 MB | Rerank (ONNX if `optimum[onnxruntime]`, else PyTorch) |
| `cross-encoder/nli-MiniLM2-L6-H768` | 90 MB | NLI / contradictions |
| `urchade/gliner_small-v2.1` | 67 MB | Entity mentions |

---

## Requirements

Declared in [`pyproject.toml`](pyproject.toml). Summary:

| | Packages | Purpose |
|---|----------|---------|
| **Core** | `chromadb>=0.5,<0.7`, `pyyaml>=6`, `sentence-transformers>=2.7` | `Memory()`, embeddings, cross-encoder rerank, BackgroundNLI |
| **Python** | 3.9 – 3.13 | — |
| **`verimem[nli]`** | `optimum[onnxruntime]` | Faster cross-encoder on CPU when ONNX loads (PyTorch fallback always available) |
| **`verimem[fast]`** | `usearch` ≥2 | `FastStore` (Rust HNSW + SQLite) instead of Chroma-only persistence |
| **`verimem[all]`** | `nli` + `fast` deps | Convenience extra for full optional stack |
| **`verimem[spellcheck]`** | `autocorrect` | Optional (legacy extra; unused by core `Memory`) |
| **Graph (manual)** | `gliner` · `spacy` + `en_core_web_sm` | Background entity / relation extraction |
| **Dev** | `pytest` · `ruff` | `verimem[dev]` / `dependency-groups.dev` |

Minimal install: `pip install verimem` (or from Git) → remember/recall, hybrid BM25, rerank modes, and NLI all work (models download on first use). Graph still needs `gliner` and the spaCy model installed separately.

---

## Project layout

```
verimem/
  memory.py          Memory API
  recall.py          ContextPacket, RecallHit, to_simple / to_dict
  fast_store.py      usearch + SQLite (optional)
  graph.py           MemoryGraph, BackgroundGraph
  background_nli.py  NLI worker + SQLite cache
  reranker.py        Cross-encoder + score cache
  policy.py          Named presets (metadata / tooling)
  revision.py        Write counter
  cli.py             Prints library usage
benchmarks/          LongMemEval, LoCoMo, ConvoMem runners
tests/
```

---

## What it does not do

- Not a full graph database; `recall_related()` is for light entity-centric expansion.
- Not multi-tenant or horizontally scaled out of the box.
- NLI and graph enrichment are **async**; first recall may not show all signals yet (`completeness` flags document that).

---

## License

[MIT](LICENSE). Copyright 2026 Vivek Rao.

Redistributions must **preserve the copyright notice and permission text** in `LICENSE`. For academic or product write-ups that use VeriMem, please **cite this repository** (or an associated publication). If you discuss lineage or related work, credit community projects that informed the design—such as [MemPalace](https://github.com/milla-jovovich/mempalace)—when your work draws on their ideas or shared code.

<!-- Link Definitions -->
[version-shield]: https://img.shields.io/badge/version-5.0.0-4dc9f6?style=flat-square&labelColor=0a0e14
[release-link]: https://github.com/itachi-hue/verimem/releases
[python-shield]: https://img.shields.io/badge/python-3.9+-7dd8f8?style=flat-square&labelColor=0a0e14&logo=python&logoColor=7dd8f8
[python-link]: https://www.python.org/
[license-shield]: https://img.shields.io/badge/license-MIT-b0e8ff?style=flat-square&labelColor=0a0e14
[license-link]: https://github.com/itachi-hue/verimem/blob/main/LICENSE
