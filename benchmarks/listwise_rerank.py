"""
Benchmark-only local rerank helpers.

- Default (local_ce_model=None): delegates to verimem CrossEncoderReranker (MS MARCO MiniLM).
- jinaai/jina-reranker-v3: listwise ``transformers`` + ``trust_remote_code`` (0.6B, CC BY-NC 4.0).
- Any other HF id: ``sentence_transformers.CrossEncoder`` (pairwise), cached per model id.

Does not change ``Memory.recall`` defaults — use only from ``longmemeval_bench.py``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_jina_models: dict[str, object] = {}
_cross_encoders: dict[str, object] = {}


def apply_local_rerank_indices(
    query: str,
    rankings: List[int],
    corpus: List[str],
    top_k: int,
    local_ce_model: Optional[str],
) -> List[int]:
    """
    Reorder ``rankings[:top_k]`` by local reranking; append ``rankings[top_k:]`` unchanged.
    """
    if not rankings:
        return rankings
    if not local_ce_model:
        from verimem.reranker import CrossEncoderReranker

        return CrossEncoderReranker.get().rerank_indices(query, rankings, corpus, top_k=top_k)

    candidates = rankings[:top_k]
    tail = rankings[top_k:]
    if not candidates:
        return tail

    mid = local_ce_model.lower()
    if "jina-reranker-v3" in mid:
        return _jina_v3_rerank(query, candidates, corpus, tail, local_ce_model)
    return _cross_encoder_rerank(query, candidates, corpus, tail, local_ce_model)


def _jina_v3_rerank(
    query: str,
    candidates: List[int],
    corpus: List[str],
    tail: List[int],
    model_id: str,
) -> List[int]:
    docs = [corpus[i] for i in candidates]
    model = _get_jina(model_id)
    try:
        results = model.rerank(query, docs, top_n=len(docs))
    except Exception as exc:
        logger.warning("Jina listwise rerank failed: %s", exc)
        return candidates + tail

    seen_order: List[int] = []
    seen: set[int] = set()
    for r in results:
        idx = r["index"]
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            if c not in seen:
                seen_order.append(c)
                seen.add(c)
    rest = [c for c in candidates if c not in seen]
    return seen_order + rest + tail


def _get_jina(model_id: str):
    if model_id not in _jina_models:
        from transformers import AutoModel

        try:
            m = AutoModel.from_pretrained(model_id, dtype="auto", trust_remote_code=True)
        except TypeError:
            m = AutoModel.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
        m.eval()
        _jina_models[model_id] = m
    return _jina_models[model_id]


def _cross_encoder_rerank(
    query: str,
    candidates: List[int],
    corpus: List[str],
    tail: List[int],
    model_id: str,
) -> List[int]:
    from sentence_transformers import CrossEncoder

    if model_id not in _cross_encoders:
        try:
            ce = CrossEncoder(
                model_id, trust_remote_code=True, model_kwargs={"torch_dtype": "auto"}
            )
        except (TypeError, ValueError):
            ce = CrossEncoder(model_id, trust_remote_code=True)
        _cross_encoders[model_id] = ce
    model = _cross_encoders[model_id]
    pairs = [(query, corpus[i]) for i in candidates]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        logger.warning("CrossEncoder rerank failed: %s", exc)
        return candidates + tail
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [i for _, i in scored] + tail
