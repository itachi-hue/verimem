"""
reranker.py — Local cross-encoder reranking, zero API cost.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, ~22MB) trained on
MS MARCO passage relevance. Takes top-k ChromaDB candidates and re-scores
every (query, document) pair. Better R@1 and R@3 without any API call.

Speed optimisations (applied in order):
  1. Rerank result cache — sha256(query, text) → score dict.
     Same (query, chunk) pair seen before = dict lookup, ~0.01ms.
     Typical agent: same query repeated → reranking is effectively free.
  2. ONNX Runtime backend — ~3× faster than PyTorch on CPU.
     Requires: pip install optimum[onnxruntime]
     Falls back to PyTorch automatically if not available.

Requires: sentence-transformers (core dependency). Optional: pip install verimem[nli] for ONNX cross-encoder speedup.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger("verimem_mcp")

_DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Rerank result cache — module-level, process-lifetime, like the embed cache.
# Key: sha256(query + "|||" + chunk_text) → logit score (float)
# ---------------------------------------------------------------------------
_rerank_cache: Dict[str, float] = {}
_rerank_cache_lock = threading.Lock()


def _rerank_cache_key(query: str, text: str) -> str:
    payload = f"{query}|||{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _cross_encoder_device_kw(device: str) -> Optional[str]:
    """CrossEncoder ``device=`` — ``None`` keeps default (CPU); ``cuda`` for GPU ONNX/PyTorch."""
    if not device or device.lower() in ("cpu",):
        return None
    return device


class CrossEncoderReranker:
    """
    Singleton cross-encoder reranker. Lazy-loads model on first use.

    Usage
    -----
    reranker = CrossEncoderReranker.get()
    reranked_hits = reranker.rerank(query, hits, top_n=5)
    """

    _instance: Optional["CrossEncoderReranker"] = None
    _lock = threading.Lock()
    _model = None
    _model_lock = threading.Lock()
    _pytorch_fallback_warned = False

    @classmethod
    def get(cls, model: str = _DEFAULT_RERANK_MODEL, device: str = "cpu") -> "CrossEncoderReranker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model, device)
            elif cls._instance._device != device:
                if cls._instance._model is not None:
                    logger.warning(
                        "CrossEncoderReranker already loaded with device=%r; ignoring device=%r",
                        cls._instance._device,
                        device,
                    )
                else:
                    cls._instance = cls(model, device)
            return cls._instance

    def __init__(self, model: str = _DEFAULT_RERANK_MODEL, device: str = "cpu") -> None:
        self._model_name = model
        self._device = device
        # When True, wrap predict() in torch.inference_mode() (PyTorch path only).
        self._use_torch_inference_guard = False

    def _load(self):
        with self._model_lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder

                    dkw: dict = {}
                    ce_dev = _cross_encoder_device_kw(self._device)
                    if ce_dev is not None:
                        dkw["device"] = ce_dev

                    # Sentence-Transformers defaults ONNX to ``get_available_providers()[0]``, often TensorRT.
                    # TensorRT without full libs fails the session; ORT then retries CPU only, ignoring CUDA.
                    onnx_model_kw: dict = {}
                    if ce_dev == "cuda":
                        try:
                            import onnxruntime as ort

                            if "CUDAExecutionProvider" in ort.get_available_providers():
                                onnx_model_kw["provider"] = "CUDAExecutionProvider"
                        except ImportError:
                            pass

                    # ONNX first — same exported weights; use ``onnxruntime-gpu`` + ``device=cuda`` for GPU.
                    # Requires: pip install optimum[onnxruntime]>=2.1  (verimem[nli])
                    onnx_err: Optional[Exception] = None
                    try:
                        dkw_onnx = dict(dkw)
                        if onnx_model_kw:
                            dkw_onnx["model_kwargs"] = onnx_model_kw
                        self._model = CrossEncoder(self._model_name, backend="onnx", **dkw_onnx)
                        self._use_torch_inference_guard = False
                        logger.debug(
                            "CrossEncoderReranker: ONNX backend (%s) device=%s",
                            self._model_name,
                            ce_dev or "cpu",
                        )
                    except Exception as e:
                        onnx_err = e
                        self._model = CrossEncoder(self._model_name, **dkw)
                        self._use_torch_inference_guard = True
                        logger.debug(
                            "CrossEncoderReranker: ONNX unavailable (%s); using PyTorch",
                            e,
                        )
                        if not CrossEncoderReranker._pytorch_fallback_warned:
                            CrossEncoderReranker._pytorch_fallback_warned = True
                            logger.warning(
                                "CrossEncoder rerank is using PyTorch (slow on CPU for ~20 pairs). "
                                "Install ONNX Runtime for the same model, ~3× faster rerank, no quality change: "
                                "pip install 'optimum[onnxruntime]>=2.1'   "
                                "(or: pip install verimem[nli]). Underlying error: %s",
                                onnx_err,
                            )
                except ImportError:
                    logger.warning(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )
                    self._model = None
        return self._model

    def _predict_pairs(self, model, pairs: List[tuple]) -> list:
        """Single batched forward; batch_size=len(pairs) avoids extra internal chunking."""
        bs = max(1, len(pairs))
        if self._use_torch_inference_guard:
            try:
                import torch

                ctx = torch.inference_mode()
            except ImportError:
                ctx = contextlib.nullcontext()
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return model.predict(
                pairs,
                show_progress_bar=False,
                batch_size=bs,
            )

    def available(self) -> bool:
        return self._load() is not None

    def rerank(self, query: str, hits: list, top_n: Optional[int] = None) -> list:
        """
        Rerank RecallHit objects by cross-encoder relevance score.

        Cache-first: if (query, hit.text) was scored before, the cached logit
        is returned immediately — no model inference. On a typical agent that
        re-asks similar questions, this makes reranking effectively free.

        Parameters
        ----------
        query : str
        hits  : List[RecallHit]
        top_n : int or None — trim to top_n after reranking

        Returns
        -------
        List[RecallHit] sorted by rerank_score descending.
        """
        model = self._load()
        if model is None or not hits:
            return hits

        pairs = [(query, h.text) for h in hits]

        # Split into cache hits and misses
        scores: List[Optional[float]] = [None] * len(pairs)
        uncached_idx: List[int] = []

        with _rerank_cache_lock:
            for i, (q, text) in enumerate(pairs):
                key = _rerank_cache_key(q, text)
                if key in _rerank_cache:
                    scores[i] = _rerank_cache[key]
                else:
                    uncached_idx.append(i)

        # Run inference only on cache misses (batched)
        if uncached_idx:
            uncached_pairs = [pairs[i] for i in uncached_idx]
            try:
                raw_scores = self._predict_pairs(model, uncached_pairs)
            except Exception as exc:
                logger.warning("CrossEncoderReranker.rerank failed: %s", exc)
                return hits

            with _rerank_cache_lock:
                for i, score in zip(uncached_idx, raw_scores):
                    q, text = pairs[i]
                    key = _rerank_cache_key(q, text)
                    _rerank_cache[key] = float(score)
                    scores[i] = float(score)

        for hit, score in zip(hits, scores):
            hit.rerank_score = score

        reranked = sorted(hits, key=lambda h: getattr(h, "rerank_score", 0.0), reverse=True)
        if top_n is not None:
            reranked = reranked[:top_n]
        return reranked

    def rerank_indices(
        self,
        query: str,
        rankings: List[int],
        corpus: List[str],
        top_k: int = 20,
    ) -> List[int]:
        """
        Rerank a list of corpus indices (benchmark-compatible interface).
        Cache-aware: already-scored pairs skip model inference.
        """
        model = self._load()
        if model is None or not rankings:
            return rankings

        candidates = rankings[:top_k]
        tail = rankings[top_k:]

        pairs = [(query, corpus[idx]) for idx in candidates]

        scores: List[Optional[float]] = [None] * len(pairs)
        uncached_idx: List[int] = []

        with _rerank_cache_lock:
            for i, (q, text) in enumerate(pairs):
                key = _rerank_cache_key(q, text)
                if key in _rerank_cache:
                    scores[i] = _rerank_cache[key]
                else:
                    uncached_idx.append(i)

        if uncached_idx:
            uncached_pairs = [pairs[i] for i in uncached_idx]
            try:
                raw_scores = self._predict_pairs(model, uncached_pairs)
            except Exception as exc:
                logger.warning("CrossEncoderReranker.rerank_indices failed: %s", exc)
                return rankings

            with _rerank_cache_lock:
                for i, score in zip(uncached_idx, raw_scores):
                    q, text = pairs[i]
                    _rerank_cache[_rerank_cache_key(q, text)] = float(score)
                    scores[i] = float(score)

        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scored] + tail
