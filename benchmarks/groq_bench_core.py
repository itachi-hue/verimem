"""Shared Groq chat + judge helpers for benchmarks (no VeriMem import)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq

# https://console.groq.com/docs/model/llama-4-scout-17b-16e-instruct
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _http_status(exc: BaseException) -> Optional[int]:
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        return sc
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc2 = getattr(resp, "status_code", None)
        if isinstance(sc2, int):
            return sc2
    return None


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None) or {}
    for key in ("retry-after", "Retry-After"):
        if key in headers:
            try:
                return float(headers[key])
            except (TypeError, ValueError):
                return None
    return None


def _should_retry_http(status: Optional[int]) -> bool:
    if status is None:
        return False
    return status in (429, 500, 502, 503, 504)


@dataclass
class GroqRunner:
    """Groq chat.completions with retries on 429 / transient errors."""

    client: Groq
    model: str
    max_retries: int = 8
    retry_base_sec: float = 1.0

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                msg = completion.choices[0].message
                text = (msg.content or "").strip()
                return text
            except Exception as e:
                last_err = e
                status = _http_status(e)
                ra = _retry_after_seconds(e)
                if _should_retry_http(status) and attempt < self.max_retries - 1:
                    delay = ra if ra is not None else self.retry_base_sec * (2**attempt)
                    delay = min(max(delay, 0.1), 120.0)
                    print(
                        f"  Groq retry {attempt + 1}/{self.max_retries} "
                        f"(HTTP {status or '?'}, sleep {delay:.1f}s): {e}"
                    )
                    time.sleep(delay)
                    continue
                print(f"Groq API error: {e}")
                return ""
        if last_err:
            print(f"Groq API error (exhausted retries): {last_err}")
        return ""


def _packet_prompt_suffix(packet_mode: str, simple: Optional[dict]) -> str:
    """Extra prompt text from ``ContextPacket.to_simple()`` (none for ``hits-only``)."""
    if packet_mode == "hits-only" or not simple:
        return ""
    if packet_mode == "full":
        blob = json.dumps(simple, indent=2)
    elif packet_mode == "extras":
        slim = {k: v for k, v in simple.items() if k != "hits"}
        blob = json.dumps(slim, indent=2)
    else:
        return ""
    return f"""

Retrieval packet (structured metadata from VeriMem recall):
{blob}
"""


def generate_answer(
    runner: GroqRunner,
    question: str,
    contexts: list[str],
    *,
    packet_mode: str = "hits-only",
    packet_simple: Optional[dict] = None,
) -> str:
    context_str = "\n\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)])
    extra = _packet_prompt_suffix(packet_mode, packet_simple)
    prompt = f"""Answer the following question based on the conversation context provided.

Context:
{context_str}
{extra}
Question: {question}

Provide a helpful answer based on the context. If the context doesn't contain enough information to answer, say "I don't know."

Answer:"""
    return runner.complete(prompt, max_tokens=512)


def judge_answer(runner: GroqRunner, question: str, ground_truth: str, generated: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated,
    )
    response = runner.complete(prompt, max_tokens=512)
    try:
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse judge response: {e}")
        print(f"Response was: {response}")
        return {"correct": 0, "reasoning": f"Failed to parse: {str(e)}"}


def aggregate_results(results: list[dict]) -> dict:
    total = len(results)
    successful = sum(1 for r in results if r["retrieval_successful"])
    if successful == 0:
        return {"error": "No successful evaluations"}

    correct_count = sum(r.get("correct", 0) for r in results if r["retrieval_successful"])
    accuracy = correct_count / successful if successful > 0 else 0.0
    avg_rr = (
        sum(r["retrieval_recall_at_k"] for r in results if r["retrieval_successful"]) / successful
    )

    by_category: dict[str, list[dict]] = {}
    for result in results:
        if not result["retrieval_successful"]:
            continue
        cat = result["category"]
        by_category.setdefault(cat, []).append(result)

    category_stats = {}
    for cat, cat_results in by_category.items():
        n = len(cat_results)
        cat_correct = sum(r.get("correct", 0) for r in cat_results)
        category_stats[cat] = {
            "count": n,
            "correct": cat_correct,
            "accuracy": round(cat_correct / n if n > 0 else 0.0, 3),
            "retrieval_recall": round(
                sum(r["retrieval_recall_at_k"] for r in cat_results) / n,
                3,
            ),
        }

    return {
        "total_items": total,
        "successful_items": successful,
        "overall": {
            "correct": correct_count,
            "accuracy": round(accuracy, 3),
            "retrieval_recall_at_k": round(avg_rr, 3),
        },
        "by_category": category_stats,
    }
