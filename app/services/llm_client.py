"""
app/services/llm_client.py — Unified async LLM client

Providers supported: groq, gemini, anthropic, openai, ollama
Default provider: groq

Environment variables (loaded from .env via python-dotenv):
  LLM_PROVIDER   — "groq" | "gemini" | "anthropic" | "openai" | "ollama"  (default "groq")
  DEV_MODE       — "true" | "false"  (default "false")
                   true  → task_type="reasoning" uses Llama 3.1 8B  (14,400 req/day)
                   false → task_type="reasoning" uses Llama 3.3 70B (1,000 req/day)
  LLM_MAX_RPM    — requests per minute allowed by our local limiter (default 30 = Groq free tier)
  GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY (as needed)

Public API (every pipeline stage uses this):
  async def call_llm(prompt, task_type="fast", system_prompt=None,
                     max_tokens=1024, temperature=0.0, json_mode=False) -> str
"""

import asyncio
import os
import re
import time
from collections import deque
from typing import Optional

from dotenv import load_dotenv

from app.services.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()
DEV_MODE: bool = os.getenv("DEV_MODE", "false").strip().lower() == "true"


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------
PROVIDER_MODEL_MAP = {
    "groq": {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "llama-3.1-8b-instant": "gemini-1.5-flash",
        "llama-3.3-70b-versatile": "gemini-1.5-pro",
    },
    "anthropic": {
        "llama-3.1-8b-instant": "claude-haiku-4-5-20251001",
        "llama-3.3-70b-versatile": "claude-sonnet-4-6",
    },
    "openai": {
        "llama-3.1-8b-instant": "gpt-4o-mini",
        "llama-3.3-70b-versatile": "gpt-4o",
    },
    "ollama": {
        "llama-3.1-8b-instant": "llama3.1:8b",
        "llama-3.3-70b-versatile": "llama3.3:70b",
    },
}


# ---------------------------------------------------------------------------
# Async rate limiter (sliding window) — WAITS for a free slot instead of failing,
# so long benchmark runs and multi-claim requests never crash on burst traffic.
# ---------------------------------------------------------------------------
_rate_lock = asyncio.Lock()
_timestamps: deque = deque()
MAX_REQUESTS: int = int(os.getenv("LLM_MAX_RPM", "30"))
WINDOW_SECONDS: int = 60
MAX_RETRIES: int = 3


async def _check_rate_limit() -> None:
    while True:
        async with _rate_lock:
            now = time.monotonic()
            while _timestamps and _timestamps[0] < now - WINDOW_SECONDS:
                _timestamps.popleft()
            if len(_timestamps) < MAX_REQUESTS:
                _timestamps.append(now)
                return
            wait = _timestamps[0] + WINDOW_SECONDS - now
        logger.info("Rate limiter: waiting %.1fs for a free slot", max(wait, 0.1))
        await asyncio.sleep(max(wait, 0.1))


# ---------------------------------------------------------------------------
# Provider client initialisation (module level, guarded)
# ---------------------------------------------------------------------------
if LLM_PROVIDER == "groq":
    from groq import AsyncGroq
    _groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

elif LLM_PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

elif LLM_PROVIDER == "anthropic":
    from anthropic import AsyncAnthropic
    _anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

elif LLM_PROVIDER == "openai":
    from openai import AsyncOpenAI
    _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

elif LLM_PROVIDER == "ollama":
    import httpx  # no client init needed; use per-call AsyncClient


def _messages(prompt: str, system_prompt: Optional[str]) -> list[dict]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


async def _call_groq(prompt: str, model: str, system_prompt: Optional[str] = None,
                     max_tokens: int = 1024, temperature: float = 0.0,
                     json_mode: bool = False) -> str:
    kwargs = dict(
        model=model,
        messages=_messages(prompt, system_prompt),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await _groq_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


async def _call_gemini(prompt: str, model: str, system_prompt: Optional[str] = None,
                       max_tokens: int = 1024, temperature: float = 0.0,
                       json_mode: bool = False) -> str:
    gmodel = genai.GenerativeModel(model, system_instruction=system_prompt)
    cfg = {"max_output_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    resp = await gmodel.generate_content_async(prompt, generation_config=cfg)
    return resp.text


async def _call_anthropic(prompt: str, model: str, system_prompt: Optional[str] = None,
                          max_tokens: int = 1024, temperature: float = 0.0,
                          json_mode: bool = False) -> str:
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_prompt:
        kwargs["system"] = system_prompt  # json_mode: enforced via the prompt itself
    resp = await _anthropic_client.messages.create(**kwargs)
    return resp.content[0].text


async def _call_openai(prompt: str, model: str, system_prompt: Optional[str] = None,
                       max_tokens: int = 1024, temperature: float = 0.0,
                       json_mode: bool = False) -> str:
    kwargs = dict(
        model=model,
        messages=_messages(prompt, system_prompt),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await _openai_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


async def _call_ollama(prompt: str, model: str, system_prompt: Optional[str] = None,
                       max_tokens: int = 1024, temperature: float = 0.0,
                       json_mode: bool = False) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if system_prompt:
        payload["system"] = system_prompt
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:11434/api/generate", json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()["response"]


_DISPATCH = {
    "groq": "_call_groq",
    "gemini": "_call_gemini",
    "anthropic": "_call_anthropic",
    "openai": "_call_openai",
    "ollama": "_call_ollama",
}


def _retry_delay(message: str, attempt: int) -> float:
    """Honour 'try again in 2.5s' hints from Groq; otherwise exponential backoff."""
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", message)
    if m:
        secs = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
        return min(secs + 0.5, 30.0)
    return float(2 ** (attempt + 1))


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
async def call_llm(
    prompt: str,
    task_type: str = "fast",
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """
    Call the configured LLM provider.

    Args:
        prompt:        The user prompt.
        task_type:     "fast"      → Llama 3.1 8B (all providers' small model)
                       "reasoning" → 8B if DEV_MODE=true, else 3.3 70B
        system_prompt: Optional system message.
        max_tokens:    Max tokens to generate.
        temperature:   Sampling temperature (0.0 = deterministic).
        json_mode:     Ask the provider for strict JSON output where supported.

    Returns:
        Plain string response.

    Raises:
        ValueError:   Unknown task_type.
        RuntimeError: Provider call failed after retries.
    """
    # 1. Resolve model
    if task_type == "fast":
        groq_model = "llama-3.1-8b-instant"
    elif task_type == "reasoning":
        groq_model = "llama-3.1-8b-instant" if DEV_MODE else "llama-3.3-70b-versatile"
    else:
        raise ValueError(f"Unknown task_type: '{task_type}'. Expected 'fast' or 'reasoning'.")

    if LLM_PROVIDER not in _DISPATCH:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'")
    model = PROVIDER_MODEL_MAP[LLM_PROVIDER][groq_model]

    logger.info("LLM call | model=%s | task_type=%s | DEV_MODE=%s", model, task_type, DEV_MODE)

    # 2. Dispatch with rate limiting + retry on 429
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        await _check_rate_limit()
        try:
            fn = globals()[_DISPATCH[LLM_PROVIDER]]  # looked up at call time (patch-friendly)
            return await fn(
                prompt, model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
            )
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                delay = _retry_delay(str(exc), attempt)
                logger.warning("Provider rate-limited (attempt %d). Retrying in %.1fs", attempt + 1, delay)
                await asyncio.sleep(delay)
                continue
            break

    raise RuntimeError(
        f"LLM call failed [provider={LLM_PROVIDER}, model={model}]: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Smoke test:  python -m app.services.llm_client
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _smoke_test():
        print(f"[llm_client] Provider : {LLM_PROVIDER}")
        print(f"[llm_client] DEV_MODE : {DEV_MODE}")
        try:
            out = await call_llm(
                prompt='Return JSON: {"ok": true}',
                task_type="fast",
                json_mode=True,
                max_tokens=32,
            )
            print(f"[llm_client] Response : {out}")
        except Exception as e:
            print(f"[llm_client] ERROR    : {e}")

    asyncio.run(_smoke_test())
