"""
app/services/llm_client.py — Unified async LLM client

Providers supported: groq, gemini, anthropic, openai, ollama
Default provider: groq

Environment variables (load from .env via python-dotenv):
  LLM_PROVIDER   — "groq" | "gemini" | "anthropic" | "openai" | "ollama"  (default: "groq")
  DEV_MODE       — "true" | "false"  (default: "false")
  GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY (as needed)

Public API:
  async def call_llm(prompt: str, task_type: str = "fast") -> str
"""

import os
import asyncio
import time
from collections import deque
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()
DEV_MODE: bool = os.getenv("DEV_MODE", "false").strip().lower() == "true"


# ---------------------------------------------------------------------------
# Section 3 — Model name resolution
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
# Section 4 — Async rate limiter (sliding window)
# ---------------------------------------------------------------------------
_rate_lock = asyncio.Lock()
_timestamps: deque = deque()
MAX_REQUESTS = 100
WINDOW_SECONDS = 60


async def _check_rate_limit() -> None:
    async with _rate_lock:
        now = time.monotonic()
        while _timestamps and _timestamps[0] < now - WINDOW_SECONDS:
            _timestamps.popleft()
        if len(_timestamps) >= MAX_REQUESTS:
            raise Exception(
                f"Rate limit exceeded: max {MAX_REQUESTS} requests "
                f"per {WINDOW_SECONDS} seconds"
            )
        _timestamps.append(now)


# ---------------------------------------------------------------------------
# Section 5 — Provider client initialization (module level, guarded)
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


async def _call_groq(prompt: str, model: str) -> str:
    resp = await _groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


async def _call_gemini(prompt: str, model: str) -> str:
    gmodel = genai.GenerativeModel(model)
    resp = await gmodel.generate_content_async(prompt)
    return resp.text


async def _call_anthropic(prompt: str, model: str) -> str:
    resp = await _anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


async def _call_openai(prompt: str, model: str) -> str:
    resp = await _openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


async def _call_ollama(prompt: str, model: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]


# ---------------------------------------------------------------------------
# Section 6 — Public function
# ---------------------------------------------------------------------------
async def call_llm(prompt: str, task_type: str = "fast") -> str:
    """
    Call the configured LLM provider.

    Args:
        prompt:    The user prompt string.
        task_type: "fast"      → llama-3.1-8b-instant (all providers)
                   "reasoning" → llama-3.1-8b-instant if DEV_MODE=true
                                 llama-3.3-70b-versatile if DEV_MODE=false

    Returns:
        Plain string response from the LLM.

    Raises:
        Exception:      Rate limit exceeded.
        ValueError:     Unknown task_type or provider.
        RuntimeError:   Provider call failed.
    """
    # 1. Rate limit
    await _check_rate_limit()

    # 2. Resolve model
    if task_type == "fast":
        groq_model = "llama-3.1-8b-instant"
    elif task_type == "reasoning":
        groq_model = "llama-3.1-8b-instant" if DEV_MODE else "llama-3.3-70b-versatile"
    else:
        raise ValueError(f"Unknown task_type: '{task_type}'. Expected 'fast' or 'reasoning'.")

    model = PROVIDER_MODEL_MAP[LLM_PROVIDER][groq_model]

    # 3. Dispatch to provider
    try:
        if LLM_PROVIDER == "groq":
            return await _call_groq(prompt, model)
        elif LLM_PROVIDER == "gemini":
            return await _call_gemini(prompt, model)
        elif LLM_PROVIDER == "anthropic":
            return await _call_anthropic(prompt, model)
        elif LLM_PROVIDER == "openai":
            return await _call_openai(prompt, model)
        elif LLM_PROVIDER == "ollama":
            return await _call_ollama(prompt, model)
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'")
    except (ValueError, Exception) as e:
        raise RuntimeError(
            f"LLM call failed [provider={LLM_PROVIDER}, model={model}]: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Section 7 — __main__ test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _smoke_test():
        print(f"[llm_client] Provider : {LLM_PROVIDER}")
        print(f"[llm_client] DEV_MODE  : {DEV_MODE}")
        print(f"[llm_client] Sending test prompt with task_type='fast' ...")
        try:
            result = await call_llm(
                prompt="Reply with exactly three words: the sky is.",
                task_type="fast",
            )
            print(f"[llm_client] Response : {result}")
        except Exception as e:
            print(f"[llm_client] ERROR    : {e}")

    asyncio.run(_smoke_test())
