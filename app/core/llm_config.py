from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMEngine(ABC):
    """Abstract base for LLM engine providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a response from the LLM."""
        pass


class AnthropicEngine(LLMEngine):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. Set it in .env or pass "
                    "api_key argument."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a response using Anthropic Claude."""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


class LocalEngine(LLMEngine):
    """Local/no-op engine for development without API keys."""

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Return a placeholder response for development."""
        return f"[LOCAL] Prompt received ({len(prompt)} chars): {prompt[:80]}..."


def get_engine(provider: Optional[str] = None) -> LLMEngine:
    """Factory function to get the configured LLM engine."""
    provider = (provider or "").strip().lower() or os.getenv("LLM_PROVIDER", "local")

    if provider == "anthropic":
        return AnthropicEngine()
    elif provider == "local":
        return LocalEngine()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'anthropic' or 'local'.")