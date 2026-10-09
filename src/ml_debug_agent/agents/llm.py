"""One place to create the chat model, so every agent uses the same settings.

The provider and model can be switched without changing code:
    OLLAMA_MODEL=qwen3:8b uv run python -m ml_debug_agent.eval.run_benchmark --system single_agent
    LLM_PROVIDER=gemini uv run python -m ml_debug_agent.eval.run_benchmark --system single_agent

Gemini needs GOOGLE_API_KEY, from the environment or the project's .env file.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

DEFAULT_OLLAMA_MODEL = "gemma4:e2b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# The Gemini free tier allows only a few requests per minute. Staying just under the
# limit is faster overall than hitting it and waiting out 429 retries.
GEMINI_REQUESTS_PER_MINUTE = float(os.environ.get("GEMINI_RPM", "8"))
_gemini_rate_limiter = InMemoryRateLimiter(
    requests_per_second=GEMINI_REQUESTS_PER_MINUTE / 60,
    check_every_n_seconds=0.5,
    max_bucket_size=1,  # no bursts: spread requests evenly
)


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama").lower()


def model_name() -> str:
    if provider() == "gemini":
        return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def get_llm(thinking: bool = False) -> BaseChatModel:
    if provider() == "gemini":
        return _gemini(thinking)
    if provider() == "ollama":
        return _ollama(thinking)
    raise ValueError(f"Unknown LLM_PROVIDER {provider()!r}; use 'ollama' or 'gemini'.")


def _ollama(thinking: bool) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name(),
        temperature=0,  # always pick the most likely token, so evals are repeatable
        seed=0,
        reasoning=thinking,  # Gemma 4's step-by-step "thinking" mode, on or off
        num_ctx=8192,  # context window; set explicitly because Ollama's default can be small
    )


def _gemini(thinking: bool) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name(),
        temperature=0,
        seed=0,
        # None lets the model decide how much to think; 0 turns thinking off.
        thinking_budget=None if thinking else 0,
        rate_limiter=_gemini_rate_limiter,
        max_retries=6,  # back off and retry on 429s instead of recording an error
    )
