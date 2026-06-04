"""
Modular LLM provider with automatic fallback on rate limits.

Priority chain (highest to lowest quality):
  1. gpt-oss:20b-cloud  — OpenAI open-weight via Ollama, best for agentic/tool use
  2. gemini-2.5-flash   — Google, fast and capable, free tier fallback

Add more providers by extending build_llm(). Each provider is skipped gracefully
if its dependency or API key is missing. On a 429 / rate-limit error the next
provider in the chain is tried automatically.
"""

import logging
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# Strings that identify a rate-limit / quota error from any provider
_RATE_LIMIT_SIGNALS = (
    "429",
    "resource_exhausted",
    "rate_limit",
    "ratelimiterror",
    "quota",
    "too many requests",
    "requests per day",
    "requests per minute",
)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(signal in msg for signal in _RATE_LIMIT_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# Internal bound wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _BoundFallback:
    """
    Returned by FallbackLLM.bind_tools(). Holds a list of already-bound LLMs
    and falls back to the next one whenever a rate limit is encountered.
    """

    def __init__(self, bound_llms: list, names: list[str]) -> None:
        self._bound = bound_llms
        self._names = names

    def invoke(self, messages: Any) -> Any:
        last_exc: Exception | None = None

        for name, llm in zip(self._names, self._bound):
            try:
                logger.info(f"[LLM] Attempting provider: {name}")
                result = llm.invoke(messages)
                logger.info(f"[LLM] Success with provider: {name}")
                return result

            except Exception as exc:
                if _is_rate_limit(exc):
                    logger.warning(
                        f"[LLM] Rate limit hit on '{name}' — switching to next provider. "
                        f"Reason: {str(exc)[:120]}"
                    )
                    last_exc = exc
                    continue
                # Non-rate-limit errors bubble up immediately
                raise

        raise RuntimeError(
            f"All LLM providers exhausted after rate limits. Last error: {last_exc}"
        ) from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Public FallbackLLM
# ─────────────────────────────────────────────────────────────────────────────

class FallbackLLM:
    """
    Wraps multiple LangChain chat models and falls back automatically on rate limits.

    Example:
        llm = FallbackLLM([ollama_model, gemini_model], ["gpt-oss", "gemini"])
        response = llm.bind_tools(tools).invoke(messages)
        # or without tools:
        response = llm.invoke(messages)
    """

    def __init__(self, providers: list[BaseChatModel], names: list[str]) -> None:
        if not providers:
            raise ValueError("FallbackLLM requires at least one provider.")
        if len(providers) != len(names):
            raise ValueError("`providers` and `names` must have the same length.")

        self._providers = providers
        self._names = names
        logger.info(f"[LLM] FallbackLLM ready. Provider chain: {' → '.join(names)}")

    # Mimic the standard LangChain interface so this can drop in anywhere
    def bind_tools(self, tools: list) -> _BoundFallback:
        bound, active_names = [], []
        for name, provider in zip(self._names, self._providers):
            try:
                bound.append(provider.bind_tools(tools))
                active_names.append(name)
            except Exception as exc:
                logger.warning(f"[LLM] '{name}' could not bind tools ({exc}) — skipping.")

        if not bound:
            raise RuntimeError("No LLM provider could bind tools.")

        return _BoundFallback(bound, active_names)

    def invoke(self, messages: Any) -> Any:
        return _BoundFallback(self._providers, self._names).invoke(messages)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_llm() -> FallbackLLM:
    """
    Builds the FallbackLLM chain from available providers.

    Provider priority:
      1. gpt-oss:20b-cloud via Ollama  (requires langchain-ollama + Ollama running)
      2. gemini-2.5-flash              (requires GOOGLE_API_KEY in .env)

    Each provider is registered only if its dependencies/credentials are present.
    At least one must succeed or a RuntimeError is raised at startup.
    """
    providers: list[BaseChatModel] = []
    names: list[str] = []

    # ── 1. gpt-oss:20b-cloud via Ollama ──────────────────────────────────────
    try:
        from langchain_ollama import ChatOllama  # noqa: PLC0415

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")

        providers.append(ChatOllama(model=ollama_model, base_url=ollama_host))
        names.append(f"{ollama_model} (Ollama)")
        logger.info(f"[LLM] Registered: {ollama_model} via {ollama_host}")

    except ImportError:
        logger.warning("[LLM] langchain-ollama not installed — skipping Ollama provider.")
    except Exception as exc:
        logger.warning(f"[LLM] Ollama provider init failed ({exc}) — skipping.")

    # ── 2. gemini-2.5-flash via Google ───────────────────────────────────────
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            providers.append(ChatGoogleGenerativeAI(model=gemini_model))
            names.append(f"{gemini_model} (Google)")
            logger.info(f"[LLM] Registered: {gemini_model}")

        except Exception as exc:
            logger.warning(f"[LLM] Gemini provider init failed ({exc}) — skipping.")
    else:
        logger.warning("[LLM] GOOGLE_API_KEY not set — skipping Gemini provider.")

    if not providers:
        raise RuntimeError(
            "No LLM providers could be initialised.\n"
            "Options:\n"
            "  • Install langchain-ollama and run Ollama locally for gpt-oss\n"
            "  • Set GOOGLE_API_KEY in .env for Gemini fallback"
        )

    return FallbackLLM(providers, names)
