"""
Unified LLM Client — Multi-provider with automatic failover.

Provider chain:
1. OpenRouter (Kimi K2.5) — primary reasoning and multimodal provider
2. Groq (GPT-OSS / Llama) — low-latency fallback
3. Local HuggingFace (Phi-4-mini) — emergency offline

All providers use OpenAI-compatible API format.
"""

import asyncio
import base64
import json
import logging
import time
from enum import Enum
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from config import config

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class LLMRole(str, Enum):
    """Semantic roles — each maps to the best model for the job."""

    REASONING = "reasoning"  # Kimi K2.5 — agentic scam analysis
    MULTIMODAL = "multimodal"  # Kimi K2.5 — image and text reasoning
    ROUTING = "routing"  # Kimi K2.5 — orchestrator decisions
    FAST = "fast"  # Llama 4 Scout — quick classification


# Map roles → model IDs per provider
_MODEL_MAP: dict[LLMProvider, dict[LLMRole, str]] = {
    LLMProvider.GROQ: {
        LLMRole.REASONING: config.groq.primary_model,  # GPT-OSS fallback
        LLMRole.MULTIMODAL: config.groq.multimodal_model,  # Llama 4 Maverick
        LLMRole.ROUTING: config.groq.primary_model,  # GPT-OSS fallback
        LLMRole.FAST: config.groq.fast_model,  # Llama 4 Scout
    },
    LLMProvider.OPENROUTER: {
        LLMRole.REASONING: config.openrouter.reasoning_model,
        LLMRole.MULTIMODAL: config.openrouter.reasoning_model,
        LLMRole.ROUTING: config.openrouter.reasoning_model,
        LLMRole.FAST: config.openrouter.free_router,
    },
}


class RateLimiter:
    """Simple token-bucket rate limiter for free-tier compliance."""

    def __init__(self, max_rpm: int = 30):
        self.max_rpm = max_rpm
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        now = time.time()
        # Remove timestamps older than 60s
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max_rpm:
            wait = 60 - (now - self._timestamps[0])
            logger.warning(f"Rate limit approaching, waiting {wait:.1f}s")
            await asyncio.sleep(max(wait, 0.1))
            self._timestamps.append(time.time())


class LLMClient:
    """
    Multi-provider LLM client with automatic failover.

    Usage:
    client = LLMClient()

    # Text reasoning (uses Kimi K2.5 via OpenRouter)
    result = await client.reason(
    system="You are a scam detection expert.",
    user="Analyze this transcript for scam patterns...",
    )

    # Multimodal (uses Kimi K2.5 via OpenRouter — sees images)
    result = await client.analyze_image(
    image_base64="...",
    prompt="Analyze this currency note for authenticity...",
    )

    # Tool-use / function calling (uses Kimi K2.5 via OpenRouter)
    result = await client.reason_with_tools(
    system="...",
    user="...",
    tools=[{"type": "function", "function": {...}}],
    )
    """

    def __init__(self):
        self._providers: dict[LLMProvider, AsyncOpenAI] = {}
        self._rate_limiters: dict[LLMProvider, RateLimiter] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize available API clients based on configured keys."""
        if config.groq.api_key:
            self._providers[LLMProvider.GROQ] = AsyncOpenAI(
                api_key=config.groq.api_key,
                base_url=config.groq.base_url,
                timeout=config.groq.timeout,
            )
            self._rate_limiters[LLMProvider.GROQ] = RateLimiter(config.groq.max_rpm)
            logger.info(" Groq provider initialized (text and multimodal fallback)")

        if config.openrouter.api_key:
            self._providers[LLMProvider.OPENROUTER] = AsyncOpenAI(
                api_key=config.openrouter.api_key,
                base_url=config.openrouter.base_url,
                timeout=config.openrouter.timeout,
                default_headers={
                    "HTTP-Referer": "https://digital-public-safety-shield.app"
                },
            )
            self._rate_limiters[LLMProvider.OPENROUTER] = RateLimiter(
                config.openrouter.max_rpm
            )
            logger.info(" OpenRouter provider initialized (Kimi K2.5 primary, K2.6 free fallback)")

        if not self._providers:
            logger.warning(" No LLM API keys configured — only local models available")

    def _get_provider_chain(self, role: Optional[LLMRole] = None) -> list[LLMProvider]:
        """Return providers in role-aware priority order."""
        chain = []
        prefer_kimi = role in {LLMRole.REASONING, LLMRole.ROUTING, LLMRole.MULTIMODAL}
        order = (
            [LLMProvider.OPENROUTER, LLMProvider.GROQ]
            if prefer_kimi
            else [LLMProvider.GROQ, LLMProvider.OPENROUTER]
        )
        chain.extend(provider for provider in order if provider in self._providers)
        return chain

    def _get_model_attempts(
        self,
        role: LLMRole,
        provider_override: Optional[LLMProvider] = None,
    ) -> list[tuple[LLMProvider, str]]:
        """Return ordered provider/model attempts, including the Kimi free fallback."""
        if provider_override:
            model = _MODEL_MAP.get(provider_override, {}).get(role)
            return [(provider_override, model)] if model else []

        attempts: list[tuple[LLMProvider, str]] = []
        kimi_roles = {LLMRole.REASONING, LLMRole.ROUTING, LLMRole.MULTIMODAL}
        if role in kimi_roles and LLMProvider.OPENROUTER in self._providers:
            attempts.extend(
                [
                    (LLMProvider.OPENROUTER, config.openrouter.reasoning_model),
                    (
                        LLMProvider.OPENROUTER,
                        config.openrouter.kimi_free_fallback_model,
                    ),
                ]
            )

        for provider in self._get_provider_chain(role):
            if provider == LLMProvider.OPENROUTER and role in kimi_roles:
                continue
            model = _MODEL_MAP.get(provider, {}).get(role)
            if model:
                attempts.append((provider, model))
        return attempts

    async def _call_llm(
        self,
        role: LLMRole,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
        provider_override: Optional[LLMProvider] = None,
        reasoning_effort: str = "minimal",
    ) -> dict[str, Any]:
        """
        Core LLM call with automatic provider failover.
        Returns dict with 'content', 'tool_calls', 'model', 'provider', 'usage'.
        """
        attempts = self._get_model_attempts(role, provider_override)

        last_error = None
        for provider, model in attempts:
            client = self._providers.get(provider)
            if not client:
                continue

            try:
                # Rate limit
                rate_limiter = self._rate_limiters.get(provider)
                if rate_limiter:
                    await rate_limiter.acquire()

                # Build kwargs
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if response_format:
                    kwargs["response_format"] = response_format

                if provider == LLMProvider.OPENROUTER:
                    kwargs["extra_body"] = {
                        "reasoning": {
                            "effort": reasoning_effort,
                            "exclude": True,
                        }
                    }

                logger.info(f" Calling {provider.value} → {model}")
                response = await client.chat.completions.create(**kwargs)

                choice = response.choices[0]
                result = {
                    "content": choice.message.content or "",
                    "tool_calls": [],
                    "model": model,
                    "provider": provider.value,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens
                        if response.usage
                        else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                    },
                    "finish_reason": choice.finish_reason,
                }

                # Extract tool calls if present
                if choice.message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]

                logger.info(
                    f" {provider.value} response: {result['usage']['completion_tokens']} tokens, "
                    f"finish={result['finish_reason']}"
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f" {provider.value} -> {model} failed: {e}. Trying next fallback..."
                )
                continue

        # All providers failed
        error_msg = f"All LLM providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

        # ─── High-Level API Methods ──────────────────────────────────────────

    async def reason(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Pure text reasoning using Kimi K2.5.
        Best for: scam pattern analysis, agentic reasoning, orchestrator decisions.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if json_mode else None
        return await self._call_llm(
            role=LLMRole.REASONING,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def reason_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Reasoning with function/tool calling using Kimi K2.5.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return await self._call_llm(
            role=LLMRole.REASONING,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        system: str = "You are an expert forensic analyst specializing in document and currency authenticity verification.",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Multimodal vision-language analysis using Kimi K2.5.
        Kimi natively processes images — no separate vision model is required here.
        Use this for: currency note reasoning, fake document detection, screenshot analysis.
        """
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        response_format = {"type": "json_object"} if json_mode else None
        return await self._call_llm(
            role=LLMRole.MULTIMODAL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def classify_fast(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = True,
    ) -> dict[str, Any]:
        """
        Fast classification using Llama 4 Scout.
        For: quick input routing, modality detection, simple classification.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if json_mode else None
        return await self._call_llm(
            role=LLMRole.FAST,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def multi_turn_reason(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Multi-turn conversation/reasoning with full message history.
        For: hierarchical multi-role CoT prompting (Investigator → Policy Checker → Risk Assessor).
        """
        return await self._call_llm(
            role=LLMRole.REASONING,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def test_connection(self) -> dict[str, Any]:
        """Test connectivity to all configured providers."""
        results = {}
        for provider in self._get_provider_chain(LLMRole.REASONING):
            try:
                result = await self._call_llm(
                    role=LLMRole.FAST,
                    messages=[
                        {"role": "system", "content": "Reply with exactly: OK"},
                        {"role": "user", "content": "Test connection"},
                    ],
                    temperature=0.0,
                    max_tokens=10,
                    provider_override=provider,
                )
                results[provider.value] = {
                    "status": "connected",
                    "model": result["model"],
                    "response": result["content"][:50],
                }
            except Exception as e:
                results[provider.value] = {
                    "status": "failed",
                    "error": str(e),
                }
                return results


# Module-level singleton
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
