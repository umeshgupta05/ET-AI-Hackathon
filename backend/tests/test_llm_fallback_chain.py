"""Model-order contract for hosted LLM failover."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config
from models.nlp.llm_client import LLMClient, LLMProvider, LLMRole


def main() -> None:
    client = LLMClient()
    client._providers = {
        LLMProvider.OPENROUTER: object(),
        LLMProvider.GROQ: object(),
    }
    attempts = client._get_model_attempts(LLMRole.REASONING)
    assert attempts == [
        (LLMProvider.OPENROUTER, config.openrouter.reasoning_model),
        (LLMProvider.OPENROUTER, "moonshotai/kimi-k2.6:free"),
        (LLMProvider.GROQ, config.groq.primary_model),
    ]
    multimodal_attempts = client._get_model_attempts(LLMRole.MULTIMODAL)
    assert multimodal_attempts[1] == (
        LLMProvider.OPENROUTER,
        "moonshotai/kimi-k2.6:free",
    )
    print("Kimi K2.5 -> Kimi K2.6 free -> Groq fallback chain: PASS")


if __name__ == "__main__":
    main()
