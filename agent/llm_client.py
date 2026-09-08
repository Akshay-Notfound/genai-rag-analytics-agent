"""
llm_client.py
-------------
LLM integration layer. Defines a common interface (`LLMClient.generate`) so the
rest of the agent is provider-agnostic, and ships two implementations:

  * OpenAIClient / AnthropicClient -- real HTTP calls to the respective APIs.
    Require an API key (set via environment variable) to actually run.
  * MockLLMClient -- a deterministic, offline stand-in used for local
    development and unit tests, so the agent's retrieval + orchestration
    logic can be exercised without network access or a paid API key.

Swapping providers means changing one line where the client is constructed
in agent.py -- nothing else in the pipeline needs to change.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Send a prompt to the LLM and return the generated text."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Real integration with the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set. Export it or pass api_key=.")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicClient(LLMClient):
    """Real integration with the Anthropic Messages API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Export it or pass api_key=.")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


class MockLLMClient(LLMClient):
    """
    Deterministic offline client used for local development, demos, and unit
    tests when no API key / network access is available. Applies a few simple
    rules to turn a natural-language question + retrieved context into a
    pandas query, so the rest of the pipeline can be verified end-to-end.

    Only the "Question:" portion of the prompt is used to decide intent and
    extract the target column -- the retrieved context is passed to a real
    LLM for grounding, but this mock keeps parsing scoped to the actual
    question so it isn't confused by unrelated terms in retrieved context.
    """

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        question = self._extract_question(prompt).lower()

        if "how many" in question or "count" in question:
            return "len(df)"
        if "average" in question or "mean" in question:
            column = self._extract_column(question, default="revenue")
            return f"df['{column}'].mean()"
        if "total" in question or "sum" in question:
            column = self._extract_column(question, default="revenue")
            return f"df['{column}'].sum()"
        if "top" in question and "region" in question:
            return "df.groupby('region')['revenue'].sum().sort_values(ascending=False).head(5)"
        return "df.describe()"

    @staticmethod
    def _extract_question(prompt: str) -> str:
        marker = "Question:"
        if marker in prompt:
            after = prompt.split(marker, 1)[1]
            return after.split("Pandas expression:")[0]
        return prompt

    @staticmethod
    def _extract_column(question_lower: str, default: str) -> str:
        for candidate in ["revenue", "units_sold", "profit", "price"]:
            if candidate in question_lower:
                return candidate
        return default
