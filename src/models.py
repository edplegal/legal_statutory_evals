"""Minimal model client abstractions for OpenAI-compatible and Ollama chat APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Protocol

import requests

SYSTEM_PROMPT = "Answer the user."


def _get_temperature(env_key: str, default: float = 0.0) -> float:
    raw_value = os.environ.get(env_key)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


class ModelClient(Protocol):
    """Protocol for chat model clients."""

    name: str

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a chat completion given a list of messages."""
        ...


@dataclass
class OpenAICompatClient:
    """Client for OpenAI-compatible chat completions."""

    base_url: str
    api_key: str
    model: str

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self.model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        temperature = _get_temperature("OPENAI_TEMPERATURE", 0.0)
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unexpected response format: {data}") from exc


@dataclass
class OllamaClient:
    """Client for Ollama /api/chat."""

    base_url: str
    model: str

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self.model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": False}
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise ValueError(f"Unexpected Ollama response: {data}")
        return content


def build_messages(user_turns: List[Dict[str, str]], system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """Prepend a system prompt to user/assistant messages."""
    return [{"role": "system", "content": system_prompt}, *user_turns]


def get_model_client_from_env(model_override: str | None = None) -> ModelClient:
    """Instantiate a model client based on environment variables."""
    backend = os.environ.get("MODEL_BACKEND", "openai_compat").lower()
    if backend == "openai_compat":
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
        model = model_override or os.environ.get("OPENAI_MODEL")
        if not api_key or not model:
            raise EnvironmentError("OPENAI_API_KEY and OPENAI_MODEL are required for OpenAI-compatible backend.")
        return OpenAICompatClient(base_url=base_url, api_key=api_key, model=model)

    if backend == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = model_override or os.environ.get("OLLAMA_MODEL")
        if not model:
            raise EnvironmentError("OLLAMA_MODEL is required for Ollama backend.")
        return OllamaClient(base_url=base_url, model=model)

    raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")
