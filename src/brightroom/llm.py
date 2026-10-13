"""Thin OpenRouter client: messages in, text out, with latency, tokens and cost as measured."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
from dotenv import load_dotenv

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float  # US dollars, as charged by OpenRouter
    seconds: float


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict], *, schema: dict | None = None, schema_name: str = "output", max_tokens: int = 1500) -> LLMResult: ...


def api_key_available() -> bool:
    load_dotenv()
    return bool(os.environ.get("OPENROUTER_API_KEY"))


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 60.0, seed: int = 7):
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set (copy .env.example to .env and add the key).")
        self.model = model or os.environ.get("BRIGHTROOM_MODEL") or DEFAULT_MODEL
        self.seed = seed
        self._http = httpx.Client(timeout=timeout)

    def complete(self, messages: list[dict], *, schema: dict | None = None, schema_name: str = "output", max_tokens: int = 1500) -> LLMResult:
        # Temperature 0 and a fixed seed: the same enquiry should give the same brief.
        body: dict = {"model": self.model, "messages": messages, "temperature": 0, "seed": self.seed, "max_tokens": max_tokens}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
            # Only route to providers that enforce the schema.
            body["provider"] = {"require_parameters": True}
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": "BrightRoom proposal assistant"}
        start = time.perf_counter()
        error = "no attempt made"
        for attempt in range(3):
            if attempt:
                time.sleep(1.5 * attempt)
            try:
                response = self._http.post(API_URL, json=body, headers=headers)
            except httpx.HTTPError as exc:
                error = f"network error: {exc}"
                continue
            if response.status_code in RETRY_STATUS:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
            if response.status_code != 200:
                raise LLMError(f"OpenRouter HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
            if "error" in data or not data.get("choices"):
                error = f"provider error: {str(data.get('error', data))[:200]}"
                continue
            usage = data.get("usage") or {}
            return LLMResult(
                text=data["choices"][0]["message"].get("content") or "",
                model=data.get("model", self.model),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                cost=float(usage.get("cost") or 0.0),
                seconds=time.perf_counter() - start,
            )
        raise LLMError(f"OpenRouter request failed after 3 attempts ({error}).")
