"""Optional OpenAI client with graceful degradation.

Every LLM-touching stage (router, NL->SQL, synthesizer) calls `complete()`;
if there is no API key, or the call/parsing fails, the stage falls back to its
deterministic implementation. The pipeline therefore never hard-depends on
network access — a property worth keeping all the way to production (it's
your incident-mode behaviour).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from .config import settings


class LLMRequiredError(RuntimeError):
    """Raised when LLM-only mode is on but the LLM is unavailable or fails."""


_client = None
_checked = False
_model_override: Optional[str] = None


def set_model(name: Optional[str]) -> None:
    """Override the model at runtime (e.g. from a UI picker)."""
    global _model_override
    _model_override = name or None


def current_model() -> str:
    return _model_override or settings.llm_model


def get_client():
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        _client = OpenAI()
    except Exception:
        _client = None
    return _client


def llm_available() -> bool:
    return get_client() is not None


def complete(system: str, user: str, max_tokens: int | None = None,
             temperature: float | None = None) -> Optional[str]:
    client = get_client()
    if client is None:
        return None
    try:
        kwargs = {
            "model": current_model(),
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception:
        return None


def complete_json(system: str, user: str,
                  temperature: float | None = None) -> Optional[dict[str, Any]]:
    raw = complete(system, user, temperature=temperature)
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
