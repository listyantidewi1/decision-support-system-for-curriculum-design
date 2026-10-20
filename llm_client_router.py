"""OpenAI-compatible LLM client routing.

Current provider policy:
    Default: every model routes to OpenRouter.
    Explicit override: `jatevo:<model>` still works for reproducing old runs,
    but Jatevo is no longer selected automatically.

Public API:
    client, provider = get_client_for_model("gpt-5.4-mini")
    # -> OpenAI client pointed at OpenRouter, provider="openrouter"

    client, provider = get_client_for_model("openrouter:gpt-5.4-mini")
    # -> same as above, explicit provider.

    client, provider = get_client_for_model("jatevo:gpt-5.4-mini")
    # -> explicit legacy override only.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JATEVO_BASE_URL = "https://lb.jatevo.ai/v1"

_GPT_FAMILY_TOKENS = re.compile(
    r"(^|/|-)(gpt-|o1|o3|chatgpt)",
    re.IGNORECASE,
)


def _strip_force_prefix(model_name: str) -> tuple[str, str | None]:
    """Allow explicit provider override via `openrouter:<model>` or `jatevo:<model>`."""
    if model_name.startswith("openrouter:"):
        return model_name.split(":", 1)[1], "openrouter"
    if model_name.startswith("jatevo:"):
        return model_name.split(":", 1)[1], "jatevo"
    return model_name, None


def is_gpt_family(model_name: str) -> bool:
    """Return True for common GPT/o-series model identifiers.

    This is kept for compatibility with older tests and call sites. It no
    longer controls default provider routing.
    """
    model_name, _forced_provider = _strip_force_prefix(model_name or "")
    if not model_name:
        return False
    return bool(_GPT_FAMILY_TOKENS.search(model_name))


def _read_key(file_path: Path) -> str:
    try:
        key = file_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"API key file not found: {file_path}. "
            f"Create it or set the appropriate environment variable."
        ) from exc
    if not key:
        raise RuntimeError(f"API key file is empty: {file_path}")
    return key


def _load_openrouter_key() -> str:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    return _read_key(Path("api_keys") / "OpenRouter.txt")


def _load_jatevo_key() -> str:
    env_key = os.getenv("JATEVO_API_KEY")
    if env_key:
        return env_key
    return _read_key(Path("api_keys") / "jatevo.txt")


def get_client_for_model(
    model_name: str,
    request_timeout: float = 120.0,
) -> Tuple[OpenAI, str]:
    """Return (OpenAI-compatible client, provider_tag)."""
    model_name, forced_provider = _strip_force_prefix(model_name)
    env_provider = os.getenv("LLM_FORCE_PROVIDER", "").strip().lower() or None
    provider = forced_provider or env_provider or "openrouter"

    if provider == "openrouter":
        api_key = _load_openrouter_key()
        base_url = OPENROUTER_BASE_URL
    elif provider == "jatevo":
        api_key = _load_jatevo_key()
        base_url = JATEVO_BASE_URL
    else:
        raise ValueError(f"Unsupported LLM provider override: {provider}")

    return OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout), provider


def strip_provider_prefix(model_name: str) -> str:
    """Return the model id to send to the selected provider.

    `openrouter:<model>` and `jatevo:<model>` are local routing prefixes, so
    they are removed. Provider namespace prefixes like `openai/gpt-5` are
    stripped only for explicit Jatevo calls because OpenRouter commonly uses
    provider-qualified ids.
    """
    model_name, forced_provider = _strip_force_prefix(model_name)
    if forced_provider == "jatevo" and "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name

