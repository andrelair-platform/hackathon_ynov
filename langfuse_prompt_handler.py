"""
LangfusePromptHandler — LiteLLM CustomLogger
=============================================
Intercepts requests to the phi3-financial virtual model and injects the current
production-labelled system prompt fetched from Langfuse.

Prompt source:  Langfuse Prompt Management → name="phi3-financial-system", label="production"
Cache TTL:      5 minutes (in-process, per LiteLLM worker)
Fail-open:      if Langfuse is unreachable, serves the stale cached prompt, then falls back
                to the PHI3_FINANCIAL_SYSTEM_PROMPT env var (set from the ConfigMap)

Rollback flow:
  1. In Langfuse UI → Prompt Management → phi3-financial-system → set old version as "production"
  2. Wait up to 5 minutes for the cache to expire (or restart the LiteLLM pod for immediate effect)
  3. All new requests automatically use the rolled-back prompt — no git push, no ollama create

Deployment:
  Mounted into the LiteLLM pod as a ConfigMap volume at /app/langfuse_prompt_handler.py.
  Registered in litellm config.yaml under general_settings.callbacks as
  "langfuse_prompt_handler.LangfusePromptHandler".
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 300  # 5 minutes
_cache: dict = {}          # {"text": str, "expires_at": float}

PROMPT_NAME  = "phi3-financial-system"
PROMPT_LABEL = "production"
TARGET_MODEL = "phi3-financial"


async def _fetch_prompt() -> Optional[str]:
    """Return production system prompt, refreshing from Langfuse when cache expires."""
    now = time.monotonic()
    if _cache.get("expires_at", 0) > now and "text" in _cache:
        return _cache["text"]

    try:
        from langfuse import Langfuse  # already installed in LiteLLM venv
        lf = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get(
                "LANGFUSE_HOST",
                "http://langfuse-web.langfuse.svc.cluster.local:3000",
            ),
        )
        prompt = lf.get_prompt(PROMPT_NAME, label=PROMPT_LABEL)
        _cache["text"]       = prompt.prompt
        _cache["expires_at"] = now + _CACHE_TTL_SECONDS
        logger.info(
            "[LangfusePromptHandler] Refreshed '%s' label=%s (TTL %ds)",
            PROMPT_NAME, PROMPT_LABEL, _CACHE_TTL_SECONDS,
        )
        return _cache["text"]

    except Exception as exc:
        logger.warning(
            "[LangfusePromptHandler] Langfuse fetch failed (%s). Serving %s.",
            exc,
            "stale cache" if "text" in _cache else "ConfigMap env var fallback",
        )
        # Fail-open priority: stale cache → ConfigMap env var → None (no injection)
        if "text" in _cache:
            return _cache["text"]
        return os.environ.get("PHI3_FINANCIAL_SYSTEM_PROMPT")


try:
    from litellm.integrations.custom_logger import CustomLogger as _Base
except ImportError:
    _Base = object  # type: ignore[assignment,misc]


class LangfusePromptHandler(_Base):  # type: ignore[misc]
    """
    Registered in litellm config.yaml under general_settings.callbacks.

    For every request targeting phi3-financial:
    - Fetches the production prompt from Langfuse (cached 5 min)
    - Strips any existing system message (prevents prompt-injection via system turn)
    - Prepends the Langfuse prompt as the sole system message
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> dict:
        if data.get("model") != TARGET_MODEL:
            return data

        system_prompt = await _fetch_prompt()
        if not system_prompt:
            logger.error(
                "[LangfusePromptHandler] No prompt available for '%s' — "
                "request proceeds without system message.", TARGET_MODEL,
            )
            return data

        # Remove any existing system message then inject ours as the first message.
        # This prevents a user from overriding guardrails by sending a system turn
        # through an API key that has system-message permissions.
        messages = [m for m in data.get("messages", []) if m.get("role") != "system"]
        data["messages"] = [{"role": "system", "content": system_prompt}] + messages
        logger.debug(
            "[LangfusePromptHandler] Injected system prompt for '%s' (%d chars)",
            TARGET_MODEL, len(system_prompt),
        )
        return data
