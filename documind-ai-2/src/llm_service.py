"""Gemini access via the Google Gen AI SDK (``google-genai``), behind a small interface.

The pipeline depends only on ``LLMBackend.generate``; tests inject fakes so no API
call is ever needed. Errors are mapped to user-safe exception types and never
include the API key.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from config import Settings

from .security import SlidingWindowLimiter, RateLimitExceeded, redact

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class. ``str(exc)`` is safe to show to end users."""


class MissingAPIKeyError(LLMError):
    pass


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


class LLMBackend(Protocol):
    def generate(
        self, system: str, prompt: str, json_output: bool = True, max_output_tokens: Optional[int] = None
    ) -> str: ...


class GeminiService:
    def __init__(self, settings: Settings, global_limiter: Optional[SlidingWindowLimiter] = None):
        self.settings = settings
        self._client = None
        self._limiter = global_limiter or SlidingWindowLimiter(settings.global_llm_calls_per_hour, 3600)
        self.calls = 0
        self.total_latency = 0.0

    def _get_client(self):
        if not self.settings.has_api_key:
            raise MissingAPIKeyError(
                "No Gemini API key is configured. Set GEMINI_API_KEY in your environment "
                "(or in the hosting platform's secrets) and restart the app."
            )
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.settings.gemini_api_key,
                # google-genai expects milliseconds
                http_options=types.HttpOptions(timeout=int(self.settings.llm_timeout_seconds * 1000)),
            )
        return self._client

    def generate(
        self, system: str, prompt: str, json_output: bool = True, max_output_tokens: Optional[int] = None
    ) -> str:
        client = self._get_client()  # raises MissingAPIKeyError first
        try:
            self._limiter.check("global")
        except RateLimitExceeded as exc:
            raise LLMRateLimitError(
                "This app has reached its hourly limit of AI requests. Please try again later."
            ) from exc

        from google.genai import errors, types

        config_kwargs = dict(
            system_instruction=system,
            temperature=self.settings.llm_temperature,
            max_output_tokens=max_output_tokens or self.settings.llm_max_output_tokens,
            response_mime_type="application/json" if json_output else "text/plain",
        )
        thinking = self._thinking_config(types)
        if thinking is not None:
            config_kwargs["thinking_config"] = thinking
        config = types.GenerateContentConfig(**config_kwargs)
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.settings.gemini_model, contents=prompt, config=config
            )
        except errors.APIError as exc:
            raise self._map_api_error(exc) from None
        except Exception as exc:  # network failures, timeouts, SDK internals
            logger.warning("LLM call failed: %s", redact(f"{type(exc).__name__}: {exc}"))
            raise LLMUnavailableError(
                "Could not reach the Gemini API. Check your internet connection and try again."
            ) from None
        finally:
            self.calls += 1
            self.total_latency += time.perf_counter() - started

        text = getattr(response, "text", None)
        if not text or not text.strip():
            reason = self._finish_reason(response)
            if "MAX_TOKENS" in reason:
                raise LLMResponseError(
                    "The model used up its output-token limit before writing an answer. "
                    "Increase LLM_MAX_OUTPUT_TOKENS in your .env file (for example 4096) and try again."
                )
            raise LLMResponseError(
                "The model returned an empty response (it may have been blocked by a safety filter). "
                "Try rephrasing the question."
            )
        return text

    def _thinking_config(self, types):
        """Keep hidden 'thinking' small: on Gemini 2.5+/3 it is counted against max_output_tokens,
        so an uncapped thinking phase can use the whole budget and leave an empty answer."""
        model = self.settings.gemini_model.lower()
        try:
            if "gemini-3" in model or "gemini-4" in model:
                return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
            if "gemini-2.5" in model:
                # 2.5 Pro cannot switch thinking off; Flash / Flash-Lite can.
                return types.ThinkingConfig(thinking_budget=512 if "pro" in model else 0)
        except Exception:  # older SDKs without these fields
            logger.debug("ThinkingConfig not supported by this SDK version")
        return None

    @staticmethod
    def _finish_reason(response) -> str:
        try:
            return str(response.candidates[0].finish_reason)
        except Exception:
            return ""

    @staticmethod
    def _map_api_error(exc) -> LLMError:
        code = getattr(exc, "code", None)
        detail = redact(str(getattr(exc, "message", "") or exc))[:300]
        logger.warning("Gemini API error code=%s detail=%s", code, detail)
        if code == 429:
            return LLMRateLimitError(
                "The Gemini API quota or rate limit was reached for this key/model. Google says: "
                f"{detail}. If it mentions a limit of 0, this model is not available on your free tier: "
                "set GEMINI_MODEL=gemini-2.5-flash-lite (or another model) in .env, or enable billing."
            )
        if code in (401, 403):
            return LLMAuthError(
                f"The Gemini API rejected the API key ({detail}). Check that GEMINI_API_KEY is valid and has access."
            )
        if code == 404:
            return LLMUnavailableError(
                f"The configured Gemini model was not found ({detail}). Check the GEMINI_MODEL setting."
            )
        if code is not None and code >= 500:
            return LLMUnavailableError("The Gemini service is temporarily unavailable. Try again shortly.")
        return LLMUnavailableError("The Gemini API returned an error. Please try again.")
