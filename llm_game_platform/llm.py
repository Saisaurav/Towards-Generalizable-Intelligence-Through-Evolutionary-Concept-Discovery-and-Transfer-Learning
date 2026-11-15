"""OpenAI-compatible chat client for supported hosted LLM providers."""

import json
import math
import os

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


class LLMConfigurationError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY")
        self.base_url = (
            base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL")
        if timeout is None:
            configured_timeout = os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")
            try:
                timeout = float(configured_timeout)
            except ValueError as error:
                raise LLMConfigurationError(
                    "LLM_REQUEST_TIMEOUT_SECONDS must be a positive number"
                ) from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise LLMConfigurationError(
                "LLM_REQUEST_TIMEOUT_SECONDS must be a positive number"
            )
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._request(system_prompt, user_prompt, json_mode=True)

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        """Free-form completion (used for generated Python source)."""
        return self._request(system_prompt, user_prompt, json_mode=False)

    def _request(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        if not self.api_key:
            raise LLMConfigurationError("set LLM_API_KEY before requesting game generation")
        if not self.model:
            raise LLMConfigurationError("set LLM_MODEL before requesting game generation")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                **options,
            )
        except APITimeoutError as error:
            raise LLMRequestError(
                f"LLM endpoint did not respond within {self.timeout:g} seconds"
            ) from error
        except APIStatusError as error:
            detail = self._error_detail(error.body)
            suffix = f": {detail}" if detail else ""
            raise LLMRequestError(
                f"LLM endpoint returned HTTP {error.status_code}{suffix}"
            ) from error
        except APIConnectionError as error:
            raise LLMRequestError(f"could not reach LLM endpoint: {error}") from error

        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError) as error:
            raise LLMRequestError(
                "LLM endpoint returned an invalid chat-completion response"
            ) from error
        if not isinstance(content, str) or not content:
            raise LLMRequestError("LLM endpoint returned an empty message")
        return content

    def _error_detail(self, body: object) -> str:
        detail = ""
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("details")
            elif isinstance(error, str):
                message = error
            else:
                message = body.get("message")
            if isinstance(message, str):
                detail = message
            else:
                detail = json.dumps(body, ensure_ascii=True)
        elif isinstance(body, str):
            detail = body
        detail = " ".join(detail.split())[:500]
        if self.api_key:
            detail = detail.replace(self.api_key, "[redacted]")
        return detail
