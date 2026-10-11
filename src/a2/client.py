"""A single local inference client; no hidden SDK retries."""

import os
import time
from urllib.parse import urlparse

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from a2.tasks import messages, response_format


class LMStudioClient:
    def __init__(self, settings: dict):
        self.settings = settings
        parsed = urlparse(settings["base_url"])
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("This project expects a local LM Studio endpoint.")
        self.api = OpenAI(
            base_url=settings["base_url"],
            api_key=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
            timeout=settings["timeout_seconds"],
            max_retries=0,
        )

    def connection_info(self) -> dict:
        model_ids = [model.id for model in self.api.models.list().data]
        if self.settings["model_id"] not in model_ids:
            raise ValueError(f"Model is not served: {self.settings['model_id']}")
        native_url = self.settings["base_url"].removesuffix("/v1") + "/api/v1/models"
        native = self.api.get(native_url, cast_to=dict)
        target = next(
            model for model in native["models"] if model["key"] == self.settings["model_id"]
        )
        if not target["loaded_instances"]:
            raise ValueError("Load the target model in LM Studio before generating.")
        return {"available_model_ids": model_ids, "target_model": target}

    def generate(self, sample: dict) -> dict:
        prompt_messages = messages(sample)
        format_schema = response_format(sample["task"])
        start = time.perf_counter()
        attempts = []
        for attempt in range(1, self.settings["max_attempts"] + 1):
            try:
                completion = self.api.chat.completions.create(
                    model=self.settings["model_id"],
                    messages=prompt_messages,
                    response_format=format_schema,
                    temperature=self.settings["temperature"],
                    max_tokens=self.settings["max_tokens"],
                    reasoning_effort=self.settings["reasoning_effort"],
                )
                return {
                    "sample_id": sample["id"],
                    "status": "completed",
                    "messages": prompt_messages,
                    "response_format": format_schema,
                    "response": completion.model_dump(),
                    "elapsed_seconds": time.perf_counter() - start,
                    "attempts": [*attempts, {"attempt": attempt, "status": "completed"}],
                }
            except (APIConnectionError, APITimeoutError, APIStatusError) as error:
                status_code = getattr(error, "status_code", None)
                retryable = status_code is None or status_code == 429 or status_code >= 500
                attempts.append(
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "status_code": status_code,
                    }
                )
                if not retryable or attempt == self.settings["max_attempts"]:
                    return {
                        "sample_id": sample["id"],
                        "status": "request_failed",
                        "messages": prompt_messages,
                        "response_format": format_schema,
                        "elapsed_seconds": time.perf_counter() - start,
                        "attempts": attempts,
                    }
                time.sleep(attempt)
        raise AssertionError("The configured request attempts must be positive.")
