from __future__ import annotations

import asyncio
import base64
import random
import re
import uuid
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from resumefilter.config import OCRSettings


_GROUNDING_METADATA = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", re.DOTALL
)
_EOS_MARKERS = ("<｜end▁of▁sentence｜>", "<|end_of_sentence|>")


class OCRError(RuntimeError):
    """A safe-to-display OCR request error."""


class SiliconFlowOCRClient:
    def __init__(
        self,
        settings: OCRSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        timeout = httpx.Timeout(
            connect=10.0,
            read=settings.request_timeout_seconds,
            write=settings.request_timeout_seconds,
            pool=settings.request_timeout_seconds,
        )
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def __aenter__(self) -> "SiliconFlowOCRClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def recognize(
        self,
        image_data: bytes,
        mime_type: str,
        request_id: str | None = None,
    ) -> str:
        data_uri = f"data:{mime_type};base64,{base64.b64encode(image_data).decode('ascii')}"
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": self._settings.prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self._settings.max_output_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            "X-Trace-Id": request_id or str(uuid.uuid4()),
        }

        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = await self._client.post(
                    self._settings.api_url,
                    headers=headers,
                    json=payload,
                )
                if response.status_code in {429, 503, 504} and attempt < self._settings.max_retries:
                    await asyncio.sleep(self._retry_delay(response, attempt))
                    continue
                if response.status_code >= 400:
                    raise OCRError(self._http_error(response))
                return self._parse_response(response)
            except OCRError:
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self._settings.max_retries:
                    await asyncio.sleep((2**attempt) + random.uniform(0, 0.5))
                    continue
                break

        raise OCRError("SiliconFlow OCR request failed after retries") from last_error

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        value = response.headers.get("retry-after")
        if value:
            try:
                return min(max(0.0, float(value)), 30.0)
            except ValueError:
                try:
                    date_header = response.headers.get("date")
                    if date_header:
                        seconds = (
                            parsedate_to_datetime(value) - parsedate_to_datetime(date_header)
                        ).total_seconds()
                        return min(max(0.0, seconds), 30.0)
                except (TypeError, ValueError):
                    pass
        return min((2**attempt) + random.uniform(0, 0.5), 30.0)

    @classmethod
    def _parse_response(cls, response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError as exc:
            raise OCRError("SiliconFlow returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise OCRError("SiliconFlow returned an invalid response object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OCRError("SiliconFlow returned no OCR choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise OCRError("SiliconFlow returned an invalid OCR choice")
        if choice.get("finish_reason") == "length":
            raise OCRError("OCR output reached the model context limit")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OCRError("SiliconFlow returned empty OCR content")
        return cls.clean_output(content)

    @staticmethod
    def clean_output(text: str) -> str:
        cleaned = _GROUNDING_METADATA.sub("", text)
        for marker in _EOS_MARKERS:
            cleaned = cleaned.replace(marker, "")
        cleaned = cleaned.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
        cleaned = re.sub(r"\n{4,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _http_error(response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict):
                message = body.get("message")
                if isinstance(message, str):
                    return f"SiliconFlow HTTP {response.status_code}: {message}"
                error = body.get("error")
                if isinstance(error, dict) and isinstance(error.get("message"), str):
                    return f"SiliconFlow HTTP {response.status_code}: {error['message']}"
        except ValueError:
            pass
        return f"SiliconFlow returned HTTP {response.status_code}"

