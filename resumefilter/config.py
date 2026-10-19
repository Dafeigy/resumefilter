from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


class ConfigurationError(ValueError):
    """Raised when an environment setting is invalid or missing."""


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{name} must be zero or greater")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class OCRSettings:
    api_key: str
    api_url: str
    model: str
    prompt: str = DEFAULT_OCR_PROMPT
    input_path: Path = Path("resumes")
    output_path: Path = Path("ocr_results")
    render_dpi: int = 144
    render_format: str = "PNG"
    page_concurrency: int = 2
    max_retries: int = 2
    request_timeout_seconds: float = 180.0
    max_output_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "OCRSettings":
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        api_url = os.getenv("SILICONFLOW_API_URL", f"{base_url}/chat/completions")
        render_format = os.getenv("OCR_RENDER_FORMAT", "PNG").upper()
        if render_format not in {"PNG", "JPEG"}:
            raise ConfigurationError("OCR_RENDER_FORMAT must be PNG or JPEG")
        return cls(
            api_key=os.getenv("SILICONFLOW_API_KEY", "").strip(),
            api_url=api_url,
            model=os.getenv("SILICONFLOW_OCR_MODEL", "deepseek-ai/DeepSeek-OCR"),
            prompt=os.getenv("OCR_PROMPT", DEFAULT_OCR_PROMPT),
            input_path=Path(os.getenv("OCR_INPUT_DIR", "resumes")),
            output_path=Path(os.getenv("OCR_OUTPUT_DIR", "ocr_results")),
            render_dpi=_positive_int("OCR_RENDER_DPI", 144),
            render_format=render_format,
            page_concurrency=_positive_int("OCR_PAGE_CONCURRENCY", 2),
            max_retries=_nonnegative_int("OCR_MAX_RETRIES", 2),
            request_timeout_seconds=_positive_float("OCR_REQUEST_TIMEOUT_SECONDS", 180.0),
            max_output_tokens=_positive_int("OCR_MAX_OUTPUT_TOKENS", 4096),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise ConfigurationError("SILICONFLOW_API_KEY is not configured")
        if not self.api_url.startswith(("http://", "https://")):
            raise ConfigurationError("SILICONFLOW_BASE_URL/SILICONFLOW_API_URL must be an HTTP(S) URL")

    @property
    def processing_fingerprint(self) -> str:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "render_dpi": self.render_dpi,
            "render_format": self.render_format,
            "max_output_tokens": self.max_output_tokens,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

