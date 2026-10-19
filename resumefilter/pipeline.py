from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from resumefilter.config import OCRSettings
from resumefilter.storage import (
    FailureRecorder,
    SourceInfo,
    atomic_write_json,
    atomic_write_text,
    checkpoint_key,
    checkpoint_path,
    inspect_source,
    is_complete,
    result_paths,
    source_metadata,
    utc_now,
)


class OCRClientProtocol(Protocol):
    async def recognize(
        self, image_data: bytes, mime_type: str, request_id: str | None = None
    ) -> str: ...


class RendererProtocol(Protocol):
    @property
    def mime_type(self) -> str: ...

    def page_count(self, path: Path) -> int: ...

    def render_page(self, path: Path, page_index: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class DocumentResult:
    relative_path: Path
    status: str
    pages_total: int = 0
    pages_ocr: int = 0
    pages_cached: int = 0
    error: str | None = None


class OCRPipeline:
    def __init__(
        self,
        settings: OCRSettings,
        client: OCRClientProtocol,
        renderer: RendererProtocol,
    ) -> None:
        self._settings = settings
        self._client = client
        self._renderer = renderer
        self._failures = FailureRecorder(settings.output_path)
        self._page_semaphore = asyncio.Semaphore(settings.page_concurrency)

    async def process(self, pdf_path: Path, relative_path: Path) -> DocumentResult:
        try:
            source = await asyncio.to_thread(inspect_source, pdf_path, relative_path)
        except Exception as exc:
            await self._failures.record(
                None, stage="inspect", error=exc, source_path=relative_path
            )
            return DocumentResult(relative_path, "failed", error=str(exc))

        paths = result_paths(self._settings.output_path, relative_path)
        if is_complete(paths, source, self._settings.processing_fingerprint):
            return DocumentResult(relative_path, "skipped")

        try:
            page_count = await asyncio.to_thread(self._renderer.page_count, pdf_path)
            if page_count <= 0:
                raise ValueError("PDF contains no pages")
        except Exception as exc:
            await self._failures.record(source, stage="open_pdf", error=exc)
            await asyncio.to_thread(
                self._write_state, paths.metadata, source, "failed", 0, str(exc)
            )
            return DocumentResult(relative_path, "failed", error=str(exc))

        key = checkpoint_key(source.sha256, self._settings.processing_fingerprint)
        await asyncio.to_thread(
            self._write_state, paths.metadata, source, "in_progress", page_count, None, key
        )

        page_results = await asyncio.gather(
            *(
                self._process_page(source, page_index, key)
                for page_index in range(page_count)
            )
        )
        failures = [result for result in page_results if result[0] == "failed"]
        if failures:
            message = f"{len(failures)} of {page_count} page(s) failed"
            await asyncio.to_thread(
                self._write_state,
                paths.metadata,
                source,
                "failed",
                page_count,
                message,
                key,
            )
            return DocumentResult(
                relative_path,
                "failed",
                pages_total=page_count,
                pages_ocr=sum(result[0] == "ocr" for result in page_results),
                pages_cached=sum(result[0] == "cached" for result in page_results),
                error=message,
            )

        parts: list[str] = []
        for page_number in range(1, page_count + 1):
            page_text = checkpoint_path(
                self._settings.output_path, key, page_number
            ).read_text(encoding="utf-8").strip()
            parts.append(f"<!-- Page {page_number} -->\n\n{page_text}")
        combined = "\n\n---\n\n".join(parts) + "\n"

        await asyncio.to_thread(atomic_write_text, paths.markdown, combined)
        await asyncio.to_thread(
            self._write_state, paths.metadata, source, "complete", page_count, None, key
        )
        return DocumentResult(
            relative_path,
            "complete",
            pages_total=page_count,
            pages_ocr=sum(result[0] == "ocr" for result in page_results),
            pages_cached=sum(result[0] == "cached" for result in page_results),
        )

    async def _process_page(
        self, source: SourceInfo, page_index: int, key: str
    ) -> tuple[str, str | None]:
        page_number = page_index + 1
        destination = checkpoint_path(self._settings.output_path, key, page_number)
        if destination.is_file() and destination.stat().st_size > 0:
            return "cached", None

        async with self._page_semaphore:
            try:
                image = await asyncio.to_thread(
                    self._renderer.render_page, source.path, page_index
                )
                blank_check = getattr(self._renderer, "is_blank_image", None)
                if callable(blank_check) and await asyncio.to_thread(blank_check, image):
                    await asyncio.to_thread(
                        atomic_write_text, destination, "[空白页]\n"
                    )
                    return "blank", None
                request_id = f"resume-ocr-{source.sha256[:12]}-p{page_number}"
                text = await self._client.recognize(
                    image, self._renderer.mime_type, request_id=request_id
                )
                if not text.strip():
                    raise ValueError("OCR output is empty")
                await asyncio.to_thread(atomic_write_text, destination, text.strip() + "\n")
                return "ocr", None
            except Exception as exc:
                await self._failures.record(
                    source,
                    stage="ocr_page",
                    error=exc,
                    page_number=page_number,
                )
                return "failed", str(exc)

    def _write_state(
        self,
        metadata_path: Path,
        source: SourceInfo,
        status: str,
        page_count: int,
        error: str | None,
        key: str | None = None,
    ) -> None:
        metadata = {
            "status": status,
            "updated_at": utc_now(),
            "source": source_metadata(source),
            "source_sha256": source.sha256,
            "processing_fingerprint": self._settings.processing_fingerprint,
            "checkpoint_key": key,
            "page_count": page_count,
            "model": self._settings.model,
            "render_dpi": self._settings.render_dpi,
            "render_format": self._settings.render_format,
            "error": error,
        }
        atomic_write_json(metadata_path, metadata)


def discover_pdfs(input_path: Path) -> list[tuple[Path, Path]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"input file is not a PDF: {input_path}")
        return [(input_path, Path(input_path.name))]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    files = sorted(
        (path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).lower(),
    )
    return [(path, path.relative_to(input_path)) for path in files]

