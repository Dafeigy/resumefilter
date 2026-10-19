from pathlib import Path

import pytest

from resumefilter.config import OCRSettings
from resumefilter.pipeline import OCRPipeline


class FakeRenderer:
    mime_type = "image/png"

    def page_count(self, _path: Path) -> int:
        return 3

    def render_page(self, _path: Path, page_index: int) -> bytes:
        return str(page_index + 1).encode("ascii")


class FlakyClient:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fail_page_two = True

    async def recognize(
        self, image_data: bytes, _mime_type: str, request_id: str | None = None
    ) -> str:
        page = int(image_data.decode("ascii"))
        self.calls.append(page)
        if page == 2 and self.fail_page_two:
            raise RuntimeError("temporary upstream failure")
        return f"content for page {page}"


@pytest.mark.asyncio
async def test_failed_document_resumes_only_missing_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "input" / "candidate.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"not-a-real-pdf; fake renderer is injected")
    output = tmp_path / "output"
    settings = OCRSettings(
        api_key="secret",
        api_url="https://example.invalid/v1/chat/completions",
        model="deepseek-ai/DeepSeek-OCR",
        output_path=output,
    )
    client = FlakyClient()
    pipeline = OCRPipeline(settings, client, FakeRenderer())

    first = await pipeline.process(pdf, Path("candidate.pdf"))
    assert first.status == "failed"
    assert client.calls == [1, 2, 3]
    assert not (output / "candidate.md").exists()

    client.fail_page_two = False
    second = await pipeline.process(pdf, Path("candidate.pdf"))
    assert second.status == "complete"
    assert client.calls == [1, 2, 3, 2]
    assert second.pages_cached == 2
    assert "content for page 2" in (output / "candidate.md").read_text("utf-8")

    third = await pipeline.process(pdf, Path("candidate.pdf"))
    assert third.status == "skipped"
    assert client.calls == [1, 2, 3, 2]

