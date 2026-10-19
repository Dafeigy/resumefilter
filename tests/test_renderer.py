from pathlib import Path

import pymupdf

from resumefilter.renderer import PDFRenderer


def test_pdf_page_is_rendered_to_png(tmp_path: Path) -> None:
    pdf_path = tmp_path / "resume.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Resume OCR smoke test")
    document.save(pdf_path)
    document.close()

    renderer = PDFRenderer(dpi=144, image_format="PNG")

    assert renderer.page_count(pdf_path) == 1
    image = renderer.render_page(pdf_path, 0)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert renderer.is_blank_image(image) is False


def test_blank_pdf_page_is_detected(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    renderer = PDFRenderer(dpi=144, image_format="PNG")
    image = renderer.render_page(pdf_path, 0)
    assert renderer.is_blank_image(image) is True

