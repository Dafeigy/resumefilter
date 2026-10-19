from __future__ import annotations

from pathlib import Path


class PDFRenderer:
    def __init__(self, dpi: int, image_format: str) -> None:
        self.dpi = dpi
        self.image_format = image_format.upper()

    @property
    def mime_type(self) -> str:
        return "image/png" if self.image_format == "PNG" else "image/jpeg"

    def page_count(self, path: Path) -> int:
        import pymupdf

        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError("password-protected PDFs are not supported")
            return document.page_count

    def render_page(self, path: Path, page_index: int) -> bytes:
        import pymupdf

        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError("password-protected PDFs are not supported")
            page = document.load_page(page_index)
            scale = self.dpi / 72.0
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            output = "png" if self.image_format == "PNG" else "jpeg"
            return pixmap.tobytes(output=output)

    @staticmethod
    def is_blank_image(image_data: bytes) -> bool:
        """Return true when a rendered page contains only near-white pixels."""
        import pymupdf

        pixmap = pymupdf.Pixmap(image_data)
        # Rendered pages have no alpha channel. A small tolerance accommodates
        # JPEG compression while still requiring every channel to be near white.
        return bool(pixmap.samples) and min(pixmap.samples) >= 250

