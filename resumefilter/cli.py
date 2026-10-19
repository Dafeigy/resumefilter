from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from resumefilter.client import SiliconFlowOCRClient
from resumefilter.config import ConfigurationError, OCRSettings
from resumefilter.pipeline import OCRPipeline, discover_pdfs
from resumefilter.renderer import PDFRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-filter-ocr",
        description="Resumable PDF OCR through SiliconFlow DeepSeek-OCR.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="PDF file or directory; defaults to OCR_INPUT_DIR/resumes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory; defaults to OCR_OUTPUT_DIR/ocr_results",
    )
    return parser


async def run(settings: OCRSettings) -> int:
    try:
        settings.validate()
        documents = discover_pdfs(settings.input_path)
    except (ConfigurationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not documents:
        print(f"No PDF files found under {settings.input_path}")
        return 0

    settings.output_path.mkdir(parents=True, exist_ok=True)
    renderer = PDFRenderer(settings.render_dpi, settings.render_format)
    completed = skipped = failed = 0

    async with SiliconFlowOCRClient(settings) as client:
        pipeline = OCRPipeline(settings, client, renderer)
        for index, (pdf_path, relative_path) in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] {relative_path}")
            result = await pipeline.process(pdf_path, relative_path)
            if result.status == "complete":
                completed += 1
                print(
                    f"  OK: {result.pages_total} page(s), "
                    f"{result.pages_ocr} OCR, {result.pages_cached} resumed"
                )
            elif result.status == "skipped":
                skipped += 1
                print("  SKIP: completed result is current")
            else:
                failed += 1
                print(f"  FAIL: {result.error}", file=sys.stderr)

    print(
        f"Done: {completed} completed, {skipped} skipped, {failed} failed; "
        f"results: {settings.output_path}"
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = OCRSettings.from_env()
    except ConfigurationError as exc:
        parser.error(str(exc))
    if args.input is not None:
        settings = replace(settings, input_path=args.input)
    if args.output is not None:
        settings = replace(settings, output_path=args.output)
    try:
        return asyncio.run(run(settings))
    except KeyboardInterrupt:
        print("Interrupted; completed page checkpoints were kept.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

