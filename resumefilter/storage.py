from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceInfo:
    path: Path
    relative_path: Path
    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class ResultPaths:
    markdown: Path
    metadata: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source(path: Path, relative_path: Path) -> SourceInfo:
    stat = path.stat()
    return SourceInfo(
        path=path,
        relative_path=relative_path,
        sha256=sha256_file(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def result_paths(output_root: Path, relative_pdf: Path) -> ResultPaths:
    relative_markdown = relative_pdf.with_suffix(".md")
    markdown = output_root / relative_markdown
    metadata = markdown.with_suffix(".ocr.json")
    return ResultPaths(markdown=markdown, metadata=metadata)


def checkpoint_key(source_sha256: str, processing_fingerprint: str) -> str:
    value = f"{source_sha256}:{processing_fingerprint}".encode("ascii")
    return hashlib.sha256(value).hexdigest()


def checkpoint_path(output_root: Path, key: str, page_number: int) -> Path:
    return output_root / ".checkpoints" / key / f"page-{page_number:04d}.md"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def is_complete(
    paths: ResultPaths,
    source: SourceInfo,
    processing_fingerprint: str,
) -> bool:
    if not paths.markdown.is_file() or paths.markdown.stat().st_size == 0:
        return False
    metadata = read_json(paths.metadata)
    return bool(
        metadata
        and metadata.get("status") == "complete"
        and metadata.get("source_sha256") == source.sha256
        and metadata.get("processing_fingerprint") == processing_fingerprint
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, serialized)


class FailureRecorder:
    def __init__(self, output_root: Path) -> None:
        self._path = output_root / "failures.jsonl"
        self._lock = asyncio.Lock()

    async def record(
        self,
        source: SourceInfo | None,
        *,
        stage: str,
        error: BaseException,
        page_number: int | None = None,
        source_path: Path | None = None,
    ) -> None:
        entry = {
            "timestamp": utc_now(),
            "source": str(source.relative_path if source else source_path),
            "source_sha256": source.sha256 if source else None,
            "page": page_number,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(line)
            file.flush()


def source_metadata(source: SourceInfo) -> dict[str, Any]:
    value = asdict(source)
    value["path"] = str(source.path)
    value["relative_path"] = str(source.relative_path)
    return value

