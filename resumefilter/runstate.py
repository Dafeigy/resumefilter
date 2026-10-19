from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from resumefilter.screening import write_json_atomic


RUN_SCHEMA_VERSION = 2


def local_now() -> str:
    return datetime.now().astimezone().isoformat()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def result_path(run_dir: Path, index: int) -> Path:
    return run_dir / "results" / f"{index:04d}.json"


def error_path(run_dir: Path, index: int) -> Path:
    return run_dir / "errors" / f"{index:04d}.json"


def scan_item_states(
    run_dir: Path, selected_files: list[str]
) -> tuple[set[int], set[int]]:
    completed: set[int] = set()
    failed: set[int] = set()
    for index, source_file in enumerate(selected_files, start=1):
        result_file = result_path(run_dir, index)
        if result_file.is_file():
            try:
                result = load_json_object(result_file)
            except ValueError:
                pass
            else:
                if result.get("source_file") == source_file:
                    completed.add(index)
                    continue
        if error_path(run_dir, index).is_file():
            failed.add(index)
    return completed, failed


def write_progress(
    run_dir: Path,
    *,
    total: int,
    completed: set[int],
    failed: set[int],
    status: str,
    current_index: int | None = None,
) -> dict[str, Any]:
    active_failed = failed.difference(completed)
    attempted = completed.union(active_failed)
    progress = {
        "status": status,
        "updated_at": local_now(),
        "total": total,
        "completed": len(completed),
        "failed": len(active_failed),
        "not_started": total - len(attempted),
        "remaining": total - len(completed),
        "current_index": current_index,
    }
    write_json_atomic(run_dir / "progress.json", progress)
    return progress


def write_item_error(
    run_dir: Path,
    index: int,
    source_file: str,
    message: str,
) -> None:
    destination = error_path(run_dir, index)
    attempts = 1
    if destination.is_file():
        try:
            previous = load_json_object(destination)
            attempts = int(previous.get("attempts", 0)) + 1
        except (ValueError, TypeError):
            pass
    write_json_atomic(
        destination,
        {
            "index": index,
            "source_file": source_file,
            "error": message,
            "attempts": attempts,
            "updated_at": local_now(),
        },
    )
