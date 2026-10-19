from __future__ import annotations

import argparse
import secrets
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from resumefilter.config import ConfigurationError
from resumefilter.runstate import (
    RUN_SCHEMA_VERSION,
    load_json_object,
    local_now,
    result_path,
    scan_item_states,
    text_sha256,
    write_item_error,
    write_progress,
)
from resumefilter.screening import (
    OpenAICompatibleScreeningClient,
    SYSTEM_PROMPT,
    ScreeningError,
    ScreeningSettings,
    discover_ocr_results,
    sample_resumes,
    write_json_atomic,
)


def _screening_fingerprint(model: str, standard: str) -> str:
    return text_sha256(f"{model}\0{SYSTEM_PROMPT}\0{standard}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen OCR resume markdown files serially with resumable runs."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        dest="all_results",
        action="store_true",
        help="screen every final OCR markdown file in stable filename order",
    )
    selection.add_argument(
        "--count",
        type=int,
        help="number of resumes to sample randomly; defaults to 5",
    )
    selection.add_argument(
        "--file",
        type=Path,
        help="screen one specific OCR markdown file",
    )
    selection.add_argument(
        "--resume",
        type=Path,
        help="continue an interrupted schema-v2 screening run directory",
    )
    parser.add_argument("--seed", type=int, help="random seed; generated when omitted")
    parser.add_argument("--input", type=Path, help="OCR markdown directory")
    parser.add_argument("--output", type=Path, help="screening output directory")
    parser.add_argument(
        "--max-items",
        type=int,
        help="pause after this many new attempts; useful for controlled batches",
    )
    return parser


def _new_selection(
    args: argparse.Namespace,
    settings: ScreeningSettings,
) -> tuple[list[Path], int | None, str]:
    files = discover_ocr_results(settings.input_path)
    if args.file is not None:
        if args.seed is not None:
            raise ValueError("--seed cannot be used with --file")
        selected_file = args.file
        if not selected_file.is_absolute():
            selected_file = settings.input_path / selected_file
        selected_file = selected_file.resolve()
        available = {path.resolve() for path in files}
        if selected_file not in available:
            raise ValueError(
                f"specific file is not a final OCR markdown result: {selected_file}"
            )
        return [selected_file], None, "specific"
    if args.all_results:
        if args.seed is not None:
            raise ValueError("--seed cannot be used with --all")
        if not files:
            raise ValueError(f"no final OCR markdown files found in {settings.input_path}")
        return files, None, "all"
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    count = args.count if args.count is not None else 5
    return sample_resumes(files, count, seed), seed, "random"


def _create_run(
    settings: ScreeningSettings,
    selected: list[Path],
    seed: int | None,
    selection_mode: str,
    standard: str,
) -> tuple[Path, dict[str, object]]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = (settings.output_path / run_id).resolve()
    manifest: dict[str, object] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": local_now(),
        "model": settings.model,
        "seed": seed,
        "selection": selection_mode,
        "execution": "serial",
        "input_path": str(settings.input_path.resolve()),
        "standard_path": str(settings.standard_path.resolve()),
        "standard_sha256": text_sha256(standard),
        "screening_fingerprint": _screening_fingerprint(settings.model, standard),
        "selected_files": [path.name for path in selected],
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    write_progress(
        run_dir,
        total=len(selected),
        completed=set(),
        failed=set(),
        status="ready",
    )
    return run_dir, manifest


def _load_resumable_run(
    args: argparse.Namespace,
    settings: ScreeningSettings,
) -> tuple[ScreeningSettings, Path, dict[str, object], list[Path], str]:
    if args.seed is not None or args.input is not None or args.output is not None:
        raise ValueError("--resume cannot be combined with --seed, --input, or --output")
    run_dir = args.resume.resolve()
    manifest = load_json_object(run_dir / "manifest.json")
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("--resume requires a schema-v2 screening run")
    selected_files = manifest.get("selected_files")
    if not isinstance(selected_files, list) or not all(
        isinstance(item, str) for item in selected_files
    ):
        raise ValueError("resume manifest has invalid selected_files")
    input_path = Path(str(manifest.get("input_path", "")))
    standard_path = Path(str(manifest.get("standard_path", "")))
    manifest_model = str(manifest.get("model", ""))
    if settings.model != manifest_model:
        raise ValueError(
            f"screening model changed from {manifest_model!r} to {settings.model!r}"
        )
    resumed_settings = replace(
        settings,
        input_path=input_path,
        standard_path=standard_path,
        model=manifest_model,
    )
    resumed_settings.validate()
    standard = standard_path.read_text(encoding="utf-8")
    if text_sha256(standard) != manifest.get("standard_sha256"):
        raise ValueError("screening standard changed since this run was created")
    if _screening_fingerprint(manifest_model, standard) != manifest.get(
        "screening_fingerprint"
    ):
        raise ValueError("screening prompt or configuration changed since run creation")
    selected = [input_path / name for name in selected_files]
    return resumed_settings, run_dir, manifest, selected, standard


def _run_screening(
    settings: ScreeningSettings,
    run_dir: Path,
    manifest: dict[str, object],
    selected: list[Path],
    standard: str,
    max_items: int | None,
) -> int:
    selected_files = [path.name for path in selected]
    completed, failed = scan_item_states(run_dir, selected_files)
    total = len(selected)
    attempts = 0
    interrupted = False
    write_progress(
        run_dir,
        total=total,
        completed=completed,
        failed=failed,
        status="running",
    )

    print(f"run={run_dir}")
    print(f"selection={manifest.get('selection')}")
    print(f"model={settings.model}")
    print(f"execution=serial")
    print(f"progress={len(completed)}/{total} already completed")

    try:
        with OpenAICompatibleScreeningClient(settings) as client:
            for index, path in enumerate(selected, start=1):
                if index in completed:
                    continue
                if max_items is not None and attempts >= max_items:
                    interrupted = True
                    break
                attempts += 1
                print(f"[{index}/{total}] {path.name}", flush=True)
                write_progress(
                    run_dir,
                    total=total,
                    completed=completed,
                    failed=failed,
                    status="running",
                    current_index=index,
                )
                try:
                    resume_text = path.read_text(encoding="utf-8", errors="replace")
                    result = client.screen(standard, path.name, resume_text)
                except (OSError, ScreeningError) as exc:
                    failed.add(index)
                    write_item_error(run_dir, index, path.name, str(exc))
                    write_progress(
                        run_dir,
                        total=total,
                        completed=completed,
                        failed=failed,
                        status="running",
                        current_index=index,
                    )
                    print(f"  FAIL: {exc}", file=sys.stderr, flush=True)
                    continue
                write_json_atomic(result_path(run_dir, index), result)
                completed.add(index)
                failed.discard(index)
                write_progress(
                    run_dir,
                    total=total,
                    completed=completed,
                    failed=failed,
                    status="running",
                    current_index=index,
                )
                print(
                    f"  {result['recommendation']} | worth_reviewing={result['worth_reviewing']} | total={result['scores']['total'] if result['scoring_performed'] else 'N/A'}",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\nPaused by user; completed item results were kept.", file=sys.stderr)

    if interrupted:
        status = "paused"
    elif len(completed) == total:
        status = "complete"
    else:
        status = "completed_with_errors"
    progress = write_progress(
        run_dir,
        total=total,
        completed=completed,
        failed=failed,
        status=status,
    )
    print(
        f"status={status}; completed={progress['completed']}; "
        f"failed={progress['failed']}; remaining={progress['remaining']}"
    )
    print(f"Generate a current report with: python summarize.py --run \"{run_dir}\"")
    if interrupted:
        print(f"Continue with: python screen.py --resume \"{run_dir}\"")
        return 130
    return 0 if status == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be greater than zero")

    try:
        settings = ScreeningSettings.from_env()
        if args.resume is not None:
            settings, run_dir, manifest, selected, standard = _load_resumable_run(
                args, settings
            )
        else:
            if args.input is not None:
                settings = replace(settings, input_path=args.input)
            if args.output is not None:
                settings = replace(settings, output_path=args.output)
            settings.validate()
            selected, seed, selection_mode = _new_selection(args, settings)
            standard = settings.standard_path.read_text(encoding="utf-8")
            run_dir, manifest = _create_run(
                settings, selected, seed, selection_mode, standard
            )
    except (ConfigurationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return _run_screening(
        settings,
        run_dir,
        manifest,
        selected,
        standard,
        args.max_items,
    )


if __name__ == "__main__":
    raise SystemExit(main())
