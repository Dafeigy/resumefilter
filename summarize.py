from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resumefilter.reporting import generate_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a compact snapshot report for a screening run."
    )
    parser.add_argument("--run", type=Path, required=True, help="screening run directory")
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="maximum candidates to show in summary.md; defaults to 50",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run.resolve()
    try:
        summary_path, csv_path = generate_reports(run_dir, top=args.top)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"summary={summary_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
