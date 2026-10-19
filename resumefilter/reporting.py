from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from resumefilter.runstate import load_json_object
from resumefilter.storage import atomic_write_text


CSV_FIELDS = [
    "index",
    "source_file",
    "name",
    "major_track",
    "hard_check_status",
    "ml_dl_optimization",
    "power_grid_digitalization",
    "project_breadth_learning",
    "competitions_awards",
    "qualities",
    "development_deployment",
    "total",
    "recommendation",
    "worth_reviewing",
    "summary",
    "strengths",
    "risks",
    "verification_needed",
]


def load_run_snapshot(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_json_object(run_dir / "manifest.json")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if manifest.get("schema_version") == 2:
        selected = manifest.get("selected_files", [])
        if not isinstance(selected, list):
            raise ValueError("manifest selected_files must be a list")
        for index, source_file in enumerate(selected, start=1):
            result_file = run_dir / "results" / f"{index:04d}.json"
            if result_file.is_file():
                try:
                    result = load_json_object(result_file)
                except ValueError:
                    result = None
                if result is not None and result.get("source_file") == source_file:
                    result["_index"] = index
                    results.append(result)
                    continue
            error_file = run_dir / "errors" / f"{index:04d}.json"
            if error_file.is_file():
                try:
                    error = load_json_object(error_file)
                except ValueError:
                    continue
                error["_index"] = index
                errors.append(error)
    else:
        # Compatibility with runs created before resumable run schema v2.
        manifest_results = manifest.get("results", [])
        if isinstance(manifest_results, list):
            for index, result in enumerate(manifest_results, start=1):
                if isinstance(result, dict):
                    copied = dict(result)
                    copied["_index"] = index
                    results.append(copied)
        manifest_errors = manifest.get("errors", [])
        if isinstance(manifest_errors, list):
            errors = [dict(item) for item in manifest_errors if isinstance(item, dict)]
    return manifest, results, errors


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    basic = result.get("basic_info", {})
    education = result.get("education_check", {})
    scores = result.get("scores", {})
    qualities = scores.get("qualities", {})
    deployment = scores.get("development_deployment", {})
    scored = bool(result.get("scoring_performed"))
    return {
        "index": result.get("_index", ""),
        "source_file": result.get("source_file", ""),
        "name": basic.get("name", ""),
        "major_track": basic.get("major_track", ""),
        "hard_check_status": education.get("hard_check_status", ""),
        "ml_dl_optimization": scores.get("ml_dl_optimization", 0),
        "power_grid_digitalization": scores.get("power_grid_digitalization", 0),
        "project_breadth_learning": scores.get("project_breadth_learning", 0),
        "competitions_awards": scores.get("competitions_awards", 0),
        "qualities": qualities.get("total", 0),
        "development_deployment": deployment.get("total", 0),
        "total": scores.get("total", 0) if scored else "",
        "recommendation": result.get("recommendation", ""),
        "worth_reviewing": result.get("worth_reviewing", ""),
        "summary": result.get("summary", ""),
        "strengths": "；".join(map(str, result.get("strengths", []))),
        "risks": "；".join(map(str, result.get("risks", []))),
        "verification_needed": "；".join(
            map(str, result.get("verification_needed", []))
        ),
    }


def write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result_row(result) for result in results)
    temporary.replace(path)


def _score_sort_key(result: dict[str, Any]) -> tuple[int, int, int, int]:
    rank = {
        "强烈推荐": 5,
        "待筛选": 4,
        "待核验": 3,
        "综合淘汰": 2,
        "学历一票否决": 1,
    }
    scores = result.get("scores", {})
    return (
        rank.get(str(result.get("recommendation", "")), 0),
        int(scores.get("total", 0)),
        int(scores.get("power_grid_digitalization", 0)),
        int(scores.get("ml_dl_optimization", 0)),
    )


def build_compact_summary(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    top: int,
) -> str:
    total = len(manifest.get("selected_files", [])) or len(results)
    completed = len(results)
    failed = len(errors)
    remaining = max(total - completed, 0)
    recommendations = Counter(str(item.get("recommendation", "未知")) for item in results)
    worth = Counter(str(item.get("worth_reviewing", "未知")) for item in results)

    lines = [
        "# 简历筛选进度分析",
        "",
        f"- 运行目录：`{manifest.get('run_id', '')}`",
        f"- 模型：`{manifest.get('model', '')}`",
        f"- 筛选模式：`{manifest.get('selection', '')}`",
        f"- 总文件数：{total}",
        f"- 已完成：{completed}",
        f"- 当前失败：{failed}",
        f"- 待完成或待重试：{remaining}",
        "",
        "## 结论分布",
        "",
        "| 结论 | 数量 |",
        "|---|---:|",
    ]
    for label in ("强烈推荐", "待筛选", "待核验", "综合淘汰", "学历一票否决", "未知"):
        if recommendations[label]:
            lines.append(f"| {label} | {recommendations[label]} |")
    lines.extend(
        [
            "",
            "## 是否值得一看",
            "",
            "| 判断 | 数量 |",
            "|---|---:|",
        ]
    )
    for label in ("是", "待核验", "否", "未知"):
        if worth[label]:
            lines.append(f"| {label} | {worth[label]} |")

    candidates = sorted(results, key=_score_sort_key, reverse=True)[:top]
    lines.extend(
        [
            "",
            f"## 当前优先候选人（最多 {top} 人）",
            "",
            "完整明细见 `results.csv`，单份证据见 `results/*.json`。",
            "",
            "| 姓名 | 总分 | 推荐 | 值得一看 | 核心判断 |",
            "|---|---:|---|---|---|",
        ]
    )
    for result in candidates:
        row = result_row(result)
        summary = str(row["summary"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {str(row['name']).replace('|', '\\|')} | {row['total'] or '—'} | "
            f"{row['recommendation']} | {row['worth_reviewing']} | {summary} |"
        )

    if errors:
        lines.extend(["", "## 当前失败（最多显示 20 项）", ""])
        for error in errors[:20]:
            source = str(error.get("source_file", "")).replace("`", "")
            message = str(error.get("error", "")).replace("\n", " ")
            lines.append(f"- `{source}`：{message}")
        if len(errors) > 20:
            lines.append(f"- 另有 {len(errors) - 20} 项未在此处展开。")
    lines.append("")
    return "\n".join(lines)


def generate_reports(run_dir: Path, *, top: int = 50) -> tuple[Path, Path]:
    if top <= 0:
        raise ValueError("top must be positive")
    manifest, results, errors = load_run_snapshot(run_dir)
    summary_path = run_dir / "summary.md"
    csv_path = run_dir / "results.csv"
    atomic_write_text(
        summary_path,
        build_compact_summary(manifest, results, errors, top=top) + "\n",
    )
    write_results_csv(csv_path, results)
    return summary_path, csv_path
