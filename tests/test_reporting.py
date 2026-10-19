from __future__ import annotations

import csv
from pathlib import Path

from resumefilter.reporting import generate_reports, load_run_snapshot
from resumefilter.runstate import scan_item_states, write_item_error
from resumefilter.screening import write_json_atomic


def _result(source_file: str, name: str, total: int) -> dict[str, object]:
    return {
        "source_file": source_file,
        "scoring_performed": True,
        "worth_reviewing": "是",
        "summary": f"{name} summary",
        "basic_info": {"name": name, "major_track": "非电气"},
        "education_check": {"hard_check_status": "通过"},
        "scores": {
            "ml_dl_optimization": 26,
            "power_grid_digitalization": 12,
            "project_breadth_learning": 8,
            "competitions_awards": 8,
            "qualities": {"total": 5},
            "development_deployment": {"total": 3},
            "total": total,
        },
        "strengths": ["strength"],
        "risks": ["risk"],
        "verification_needed": [],
        "recommendation": "待筛选",
    }


def test_partial_run_can_be_summarized_without_completion(tmp_path: Path) -> None:
    selected = ["a.md", "b.md", "c.md"]
    write_json_atomic(
        tmp_path / "manifest.json",
        {
            "schema_version": 2,
            "run_id": "test-run",
            "model": "test-model",
            "selection": "all",
            "selected_files": selected,
        },
    )
    write_json_atomic(tmp_path / "results" / "0001.json", _result("a.md", "A", 62))
    write_item_error(tmp_path, 2, "b.md", "temporary failure")

    completed, failed = scan_item_states(tmp_path, selected)
    assert completed == {1}
    assert failed == {2}

    summary_path, csv_path = generate_reports(tmp_path, top=10)
    summary = summary_path.read_text(encoding="utf-8")
    assert "总文件数：3" in summary
    assert "已完成：1" in summary
    assert "当前失败：1" in summary
    assert "待完成或待重试：2" in summary
    assert "A" in summary

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["source_file"] == "a.md"


def test_completed_result_supersedes_stale_error(tmp_path: Path) -> None:
    selected = ["a.md"]
    write_json_atomic(
        tmp_path / "manifest.json",
        {"schema_version": 2, "selected_files": selected},
    )
    write_item_error(tmp_path, 1, "a.md", "old failure")
    write_json_atomic(tmp_path / "results" / "0001.json", _result("a.md", "A", 62))

    _manifest, results, errors = load_run_snapshot(tmp_path)
    assert len(results) == 1
    assert errors == []
