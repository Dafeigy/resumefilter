from pathlib import Path

from dotenv import dotenv_values

import screen
from resumefilter.runstate import load_json_object
from resumefilter.screening import ScreeningSettings
from screen import build_parser


def test_dotenv_dependency_parses_expected_keys(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SILICONFLOW_API_KEY=test-key\n"
        "SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1\n",
        encoding="utf-8",
    )

    values = dotenv_values(env_file)

    assert values["SILICONFLOW_API_KEY"] == "test-key"
    assert values["SILICONFLOW_BASE_URL"] == "https://api.siliconflow.cn/v1"


def test_screen_parser_supports_all_mode() -> None:
    args = build_parser().parse_args(["--all", "--input", "ocr_results"])
    assert args.all_results is True
    assert args.count is None
    assert args.file is None


def test_screen_selection_modes_are_mutually_exclusive() -> None:
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--all", "--count", "5"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--all", "--file", "candidate.md"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--all", "--resume", "run-dir"])


def test_screen_parser_supports_resume_and_controlled_pause() -> None:
    args = build_parser().parse_args(["--resume", "run-dir", "--max-items", "25"])
    assert args.resume == Path("run-dir")
    assert args.max_items == 25


def test_controlled_pause_then_resume_skips_completed_item(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "ocr"
    input_path.mkdir()
    selected = [input_path / "a.md", input_path / "b.md"]
    for path in selected:
        path.write_text("resume", encoding="utf-8")
    standard_path = tmp_path / "STANDARD.md"
    standard_path.write_text("standard", encoding="utf-8")
    settings = ScreeningSettings(
        api_key="key",
        api_url="https://example.invalid/chat/completions",
        model="model",
        standard_path=standard_path,
        input_path=input_path,
        output_path=tmp_path / "runs",
    )
    run_dir, manifest = screen._create_run(
        settings, selected, None, "all", "standard"
    )

    calls: list[str] = []

    class FakeClient:
        def __init__(self, _settings) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def screen(self, _standard: str, name: str, _text: str):
            calls.append(name)
            return {
                "source_file": name,
                "recommendation": "综合淘汰",
                "worth_reviewing": "否",
                "scoring_performed": True,
                "scores": {"total": 10},
            }

    monkeypatch.setattr(screen, "OpenAICompatibleScreeningClient", FakeClient)

    assert screen._run_screening(
        settings, run_dir, manifest, selected, "standard", max_items=1
    ) == 130
    assert calls == ["a.md"]
    assert load_json_object(run_dir / "progress.json")["status"] == "paused"

    assert screen._run_screening(
        settings, run_dir, manifest, selected, "standard", max_items=None
    ) == 0
    assert calls == ["a.md", "b.md"]
    progress = load_json_object(run_dir / "progress.json")
    assert progress["status"] == "complete"
    assert progress["completed"] == 2
