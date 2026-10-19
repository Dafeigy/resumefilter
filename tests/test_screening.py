from __future__ import annotations

from pathlib import Path

import pytest

from resumefilter.config import ConfigurationError
from resumefilter.screening import (
    ScreeningSettings,
    discover_ocr_results,
    normalize_and_validate,
    sample_resumes,
)


def test_screening_model_must_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIABLE_URL", "https://example.test")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    settings = ScreeningSettings.from_env()
    with pytest.raises(ConfigurationError, match="OPENAI_MODEL"):
        settings.validate()


def test_completion_url_accepts_base_and_full_endpoint() -> None:
    assert (
        ScreeningSettings._completion_url("https://example.test/v1")
        == "https://example.test/v1/chat/completions"
    )
    assert (
        ScreeningSettings._completion_url("https://example.test/chat/completions")
        == "https://example.test/chat/completions"
    )


def test_sample_resumes_is_reproducible() -> None:
    files = [Path(f"resume-{index}.md") for index in range(10)]
    assert sample_resumes(files, 5, 123) == sample_resumes(files, 5, 123)


def test_discovery_excludes_page_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "resume.md").write_text("resume", encoding="utf-8")
    checkpoint = tmp_path / ".checkpoints" / "document"
    checkpoint.mkdir(parents=True)
    (checkpoint / "page-0001.md").write_text("page", encoding="utf-8")
    assert discover_ocr_results(tmp_path) == [tmp_path / "resume.md"]


def test_nonpassing_hard_check_clears_scores() -> None:
    result = {
        "basic_info": {},
        "education_check": {"hard_check_status": "待核验"},
        "scores": {"total": 99},
        "recommendation": "强烈推荐",
    }
    normalized = normalize_and_validate(result, "a.md")
    assert normalized["scores"]["total"] == 0
    assert normalized["recommendation"] == "待核验"
    assert normalized["worth_reviewing"] == "待核验"
    assert normalized["scoring_performed"] is False
    assert normalized["summary"] == (
        "学历死线为待核验，按标准不进入技术评分；结论为待核验，值得一看：待核验。"
    )


def test_scored_result_recalculates_totals_and_band() -> None:
    result = {
        "basic_info": {},
        "education_check": {"hard_check_status": "通过"},
        "scores": {
            "ml_dl_optimization": 18,
            "power_grid_digitalization": 12,
            "project_breadth_learning": 8,
            "competitions_awards": 8,
            "qualities": {
                "responsibility": 3,
                "pressure_resilience": 2,
                "curiosity_exploration": 2,
                "total": 99,
            },
            "development_deployment": {
                "frontend_backend": 1,
                "docker_k8s_ops": 0,
                "end_to_end_delivery": 0,
                "total": 99,
            },
            "total": 99,
        },
        "recommendation": "强烈推荐",
    }
    normalized = normalize_and_validate(result, "a.md")
    assert normalized["scores"]["total"] == 54
    assert normalized["recommendation"] == "综合淘汰"
    assert normalized["worth_reviewing"] == "否"
    assert normalized["summary"] == (
        "学历死线通过；六维合计54/100，结论为综合淘汰，值得一看：否。"
    )
