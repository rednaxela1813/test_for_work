from pathlib import Path
from types import SimpleNamespace

import pytest

main = pytest.importorskip("main")


def test_main_orchestrates_pipeline(monkeypatch, tmp_path):
    args = SimpleNamespace(
        topic="electric vehicles",
        max_articles=3,
        source="techcrunch",
        sleep=0.0,
        no_llm=False,
    )
    paths = {
        "log_file": tmp_path / "logs" / "pipeline.log",
        "raw_json": tmp_path / "data" / "raw_news.json",
        "processed_xlsx": tmp_path / "data" / "processed_news.xlsx",
        "visuals": tmp_path / "reports" / "visuals",
        "report_html": tmp_path / "reports" / "news_intelligence_report.html",
    }

    calls = []

    def _setup_logging(log_file: Path):
        calls.append(("setup_logging", log_file))

    def _run_scrape(a, p, _log):
        calls.append(("run_scrape", a.topic, p["raw_json"].name))
        return [{"url": "https://example.com"}]

    def _run_processing(raw_articles, a, p, _log):
        calls.append(("run_processing", len(raw_articles), a.no_llm, p["processed_xlsx"].name))
        return "df-object"

    def _run_report(df, a, p, _log):
        calls.append(("run_report", df, a.source, p["report_html"].name))

    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main, "setup_paths", lambda: paths)
    monkeypatch.setattr(main, "setup_logging", _setup_logging)
    monkeypatch.setattr(main, "run_scrape", _run_scrape)
    monkeypatch.setattr(main, "run_processing", _run_processing)
    monkeypatch.setattr(main, "run_report", _run_report)

    main.main()

    assert calls[0][0] == "setup_logging"
    assert calls[1][0] == "run_scrape"
    assert calls[2][0] == "run_processing"
    assert calls[3][0] == "run_report"
