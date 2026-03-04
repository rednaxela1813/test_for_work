import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("huggingface_hub")
llm_utils = pytest.importorskip("llm_utils")


def test_try_parse_json_with_wrappers():
    raw = 'prefix text {"ai_headline":"H","ai_summary":"S","sentiment":"Positive"} suffix'
    parsed = llm_utils._try_parse_json(raw)
    assert parsed["ai_headline"] == "H"
    assert parsed["sentiment"] == "Positive"


def test_validate_ai_payload_normalizes_and_truncates():
    payload = llm_utils._validate_ai_payload(
        {
            "ai_headline": "x" * 200,
            "ai_summary": "y" * 1000,
            "sentiment": "neg",
            "extra": "ignored",
        }
    )
    assert payload is not None
    assert payload["sentiment"] == "Negative"
    assert len(payload["ai_headline"]) == 120
    assert len(payload["ai_summary"]) == 800


def test_ai_analyze_article_short_text_skips_llm(monkeypatch):
    calls = {"chat": 0}

    def _fake_chat(prompt, cfg):
        calls["chat"] += 1
        return {"ai_headline": "H", "ai_summary": "S", "sentiment": "Neutral"}

    monkeypatch.setattr(llm_utils, "_chat_json", _fake_chat)

    out = llm_utils.ai_analyze_article("Fast update", "short text", "ai")
    assert calls["chat"] == 0
    assert out["ai_headline"] == "Fast update"
    assert out["ai_summary"] == "short text"


def test_ai_analyze_article_fallback_on_chat_failure(monkeypatch):
    def _raise_chat(_prompt, _cfg):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm_utils, "_chat_json", _raise_chat)
    text = ("This report describes a major security breach and service outage. " * 10).strip()

    out = llm_utils.ai_analyze_article("Security incident", text, "security")
    assert out["ai_headline"] == "Security incident"
    assert out["sentiment"] == "Negative"
    assert out["ai_summary"]


def test_enrich_with_llm_adds_columns(monkeypatch):
    df = pd.DataFrame(
        [
            {"title_clean": "One", "content_clean": "First content"},
            {"title_clean": "Two", "content_clean": "Second content"},
        ]
    )

    monkeypatch.setattr(
        llm_utils,
        "ai_analyze_article",
        lambda title, text, topic: {
            "ai_headline": f"H-{title}",
            "ai_summary": f"S-{topic}",
            "sentiment": "Neutral",
        },
    )

    out = llm_utils.enrich_with_llm(df, topic="ev")
    assert list(out["ai_headline"]) == ["H-One", "H-Two"]
    assert list(out["sentiment"]) == ["Neutral", "Neutral"]
