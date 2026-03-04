import pytest

processor = pytest.importorskip("processor")


def test_normalize_date_iso_and_fallback():
    assert processor._normalize_date("2026-03-01T10:20:30Z") == "2026-03-01"
    assert processor._normalize_date("2026-03-02 something") == "2026-03-02"
    assert processor._normalize_date("") is None


def test_get_english_stopwords_fallback_without_corpus(monkeypatch):
    processor._STOPWORDS_CACHE = None
    calls = {"count": 0}

    def _raise_lookup(_lang):
        calls["count"] += 1
        raise LookupError("missing corpus")

    monkeypatch.setattr(processor.stopwords, "words", _raise_lookup)

    first = processor._get_english_stopwords()
    second = processor._get_english_stopwords()

    assert first == set()
    assert second == set()
    assert calls["count"] == 1


def test_process_articles_dedup_and_required_columns(monkeypatch):
    monkeypatch.setattr(processor, "_get_english_stopwords", lambda: set())

    raw_articles = [
        {
            "title": "EV market surges",
            "date": "2026-03-01T10:00:00Z",
            "author": "Alice",
            "snippet": "Short snippet about electric vehicles growth",
            "url": "https://example.com/a1",
            "source": "techcrunch",
            "content": "Electric vehicles market shows strong growth in customer demand and sales momentum.",
        },
        {
            "title": "EV market surges",
            "date": "2026-03-01",
            "url": "https://example.com/a2",
            "source": "techcrunch",
            "content": "Another version with duplicate title",
        },
        {
            "title": "Bad",
            "url": "https://example.com/a3",
            "source": "techcrunch",
        },
        {
            "title": "Battery launch update",
            "url": "https://example.com/a1",
            "source": "techcrunch",
            "content": "Duplicate URL entry",
        },
    ]

    df = processor.process_articles(raw_articles)

    assert len(df) == 1
    assert df.iloc[0]["url"] == "https://example.com/a1"
    assert df.iloc[0]["date_norm"] == "2026-03-01"
    for col in ["title_clean", "snippet_clean", "content_clean", "keywords", "category"]:
        assert col in df.columns
    assert isinstance(df.iloc[0]["keywords"], str)
