import pytest

scraper = pytest.importorskip("scraper")


class _Resp:
    def __init__(self, text):
        self.text = text


class _AllowAllRobots:
    def can_fetch(self, _user_agent, _url):
        return True


def test_slugify_and_topic_url():
    assert scraper._slugify_topic("Electric Vehicles!") == "electric-vehicles"
    assert scraper._techcrunch_topic_url("ai") == "https://techcrunch.com/category/artificial-intelligence/"
    assert scraper._techcrunch_topic_url("Climate Tech") == "https://techcrunch.com/tag/climate-tech/"


def test_parse_techcrunch_search_results_dedup_and_fallback():
    html = """
    <html><body>
      <a class="post-block__title__link" href="https://techcrunch.com/2026/03/01/a1/">A1</a>
      <a class="post-block__title__link" href="https://techcrunch.com/2026/03/01/a1/">A1-dup</a>
      <a href="https://techcrunch.com/2026/03/01/a2/">A2</a>
      <a href="https://techcrunch.com/tag/ai/">Tag</a>
    </body></html>
    """
    links = scraper._parse_techcrunch_search_results(html, max_links=5)
    assert links == [
        "https://techcrunch.com/2026/03/01/a1/",
        "https://techcrunch.com/2026/03/01/a2/",
    ]


def test_parse_techcrunch_article_extracts_fields():
    html = """
    <html>
      <head><meta name="description" content="Meta description"/></head>
      <body>
        <h1>Battery breakthrough announced</h1>
        <time datetime="2026-03-01T10:00:00Z"></time>
        <a rel="author">Jane Doe</a>
        <article>
          <p>Too short.</p>
          <p>This is a long paragraph with enough detail to pass the minimum content threshold for extraction.</p>
        </article>
      </body>
    </html>
    """
    article = scraper._parse_techcrunch_article("https://techcrunch.com/2026/03/01/a1/", html)
    assert article.title == "Battery breakthrough announced"
    assert article.date == "2026-03-01T10:00:00Z"
    assert article.author == "Jane Doe"
    assert article.snippet == "Meta description"
    assert "minimum content threshold" in (article.content or "")


def test_scrape_latest_articles_keyword_filter_and_limit(monkeypatch):
    search_url = "https://techcrunch.com/tag/electric-vehicles/"
    a1 = "https://techcrunch.com/2026/03/01/a1/"
    a2 = "https://techcrunch.com/2026/03/01/a2/"
    search_html = f"""
    <a class="post-block__title__link" href="{a1}">A1</a>
    <a class="post-block__title__link" href="{a2}">A2</a>
    """
    article_htmls = {
        search_url: search_html,
        a1: """
            <h1>Electric vehicles demand rises</h1>
            <meta name="description" content="EV market update" />
            <article><p>Electric vehicles keep growing as adoption improves across major regions.</p></article>
        """,
        a2: """
            <h1>Cloud pricing changes</h1>
            <meta name="description" content="Unrelated topic" />
            <article><p>This article talks about cloud contracts and margins only.</p></article>
        """,
    }

    def fake_get(url, sleep_seconds=0.0):
        return _Resp(article_htmls[url])

    monkeypatch.setattr(scraper, "_get_robot_parser", lambda _u: _AllowAllRobots())
    monkeypatch.setattr(scraper, "_safe_get", fake_get)

    out = scraper.scrape_latest_articles(
        topic="electric vehicles",
        max_articles=1,
        source="techcrunch",
        sleep_seconds=0.0,
    )

    assert len(out) == 1
    assert out[0]["url"] == a1
