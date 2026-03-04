#project/scraper.py

import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


log = logging.getLogger("scraper")


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}


@dataclass
class Article:
    title: str
    date: Optional[str]
    author: Optional[str]
    snippet: Optional[str]
    url: str
    source: str
    content: Optional[str] = None


def _get_robot_parser(base_url: str) -> RobotFileParser:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        log.info("Loaded robots.txt: %s", robots_url)
    except Exception as e:
        # If robots can't be fetched, we act conservatively but still allow.
        log.warning("Could not read robots.txt (%s): %s", robots_url, e)
    return rp


def _safe_get(url: str, sleep_seconds: float = 0.8) -> requests.Response:
    time.sleep(max(0.0, sleep_seconds))
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
    r.raise_for_status()
    return r


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def _slugify_topic(topic: str) -> str:
    s = (topic or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s or "ai"


def _techcrunch_topic_url(topic: str) -> str:
    t = (topic or "").strip().lower()

    # Common case: for AI, the category page is more reliable.
    if t in {"ai", "artificial intelligence", "artificial-intelligence"}:
        return "https://techcrunch.com/category/artificial-intelligence/"

    # For other topics, use the corresponding tag page.
    slug = _slugify_topic(topic)
    return f"https://techcrunch.com/tag/{slug}/"


def _parse_techcrunch_search_results(html: str, max_links: int) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    # TechCrunch search cards typically include <a class="post-block__title__link" href="...">
    for a in soup.select("a.post-block__title__link"):
        href = a.get("href")
        if href and href.startswith("http"):
            links.append(href)
        if len(links) >= max_links:
            break
    # fallback: any article links
    if len(links) < max_links:
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("https://techcrunch.com/") and re.search(r"/\d{4}/\d{2}/\d{2}/", href):
                links.append(href)
            if len(links) >= max_links:
                break
    # unique, keep order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:max_links]


def _parse_techcrunch_article(url: str, html: str) -> Article:
    soup = BeautifulSoup(html, "lxml")

    title = None
    h1 = soup.find("h1")
    if h1:
        title = _clean_text(h1.get_text())

    # date
    date = None
    time_tag = soup.find("time")
    if time_tag:
        date = time_tag.get("datetime") or _clean_text(time_tag.get_text())

    # author
    author = None
    author_el = soup.select_one("a[rel='author']") or soup.select_one(".article__byline a")
    if author_el:
        author = _clean_text(author_el.get_text())

    # snippet: meta description
    snippet = None
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        snippet = _clean_text(meta_desc["content"])

    # content: aggregate paragraphs in article body
    content_parts = []
    body = soup.select_one("div.article-content") or soup.select_one("div.entry-content") or soup.select_one("article")
    if body:
        for p in body.find_all("p"):
            txt = _clean_text(p.get_text(" ", strip=True))
            # skip very short boilerplate
            if len(txt) >= 40:
                content_parts.append(txt)

    content = "\n".join(content_parts) if content_parts else None

    return Article(
        title=title or "(no title)",
        date=date,
        author=author,
        snippet=snippet,
        url=url,
        source="techcrunch",
        content=content,
    )


def scrape_latest_articles(topic: str, max_articles: int, source: str = "techcrunch", sleep_seconds: float = 0.8):
    if source != "techcrunch":
        raise ValueError("Only techcrunch is implemented in this template.")

    search_url = _techcrunch_topic_url(topic)
    rp = _get_robot_parser(search_url)

    # Respect robots for search page
    if hasattr(rp, "can_fetch") and not rp.can_fetch(DEFAULT_HEADERS["User-Agent"], search_url):
        raise RuntimeError(f"robots.txt disallows fetching: {search_url}")

    log.info("Fetching search page: %s", search_url)
    r = _safe_get(search_url, sleep_seconds=sleep_seconds)
    links = _parse_techcrunch_search_results(r.text, max_links=max_articles * 2)  # collect extra, some pages may fail
    log.info("Found %d candidate links", len(links))

    articles = []
    for i, url in enumerate(links, start=1):
        if len(articles) >= max_articles:
            break

        # Respect robots per article URL
        if hasattr(rp, "can_fetch") and not rp.can_fetch(DEFAULT_HEADERS["User-Agent"], url):
            log.warning("robots.txt disallows: %s (skipping)", url)
            continue

        try:
            log.info("Fetching article %d/%d: %s", i, max_articles, url)
            rr = _safe_get(url, sleep_seconds=sleep_seconds)
            art = _parse_techcrunch_article(url, rr.text)
            # Keyword-based filter (task requirement): keep only articles
            # where the topic appears in title/snippet/content.
            haystack = " ".join([
            (art.title or ""),
            (art.snippet or ""),
            (art.content or ""),
                ]).lower()

            if topic.strip().lower() not in haystack:
                continue
            # minimal validation
            if not art.title or art.title == "(no title)":
                continue
            articles.append(asdict(art))
        except Exception as e:
            log.warning("Failed to parse article: %s error=%s", url, e)

    # If article dates missing, it’s acceptable; but we keep it.
    return articles
