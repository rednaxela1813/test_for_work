# project/processor.py
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
from nltk.corpus import stopwords

log = logging.getLogger("processor")


CATEGORY_RULES = {
    "Policy": ["regulation", "law", "government", "eu", "commission", "ban", "subsidy", "tax", "policy"],
    "Finance": ["funding", "investment", "ipo", "earnings", "revenue", "valuation", "market", "stocks"],
    "Innovation": ["ai", "model", "chip", "startup", "launch", "product", "research", "robot", "battery"],
    "Market": ["demand", "sales", "customers", "competition", "growth", "pricing", "supply"],
}


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_date(s: str):
    # TechCrunch often provides ISO datetime; we convert to date for timeline
    if not s:
        return None
    s = s.strip()
    try:
        # handle ISO datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        # best effort fallback
        return s[:10]


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = [t for t in text.split() if len(t) >= 3]
    sw = set(stopwords.words("english"))
    return [t for t in tokens if t not in sw]


def extract_keywords(title: str, content: str, top_n: int = 8) -> List[str]:
    tokens = _tokenize((title or "") + " " + (content or ""))
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_n)]


def categorize(keywords: List[str]) -> str:
    ks = set([k.lower() for k in (keywords or [])])
    best_cat = "Other"
    best_score = 0
    for cat, rules in CATEGORY_RULES.items():
        score = sum(1 for r in rules if r in ks)
        if score > best_score:
            best_cat = cat
            best_score = score
    return best_cat


def process_articles(raw_articles: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_articles)

    # Ensure columns
    for col in ["title", "date", "author", "snippet", "url", "source", "content"]:
        if col not in df.columns:
            df[col] = None

    df["title_clean"] = df["title"].map(_clean_text)
    df["snippet_clean"] = df["snippet"].map(_clean_text)
    df["content_clean"] = df["content"].map(_clean_text)
    df["date_norm"] = df["date"].map(_normalize_date)

    # Dedup by URL + title
    before = len(df)
    df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["title_clean"])
    log.info("Dedup: %d -> %d", before, len(df))

    # Filter invalid
    df = df[df["url"].notna() & (df["title_clean"].str.len() > 5)]
    df = df.reset_index(drop=True)

    # Keywords & category
    keywords_list = []
    categories = []
    for _, row in df.iterrows():
        kws = extract_keywords(row["title_clean"], row["content_clean"] or row["snippet_clean"])
        keywords_list.append(", ".join(kws))
        categories.append(categorize(kws))

    df["keywords"] = keywords_list
    df["category"] = categories

    return df


def build_visuals(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1) Sentiment distribution
    if "sentiment" in df.columns and df["sentiment"].notna().any():
        sentiment_counts = df["sentiment"].fillna("Neutral").value_counts()
    else:
        sentiment_counts = pd.Series({"Neutral": len(df)})

    fig = plt.figure()
    sentiment_counts.plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    out1 = output_dir / "sentiment_distribution.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close(fig)
    paths["sentiment_distribution"] = str(out1)

    # 2) Keyword frequency (top 10)
    all_keywords = []
    for s in df.get("keywords", pd.Series([], dtype=str)).fillna("").tolist():
        all_keywords.extend([k.strip().lower() for k in s.split(",") if k.strip()])

    freq = Counter(all_keywords).most_common(10)
    kw_df = pd.DataFrame(freq, columns=["keyword", "count"])

    fig = plt.figure()
    if not kw_df.empty:
        plt.bar(kw_df["keyword"], kw_df["count"])
        plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Keyword Frequency")
    plt.xlabel("Keyword")
    plt.ylabel("Count")
    out2 = output_dir / "keyword_frequency.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close(fig)
    paths["keyword_frequency"] = str(out2)

    # 3) Publication timeline (daily)
    if df.get("date_norm") is not None:
        timeline = df["date_norm"].dropna().value_counts().sort_index()
    else:
        timeline = pd.Series(dtype=int)

    fig = plt.figure()
    if not timeline.empty:
        plt.plot(timeline.index, timeline.values)
        plt.xticks(rotation=45, ha="right")
    plt.title("Publication Timeline")
    plt.xlabel("Date")
    plt.ylabel("Articles")
    out3 = output_dir / "publication_timeline.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=160)
    plt.close(fig)
    paths["publication_timeline"] = str(out3)

    return paths


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AI-Powered News Intelligence Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.4; }
    .meta { color: #555; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
    img { max-width: 100%; border: 1px solid #ddd; padding: 6px; background: #fff; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
    th { background: #f5f5f5; }
    .small { font-size: 12px; color: #666; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eee; }
  </style>
</head>
<body>
  <h1>AI-Powered News Intelligence Report</h1>
  <div class="meta">
    <div><b>Topic:</b> {{ topic }} | <b>Source:</b> {{ source }}</div>
    <div class="small">Generated: {{ generated_at }}</div>
    <div class="small">Raw data: {{ raw_json_path }}</div>
  </div>

  <h2>Data Overview</h2>
  <ul>
    <li><b>Articles scraped:</b> {{ article_count }}</li>
    <li><b>Date range:</b> {{ date_min }} → {{ date_max }}</li>
    <li><b>Unique sources:</b> {{ unique_sources }}</li>
  </ul>

  <h2>Sentiment Summary</h2>
  <table>
    <tr><th>Sentiment</th><th>Count</th></tr>
    {% for k,v in sentiment_counts.items() %}
      <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
    {% endfor %}
  </table>

  <h2>Visuals</h2>
  <div class="grid">
    <div>
      <h3>Sentiment Distribution</h3>
      <img src="{{ visuals.sentiment_distribution_rel }}" alt="Sentiment Distribution" />
    </div>
    <div>
      <h3>Keyword Frequency</h3>
      <img src="{{ visuals.keyword_frequency_rel }}" alt="Keyword Frequency" />
    </div>
    <div>
      <h3>Publication Timeline</h3>
      <img src="{{ visuals.publication_timeline_rel }}" alt="Publication Timeline" />
    </div>
  </div>

  <h2>Top 5 AI Summaries</h2>
  <table>
    <tr>
      <th>#</th>
      <th>AI Headline</th>
      <th>Sentiment</th>
      <th>Summary</th>
      <th>Link</th>
    </tr>
    {% for row in top5 %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ row.ai_headline }}</td>
        <td><span class="pill">{{ row.sentiment }}</span></td>
        <td>{{ row.ai_summary }}</td>
        <td><a href="{{ row.url }}" target="_blank">open</a></td>
      </tr>
    {% endfor %}
  </table>

  <h2>All Articles (snapshot)</h2>
  <table>
    <tr>
      <th>Date</th><th>Title</th><th>Author</th><th>Category</th><th>Keywords</th><th>Link</th>
    </tr>
    {% for row in all_rows %}
      <tr>
        <td>{{ row.date_norm or "" }}</td>
        <td>{{ row.title_clean }}</td>
        <td>{{ row.author or "" }}</td>
        <td>{{ row.category }}</td>
        <td class="small">{{ row.keywords }}</td>
        <td><a href="{{ row.url }}" target="_blank">open</a></td>
      </tr>
    {% endfor %}
  </table>

  <p class="small">Note: LLM calls may fallback to local heuristics if the free API is rate-limited.</p>
</body>
</html>
"""


def build_html_report(
    df: pd.DataFrame,
    topic: str,
    source: str,
    raw_json_path: str,
    visuals: Dict[str, str],
    output_html: Path,
) -> None:
    df = df.copy()
    
    # Ensure columns exist even if LLM enrichment was skipped
    if "ai_summary" not in df.columns:
        df["ai_summary"] = None
    if "ai_headline" not in df.columns:
        df["ai_headline"] = None
    if "sentiment" not in df.columns:
        df["sentiment"] = "Neutral"

    sentiment_counts = df.get("sentiment", pd.Series([], dtype=str)).fillna("Neutral").value_counts().to_dict()
    date_series = df.get("date_norm", pd.Series([], dtype=str)).dropna()
    date_min = date_series.min() if not date_series.empty else ""
    date_max = date_series.max() if not date_series.empty else ""

    # convert visuals to relative paths for HTML
    out_dir = output_html.parent
    visuals_rel = {
        "sentiment_distribution_rel": str(Path(visuals["sentiment_distribution"]).relative_to(out_dir)),
        "keyword_frequency_rel": str(Path(visuals["keyword_frequency"]).relative_to(out_dir)),
        "publication_timeline_rel": str(Path(visuals["publication_timeline"]).relative_to(out_dir)),
    }

    top5_cols = ["ai_headline", "sentiment", "ai_summary", "url"]
    top5 = df[df["ai_summary"].notna()].head(5) if df["ai_summary"].notna().any() else df.head(5)
    top5_records = top5[top5_cols].to_dict(orient="records") if not top5.empty else []

    all_rows_cols = ["date_norm", "title_clean", "author", "category", "keywords", "url"]
    all_rows = df[all_rows_cols].head(30).to_dict(orient="records")

    tpl = Template(HTML_TEMPLATE)
    html = tpl.render(
        topic=topic,
        source=source,
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        raw_json_path=raw_json_path,
        article_count=len(df),
        date_min=date_min,
        date_max=date_max,
        unique_sources=int(df.get("source", pd.Series(["unknown"])).nunique()),
        sentiment_counts=sentiment_counts,
        visuals=visuals_rel,
        top5=top5_records,
        all_rows=all_rows,
    )

    output_html.write_text(html, encoding="utf-8")