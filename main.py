#project/main.py
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scraper import scrape_latest_articles
from processor import process_articles, build_visuals
from llm_utils import enrich_with_llm
from processor import build_html_report


def setup_paths() -> dict:
    base = Path(__file__).resolve().parent
    paths = {
        "base": base,
        "data": base / "data",
        "logs": base / "logs",
        "reports": base / "reports",
        "visuals": base / "reports" / "visuals",
        "raw_json": base / "data" / "raw_news.json",
        "processed_xlsx": base / "data" / "processed_news.xlsx",
        "report_html": base / "reports" / "news_intelligence_report.html",
        "log_file": base / "logs" / "pipeline.log",
    }
    for k in ["data", "logs", "reports", "visuals"]:
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

def parse_args():
    p = argparse.ArgumentParser(description="AI-Powered News Intelligence Pipeline")
    p.add_argument("--topic", default="electric vehicles", help="Keyword/topic to search")
    p.add_argument("--max-articles", type=int, default=25, help="How many articles to collect")
    p.add_argument("--source", default="techcrunch", choices=["techcrunch"], help="News source")
    p.add_argument("--sleep", type=float, default=0.8, help="Delay between requests (seconds)")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM enrichment (debug)")
    return p.parse_args()


def main():
    
    args = parse_args()
    paths = setup_paths()
    setup_logging(paths["log_file"])
    log = logging.getLogger("main")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log.info("Run started. run_id=%s topic=%s max_articles=%s source=%s", run_id, args.topic, args.max_articles, args.source)

    # 1) Scrape
    raw_articles = scrape_latest_articles(
        topic=args.topic,
        max_articles=args.max_articles,
        source=args.source,
        sleep_seconds=args.sleep,
    )
    log.info("Scraped %d raw articles", len(raw_articles))

    # Save raw JSON
    with open(paths["raw_json"], "w", encoding="utf-8") as f:
        json.dump(raw_articles, f, ensure_ascii=False, indent=2)
    log.info("Saved raw JSON: %s", paths["raw_json"])

    # 2) Process
    df = process_articles(raw_articles)
    log.info("Processed dataset rows=%d cols=%d", df.shape[0], df.shape[1])

    # 3) LLM enrichment
    if not args.no_llm:
        df = enrich_with_llm(df, topic=args.topic)
        log.info("LLM enrichment completed")
    else:
        log.warning("LLM enrichment skipped by --no-llm")

    # Save processed XLSX
    df.to_excel(paths["processed_xlsx"], index=False)
    log.info("Saved processed Excel: %s", paths["processed_xlsx"])

    # 4) Visuals
    visuals_paths = build_visuals(df, output_dir=paths["visuals"])
    log.info("Saved visuals: %s", visuals_paths)

    # 5) HTML report
    build_html_report(
        df=df,
        topic=args.topic,
        source=args.source,
        raw_json_path=str(paths["raw_json"].name),
        visuals=visuals_paths,
        output_html=paths["report_html"],
    )
    log.info("Saved report: %s", paths["report_html"])

    log.info("Run finished successfully.")
    
    
if __name__ == "__main__":
    main()
    