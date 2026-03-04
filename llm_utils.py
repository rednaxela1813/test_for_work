# project/llm_utils.py
"""
llm_utils.py

Enrich a pandas DataFrame of articles with:
- ai_headline (max ~12 words)
- ai_summary  (2-3 sentences)
- sentiment   (Positive / Neutral / Negative)

Key goals:
- No NLTK/TextBlob corpora deps.
- Work with Hugging Face Inference Providers router (https://router.huggingface.co)
- Prefer chat_completion (works well with provider=cerebras in your logs)
- Robust JSON parsing + deterministic fallbacks
- Rate-limit + small retries to avoid flaky runs

Env:
- HF_TOKEN: Hugging Face token (with Inference Providers permissions)
- HF_PROVIDER: e.g. "cerebras" (works for you), or "hf-inference"
- HF_GEN_MODEL: e.g. "meta-llama/Llama-3.1-8B-Instruct"
- HF_LLM_SLEEP: optional sleep between calls in seconds (default 0.0)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from huggingface_hub import InferenceClient

log = logging.getLogger("llm")

# -------------------------
# Config / Defaults
# -------------------------

DEFAULT_CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PROVIDER = "hf-inference"

_SENTIMENT_ALLOWED = {"Positive", "Neutral", "Negative"}
LLM_RECOVERABLE_ERRORS = (
    TimeoutError,
    ConnectionError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    IndexError,
    KeyError,
)


@dataclass(frozen=True)
class LLMConfig:
    token: Optional[str]
    provider: str
    model: str
    sleep_s: float
    retries: int
    timeout_s: float


def _get_config() -> LLMConfig:
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    provider = (os.getenv("HF_PROVIDER") or DEFAULT_PROVIDER).strip()
    model = (os.getenv("HF_GEN_MODEL") or DEFAULT_CHAT_MODEL).strip()
    sleep_s = float((os.getenv("HF_LLM_SLEEP") or "0").strip() or "0")
    retries = int((os.getenv("HF_LLM_RETRIES") or "2").strip() or "2")
    timeout_s = float((os.getenv("HF_LLM_TIMEOUT") or "60").strip() or "60")
    return LLMConfig(
        token=token,
        provider=provider,
        model=model,
        sleep_s=sleep_s,
        retries=retries,
        timeout_s=timeout_s,
    )


# -------------------------
# Client (cached)
# -------------------------

_CLIENT: Optional[InferenceClient] = None
_CLIENT_KEY: Optional[Tuple[str, Optional[str], float]] = None  # (provider, token, timeout)


def _client(cfg: LLMConfig) -> InferenceClient:
    global _CLIENT, _CLIENT_KEY
    key = (cfg.provider, cfg.token, cfg.timeout_s)
    if _CLIENT is None or _CLIENT_KEY != key:
        # InferenceClient accepts `provider` and `api_key`. Timeout parameter name
        # varies by huggingface_hub version; `timeout` works in many versions.
        try:
            _CLIENT = InferenceClient(provider=cfg.provider, api_key=cfg.token, timeout=cfg.timeout_s)
        except TypeError:
            _CLIENT = InferenceClient(provider=cfg.provider, api_key=cfg.token)
        _CLIENT_KEY = key
    return _CLIENT


# -------------------------
# Local fallback helpers
# -------------------------

def _cheap_extractive_summary(text: str, max_chars: int = 700) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    s = " ".join(parts[:2]).strip()
    return (s or text)[:max_chars]


def _cheap_sentiment(text: str) -> str:
    low = (text or "").lower()
    pos = ["beat", "surge", "record", "win", "growth", "breakthrough", "launch", "raise", "funding", "profit"]
    neg = ["lawsuit", "outage", "fail", "crash", "ban", "risk", "fleeing", "security", "breach", "fraud", "delusion"]
    if any(w in low for w in pos):
        return "Positive"
    if any(w in low for w in neg):
        return "Negative"
    return "Neutral"


def _normalize_sentiment(val: str) -> str:
    v = (val or "").strip().lower()
    if v in {"positive", "pos"}:
        return "Positive"
    if v in {"negative", "neg"}:
        return "Negative"
    return "Neutral"


# -------------------------
# JSON parsing (robust-ish)
# -------------------------

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None

    s = s.strip()
    s = _JSON_FENCE_RE.sub("", s).strip()

    # 1) direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) find the *last* json object block (often model prepends text)
    # Greedy search can be dangerous; instead find candidates by scanning braces.
    candidates = []
    stack = []
    start_idx = None
    for i, ch in enumerate(s):
        if ch == "{":
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidates.append(s[start_idx : i + 1])
                    start_idx = None

    for block in reversed(candidates):
        block = block.strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def _validate_ai_payload(d: Dict[str, Any]) -> Optional[Dict[str, str]]:
    # Must have exactly the keys we care about (extra keys are tolerated but ignored)
    if not all(k in d for k in ("ai_headline", "ai_summary", "sentiment")):
        return None

    headline = str(d.get("ai_headline", "")).strip()
    summary = str(d.get("ai_summary", "")).strip()
    sentiment = _normalize_sentiment(str(d.get("sentiment", "")))

    if sentiment not in _SENTIMENT_ALLOWED:
        sentiment = "Neutral"

    if not headline:
        headline = "Key update"
    if not summary:
        summary = ""

    return {
        "ai_headline": headline[:120],
        "ai_summary": summary[:800],
        "sentiment": sentiment,
    }


# -------------------------
# HF call (prefer chat only)
# -------------------------

def _chat_json(prompt: str, cfg: LLMConfig) -> Dict[str, str]:
    client = _client(cfg)

    last_err: Optional[str] = None
    for attempt in range(cfg.retries + 1):

        # rate limit: ALWAYS between calls
        if cfg.sleep_s > 0:
            time.sleep(cfg.sleep_s)

        try:
            out = client.chat_completion(
                model=cfg.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise news analyst. "
                            "Return ONLY a valid JSON object. No markdown. No extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=280,
                temperature=0.2,
            )

            raw = getattr(out.choices[0].message, "content", None) if hasattr(out, "choices") else None
            if not raw:
                raw = str(out)

            data = _try_parse_json(raw)
            if data:
                payload = _validate_ai_payload(data)
                if payload:
                    return payload

            last_err = "non-JSON or invalid JSON schema"
            log.warning("LLM returned invalid JSON. attempt=%d model=%s", attempt + 1, cfg.model)

        except LLM_RECOVERABLE_ERRORS as e:
            last_err = str(e)
            log.warning("chat_completion failed. attempt=%d model=%s err=%s", attempt + 1, cfg.model, e)

        # backoff only after failure (no extra getenv)
        if attempt < cfg.retries:
            backoff = min(2.0, cfg.sleep_s * (attempt + 1))  # 0.2, 0.4, 0.6... capped
            if backoff > 0:
                time.sleep(backoff)

    raise RuntimeError(last_err or "chat_completion failed")

# -------------------------
# Public API
# -------------------------

def ai_analyze_article(title: str, text: str, topic: str) -> Dict[str, str]:
    """
    One-call analysis: headline + summary + sentiment.
    If HF fails, returns deterministic fallback.
    """
    title = (title or "").strip()
    text = (text or "").strip()
    topic = (topic or "").strip()

    if not text:
        return {"ai_headline": title[:120] or "Key update", "ai_summary": "", "sentiment": "Neutral"}

    # short texts: skip LLM
    if len(text) < 220:
        return {
            "ai_headline": title[:120] or "Key update",
            "ai_summary": text[:400],
            "sentiment": _cheap_sentiment(text),
        }

    cfg = _get_config()

    prompt = (
        "Return ONLY a valid JSON object with keys:\n"
        '  "ai_headline": string (max 12 words)\n'
        '  "ai_summary": string (2-3 sentences)\n'
        '  "sentiment": one of ["Positive","Neutral","Negative"]\n'
        "No markdown. No extra keys. No commentary.\n\n"
        f"Topic: {topic}\n"
        f"Article title: {title}\n"
        f"Article text:\n{text[:5000]}\n"
    )

    try:
        payload = _chat_json(prompt, cfg)
        # ensure headline not empty and keep title as fallback
        if not payload["ai_headline"].strip():
            payload["ai_headline"] = title[:120] or "Key update"
        if not payload["ai_summary"].strip():
            payload["ai_summary"] = _cheap_extractive_summary(text)
        return payload
    except LLM_RECOVERABLE_ERRORS as e:
        log.warning("AI call failed, fallback used. err=%s", e)
        return {
            "ai_headline": title[:120] or "Key update",
            "ai_summary": _cheap_extractive_summary(text),
            "sentiment": _cheap_sentiment(text),
        }


def enrich_with_llm(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """
    Add ai_summary, ai_headline, sentiment columns.
    Safe even if HF_TOKEN missing or provider not accessible: falls back.
    """
    df = df.copy()

    total = len(df)
    headlines: list[str] = []
    summaries: list[str] = []
    sentiments: list[str] = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i == 1 or i % 5 == 0:
            log.info("LLM progress: %d/%d", i, total)

        title = row.get("title_clean") or row.get("title") or ""
        text = row.get("content_clean") or row.get("snippet_clean") or ""

        out = ai_analyze_article(title=title, text=text, topic=topic)
        headlines.append(out["ai_headline"])
        summaries.append(out["ai_summary"])
        sentiments.append(out["sentiment"])

    df["ai_headline"] = headlines
    df["ai_summary"] = summaries
    df["sentiment"] = sentiments
    return df
