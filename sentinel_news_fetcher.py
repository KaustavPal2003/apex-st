"""
sentinel_news_fetcher.py — Company-Level News Fetcher for APEX-ST
══════════════════════════════════════════════════════════════════
Fetches financial news headlines for each NSE symbol and writes them
in the exact format sprint3_finbert.py expects:

    {SYMBOL}_headlines.csv   (columns: date, headline, source, url)

Three tiered sources — each tried in order, results merged and deduped:

  Tier 1 — GDELT v2 DOC API  (free, no key, company-specific boolean search,
                               full historical depth up to 5 years)
  Tier 2 — NewsAPI            (free API key required via NEWSAPI_KEY env var;
                               recent 30 days only on free plan)
  Tier 3 — Google News RSS    (no key, India-geotargeted, ~7 days window,
                               reliable fallback for daily refresh)

Headlines are cached in news_cache/{SYMBOL}.jsonl so re-runs only fetch
what's new. sprint3_finbert.py reads only the 'date' and 'headline' columns;
the extra 'source' and 'url' columns are kept for auditability.

Usage
─────
  # Historical backfill — 2 years (default) for all watchlist symbols
  python sentinel_news_fetcher.py --backfill

  # Shorter backfill for testing
  python sentinel_news_fetcher.py --backfill --weeks 8

  # Daily refresh — last 7 days, fast
  python sentinel_news_fetcher.py --refresh

  # Single symbol (all sources, full backfill)
  python sentinel_news_fetcher.py --symbol RELIANCE --backfill

  # Only GDELT (no API key needed at all)
  python sentinel_news_fetcher.py --source gdelt --backfill

  # Check what we already have
  python sentinel_news_fetcher.py --check

  # Dry run — show what queries would be fired
  python sentinel_news_fetcher.py --dry-run --symbol TCS

Environment variables
─────────────────────
  NEWSAPI_KEY   — free key from https://newsapi.org/register
                  Optional: GDELT + Google RSS work without it.
                  NewsAPI only adds value for the most recent 30 days.

Requirements
────────────
  pip install requests  (standard lib xml.etree + csv + json used otherwise)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL → COMPANY MAPPING
# ─────────────────────────────────────────────────────────────────────────────
# primary_name   : used as the first quoted term in queries
# aliases        : additional terms joined with OR in GDELT/Google queries
# context_terms  : sector context to sharpen relevance (joined with OR)
# gdelt_query    : pre-built GDELT boolean query (auto-generated if not set)

SYMBOL_META: Dict[str, dict] = {
    "ADANIENT": {
        "primary": "Adani Enterprises",
        "aliases": ["Adani Enterprises", "ADANIENT", "Adani Group"],
        "context": ["NSE", "stock", "earnings", "profit", "revenue", "coal", "infrastructure"],
    },
    "ADANIPORTS": {
        "primary": "Adani Ports",
        "aliases": ["Adani Ports", "APSEZ", "Adani Ports SEZ"],
        "context": ["NSE", "port", "logistics", "shipping", "cargo", "earnings"],
    },
    "APOLLOHOSP": {
        "primary": "Apollo Hospitals",
        "aliases": ["Apollo Hospitals", "Apollo Healthcare", "Apollo Hospital"],
        "context": ["NSE", "hospital", "healthcare", "earnings", "revenue", "medical"],
    },
    "BAJAJ-AUTO": {
        "primary": "Bajaj Auto",
        "aliases": ["Bajaj Auto", "Bajaj motorcycle", "Bajaj two-wheeler"],
        "context": ["NSE", "sales", "EV", "electric", "motorcycle", "earnings", "export"],
    },
    "BAJFINANCE": {
        "primary": "Bajaj Finance",
        "aliases": ["Bajaj Finance", "Bajaj Finserv"],
        "context": ["NSE", "NBFC", "loan", "earnings", "NPA", "AUM", "interest"],
    },
    "CIPLA": {
        "primary": "Cipla",
        "aliases": ["Cipla Limited", "Cipla pharma"],
        "context": ["NSE", "drug", "FDA", "pharma", "generic", "earnings", "revenue"],
    },
    "DIVISLAB": {
        "primary": "Divi Laboratories",
        "aliases": ["Divi Laboratories", "Divis Lab", "Divi's Lab"],
        "context": ["NSE", "API", "pharma", "drug", "FDA", "earnings", "export"],
    },
    "EICHERMOT": {
        "primary": "Eicher Motors",
        "aliases": ["Eicher Motors", "Royal Enfield"],
        "context": ["NSE", "motorcycle", "sales", "EV", "electric", "earnings", "exports"],
    },
    "GRASIM": {
        "primary": "Grasim Industries",
        "aliases": ["Grasim Industries", "Grasim", "Aditya Birla Industries"],
        "context": ["NSE", "cement", "VSF", "viscose", "earnings", "revenue", "Birla"],
    },
    "HDFCBANK": {
        "primary": "HDFC Bank",
        "aliases": ["HDFC Bank", "HDFC Ltd"],
        "context": ["NSE", "bank", "NPA", "loan", "earnings", "RBI", "credit", "deposit"],
    },
    "ICICIBANK": {
        "primary": "ICICI Bank",
        "aliases": ["ICICI Bank", "ICICI"],
        "context": ["NSE", "bank", "NPA", "loan", "earnings", "RBI", "credit"],
    },
    "INDUSINDBK": {
        "primary": "IndusInd Bank",
        "aliases": ["IndusInd Bank", "IndusInd"],
        "context": ["NSE", "bank", "NPA", "loan", "earnings", "RBI", "microfinance"],
    },
    "INFY": {
        "primary": "Infosys",
        "aliases": ["Infosys Limited", "Infosys IT"],
        "context": ["NSE", "IT", "TCV", "deal", "earnings", "guidance", "revenue", "offshore"],
    },
    "JSWSTEEL": {
        "primary": "JSW Steel",
        "aliases": ["JSW Steel", "JSW Group steel"],
        "context": ["NSE", "steel", "production", "crude steel", "earnings", "capex"],
    },
    "KOTAKBANK": {
        "primary": "Kotak Mahindra Bank",
        "aliases": ["Kotak Bank", "Kotak Mahindra"],
        "context": ["NSE", "bank", "NPA", "loan", "earnings", "RBI", "asset management"],
    },
    "LT": {
        "primary": "Larsen Toubro",
        "aliases": ["Larsen Toubro", "L&T", "L and T"],
        "context": ["NSE", "engineering", "EPC", "order book", "earnings", "infra", "defence"],
    },
    "NESTLEIND": {
        "primary": "Nestle India",
        "aliases": ["Nestle India", "Nestle FMCG"],
        "context": ["NSE", "FMCG", "Maggi", "food", "earnings", "revenue", "consumer"],
    },
    "RELIANCE": {
        "primary": "Reliance Industries",
        "aliases": ["Reliance Industries", "Reliance Jio", "RIL", "Jio"],
        "context": ["NSE", "O2C", "retail", "Jio", "earnings", "refinery", "telecom"],
    },
    "SUNPHARMA": {
        "primary": "Sun Pharmaceutical",
        "aliases": ["Sun Pharma", "Sun Pharmaceutical Industries"],
        "context": ["NSE", "drug", "FDA", "pharma", "generic", "ANDA", "earnings"],
    },
    "TATASTEEL": {
        "primary": "Tata Steel",
        "aliases": ["Tata Steel", "Tata Steel UK", "Tata Steel Europe"],
        "context": ["NSE", "steel", "crude steel", "earnings", "UK", "capex", "production"],
    },
    "TCS": {
        "primary": "Tata Consultancy Services",
        "aliases": ["TCS", "Tata Consultancy"],
        "context": ["NSE", "IT", "TCV", "deal", "earnings", "guidance", "revenue", "headcount"],
    },
    "TITAN": {
        "primary": "Titan Company",
        "aliases": ["Titan Company", "Titan watches", "Tanishq", "Titan jewellery"],
        "context": ["NSE", "jewellery", "watches", "Tanishq", "earnings", "gold", "retail"],
    },
    "ULTRACEMCO": {
        "primary": "UltraTech Cement",
        "aliases": ["UltraTech Cement", "UltraTech", "Aditya Birla cement"],
        "context": ["NSE", "cement", "volume", "realisation", "earnings", "capacity", "capex"],
    },
    "ASIANPAINT": {
        "primary": "Asian Paints",
        "aliases": ["Asian Paints", "Asian Paints India"],
        "context": ["NSE", "paint", "decorative", "earnings", "volume", "raw material"],
    },
    "BHARTIARTL": {
        "primary": "Bharti Airtel",
        "aliases": ["Bharti Airtel", "Airtel", "Bharti Enterprises"],
        "context": ["NSE", "telecom", "5G", "ARPU", "subscribers", "earnings", "spectrum"],
    },
    "AXISBANK": {
        "primary": "Axis Bank",
        "aliases": ["Axis Bank"],
        "context": ["NSE", "bank", "NPA", "loan", "earnings", "RBI", "credit"],
    },
}

BANNER = "═" * 70

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _gdelt_ts(dt: datetime) -> str:
    """datetime → GDELT timestamp string YYYYMMDDHHMMSS"""
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_gdelt_date(s: str) -> Optional[str]:
    """GDELT seendate (YYYYMMDDTHHMMSSZ or YYYYMMDDHHMMSS) → YYYY-MM-DD"""
    s = s.replace("T", "").replace("Z", "").replace("-", "")
    if len(s) >= 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


def _http_get(url: str, timeout: int = 20, max_retries: int = 3) -> Optional[bytes]:
    """GET with browser UA. Returns raw bytes or None on error.
    Retries up to max_retries times on 429 AND on connection/timeout errors,
    with increasing backoff (these are often transient, not just rate limits)."""
    for attempt in range(max_retries + 1):
        req = Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/xml, */*",
        })
        try:
            with urlopen(req, timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code == 429:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                print(f"    ⚠  Rate limited (429) — retry {attempt+1}/{max_retries+1}, sleeping {wait}s")
                time.sleep(wait)
                continue  # retry the SAME request instead of giving up
            elif e.code not in (404, 400):
                print(f"    ⚠  HTTP {e.code}: {url[:80]}")
            return None
        except URLError as e:
            if attempt < max_retries:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s — connection issues get their own backoff
                print(f"    ⚠  URLError: {e.reason} — retry {attempt+1}/{max_retries+1}, sleeping {wait}s")
                time.sleep(wait)
                continue
            print(f"    ✗  URLError after {max_retries+1} attempts: {e.reason} — {url[:80]}")
            return None
        except Exception as e:
            print(f"    ⚠  Error: {e} — {url[:80]}")
            return None
    print(f"    ✗  Gave up after {max_retries+1} attempts: {url[:80]}")
    return None


def _try_requests(url: str, timeout: int = 20, headers: dict = None,
                   max_retries: int = 3) -> Optional[bytes]:
    """Use `requests` library if available, else fall back to urllib.
    Retries up to max_retries times on 429 AND on connection/read timeouts,
    with increasing backoff (these are often transient, not just rate limits)."""
    try:
        import requests as _req
    except ImportError:
        return _http_get(url, timeout, max_retries=max_retries)

    h = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    }
    if headers:
        h.update(headers)

    for attempt in range(max_retries + 1):
        try:
            r = _req.get(url, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code == 429:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                print(f"    ⚠  Rate limited (429) — retry {attempt+1}/{max_retries+1}, sleeping {wait}s")
                time.sleep(wait)
                continue  # retry the SAME request instead of giving up
            elif r.status_code not in (404, 400):
                print(f"    ⚠  HTTP {r.status_code}: {url[:80]}")
            return None
        except (_req.exceptions.ConnectTimeout, _req.exceptions.ReadTimeout,
                _req.exceptions.ConnectionError) as e:
            if attempt < max_retries:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s — connection issues get their own backoff
                print(f"    ⚠  {type(e).__name__} — retry {attempt+1}/{max_retries+1}, sleeping {wait}s")
                time.sleep(wait)
                continue
            print(f"    ✗  {type(e).__name__} after {max_retries+1} attempts: {url[:80]}")
            return None
        except Exception as e:
            print(f"    ⚠  Request error: {e} — {url[:80]}")
            return None

    print(f"    ✗  Gave up after {max_retries+1} attempts: {url[:80]}")
    return None


def _is_relevant(title: str, meta: dict) -> bool:
    """
    True if title plausibly mentions the company.
    Checks primary name and all aliases (case-insensitive).
    """
    t = title.lower()
    for alias in meta["aliases"]:
        if alias.lower() in t:
            return True
    # Also allow if primary words appear together
    words = meta["primary"].lower().split()
    if len(words) >= 2 and all(w in t for w in words):
        return True
    return False


def _clean_title(title: str) -> str:
    """Remove HTML entities, extra whitespace."""
    title = re.sub(r"&amp;", "&", title)
    title = re.sub(r"&lt;", "<", title)
    title = re.sub(r"&gt;", ">", title)
    title = re.sub(r"&quot;", '"', title)
    title = re.sub(r"&#\d+;", "", title)
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("news_cache")


def _load_cache(symbol: str) -> Tuple[List[dict], Set[str]]:
    """
    Load cached articles for a symbol.
    Returns (articles_list, seen_urls_set).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{symbol}.jsonl"
    articles = []
    seen_urls: Set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    art = json.loads(line)
                    articles.append(art)
                    if art.get("url"):
                        seen_urls.add(art["url"])
                except json.JSONDecodeError:
                    pass
    return articles, seen_urls


def _append_cache(symbol: str, new_articles: List[dict]) -> None:
    """Append new articles to the JSONL cache."""
    if not new_articles:
        return
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{symbol}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for art in new_articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: GDELT v2 DOC API
# ─────────────────────────────────────────────────────────────────────────────

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DELAY = 6.0  # seconds between requests — sustained rate ~1 req/6s avoids persistent 429s


def _build_gdelt_query(meta: dict) -> str:
    """
    Build GDELT boolean query for a company.
    Structure: ("Primary Name" OR "Alias 2") context_terms India
    """
    # Name part: OR of quoted aliases
    name_terms = " OR ".join(f'"{a}"' for a in meta["aliases"])
    name_part = f"({name_terms})" if len(meta["aliases"]) > 1 else f'"{meta["aliases"][0]}"'

    # Context: up to 5 terms joined with OR
    ctx = meta["context"][:5]
    ctx_part = " OR ".join(ctx)

    return f'{name_part} ({ctx_part}) India'


def fetch_gdelt_window(symbol: str, meta: dict,
                       start_dt: datetime, end_dt: datetime,
                       seen_urls: Set[str],
                       dry_run: bool = False) -> Optional[List[dict]]:
    """
    Fetch up to 250 articles for a symbol within [start_dt, end_dt].
    Returns list of article dicts {date, headline, source, url}.

    Return contract (important for callers implementing backoff):
      - [] (empty list)  → request succeeded, genuinely zero matching articles
      - None              → request FAILED (rate-limited/timed out after all
                             retries) — caller should treat this differently
                             from "no news" so it can back off further
    """
    query = _build_gdelt_query(meta)
    url = (
        f"{GDELT_BASE}"
        f"?query={quote_plus(query)}"
        f"&mode=artlist"
        f"&maxrecords=250"
        f"&format=json"
        f"&sourcelang=english"
        f"&sort=DateDesc"
        f"&startdatetime={_gdelt_ts(start_dt)}"
        f"&enddatetime={_gdelt_ts(end_dt)}"
    )

    if dry_run:
        print(f"    [DRY-RUN] GDELT: {url[:120]}")
        return []

    raw = _try_requests(url)
    if raw is None:
        return None  # signals a FAILED fetch, not "zero articles"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None  # malformed response — also a failure, not zero results

    articles = data.get("articles", []) or []
    results = []
    for art in articles:
        url_str = art.get("url", "")
        if url_str in seen_urls:
            continue
        title = _clean_title(art.get("title", ""))
        if len(title) < 20:
            continue
        if not _is_relevant(title, meta):
            continue
        date_str = _parse_gdelt_date(art.get("seendate", ""))
        if not date_str:
            continue
        domain = art.get("domain", "")
        rec = {
            "date":     date_str,
            "headline": title,
            "source":   f"gdelt:{domain}",
            "url":      url_str,
        }
        results.append(rec)
        seen_urls.add(url_str)

    return results


def fetch_gdelt_symbol(symbol: str, meta: dict, weeks: int,
                       seen_urls: Set[str],
                       dry_run: bool = False) -> List[dict]:
    """
    Full historical backfill via GDELT: iterate MONTHLY windows (not weekly)
    from `weeks` weeks ago up to today.  Monthly windows reduce total API
    calls by ~4x (156 weeks → ~36 months), dramatically cutting 429s while
    still covering the same date range.  The `weeks` parameter is converted
    to an equivalent number of months internally.

    GDELT's 429s/timeouts are noisy but NOT persistent — a month that fails
    outright often succeeds right after, with no extra cooldown needed
    beyond the normal per-request retries and GDELT_DELAY pacing. So this
    just tries every month regardless of prior failures (no early abort) —
    failed months are reported at the end so you know what's missing.
    """
    all_articles = []
    now = datetime.now(timezone.utc)
    end_dt = now
    failed_months = 0

    total_months = max(1, weeks // 4)   # e.g. 156 weeks → 39 months ≈ 3 years
    for m in range(total_months):
        start_dt = end_dt - timedelta(days=30)
        arts = fetch_gdelt_window(symbol, meta, start_dt, end_dt, seen_urls, dry_run)

        if arts is None:
            failed_months += 1
            print(f"    ✗  GDELT month -{m+1:03d} FAILED "
                  f"({start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}) — "
                  f"skipping, will still try later months")
        elif arts:
            all_articles.extend(arts)
            print(f"    GDELT month -{m+1:03d}: {len(arts):3d} articles  "
                  f"({start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')})")

        end_dt = start_dt
        if not dry_run:
            time.sleep(GDELT_DELAY)

    if failed_months:
        print(f"    ⚠  {failed_months}/{total_months} months failed after retries — "
              f"re-run the same command later to fill gaps (cache dedupes automatically)")

    return all_articles



def fetch_gdelt_recent(symbol: str, meta: dict,
                       days: int, seen_urls: Set[str],
                       dry_run: bool = False) -> List[dict]:
    """Lightweight refresh: last `days` days in one or two GDELT calls."""
    now = datetime.now(timezone.utc)
    end_dt = now
    start_dt = now - timedelta(days=days)
    arts = fetch_gdelt_window(symbol, meta, start_dt, end_dt, seen_urls, dry_run)
    if arts is None:
        print(f"    ✗  GDELT last {days}d: fetch failed (rate-limited/timed out)")
        return []
    print(f"    GDELT last {days}d: {len(arts)} articles")
    return arts


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: NewsAPI
# ─────────────────────────────────────────────────────────────────────────────

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
NEWSAPI_DELAY = 0.6


def fetch_newsapi_symbol(symbol: str, meta: dict, days: int,
                         api_key: str, seen_urls: Set[str],
                         dry_run: bool = False) -> List[dict]:
    """
    Fetch from NewsAPI /v2/everything.
    Free tier: last 30 days, 100 req/day. Caps automatically at 30 days.
    """
    days = min(days, 29)  # free tier hard limit
    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    # Build OR query of key aliases (NewsAPI supports OR operator)
    name_q = " OR ".join(f'"{a}"' for a in meta["aliases"][:2])
    query = f"({name_q}) India"

    url = (
        f"{NEWSAPI_BASE}"
        f"?q={quote_plus(query)}"
        f"&language=en"
        f"&pageSize=100"
        f"&from={from_date}"
        f"&sortBy=publishedAt"
        f"&apiKey={api_key}"
    )

    if dry_run:
        print(f"    [DRY-RUN] NewsAPI: {url[:100]}...")
        return []

    raw = _try_requests(url, headers={"X-Api-Key": api_key})
    if raw is None:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if data.get("status") != "ok":
        msg = data.get("message", "unknown error")
        print(f"    ⚠  NewsAPI error: {msg}")
        return []

    results = []
    for art in data.get("articles", []):
        url_str = art.get("url", "")
        if url_str in seen_urls:
            continue
        title = _clean_title(art.get("title") or "")
        if len(title) < 20:
            continue
        if not _is_relevant(title, meta):
            continue
        pub = art.get("publishedAt", "")[:10]  # YYYY-MM-DD
        if not pub:
            continue
        rec = {
            "date":     pub,
            "headline": title,
            "source":   f"newsapi:{art.get('source', {}).get('name', '')}",
            "url":      url_str,
        }
        results.append(rec)
        seen_urls.add(url_str)

    print(f"    NewsAPI last {days}d: {len(results)} articles")
    if not dry_run:
        time.sleep(NEWSAPI_DELAY)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: Google News RSS
# ─────────────────────────────────────────────────────────────────────────────

GOOGLE_NEWS_BASE = "https://news.google.com/rss/search"
GOOGLE_DELAY = 0.5


def _parse_rss_date(pub_date: str) -> Optional[str]:
    """
    Parse RSS pubDate to YYYY-MM-DD.
    Common format: 'Mon, 29 Jun 2026 10:30:00 GMT'
    """
    if not pub_date:
        return None
    # Try parsing common RSS date formats
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(pub_date.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Fallback: find 4-digit year and month/day
    m = re.search(r"(\d{1,2})\s+(\w{3})\s+(\d{4})", pub_date)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %b %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def fetch_google_rss_symbol(symbol: str, meta: dict,
                             seen_urls: Set[str],
                             dry_run: bool = False) -> List[dict]:
    """
    Fetch via Google News RSS (~7 days, no key required).
    Good for daily refresh; limited historical depth.
    """
    # Build query: "Company Name" India NSE
    primary = meta["primary"]
    query = f'"{primary}" India NSE'
    url = (
        f"{GOOGLE_NEWS_BASE}"
        f"?q={quote_plus(query)}"
        f"&hl=en-IN&gl=IN&ceid=IN:en"
    )

    if dry_run:
        print(f"    [DRY-RUN] Google RSS: {url[:120]}")
        return []

    raw = _try_requests(url)
    if raw is None:
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"    ⚠  RSS parse error: {e}")
        return []

    results = []
    for item in root.findall(".//item"):
        url_str = item.findtext("link") or item.findtext("guid") or ""
        if url_str in seen_urls:
            continue
        title = _clean_title(item.findtext("title") or "")
        if len(title) < 20:
            continue
        if not _is_relevant(title, meta):
            continue
        pub_raw = item.findtext("pubDate") or ""
        date_str = _parse_rss_date(pub_raw)
        if not date_str:
            date_str = _today_str()
        rec = {
            "date":     date_str,
            "headline": title,
            "source":   "google_rss",
            "url":      url_str,
        }
        results.append(rec)
        seen_urls.add(url_str)

    print(f"    Google RSS: {len(results)} articles")
    if not dry_run:
        time.sleep(GOOGLE_DELAY)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def write_headlines_csv(symbol: str, articles: List[dict]) -> Path:
    """
    Write {SYMBOL}_headlines.csv sorted by date ascending.
    Columns: date, headline, source, url
    The first two are what sprint3_finbert.py reads; the latter two
    are kept for traceability.
    """
    if not articles:
        return Path(f"{symbol}_headlines.csv")

    # Sort by date
    sorted_arts = sorted(articles, key=lambda x: x.get("date", ""))

    out = Path(f"{symbol}_headlines.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "headline", "source", "url"])
        writer.writeheader()
        for art in sorted_arts:
            writer.writerow({
                "date":     art.get("date", ""),
                "headline": art.get("headline", ""),
                "source":   art.get("source", ""),
                "url":      art.get("url", ""),
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def process_symbol(
    symbol: str,
    mode: str,          # 'backfill' | 'refresh'
    weeks: int,
    sources: List[str], # subset of ['gdelt', 'newsapi', 'google']
    newsapi_key: Optional[str],
    dry_run: bool = False,
) -> dict:
    """
    Fetch, merge, cache, and write headlines for one symbol.
    Returns stats dict.
    """
    meta = SYMBOL_META.get(symbol)
    if meta is None:
        print(f"  ⚠  No metadata for {symbol} — skipping")
        return {"symbol": symbol, "status": "no_metadata", "total": 0, "new": 0}

    print(f"\n{BANNER}")
    print(f"  {symbol}  —  {meta['primary']}")
    print(BANNER)

    # Load existing cache
    cached_articles, seen_urls = _load_cache(symbol)
    print(f"  Cache: {len(cached_articles)} existing articles")

    new_articles: List[dict] = []

    # ── GDELT ─────────────────────────────────────────────────────────────
    if "gdelt" in sources:
        print(f"\n  [GDELT]")
        if mode == "backfill":
            arts = fetch_gdelt_symbol(symbol, meta, weeks, seen_urls, dry_run)
        else:  # refresh
            arts = fetch_gdelt_recent(symbol, meta, days=7, seen_urls=seen_urls, dry_run=dry_run)
        new_articles.extend(arts)

    # ── NewsAPI ───────────────────────────────────────────────────────────
    if "newsapi" in sources and newsapi_key:
        print(f"\n  [NewsAPI]")
        days = min(weeks * 7, 29) if mode == "backfill" else 7
        arts = fetch_newsapi_symbol(symbol, meta, days, newsapi_key, seen_urls, dry_run)
        new_articles.extend(arts)
    elif "newsapi" in sources and not newsapi_key:
        print(f"\n  [NewsAPI] ⚠  Skipped — set NEWSAPI_KEY env var")

    # ── Google News RSS ───────────────────────────────────────────────────
    if "google" in sources:
        print(f"\n  [Google News RSS]")
        arts = fetch_google_rss_symbol(symbol, meta, seen_urls, dry_run)
        new_articles.extend(arts)

    # ── Persist + write CSV ───────────────────────────────────────────────
    if new_articles and not dry_run:
        _append_cache(symbol, new_articles)

    all_articles = cached_articles + new_articles
    if not dry_run:
        out = write_headlines_csv(symbol, all_articles)

        # Date coverage stats
        dates = sorted({a["date"] for a in all_articles if a.get("date")})
        date_range = f"{dates[0]} → {dates[-1]}" if dates else "no dates"
        print(f"\n  ✅ {symbol}: {len(all_articles)} total ({len(new_articles)} new)")
        print(f"     Date range: {date_range}")
        print(f"     CSV: {out}")
    else:
        print(f"\n  [DRY-RUN] Would write {symbol}_headlines.csv "
              f"({len(cached_articles)} cached + {len(new_articles)} new)")

    return {
        "symbol":  symbol,
        "status":  "ok",
        "total":   len(all_articles),
        "new":     len(new_articles),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECK MODE
# ─────────────────────────────────────────────────────────────────────────────

def check_coverage(symbols: List[str]) -> None:
    """Print a summary table of headline coverage per symbol."""
    print(f"\n{BANNER}")
    print(f"  SENTINEL NEWS COVERAGE CHECK")
    print(f"{BANNER}")
    print(f"  {'Symbol':<15} {'CSV':<5} {'Articles':>9} {'Earliest':<12} {'Latest':<12} {'Days w/ news':>12}")
    print(f"  {'─'*15} {'─'*5} {'─'*9} {'─'*12} {'─'*12} {'─'*12}")

    for sym in symbols:
        csv_path = Path(f"{sym}_headlines.csv")
        if not csv_path.exists():
            print(f"  {sym:<15} {'No':<5} {'—':>9}")
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  {sym:<15} {'Yes':<5} {'0':>9}")
            continue
        dates = sorted(r["date"] for r in rows if r.get("date"))
        unique_days = len(set(dates))
        earliest = dates[0] if dates else "—"
        latest   = dates[-1] if dates else "—"
        print(f"  {sym:<15} {'Yes':<5} {len(rows):>9} {earliest:<12} {latest:<12} {unique_days:>12}")

    print()
    # Also check cache
    print(f"  Cache dir: {CACHE_DIR}/")
    if CACHE_DIR.exists():
        total_cached = 0
        for sym in symbols:
            p = CACHE_DIR / f"{sym}.jsonl"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    n = sum(1 for _ in f)
                total_cached += n
        print(f"  Total cached raw articles: {total_cached}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST LOADER
# ─────────────────────────────────────────────────────────────────────────────

def get_symbols(args_symbols: Optional[List[str]]) -> List[str]:
    if args_symbols:
        return [s.upper() for s in args_symbols]
    wl = Path("watchlist.json")
    if wl.exists():
        data = json.loads(wl.read_text()).get("watchlist", [])
        if data:
            return [s.upper() for s in data]
    # Fall back to all known symbols in SYMBOL_META
    return list(SYMBOL_META.keys())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="APEX-ST company-level news fetcher (GDELT + NewsAPI + Google RSS)"
    )
    p.add_argument("--symbol", nargs="*",
                   help="NSE symbols to process (default: read watchlist.json)")
    p.add_argument("--backfill", action="store_true",
                   help="Full historical backfill mode (default if no mode given)")
    p.add_argument("--refresh", action="store_true",
                   help="Refresh mode — last 7 days only (fast)")
    p.add_argument("--weeks", type=int, default=104,
                   help="Backfill window in weeks (default: 104 = ~2 years)")
    p.add_argument("--source", choices=["gdelt", "newsapi", "google", "all"],
                   default="all",
                   help="Which news source(s) to use (default: all)")
    p.add_argument("--newsapi-key",
                   help="NewsAPI key (also read from NEWSAPI_KEY env var)")
    p.add_argument("--check", action="store_true",
                   help="Show coverage table and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Print queries without making any HTTP requests")
    return p.parse_args()


def main():
    args = parse_args()

    symbols = get_symbols(args.symbol)

    newsapi_key = args.newsapi_key or os.environ.get("NEWSAPI_KEY")

    if args.source == "all":
        sources = ["gdelt", "newsapi", "google"]
    else:
        sources = [args.source]

    mode = "refresh" if args.refresh else "backfill"
    weeks = args.weeks

    print("\n" + "█" * 70)
    print("  SENTINEL NEWS FETCHER — APEX-ST Company-Level Headlines")
    print("█" * 70)
    print(f"  Mode     : {mode}  ({'last 7 days' if mode=='refresh' else f'{weeks} weeks'})")
    print(f"  Sources  : {', '.join(sources)}")
    print(f"  Symbols  : {len(symbols)} — {', '.join(symbols[:6])}{'…' if len(symbols) > 6 else ''}")
    print(f"  NewsAPI  : {'key loaded ✓' if newsapi_key else 'no key (set NEWSAPI_KEY to enable)'}")
    print(f"  Dry run  : {'yes' if args.dry_run else 'no'}")
    print("█" * 70)

    if args.check:
        check_coverage(symbols)
        return

    results = []
    for i, sym in enumerate(symbols):
        result = process_symbol(
            symbol=sym,
            mode=mode,
            weeks=weeks,
            sources=sources,
            newsapi_key=newsapi_key,
            dry_run=args.dry_run,
        )
        results.append(result)

        # ── Cooldown between symbols (avoid rate-limiting GDELT/NewsAPI/Google) ──
        if not args.dry_run and i < len(symbols) - 1:
            print(f"\n  ⏳ Cooling down 60s before next symbol...")
            time.sleep(60)

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{BANNER}")
    print(f"  SENTINEL FETCH COMPLETE")
    print(BANNER)
    ok   = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] != "ok"]
    total_new = sum(r.get("new", 0) for r in ok)
    total_all = sum(r.get("total", 0) for r in ok)
    print(f"  ✅ {len(ok)}/{len(results)} symbols  |  "
          f"{total_new} new articles  |  {total_all} total articles")
    if fail:
        print(f"  ❌ Failed: {', '.join(r['symbol'] for r in fail)}")
    print(f"""
  Output files: {{SYMBOL}}_headlines.csv  (date, headline, source, url)
  Cache:        news_cache/{{SYMBOL}}.jsonl

  Next steps:
    python sprint3_finbert.py          # re-encode with real headlines
    python sprint3_finbert.py --dummy  # or use synthetic (no GPU needed)
    """)
    print(BANNER)


if __name__ == "__main__":
    main()