"""
Web search + full-article retrieval.

Two stages:
  1. Search DuckDuckGo on the article headline to find independent coverage of
     the same event (`ddg_search` / `collect_evidence_by_headline`).
  2. For each result, fetch the actual page and extract the article body
     (`fetch_article_text`) so augmentation.py has real evidence to reason
     over instead of just DuckDuckGo's one-line snippet. Fetches for a result
     set happen concurrently (they're I/O-bound) and are cached on disk by
     URL, the same way search results are cached by query.

If a page can't be fetched or parsed (paywall, JS-only content, blocked
scraper, timeout, ...) `full_text` is left as None and callers fall back to
the snippet — see augmentation.py's `_doc_text`.
"""
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from duckduckgo_search import DDGS
import trafilatura
from trafilatura.settings import use_config

CACHE_DIR         = 'search_cache_v3'   # separate cache from models2 to avoid stale results
ARTICLE_CACHE_DIR = 'article_cache_v1'  # cache for fetched full-article text, keyed by URL

FETCH_TIMEOUT     = 10    # seconds allowed per page fetch
FETCH_MAX_WORKERS = 5     # fetches are I/O-bound, so a small thread pool is enough
MAX_ARTICLE_CHARS = 4000  # cap on extracted article text — keeps per-document embedding/NLI
                           # cost bounded and graphs from ballooning on very long articles;
                           # the lead usually carries most of a news article's actual content

# trafilatura.fetch_url() doesn't take a `timeout=` kwarg directly -- the request
# timeout is a config setting (DOWNLOAD_TIMEOUT), applied via a config object.
_TRAFILATURA_CONFIG = use_config()
_TRAFILATURA_CONFIG.set('DEFAULT', 'DOWNLOAD_TIMEOUT', str(FETCH_TIMEOUT))


def _cache_path(query: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hashlib.md5(query.encode()).hexdigest() + '.json')


def ddg_search(query: str, num_results: int = 10) -> list[dict]:
    cache_file = _cache_path(query)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append({
                'link':    r.get('href', ''),
                'title':   r.get('title', ''),
                'snippet': r.get('body', '')
            })
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    time.sleep(0.5)
    return results


def clean_headline(title: str) -> str:
    """
    Strip publication suffixes from headlines before searching.
    e.g. 'Biden signs bill | Reuters' -> 'Biden signs bill'
    These suffixes bias results back toward the same publisher.
    """
    for sep in [' | ', ' - ', ' – ', ' — ']:
        if sep in title:
            title = title.split(sep)[0]
    return title.strip()


def _article_cache_path(url: str) -> str:
    os.makedirs(ARTICLE_CACHE_DIR, exist_ok=True)
    return os.path.join(ARTICLE_CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + '.txt')


def fetch_article_text(url: str) -> str | None:
    """
    Download a retrieved article and extract its main body text (stripping
    nav, ads, related-article widgets, etc.) via trafilatura.

    Returns None if the page can't be fetched or no article-like content can
    be extracted — callers should fall back to the search snippet in that
    case. Results (including failures) are cached on disk by URL so
    re-running the pipeline doesn't re-fetch the same pages.
    """
    if not url:
        return None

    cache_file = _article_cache_path(url)
    if os.path.exists(cache_file):
        with open(cache_file, encoding='utf-8') as f:
            cached = f.read()
        return cached if cached else None  # empty file == cached failure, don't retry

    text = None
    try:
        downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CONFIG)
        if downloaded:
            text = trafilatura.extract(downloaded, favor_recall=True)
    except Exception as e:
        print(f'  Fetch failed for "{url[:60]}": {e}')
        return None

    text = (text or '').strip()[:MAX_ARTICLE_CHARS]
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(text)
    return text if text else None


def fetch_articles_text(urls: list[str], max_workers: int = FETCH_MAX_WORKERS) -> dict:
    """Fetch full text for multiple URLs concurrently. Returns {url: text_or_None}."""
    results = {}
    urls = [u for u in dict.fromkeys(urls) if u]  # dedupe, drop empties, keep order
    if not urls:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_url = {pool.submit(fetch_article_text, u): u for u in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                print(f'  Fetch worker failed for "{url[:60]}": {e}')
                results[url] = None
    return results


def enrich_with_full_text(docs: list[dict], max_workers: int = FETCH_MAX_WORKERS) -> list[dict]:
    """
    Fetch full article text for each doc's link and attach it as
    doc['full_text'] (None if the fetch failed). Mutates and returns the
    input list.
    """
    if not docs:
        return docs
    texts = fetch_articles_text([d.get('link', '') for d in docs], max_workers=max_workers)
    for d in docs:
        d['full_text'] = texts.get(d.get('link', ''))
    return docs


def collect_evidence_by_headline(title: str, k: int = 10,
                                  fetch_full_text: bool = True) -> list[dict]:
    """
    Search using the article headline to find coverage of the same event
    from different publishers.

    Returns doc dicts with 'link', 'title', 'snippet', and — unless
    fetch_full_text=False — 'full_text': the scraped article body, which
    graph construction prefers over the short search snippet when available
    (set fetch_full_text=False to skip the extra network round-trips, e.g.
    for a quick smoke run).
    """
    query = clean_headline(title)
    if not query or len(query) < 10:
        return []
    try:
        docs = ddg_search(query, num_results=k)
    except Exception as e:
        print(f'  Search failed for "{query[:60]}": {e}')
        return []

    if fetch_full_text and docs:
        docs = enrich_with_full_text(docs)

    return docs