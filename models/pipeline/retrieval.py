"""
Web search + full-article retrieval.

Two stages:
  1. Search DuckDuckGo on the article headline (and a couple of rephrasings)
     to find independent coverage of the same event (`ddg_search` /
     `collect_evidence_by_headline`).
  2. Fetch the actual page for each candidate and extract the article body
     (`fetch_article_text`) so augmentation.py has real evidence to reason
     over instead of just DuckDuckGo's one-line snippet. Fetches happen
     concurrently in waves and are cached on disk by URL, the same way
     search results are cached by query.

`collect_evidence_by_headline` pulls candidates from a wider pool than it
needs and fetches them in batches, stopping once `target_full_text` articles
have real full text attached (or the pool runs out) — rather than always
attempting a fixed number of URLs regardless of how many of them turn out to
be reachable. If a page can't be fetched or parsed (paywall, JS-only
content, blocked scraper, timeout, ...) `full_text` is left as None and
callers fall back to the snippet — see augmentation.py's `_doc_text`.

Note on scope: this file makes retrieval try harder against *transient*
failures (timeouts, one blocked outlet among many candidates) by casting a
wider net across more, more diverse sources. It deliberately does not try to
defeat paywalls or anti-bot/anti-scraping protections on a given site (proxy
rotation, headless-browser stealth, referrer spoofing, etc.) — those are
access controls publishers put up on purpose, and getting around them raises
real legal/ToS issues on top of the technical ones. See the module docstring
above for what this *does* do instead.
"""
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from duckduckgo_search import DDGS
import trafilatura
from trafilatura.settings import use_config

CACHE_DIR         = 'search_cache_v3'   # separate cache from models2 to avoid stale results
ARTICLE_CACHE_DIR = 'article_cache_v1'  # cache for fetched full-article text, keyed by URL

FETCH_TIMEOUT     = 10    # seconds allowed per page fetch attempt
FETCH_MAX_RETRIES = 2     # retries for a single URL on transient errors (timeout, conn reset)
FETCH_MAX_WORKERS = 5     # fetches are I/O-bound, so a small thread pool is enough
MAX_ARTICLE_CHARS = 4000  # cap on extracted article text — keeps per-document embedding/NLI
                           # cost bounded and graphs from ballooning on very long articles;
                           # the lead usually carries most of a news article's actual content

DEFAULT_TARGET_FULL_TEXT = 5   # stop once an article has this many real full-text docs
SEARCH_POOL_SIZE         = 25  # ...but never try more than this many candidate URLs total
FETCH_BATCH_SIZE         = 8   # how many URLs to fetch concurrently per wave
MAX_PER_DOMAIN           = 2   # cap candidates from any single publisher in the pool, so
                                # the pool isn't dominated by e.g. 6 syndicated reprints of
                                # the same wire story from outlets that all block scrapers

# trafilatura.fetch_url() doesn't take a `timeout=` kwarg directly -- the request
# timeout is a config setting (DOWNLOAD_TIMEOUT), applied via a config object.
_TRAFILATURA_CONFIG = use_config()
_TRAFILATURA_CONFIG.set('DEFAULT', 'DOWNLOAD_TIMEOUT', str(FETCH_TIMEOUT))


def _cache_path(query: str, num_results: int) -> str:
    # num_results is part of the key now: a query cached at num_results=10
    # must NOT be silently returned when a caller later asks for 25 — that
    # was a latent bug (old cache entries under the old key are just
    # orphaned, no migration needed).
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f'{query}::{num_results}'
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest() + '.json')


def ddg_search(query: str, num_results: int = 10) -> list[dict]:
    cache_file = _cache_path(query, num_results)
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


def _query_variants(title: str) -> list[str]:
    """
    A couple of different phrasings of the same headline, so a single
    narrow/low-yield search doesn't limit the whole candidate pool to one
    slice of the web. Different phrasings tend to surface different
    publishers for the same story.
    """
    base = clean_headline(title)
    variants = [base]
    words = base.split()
    if len(words) > 6:
        variants.append(' '.join(words[:6]))  # shorter, keyword-heavy version
    return variants


def _domain_of(link: str) -> str:
    try:
        netloc = urlparse(link).netloc.lower()
        return netloc[4:] if netloc.startswith('www.') else netloc
    except Exception:
        return ''


def _article_cache_path(url: str) -> str:
    os.makedirs(ARTICLE_CACHE_DIR, exist_ok=True)
    return os.path.join(ARTICLE_CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + '.txt')


def fetch_article_text(url: str, retries: int = FETCH_MAX_RETRIES) -> str | None:
    """
    Download a retrieved article and extract its main body text (stripping
    nav, ads, related-article widgets, etc.) via trafilatura.

    Retries a couple of times, with a short backoff, on request-level
    exceptions (timeout, connection reset) before giving up — a good chunk
    of "failed" fetches are just a slow server that responds fine on a
    second try, not a hard block. Doesn't retry when the request succeeds
    but extraction comes back empty (e.g. a paywall stub page) — that's not
    transient, trying again won't change the result.

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
    for attempt in range(retries + 1):
        try:
            downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CONFIG)
            if downloaded:
                text = trafilatura.extract(downloaded, favor_recall=True)
            break  # got a response (even if extraction was empty) — retrying won't help
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f'  Fetch failed for "{url[:60]}" after {retries + 1} attempt(s): {e}')
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
    input list. Kept for any caller that wants to enrich an existing doc
    list directly (collect_evidence_by_headline uses its own batched
    version of this internally now).
    """
    if not docs:
        return docs
    texts = fetch_articles_text([d.get('link', '') for d in docs], max_workers=max_workers)
    for d in docs:
        d['full_text'] = texts.get(d.get('link', ''))
    return docs


def _build_candidate_pool(title: str, pool_size: int) -> list[dict]:
    """
    Search across a few headline phrasings and merge the results into one
    deduped, per-domain-capped candidate pool — the set of URLs we're
    willing to try fetching full text from for this article.
    """
    seen_links = set()
    domain_counts: dict[str, int] = {}
    pool = []

    for variant in _query_variants(title):
        if len(pool) >= pool_size:
            break
        try:
            docs = ddg_search(variant, num_results=pool_size)
        except Exception as e:
            print(f'  Search failed for "{variant[:60]}": {e}')
            continue
        for d in docs:
            link = d.get('link', '')
            if not link or link in seen_links:
                continue
            domain = _domain_of(link)
            if domain_counts.get(domain, 0) >= MAX_PER_DOMAIN:
                continue
            seen_links.add(link)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            pool.append(d)
            if len(pool) >= pool_size:
                break

    return pool


def collect_evidence_by_headline(title: str,
                                  target_full_text: int = DEFAULT_TARGET_FULL_TEXT,
                                  pool_size: int = SEARCH_POOL_SIZE,
                                  batch_size: int = FETCH_BATCH_SIZE,
                                  fetch_full_text: bool = True) -> list[dict]:
    """
    Search using the article headline (and a couple of rephrasings) to find
    coverage of the same event from different publishers, then fetch full
    article text in batches — trying progressively more of the candidate
    pool — until `target_full_text` documents have real full text attached,
    or the pool (`pool_size` candidates total, capped at `MAX_PER_DOMAIN`
    per publisher) is exhausted.

    Trade-off to be aware of: articles whose top hits are mostly blocked or
    paywalled will now take longer to build (more candidates get tried
    before giving up), while articles that hit the target quickly stop
    early instead of always paying for a fixed 10 fetches. Net effect on
    total construction time depends on your corpus's typical fetch success
    rate — tune `pool_size` / `target_full_text` down if hard articles are
    dominating build time too much.

    Returns doc dicts with 'link', 'title', 'snippet', and — unless
    fetch_full_text=False — 'full_text': the scraped article body (None for
    any doc the fetch loop didn't reach, or where the fetch failed). Graph
    construction prefers 'full_text' over the snippet when available (set
    fetch_full_text=False to skip all network round-trips beyond search,
    e.g. for a quick smoke run).
    """
    pool = _build_candidate_pool(title, pool_size)

    if not fetch_full_text or not pool:
        return pool

    full_text_count = 0
    idx = 0
    while idx < len(pool) and full_text_count < target_full_text:
        batch = pool[idx: idx + batch_size]
        texts = fetch_articles_text([d.get('link', '') for d in batch])
        for d in batch:
            d['full_text'] = texts.get(d.get('link', ''))
            if d['full_text']:
                full_text_count += 1
        idx += batch_size

    # Anything past `idx` was never attempted once the target was hit — leave
    # full_text unset (falls back to the snippet downstream, same as a
    # failed fetch) rather than fetching the rest of the pool for nothing.
    for d in pool[idx:]:
        d.setdefault('full_text', None)

    return pool
