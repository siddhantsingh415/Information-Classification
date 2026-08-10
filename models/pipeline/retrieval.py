"""
Web search retrieval: search on the article headline via DuckDuckGo to find
independent coverage of the same event, with on-disk JSON caching so repeated
pipeline runs don't re-hit the network for the same query.

This module only fetches raw search results — turning them into graph
structure is augmentation.py's job.
"""
import os
import json
import hashlib
import time
import requests

from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

CACHE_DIR = 'search_cache_v3'  # separate cache from models2 to avoid stale results


def _cache_path(query: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hashlib.md5(query.encode()).hexdigest() + '.json')

def fetch_full_text(url: str, timeout: int = 5) -> str:
    """Scrape and extract the core article body text from a URL."""
    if not url:
        return ""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')

        # Strip script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()

        # Standard news sites wrap main content in <article> or specific body paragraphs
        article_elem = soup.find('article')
        target = article_elem if article_elem else soup

        paragraphs = [p.get_text().strip() for p in target.find_all('p')]
        # Filter out short crumbs/sharing buttons
        full_text = " ".join([p for p in paragraphs if len(p) > 20])
        return full_text
    except Exception:
        return "" # Fail gracefully to avoid blocking the pipeline

def ddg_search(query: str, num_results: int = 10) -> list[dict]:
    cache_file = _cache_path(query)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            link = r.get('href', '')
            # --- CRITICAL CHANGE: Fetch full text instead of relying on snippet ---
            full_body = fetch_full_text(link)

            results.append({
                'link':    link,
                'title':   r.get('title', ''),
                'snippet': r.get('body', ''), # Keep it as a fallback flag
                'full_text': full_body if full_body else r.get('body', '') # Fallback to snippet if scrape fails
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


def collect_evidence_by_headline(title: str, k: int = 10) -> list[dict]:
    """
    Search using the article headline to find coverage of the same event
    from different publishers. Returns raw search result dicts.
    """
    query = clean_headline(title)
    if not query or len(query) < 10:
        return []
    try:
        return ddg_search(query, num_results=k)
    except Exception as e:
        print(f'  Search failed for "{query[:60]}": {e}')
        return []
