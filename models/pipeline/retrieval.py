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

from duckduckgo_search import DDGS

CACHE_DIR = 'search_cache_v3'  # separate cache from models2 to avoid stale results


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
