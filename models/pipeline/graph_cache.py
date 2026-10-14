"""
File-backed cache for built claim graphs (unaugmented graph, augmented
graph, scored_docs), keyed by a hash of the claim text plus whatever
construction parameters would change the result.

Graph construction is the most expensive and most failure-prone step in this
pipeline -- a live DuckDuckGo search plus several page fetches, embeddings,
and NLI calls per claim -- so caching it means re-running a notebook cell
(or coming back in a later session) doesn't repeat all of that work for
claims that were already successfully built.

Each entry is one pickle file under CACHE_DIR, named by its cache key.
Entries are read/written whole (no partial invalidation): if augmentation or
graph-building logic changes in a way that should invalidate old entries,
either clear CACHE_DIR or bump CACHE_VERSION below -- it's folded into every
key, so bumping it effectively starts a fresh cache without touching
anything already on disk.
"""
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Tuple

CACHE_DIR     = Path('graph_cache')
CACHE_VERSION = 'v1'


def _cache_key(claim_text: str, speaker, source_url, max_results: int) -> str:
    raw = f'{CACHE_VERSION}|{claim_text}|{speaker}|{source_url}|{max_results}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def get(claim_text: str, speaker=None, source_url=None, max_results: int = 3) -> Optional[Tuple]:
    """Returns the cached (unaugmented_graph, augmented_graph, scored_docs)
    for this claim + these parameters, or None on a cache miss."""
    key = _cache_key(claim_text, speaker, source_url, max_results)
    path = CACHE_DIR / f'{key}.pkl'
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None  # a corrupt/partial cache entry is treated as a miss, not a crash


def set(claim_text: str, result: Tuple, speaker=None, source_url=None, max_results: int = 3) -> None:
    """Caches (unaugmented_graph, augmented_graph, scored_docs) for this claim."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(claim_text, speaker, source_url, max_results)
    path = CACHE_DIR / f'{key}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(result, f)


def stats() -> dict:
    """Quick visibility into cache size, for a notebook diagnostic cell."""
    if not CACHE_DIR.exists():
        return {'entries': 0, 'size_mb': 0.0}
    files = list(CACHE_DIR.glob('*.pkl'))
    size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    return {'entries': len(files), 'size_mb': round(size_mb, 2)}
