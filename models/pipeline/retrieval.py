import re
import time
import sqlite3
import trafilatura
try:
    from ddgs import DDGS  # duckduckgo_search was renamed to ddgs
except ImportError:
    from duckduckgo_search import DDGS  # fallback for older installs
from typing import List, Dict, Optional, Tuple
from pipeline.config import nlp  # Leveraging your existing shared spaCy model

DB_PATH = "evidence_cache.db"

_WAYBACK_PREFIX = re.compile(r'^https?://web\.archive\.org/web/\d+(?:[a-z_]*)?/')


def _normalize_url(url: str) -> str:
    """Strips a web.archive.org wrapper, if present, so an archived mirror
    and its live original compare equal for `exclude_url` filtering."""
    if not url:
        return url
    return _WAYBACK_PREFIX.sub('', url).rstrip('/')


def extract_live_entities_and_claims(text: str) -> Tuple[List[str], str]:
    """Uses the shared spaCy model to extract speakers and lead sentences from raw text."""
    doc = nlp(text[:3000])

    speakers = [
        ent.text.strip() for ent in doc.ents
        if ent.label_ in ["PERSON", "ORG", "GPE"] and len(ent.text.strip()) > 2
    ]
    unique_speakers = list(dict.fromkeys(speakers))

    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]
    lead_sentence = sentences[0] if sentences else text[:120]

    return unique_speakers, lead_sentence


def generate_search_query(raw_text: str, title: str = "", speaker_override: str = None) -> str:
    """
    Constructs queries depending on whether input is structured dataset or unstructured live article.

    The title-cleanup regex requires at least one real whitespace character
    before the separator (`\\s+`, not `\\s*`), not just an optional one --
    the previous `\\s*` version matched a bare hyphen anywhere in the title,
    truncating titles at any internal hyphen (e.g. "...for re-election -
    Publisher" became "...for re", and "COVID-19 ..." became just "COVID"),
    not only the intended " - Publisher Name" branding suffix.
    """
    # Dataset Mode
    if speaker_override:
        return f"{speaker_override} {raw_text}"

    # Live URL Mode Priority 1: Cleaned Title
    if title and len(title) > 10:
        clean_title = re.sub(r'\s+[-|]\s*.*$', '', title).strip()
        if len(clean_title) > 10:
            return clean_title

    # Live URL Mode Priority 2: Inferred Speaker + Lead Sentence
    speakers, lead_sentence = extract_live_entities_and_claims(raw_text)
    if speakers and speakers[0].lower() not in lead_sentence.lower():
        return f"{speakers[0]} {lead_sentence[:100]}"

    return lead_sentence[:120]


def fetch_source_article(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches and extracts the article at `url` -- a claim's own source/context
    article (e.g. AVeriTeC's `cached_original_claim_url`). Returns
    (text, title); either may be None if the fetch or extraction failed.

    This is the single most failure-prone step in the whole pipeline: many of
    these links are old social-media posts, news pages that have since been
    taken down, or archive.org mirrors that themselves time out or 404.
    Callers should treat a (None, None) result as expected and fall back
    accordingly, not as something to retry or raise on.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, None
        text = trafilatura.extract(downloaded, include_comments=False)
        meta = trafilatura.extract_metadata(downloaded)
        title = meta.title if meta and meta.title else None
        return text, title
    except Exception:
        return None, None


def search_claim_evidence(raw_input: str, speaker: Optional[str] = None, is_url: bool = False,
                           source_url: Optional[str] = None, max_results: int = 3,
                           max_fetch_attempts: int = 10, verbose: bool = False) -> List[Dict]:
    """
    Gathers up to `max_results` evidence documents for a claim.

    Three modes, in priority order:
      1. `source_url` given (AVeriTeC's `original_claim_url` /
         `cached_original_claim_url`) -- fetches that article first
         (`fetch_source_article`) and uses *its own title* as the search
         headline, the models4-style flow (search by headline, not by the
         bare claim sentence) adapted to AVeriTeC, where the source article
         is a separate link rather than something already in hand. If that
         fetch fails (expected to happen often -- see `fetch_source_article`),
         falls through to claim-text-based query generation rather than
         giving up on the claim.
      2. `is_url=True` (live-URL mode, `raw_input` itself is the article):
         unchanged from before.
      3. Neither: dataset mode, query built from `speaker` + `raw_input`
         directly, same as the original version of this function.

    Whichever mode supplied a source article, its URL is excluded from the
    DDG results so it can never show up as its own "evidence" -- comparison
    strips any web.archive.org wrapper first, so an archived mirror and its
    live original are still recognized as the same article.

    Note on `source_url` mode specifically: if the fetch succeeds but no
    usable title comes back, the NER/lead-sentence fallback (Priority 2 in
    `generate_search_query`) runs against `raw_input` (the claim text), not
    the fetched article's raw body. A scraped page's extracted text can be
    anything -- a Twitter/thread dump, nav boilerplate, an unrelated
    sidebar -- trafilatura doesn't guarantee clean article prose, whereas
    the claim text is always a single, well-formed, on-topic sentence. Using
    noisy scraped text as the query-generation basis was producing visibly
    garbled queries (e.g. a chain of "Show this thread" / "New conversation"
    fragments from a Twitter page) that predictably returned zero results.

    Retries: requests `max(max_fetch_attempts, max_results * 4)` candidate
    search results up front in a single DDGS call (rather than issuing
    multiple separate search calls, which would add more rate-limit risk),
    then works through them in order -- skipping the excluded URL and any
    candidate whose page fetch fails -- until either `max_results` evidence
    docs have been collected or `max_fetch_attempts` candidates have been
    tried, whichever comes first.
    """
    title = ""
    text_body = raw_input
    exclude_url = None

    if source_url:
        source_text, source_title = fetch_source_article(source_url)
        exclude_url = source_url
        if source_title:
            title = source_title
        # text_body is deliberately NOT set to source_text here -- see the
        # docstring note above. It stays as raw_input (the claim text), so
        # the Priority-2 NER/lead-sentence fallback in generate_search_query
        # always has clean, on-topic text to work with, even when the
        # fetched article's title comes back empty.
        if verbose and not source_text:
            print(f"[retrieval] could not fetch/parse source article: {source_url}")
        elif verbose and not source_title:
            print(f"[retrieval] fetched source article but found no title: {source_url}")
    elif is_url:
        downloaded = trafilatura.fetch_url(raw_input)
        if downloaded:
            text_body = trafilatura.extract(downloaded, include_comments=False) or raw_input
            meta = trafilatura.extract_metadata(downloaded)
            title = meta.title if meta and meta.title else ""
        exclude_url = raw_input

    query = generate_search_query(raw_text=text_body, title=title, speaker_override=speaker)
    results = []
    exclude_norm = _normalize_url(exclude_url) if exclude_url else None
    candidate_budget = max(max_fetch_attempts, max_results * 4)

    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=candidate_budget))

        if not search_results and verbose:
            print(f"[retrieval] 0 search results for query: '{query}'")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS scraped_pages (url TEXT PRIMARY KEY, content TEXT)''')

            attempts = 0
            for res in search_results:
                if len(results) >= max_results or attempts >= max_fetch_attempts:
                    break

                url = res.get('href')
                domain = res.get('title')
                if not url:
                    continue
                if exclude_norm and _normalize_url(url) == exclude_norm:
                    continue  # this is the claim's own source article -- skip it

                attempts += 1
                cursor.execute("SELECT content FROM scraped_pages WHERE url=?", (url,))
                row = cursor.fetchone()

                if row:
                    text_content = row[0]
                else:
                    downloaded = trafilatura.fetch_url(url)
                    text_content = trafilatura.extract(downloaded) if downloaded else None
                    if text_content:
                        cursor.execute("INSERT INTO scraped_pages VALUES (?, ?)", (url, text_content))
                        conn.commit()

                if text_content:
                    results.append({'url': url, 'source_domain': domain, 'text': text_content[:2000]})
                elif verbose:
                    print(f"[retrieval] fetch failed for candidate: {url}")
                time.sleep(0.5)

        if verbose:
            print(f"[retrieval] query='{query}' -> {len(results)}/{max_results} evidence docs "
                  f"from {attempts} fetch attempts ({len(search_results)} candidates returned)")
    except Exception as e:
        if verbose:
            print(f"[retrieval] error for query '{query}': {e}")

    return results
