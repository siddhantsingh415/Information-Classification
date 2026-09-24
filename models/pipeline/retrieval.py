import sqlite3
import trafilatura
from duckduckgo_search import DDGS
from typing import List, Dict, Optional
from article_unpacker import generate_dynamic_search_query, fetch_and_clean_raw_url

DB_PATH = "evidence_cache.db"

def search_live_evidence(
    raw_input: str,
    is_url: bool = False,
    speaker_override: Optional[str] = None,
    max_results: int = 3
) -> List[Dict]:
    """
    Unified retrieval entry point:
    Works for raw URLs, arbitrary pasted text, or benchmark dataset records.
    """
    title = ""
    text_body = raw_input

    # Handle Live Web URLs
    if is_url:
        parsed = fetch_and_clean_raw_url(raw_input)
        title = parsed["title"]
        text_body = parsed["text"]

    # Query Formulation Strategy
    if speaker_override and not is_url:
        # Explicit dataset mode (e.g. AVeriTeC)
        query = f"{speaker_override} {text_body}"
    else:
        # Live article mode (Automatic Unpacker)
        query = generate_dynamic_search_query(raw_text=text_body, title=title)

    # Execute DuckDuckGo Search with caching
    return _execute_cached_search(query, max_results=max_results)

def _execute_cached_search(query: str, max_results: int = 3) -> List[Dict]:
    results = []
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS scraped_pages
                            (url TEXT PRIMARY KEY, content TEXT)''')

            for res in search_results:
                url = res.get('href')
                source = res.get('title')

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
                    results.append({
                        'url': url,
                        'source_domain': source,
                        'text': text_content[:2000]
                    })
    except Exception as e:
        print(f"Retrieval error for query '{query}': {e}")

    return results