import re
import time
import sqlite3
import trafilatura
from duckduckgo_search import DDGS
from typing import List, Dict, Optional, Tuple
from pipeline.config import nlp  # Leveraging your existing shared spaCy model

DB_PATH = "evidence_cache.db"

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
    """Constructs queries depending on whether input is structured dataset or unstructured live article."""
    # Dataset Mode
    if speaker_override:
        return f"{speaker_override} {raw_text}"

    # Live URL Mode Priority 1: Cleaned Title
    if title and len(title) > 10:
        clean_title = re.sub(r'\s*[\-|\|]\s*.*$', '', title).strip()
        if len(clean_title) > 10:
            return clean_title

    # Live URL Mode Priority 2: Inferred Speaker + Lead Sentence
    speakers, lead_sentence = extract_live_entities_and_claims(raw_text)
    if speakers and speakers[0].lower() not in lead_sentence.lower():
        return f"{speakers[0]} {lead_sentence[:100]}"

    return lead_sentence[:120]

def search_claim_evidence(raw_input: str, speaker: Optional[str] = None, is_url: bool = False, max_results: int = 3) -> List[Dict]:
    """Executes caching DDG search using dynamic query formulation."""
    title = ""
    text_body = raw_input

    if is_url:
        downloaded = trafilatura.fetch_url(raw_input)
        if downloaded:
            text_body = trafilatura.extract(downloaded, include_comments=False) or raw_input
            meta = trafilatura.extract_metadata(downloaded)
            title = meta.title if meta and meta.title else ""

    query = generate_search_query(raw_text=text_body, title=title, speaker_override=speaker)
    results = []

    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS scraped_pages (url TEXT PRIMARY KEY, content TEXT)''')

            for res in search_results:
                url = res.get('href')
                domain = res.get('title')

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
                time.sleep(0.5)
    except Exception as e:
        print(f"Retrieval error for query '{query}': {e}")

    return results