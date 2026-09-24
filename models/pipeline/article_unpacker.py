import re
import spacy
import trafilatura
from typing import Dict, List, Tuple
from models4.sentence_roles import classify_sentence_roles

# Load spaCy model for Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def fetch_and_clean_raw_url(url: str) -> Dict[str, str]:
    """
    Downloads a live web page and extracts clean body text and title.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not retrieve content from URL: {url}")

    extracted_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    # Extract metadata/title using trafilatura XML/meta
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else ""

    return {
        "title": title,
        "text": extracted_text or ""
    }

def extract_entities_and_claims(text: str) -> Tuple[List[str], List[str]]:
    """
    Extracts high-priority named entities (People/Organizations) to serve as implicit
    'speakers' or key actors, along with candidate claim sentences.
    """
    doc = nlp(text[:3000]) # Process first 3000 chars for efficiency

    # 1. Extract prominent speakers/organizations using NER
    speakers = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in ["PERSON", "ORG", "GPE"] and len(ent.text.strip()) > 2
    ]
    # Deduplicate while preserving order
    unique_speakers = list(dict.fromkeys(speakers))

    # 2. Extract key sentences (first 5 sentences contain primary news claims)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15][:5]

    return unique_speakers, sentences

def generate_dynamic_search_query(raw_text: str, title: str = "") -> str:
    """
    Constructs an optimal, robust search query for live web articles without
    relying on dataset metadata.
    """
    # Priority 1: Clean Article Title / Headline
    if title and len(title) > 10:
        # Strip common publisher site suffixes (e.g., " | CNN", " - The New York Times")
        clean_title = re.sub(r'\s*[\-|\|]\s*.*$', '', title).strip()
        if len(clean_title) > 10:
            return clean_title

    # Priority 2: Extract primary Named Entity + First Claim Sentence
    speakers, sentences = extract_entities_and_claims(raw_text)

    if sentences:
        lead_sentence = sentences[0]
        # Use zero-shot classifier if available, or fall back to lead sentence
        if speakers:
            top_speaker = speakers[0]
            # If lead sentence already contains speaker, use lead sentence directly
            if top_speaker.lower() in lead_sentence.lower():
                return lead_sentence[:120]
            # Otherwise prepend top speaker entity
            return f"{top_speaker} {lead_sentence[:100]}"
        return lead_sentence[:120]

    # Priority 3: Fallback - First 100 characters of text
    return raw_text[:100].replace("\n", " ").strip()