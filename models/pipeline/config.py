"""
Shared configuration: pretrained models, constants, and shared NLP resources
used across the fake-news detection pipeline.

Import this module first (or import anything else in the package, which will
import this transitively) — it downloads NLTK data and loads the sentence
encoder / NLI cross-encoder / spaCy model that every other module depends on.
Loading these once here (instead of once per notebook cell) means every
module gets the exact same model instances.
"""
import nltk

try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    nltk.download('punkt', quiet=True)

import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder

nlp = spacy.load('en_core_web_sm')

ENCODER   = SentenceTransformer('all-MiniLM-L6-v2')
NLI_MODEL = CrossEncoder('cross-encoder/nli-deberta-v3-small')

# NLI label indices for nli-deberta-v3-small
NLI_CONTRADICTION = 0
NLI_ENTAILMENT    = 1
NLI_NEUTRAL       = 2

# Edge type constants
EDGE_NEUTRAL       = 0
EDGE_ENTAILMENT    = 1
EDGE_CONTRADICTION = 2

# Sentence role constants
ROLE_CLAIM      = 0
ROLE_EVIDENCE   = 1
ROLE_ANALYSIS   = 2
ROLE_BACKGROUND = 3
ROLE_NAMES      = ['claim', 'evidence', 'analysis', 'background']

# Node feature dimension — expanded to include role encoding
FEATURE_DIM = 20

print('[pipeline.config] Setup complete.')
