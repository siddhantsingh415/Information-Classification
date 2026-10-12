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

# Node feature dimension. Two parts, concatenated in
# pipeline/features.py::compute_node_features:
#   1. HANDCRAFTED_FEATURE_DIM (20) -- the original hand-crafted structural
#      features (neighbor counts, credibility weights, entailment/
#      contradiction tallies, role one-hot). These never encoded what a
#      node's text actually *says*.
#   2. EMBEDDING_DIM -- each node's own sentence embedding, from the same
#      ENCODER used everywhere else in the pipeline, giving the GAT direct
#      semantic signal instead of purely structural/relational features.
# Pulled from the encoder itself (get_sentence_embedding_dimension()) rather
# than hardcoded, so this doesn't silently go stale if ENCODER's underlying
# model is ever swapped for one with a different embedding size.
HANDCRAFTED_FEATURE_DIM = 20
EMBEDDING_DIM           = ENCODER.get_sentence_embedding_dimension()
FEATURE_DIM             = HANDCRAFTED_FEATURE_DIM + EMBEDDING_DIM

print('[pipeline.config] Setup complete.')
