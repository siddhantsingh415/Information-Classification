"""
Base article graph construction.

Turns a single article's raw text into a networkx graph: sentences become
nodes (embedded + role-labeled), and similar sentence pairs within the
article become edges (typed neutral / entailment / contradiction via NLI).

This only looks at the article's own text. Folding in *retrieved* articles as
extra evidence nodes/edges is augmentation.py's job — that's a separate
concern so you can change how cross-source evidence gets scored/wired in
without touching how the base graph itself is built, and vice versa.
"""
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

from .config import (
    ENCODER, NLI_MODEL,
    EDGE_NEUTRAL, EDGE_ENTAILMENT, EDGE_CONTRADICTION,
    NLI_CONTRADICTION, NLI_ENTAILMENT,
)
from .sentence_roles import classify_sentence_roles


def extract_and_embed(text: str) -> tuple[list[str], np.ndarray]:
    """Tokenize text into sentences and embed all at once."""
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 10]
    if not sentences:
        return [], np.array([])
    embeddings = ENCODER.encode(sentences, batch_size=32, show_progress_bar=False)
    return sentences, embeddings


def nli_batch(pairs: list[tuple[str, str]]) -> np.ndarray:
    """Run NLI on a list of (premise, hypothesis) pairs. Returns (n, 3) prob array."""
    if not pairs:
        return np.array([])
    logits = NLI_MODEL.predict(pairs)
    return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


def edge_type_from_probs(probs_row: np.ndarray) -> tuple[int, float]:
    label = int(np.argmax(probs_row))
    etype = EDGE_CONTRADICTION if label == NLI_CONTRADICTION else \
            EDGE_ENTAILMENT    if label == NLI_ENTAILMENT    else EDGE_NEUTRAL
    return etype, float(probs_row[label])


def build_article_graph(text: str,
                         sim_threshold: float = 0.60,
                         use_nli: bool = True) -> tuple[nx.Graph, list[str], list[int]]:
    """
    Build the base knowledge graph from the article text.
    Returns the graph, the list of sentences, and their role labels.
    """
    sentences, embeddings = extract_and_embed(text)
    G = nx.Graph()
    if not sentences:
        return G, [], []

    roles = classify_sentence_roles(sentences)

    for i, (sent, emb, role) in enumerate(zip(sentences, embeddings, roles)):
        G.add_node(i, text=sent, embedding=emb, source='input',
                   role=role, weight=1.0)

    sim_matrix = cosine_similarity(embeddings)
    candidate_pairs = [
        (i, j) for i in range(len(sentences))
        for j in range(i + 1, len(sentences))
        if sim_matrix[i, j] > sim_threshold
    ]

    if use_nli and candidate_pairs:
        pairs_text = [(sentences[i], sentences[j]) for i, j in candidate_pairs]
        probs_all  = nli_batch(pairs_text)
        for (i, j), probs in zip(candidate_pairs, probs_all):
            etype, conf = edge_type_from_probs(probs)
            G.add_edge(i, j, similarity=float(sim_matrix[i, j]),
                       edge_type=etype, nli_confidence=conf,
                       edge_source='intra')
    else:
        for i, j in candidate_pairs:
            G.add_edge(i, j, similarity=float(sim_matrix[i, j]),
                       edge_type=EDGE_NEUTRAL, nli_confidence=0.5,
                       edge_source='intra')

    return G, sentences, roles
