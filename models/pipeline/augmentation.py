"""
Search-result augmentation logic.

This is where retrieved (external) articles get folded into an existing
article graph: retrieved docs are scored for consensus + source credibility,
then sentences from their full article text (see `_doc_text` — falls back to
the search snippet if the page couldn't be fetched) are added as new
'evidence' nodes connected to the original article's sentences by
cross-source similarity + NLI edges.

Edit this file to change: how retrieved documents are scored/ranked
(`score_documents`), which role pairings are allowed to form cross-source
edges, or the similarity/score thresholds used to decide what gets wired in
(`augment_with_cross_source`).
"""
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from .config import ENCODER, ROLE_CLAIM, ROLE_EVIDENCE, ROLE_ANALYSIS
from .sentence_roles import classify_sentence_roles
from .credibility import get_credibility, extract_domain
from .graph_building import extract_and_embed, nli_batch, edge_type_from_probs


def _doc_text(doc: dict) -> str:
    """
    Prefer the full scraped article body (retrieval.fetch_article_text) over
    the short DuckDuckGo snippet — richer text means more sentences to draw
    on when scoring consensus and building cross-source evidence edges. Falls
    back to the snippet if the page couldn't be fetched (paywall, blocked
    scraper, timeout, ...) or fetch_full_text=False was used upstream.
    """
    return doc.get('full_text') or doc.get('snippet') or ''


def score_documents(documents: list[dict],
                    credibility_alpha: float = 0.6,
                    eps: float = 0.3,
                    min_samples: int = 2) -> list[tuple[dict, float]]:
    """DBSCAN consensus clustering + domain credibility blending."""
    doc_texts = [_doc_text(doc) for doc in documents]
    if not doc_texts:
        return []
    embeddings  = ENCODER.encode(doc_texts, batch_size=32, show_progress_bar=False)
    sim_matrix  = cosine_similarity(embeddings)
    dist_matrix = np.clip(1.0 - sim_matrix, 0, 2).astype(np.float64)
    labels      = DBSCAN(eps=eps, min_samples=min_samples,
                         metric='precomputed').fit_predict(dist_matrix)
    unique      = [l for l in set(labels) if l != -1]
    cons_label  = max(unique, key=lambda l: (labels == l).sum()) if unique else None

    scored = []
    for i, doc in enumerate(documents):
        if cons_label is not None:
            if labels[i] == cons_label:
                members = np.where(labels == cons_label)[0]
                cs      = float(sim_matrix[i, members].mean())
            elif labels[i] == -1:
                cs = 0.1
            else:
                cs = float(sim_matrix[i].mean()) * 0.5
        else:
            cs = float((sim_matrix[i].sum() - 1.0) / max(len(documents) - 1, 1))

        domain   = extract_domain(doc.get('link', ''))
        combined = credibility_alpha * cs + (1 - credibility_alpha) * get_credibility(domain)
        scored.append((doc, float(combined)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def augment_with_cross_source(G: nx.Graph,
                               article_sentences: list[str],
                               article_roles: list[int],
                               retrieved_docs: list[dict],
                               sim_threshold: float = 0.55,
                               score_threshold: float = 0.2,
                               use_nli: bool = True) -> tuple[nx.Graph, list[tuple[dict, float]]]:
    """
    Augment the graph with sentences from retrieved articles.

    Each retrieved doc's full scraped article text is used when available
    (falling back to the search snippet otherwise — see `_doc_text`), so a
    doc typically contributes many more candidate sentences than the old
    one-line-snippet version did.

    Cross-source edges are only drawn between semantically similar sentences
    of *matching roles*:
      - input analysis <-> ext analysis  (is our interpretation corroborated?)
      - input claim    <-> ext evidence  (is our assertion backed externally?)
      - input claim    <-> ext claim     (do other sources assert the same thing?)

    Comparing background to analysis across sources produces noise.

    Returns the augmented graph and a list of (doc, score) for the DB update.
    """
    if not retrieved_docs:
        return G, []

    # Score documents by DBSCAN consensus + credibility
    scored_docs = score_documents(retrieved_docs)

    # Get input node info for cross-source comparison
    input_nodes = [(nid, data) for nid, data in G.nodes(data=True)
                   if data.get('source') == 'input']

    # Index article sentences by role for efficient lookup
    article_embs = np.array([data['embedding'] for _, data in input_nodes])

    next_id = max(G.nodes()) + 1 if G.nodes() else 0
    new_nodes, new_edges = [], []

    for doc, doc_score in scored_docs:
        doc_text = _doc_text(doc)
        if doc_score < score_threshold or not doc_text:
            continue

        ext_sents, ext_embs = extract_and_embed(doc_text)
        if not ext_sents:
            continue

        ext_roles = classify_sentence_roles(ext_sents)

        # Compute similarity between all external and all input sentences at once
        sim_matrix = cosine_similarity(ext_embs, article_embs)  # (n_ext, n_input)

        # Collect candidate cross-source pairs
        cross_pairs_text  = []
        cross_pairs_index = []  # (ext_idx, input_node_idx)

        for ext_i, (ext_sent, ext_role) in enumerate(zip(ext_sents, ext_roles)):
            for inp_j, (inp_nid, inp_data) in enumerate(input_nodes):
                if sim_matrix[ext_i, inp_j] < sim_threshold:
                    continue
                inp_role = inp_data['role']

                # Only compare matching or complementary role pairs
                valid = (
                    (ext_role == ROLE_ANALYSIS   and inp_role == ROLE_ANALYSIS)  or
                    (ext_role == ROLE_EVIDENCE   and inp_role == ROLE_CLAIM)     or
                    (ext_role == ROLE_CLAIM      and inp_role == ROLE_CLAIM)     or
                    (ext_role == ROLE_CLAIM      and inp_role == ROLE_ANALYSIS)
                )
                if valid:
                    cross_pairs_text.append((ext_sent, inp_data['text']))
                    cross_pairs_index.append((ext_i, inp_j, inp_nid,
                                              float(sim_matrix[ext_i, inp_j]),
                                              ext_role))

        # Batch NLI on all cross-source pairs for this document
        if use_nli and cross_pairs_text:
            cross_probs = nli_batch(cross_pairs_text)
        else:
            cross_probs = np.full((len(cross_pairs_text), 3), 1/3)

        # Add external sentences as nodes and draw edges
        ext_node_ids = {}
        for pi, (ext_i, inp_j, inp_nid, sim, ext_role) in enumerate(cross_pairs_index):
            # Add external node if not already added for this doc
            if ext_i not in ext_node_ids:
                nid = next_id
                next_id += 1
                ext_node_ids[ext_i] = nid
                new_nodes.append((
                    nid, ext_sents[ext_i], ext_embs[ext_i], doc_score, ext_role
                ))

            ext_nid = ext_node_ids[ext_i]
            etype, conf = edge_type_from_probs(cross_probs[pi])
            new_edges.append((
                ext_nid, inp_nid, sim, doc_score, etype, conf, 'cross_source'
            ))

    # Commit all changes to G
    for nid, text, emb, score, role in new_nodes:
        G.add_node(nid, text=text, embedding=emb, source='evidence',
                   role=role, weight=score)
    for nid, inp_nid, sim, score, etype, conf, esrc in new_edges:
        G.add_edge(nid, inp_nid, similarity=sim, evidence_weight=score,
                   edge_type=etype, nli_confidence=conf, edge_source=esrc)

    return G, scored_docs
