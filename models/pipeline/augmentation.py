"""
Cross-source augmentation (v3) for the AVeriTeC single-claim graph schema.

v3 changes, made in response to observed training pathology (constant
per-batch accuracy exactly matching the majority-class ratio, F1 stuck at 0,
and "unaugmented" vs "augmented" LR ablations producing byte-identical
metrics):

1. RELEVANCE FILTERING. v2 added *every* sentence from every scored document
   as an evidence node, regardless of whether that sentence had anything to
   do with the claim -- scraped article text is full of boilerplate,
   navigation text, and off-topic paragraphs, all of which got wired
   straight into the graph as "evidence" before. Sentences are now embedded
   (via the shared ENCODER) and compared to the claim's own embedding by
   cosine similarity; only sentences above `sim_threshold` are kept. This is
   the "attach them to relevant claim nodes" half of the request.

2. STANCE-AWARE CLUSTERING. Relevant sentences are grouped by their NLI
   verdict against the claim (entailment / contradiction / neutral) *before*
   DBSCAN runs -- clustering happens separately within each verdict group,
   never across them. This is what preserves a contradiction's semantic
   context: a sentence that flatly contradicts the claim can be textually
   similar to a dozen sentences that support it (they're all about the same
   topic), but it must never be merged into the same consensus cluster as
   sentences it disagrees with. Within a verdict group, DBSCAN clusters
   near-duplicate/paraphrased sentences together -- multiple independent
   sources making the same point, in the same direction, is a stronger
   signal than any one of them alone.

3. CONSENSUS EDGES. Sentence nodes in the same cluster are now connected to
   *each other*, tagged `edge_source='evidence_consensus'` (distinct from
   `'cross_source'` claim<->evidence edges). This matters for
   `pipeline/features.py::compute_node_features`: individual evidence nodes
   now pick up non-trivial `ev_nbrs`/entailment/contradiction features from
   their consensus neighbors, not just the claim node -- previously a
   sentence node's only neighbor was the claim (source='input'), so its own
   `ev_nbrs` was always empty and it contributed almost nothing but noise
   under mean-pooling. Each member's edge weight is also scaled by cluster
   size (capped at `MAX_CLUSTER_BOOST`), so several independent sources
   agreeing (or disagreeing) counts for more than one saying it alone.

Still not attempted: role-matched sentence comparison from the original
models4 (there's only one role on the target side -- the claim itself -- so
there's nothing to role-match against).
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from .config import (
    nlp, ENCODER, NLI_MODEL,
    NLI_CONTRADICTION, NLI_ENTAILMENT, NLI_NEUTRAL,
    EDGE_NEUTRAL, EDGE_ENTAILMENT, EDGE_CONTRADICTION,
    ROLE_EVIDENCE,
)
from .credibility import get_credibility, extract_domain

_EDGE_TYPE_BY_NLI_INDEX = {
    NLI_CONTRADICTION: EDGE_CONTRADICTION,
    NLI_ENTAILMENT: EDGE_ENTAILMENT,
    NLI_NEUTRAL: EDGE_NEUTRAL,
}

MAX_CLUSTER_BOOST = 2.0  # caps how much one large cluster can dominate a graph's edge weights


def _split_sentences(text: str, min_len: int = 15, max_sentences: int = 8) -> List[str]:
    doc = nlp(text[:3000])
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= min_len]
    return sents[:max_sentences]


def score_documents(retrieved_docs: List[Dict],
                     credibility_alpha: float = 0.6,
                     eps: float = 0.3,
                     min_samples: int = 2) -> List[Tuple[Dict, float]]:
    """(unchanged from v2) DBSCAN consensus clustering across the retrieved
    documents' full texts, blended with each document's live per-domain
    credibility score from `pipeline.credibility`."""
    doc_texts = [doc.get('text', '') for doc in retrieved_docs]
    if not doc_texts:
        return []

    embeddings = ENCODER.encode(doc_texts, batch_size=32, show_progress_bar=False)

    if len(doc_texts) > 1:
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = np.clip(1.0 - sim_matrix, 0, 2).astype(np.float64)
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                         metric='precomputed').fit_predict(dist_matrix)
        unique = [l for l in set(labels) if l != -1]
        cons_label = max(unique, key=lambda l: (labels == l).sum()) if unique else None
    else:
        sim_matrix, labels, cons_label = np.array([[1.0]]), np.array([-1]), None

    scored = []
    for i, doc in enumerate(retrieved_docs):
        if cons_label is not None:
            if labels[i] == cons_label:
                members = np.where(labels == cons_label)[0]
                cs = float(sim_matrix[i, members].mean())
            elif labels[i] == -1:
                cs = 0.1
            else:
                cs = float(sim_matrix[i].mean()) * 0.5
        elif len(retrieved_docs) > 1:
            cs = float((sim_matrix[i].sum() - 1.0) / max(len(retrieved_docs) - 1, 1))
        else:
            cs = 0.5  # no consensus signal possible with a single document

        domain = doc.get('source_domain') or extract_domain(doc.get('url', ''))
        combined = credibility_alpha * cs + (1 - credibility_alpha) * get_credibility(domain)
        scored.append((doc, float(combined)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def augment_with_cross_source(G: nx.Graph,
                               claim_text: str,
                               retrieved_docs: List[Dict],
                               score_threshold: float = 0.2,
                               sim_threshold: float = 0.3,
                               max_sentences_per_doc: int = 8,
                               cluster_eps: float = 0.25,
                               cluster_min_samples: int = 2,
                               use_nli: bool = True) -> Tuple[nx.Graph, List[Tuple[Dict, float]]]:
    """
    1. Splits every document scoring above `score_threshold` into sentences.
    2. Keeps only sentences whose embedding is within `sim_threshold` cosine
       similarity of the claim's own embedding (relevance filter).
    3. Runs NLI between the claim and every surviving sentence, batched.
    4. Groups surviving sentences by NLI verdict, then runs DBSCAN *within*
       each verdict group to find clusters of mutually corroborating (or
       mutually contradicting) evidence -- clustering never crosses verdict
       groups, so a contradiction is never folded into a supporting cluster.
    5. Adds one node per surviving sentence (wired to the claim, as in v2)
       plus new edges *between* sentences in the same cluster, and scales
       each member's edge weight by cluster size (capped).

    Returns the augmented graph and the (doc, score) list from
    `score_documents`, unchanged in shape from v2 -- still usable directly
    with `credibility.bulk_update_from_prediction()`.
    """
    G = G.copy()
    if not retrieved_docs:
        return G, []

    scored_docs = score_documents(retrieved_docs)

    # Gather every candidate sentence across all docs that clear score_threshold.
    candidates = []  # list of (sentence_text, domain, doc_score)
    for doc, doc_score in scored_docs:
        if doc_score < score_threshold:
            continue
        domain = doc.get('source_domain') or extract_domain(doc.get('url', ''))
        for sentence in _split_sentences(doc.get('text', ''), max_sentences=max_sentences_per_doc):
            candidates.append((sentence, domain, doc_score))

    if not candidates:
        return G, scored_docs

    # --- Relevance filter: keep sentences semantically close to the claim ---
    sentences = [c[0] for c in candidates]
    all_texts = [claim_text] + sentences
    all_embs = ENCODER.encode(all_texts, batch_size=32, show_progress_bar=False)
    claim_emb, sent_embs = all_embs[0:1], all_embs[1:]

    sim_to_claim = cosine_similarity(claim_emb, sent_embs)[0]
    keep_idx = [i for i in range(len(candidates)) if sim_to_claim[i] >= sim_threshold]

    if not keep_idx:
        return G, scored_docs

    kept_sentences = [candidates[i][0] for i in keep_idx]
    kept_domains   = [candidates[i][1] for i in keep_idx]
    kept_scores    = [candidates[i][2] for i in keep_idx]
    kept_embs      = np.array([sent_embs[i] for i in keep_idx])

    # --- NLI verdict for each surviving sentence ---
    if use_nli:
        pairs = [(claim_text, s) for s in kept_sentences]
        nli_scores = NLI_MODEL.predict(pairs, apply_softmax=True)
        verdicts, confidences = [], []
        for row in nli_scores:
            idx = int(row.argmax())
            verdicts.append(_EDGE_TYPE_BY_NLI_INDEX[idx])
            confidences.append(float(row[idx]))
    else:
        verdicts = [EDGE_NEUTRAL] * len(kept_sentences)
        confidences = [1 / 3] * len(kept_sentences)

    # --- Stance-aware clustering: DBSCAN within each verdict group only ---
    cluster_id_by_idx = {}
    for verdict_type in set(verdicts):
        idxs = [i for i, v in enumerate(verdicts) if v == verdict_type]
        if len(idxs) < cluster_min_samples:
            for i in idxs:
                cluster_id_by_idx[i] = -1  # too few to cluster -- singleton
            continue
        group_embs = kept_embs[idxs]
        sim = cosine_similarity(group_embs)
        dist = np.clip(1.0 - sim, 0, 2).astype(np.float64)
        labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples,
                         metric='precomputed').fit_predict(dist)
        for local_i, global_i in enumerate(idxs):
            raw_label = labels[local_i]
            # Namespace by verdict so e.g. entailment-cluster-0 and
            # contradiction-cluster-0 are never treated as the same group.
            cluster_id_by_idx[global_i] = f'{verdict_type}-{raw_label}' if raw_label != -1 else -1

    cluster_members = {}
    for i, cid in cluster_id_by_idx.items():
        if cid != -1:
            cluster_members.setdefault(cid, []).append(i)

    # --- Add nodes + claim edges, scaled by cluster size ---
    next_id = max(G.nodes()) + 1 if G.nodes() else 1
    node_id_by_idx = {}

    for i, sentence in enumerate(kept_sentences):
        cid = cluster_id_by_idx[i]
        cluster_size = len(cluster_members[cid]) if cid != -1 else 1
        boost = min(1.0 + 0.15 * (cluster_size - 1), MAX_CLUSTER_BOOST)
        weight = kept_scores[i] * boost

        nid = next_id
        next_id += 1
        node_id_by_idx[i] = nid
        G.add_node(
            nid,
            text=sentence,
            role=ROLE_EVIDENCE,
            source='evidence',
            is_evidence=1,
            source_domain=kept_domains[i],
            weight=weight,
            node_weight=weight,
            cluster_id=cid,
        )
        G.add_edge(
            0, nid,
            edge_type=verdicts[i],
            nli_confidence=confidences[i],
            evidence_weight=weight,
            edge_source='cross_source',
        )

    # --- Consensus edges: connect sentences within the same cluster ---
    for cid, members in cluster_members.items():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                G.add_edge(
                    node_id_by_idx[i], node_id_by_idx[j],
                    edge_type=verdicts[i],  # same verdict for both members, by construction
                    nli_confidence=(confidences[i] + confidences[j]) / 2,
                    evidence_weight=(kept_scores[i] + kept_scores[j]) / 2,
                    edge_source='evidence_consensus',
                )

    return G, scored_docs
