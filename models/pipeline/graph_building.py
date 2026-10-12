import networkx as nx
from typing import Dict, List, Optional

from .config import ROLE_CLAIM, ROLE_EVIDENCE
from .credibility import get_credibility


def build_article_graph(claim_text: str) -> nx.Graph:
    """
    Constructs the base intra-article graph for AVeriTeC.
    Isolated claims become a single root node.
    """
    G = nx.Graph()
    G.add_node(
        0,
        text=claim_text,
        role=ROLE_CLAIM,          # int, matches pipeline.config's role scheme
        source='input',           # what compute_node_features actually checks
        is_evidence=0,            # kept for backwards compatibility
        source_domain='target',
        weight=1.0,               # what compute_node_features actually reads
    )
    return G


def add_evidence_nodes(base_graph: nx.Graph, retrieved_docs: List[Dict],
                        cred_db: Optional[Dict] = None) -> nx.Graph:
    """
    Appends external web documents to the graph as doc-level 'evidence'
    nodes, before any NLI/sentence-level augmentation happens. No edges are
    added here -- this is intentionally the "unaugmented" graph state used
    as a baseline in the notebook's ablation experiments.

    Credibility weighting: if `cred_db` is passed (a plain dict, e.g. for a
    quick mock/offline run), it's used as `cred_db.get(domain, 0.5)`. If
    omitted (the default), the real `pipeline.credibility` SQLite database is
    used instead -- so evidence weighting reflects whatever this pipeline has
    actually learned about each domain so far.
    """
    G = base_graph.copy()
    current_node_id = max(G.nodes) + 1

    for doc in retrieved_docs:
        domain = doc.get('source_domain', 'unknown')
        weight = cred_db.get(domain, 0.5) if cred_db is not None else get_credibility(domain)

        G.add_node(
            current_node_id,
            text=doc['text'],
            role=ROLE_EVIDENCE,       # int, matches pipeline.config's role scheme
            source='evidence',        # what compute_node_features actually checks
            is_evidence=1,            # kept for backwards compatibility
            source_domain=domain,
            weight=weight,            # what compute_node_features actually reads
            node_weight=weight,       # kept for backwards compatibility
        )
        current_node_id += 1

    return G
