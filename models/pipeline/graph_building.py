import networkx as nx
from typing import Dict

def build_article_graph(claim_text: str) -> nx.Graph:
    """
    Constructs the base intra-article graph for AVeriTeC.
    Unlike ISOT (which parses multi-sentence articles), AVeriTeC provides isolated claims.
    This generates a single root node. Edges will be added dynamically by augmentation.py.
    """
    G = nx.Graph()

    # The graph starts with only the target claim as the root node
    G.add_node(
        0,
        text=claim_text,
        role='claim',         # Explicit role tagging for zero-shot NLI matching
        is_evidence=0,        # 0 = Internal target node
        source_domain='target'
    )

    return G

def add_evidence_nodes(base_graph: nx.Graph, retrieved_docs: list, cred_db: Dict) -> nx.Graph:
    """
    Appends retrieved external web documents as 'evidence' nodes to the base graph.
    """
    G = base_graph.copy()
    current_node_id = max(G.nodes) + 1

    for doc in retrieved_docs:
        domain = doc.get('source_domain', 'unknown')
        # Fetch Beta distribution prior for this domain, default to neutral 0.5
        weight = cred_db.get(domain, 0.5)

        G.add_node(
            current_node_id,
            text=doc['text'],
            role='evidence',
            is_evidence=1,        # 1 = External retrieved node
            source_domain=domain,
            node_weight=weight
        )
        current_node_id += 1

    return G