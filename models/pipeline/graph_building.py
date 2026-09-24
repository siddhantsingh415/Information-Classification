import networkx as nx
from typing import Dict, List

def build_article_graph(claim_text: str) -> nx.Graph:
    """
    Constructs the base intra-article graph for AVeriTeC.
    Isolated claims become a single root node.
    """
    G = nx.Graph()
    G.add_node(
        0,
        text=claim_text,
        role='claim',
        is_evidence=0,
        source_domain='target'
    )
    return G

def add_evidence_nodes(base_graph: nx.Graph, retrieved_docs: List[Dict], cred_db: Dict) -> nx.Graph:
    """Appends external web documents to the graph before NLI edge generation."""
    G = base_graph.copy()
    current_node_id = max(G.nodes) + 1

    for doc in retrieved_docs:
        domain = doc.get('source_domain', 'unknown')
        weight = cred_db.get(domain, 0.5)

        G.add_node(
            current_node_id,
            text=doc['text'],
            role='evidence',
            is_evidence=1,
            source_domain=domain,
            node_weight=weight
        )
        current_node_id += 1

    return G