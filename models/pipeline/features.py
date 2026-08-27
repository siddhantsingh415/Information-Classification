"""
Node feature engineering (20-dimensional hand-crafted features per node) and
conversion from a networkx argumentation graph into a PyTorch Geometric
`Data` object ready to feed the GAT.
"""
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from .config import EDGE_NEUTRAL, EDGE_ENTAILMENT, EDGE_CONTRADICTION, ROLE_BACKGROUND


def compute_node_features(G: nx.Graph) -> np.ndarray:
    feature_list = []
    for node_id, data in G.nodes(data=True):
        neighbors     = list(G.neighbors(node_id))
        ev_nbrs       = [n for n in neighbors if G.nodes[n].get('source') == 'evidence']
        in_nbrs       = [n for n in neighbors if G.nodes[n].get('source') == 'input']
        ev_weights    = [G.edges[node_id, n].get('evidence_weight', 0.0) for n in ev_nbrs]
        all_sims      = [G.edges[node_id, n].get('similarity', 0.0) for n in neighbors]

        # NLI breakdown — all edges
        entailing     = [(n, G.edges[node_id, n]) for n in ev_nbrs
                         if G.edges[node_id, n].get('edge_type') == EDGE_ENTAILMENT]
        contradicting = [(n, G.edges[node_id, n]) for n in ev_nbrs
                         if G.edges[node_id, n].get('edge_type') == EDGE_CONTRADICTION]

        ent_wsum  = sum(e.get('evidence_weight', 0) * e.get('nli_confidence', 0.5)
                        for _, e in entailing)
        cont_wsum = sum(e.get('evidence_weight', 0) * e.get('nli_confidence', 0.5)
                        for _, e in contradicting)
        n_ev = max(len(ev_nbrs), 1)

        # Cross-source specific breakdown
        cross_nbrs = [n for n in ev_nbrs
                      if G.edges[node_id, n].get('edge_source') == 'cross_source']
        cross_ent  = [(n, G.edges[node_id, n]) for n in cross_nbrs
                      if G.edges[node_id, n].get('edge_type') == EDGE_ENTAILMENT]
        cross_cont = [(n, G.edges[node_id, n]) for n in cross_nbrs
                      if G.edges[node_id, n].get('edge_type') == EDGE_CONTRADICTION]

        cross_ent_w  = sum(e.get('evidence_weight', 0) * e.get('nli_confidence', 0.5)
                           for _, e in cross_ent)
        cross_cont_w = sum(e.get('evidence_weight', 0) * e.get('nli_confidence', 0.5)
                           for _, e in cross_cont)

        # Role one-hot
        role    = data.get('role', ROLE_BACKGROUND)
        role_oh = [1.0 if role == r else 0.0 for r in range(4)]

        feature_list.append([
            float(len(ev_nbrs)),                                       # 0
            float(len(in_nbrs)),                                       # 1
            float(np.mean(ev_weights) if ev_weights else 0.0),         # 2
            float(np.max(ev_weights)  if ev_weights else 0.0),         # 3
            float(np.mean(all_sims)   if all_sims   else 0.0),         # 4
            float(len(ev_nbrs) / max(len(neighbors), 1)),              # 5
            1.0 if data.get('source') == 'evidence' else 0.0,          # 6
            float(data.get('weight', 1.0)),                            # 7
            float(len(entailing)),                                     # 8
            float(len(contradicting)),                                 # 9
            ent_wsum,                                                  # 10
            cont_wsum,                                                 # 11
            float(len(entailing)    / n_ev),                           # 12
            float(len(contradicting) / n_ev),                          # 13
            *role_oh,                                                  # 14-17
            cross_ent_w,                                               # 18
            cross_cont_w,                                              # 19
        ])

    arr     = np.array(feature_list, dtype=np.float32)
    col_max = arr.max(axis=0)
    col_max[col_max == 0] = 1
    return arr / col_max


def graph_to_pyg(G: nx.Graph, label: int | None = None) -> Data:
    node_ids = list(G.nodes())
    id_map   = {nid: i for i, nid in enumerate(node_ids)}
    x        = torch.tensor(compute_node_features(G), dtype=torch.float)

    if G.edges():
        edges      = [(id_map[u], id_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_feats = []
        for u, v in G.edges():
            ed  = G.edges[u, v]
            et  = ed.get('edge_type', EDGE_NEUTRAL)
            esrc = 1.0 if ed.get('edge_source') == 'cross_source' else 0.0
            oh  = [1.0 if et == k else 0.0 for k in range(3)]
            edge_feats.append(oh + [
                ed.get('similarity', 0.0),
                ed.get('nli_confidence', 0.5),
                ed.get('evidence_weight', 0.0),
                esrc  # whether this is a cross-source edge
            ])
        ef_tensor = torch.tensor(edge_feats * 2, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        ef_tensor  = torch.zeros((0, 7), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=ef_tensor)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data
