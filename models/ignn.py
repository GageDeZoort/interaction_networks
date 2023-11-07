from __future__ import annotations

import torch.nn as nn
from torch.nn import ModuleList, ReLU

from models.interaction_network import InteractionNetwork


class IGNN(nn.Module):
    """GNN based on repeated interaction network (IN) convolutions
    Args:
    node_dims: node dims at input, intermediate, and output layers:
               {"in": in_dim, "hidden": hidden_dim, "out": out}
    edge_dims: edge dims at input, intermediate, and output layers:
               {"in": in_dim, "hidden": hidden_dim, "out": out}
    L: depth of the GNN, i.e. the number of IN layers
    L_internal: depth of the fully-connected NNs in each IN layer
    width: width of the node/edge networks in each IN layer
    Cw: variance scale for Gaussian init
    alpha_node: res. connection strength in each IN node network
    alpha_edge: res. connection strength in each IN edge network
    beta_node: res. connection strength between IN layers (node feats)
    beta_edge: res. connection strength between IN layers (edge feats)
    """

    def __init__(
        self,
        node_dims: int,
        edge_dims: int,
        L: int = 3,
        L_internal: int = 3,
        width: int = 128,
        alpha_node: float = 0,
        alpha_edge: float = 0,
        beta_node: float = 0,
        beta_edge: float = 0,
        Cw: float = 2.0,
    ):
        super(IGNN, self).__init__()

        self.layers = ModuleList()
        for layer in range(L + 1):
            l0 = layer == 0
            lL = layer == L
            self.layers.append(
                InteractionNetwork(
                    node_dims={
                        "in": node_dims["in"] if l0 else node_dims["hidden"],
                        "out": node_dims["out"] if lL else node_dims["hidden"],
                        "hidden": width,
                    },
                    edge_dims={
                        "in": edge_dims["in"] if l0 else edge_dims["hidden"],
                        "out": edge_dims["out"] if lL else edge_dims["hidden"],
                        "hidden": width,
                    },
                    L=L_internal,
                    Cw=Cw,
                    alpha_node=alpha_node,
                    alpha_edge=alpha_edge,
                )
            )

        self.width = width
        self.L = L
        self.L_internal = L_internal
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, deg = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.deg,
        )
        for _, weights in enumerate(self.layers):
            x, edge_attr = weights(x, edge_index, edge_attr, deg)
            x = x / x.pow(2).sum(1).sqrt().unsqueeze(1)
            edge_attr = edge_attr / edge_attr.pow(2).sum(1).sqrt().unsqueeze(1)
        return x, edge_attr
