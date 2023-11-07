from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing

from models.resnet import ResNet


class InteractionNetwork(MessagePassing):
    """Interaction network (IN) GNN layer
    Args:
    node_dims: dimension of node network, in the form
               {"in": in_dim, "hidden": hidden_dim, "out": out_dim}
    edge_dims: dimension of the edge network, in the form
               {"in": in_dim, "hidden": hidden_dim, "out": out_dim}
    L: depth of the ResNet node/edge networks
    alpha_node: res. connection strength in the node network
    alpha_edge: res. connection strength in the edge network
    Cw: variance scale for Gaussian init
    """

    def __init__(
        self,
        node_dims: int,
        edge_dims: int,
        L: int = 2,
        alpha_node: float = 0,
        alpha_edge: float = 0,
        Cw: float = 2.0,
    ):
        super(InteractionNetwork, self).__init__(aggr="add")

        self.edge_net = ResNet(
            edge_dims["in"] + 2 * node_dims["in"],
            edge_dims["out"],
            edge_dims["hidden"],
            L=L,
            alpha=alpha_edge,
            Cw=Cw,
        )
        self.node_net = ResNet(
            node_dims["in"] + edge_dims["out"],
            node_dims["out"],
            node_dims["hidden"],
            L=L,
            alpha=alpha_node,
            Cw=Cw,
        )
        self.edge_attr = Tensor()

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, deg: Tensor
    ) -> Tensor:
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, deg=deg)
        return x, self.edge_attr

    def message(self, x_i, x_j, edge_attr, deg_i, deg_j):
        m = torch.cat([x_i, x_j, edge_attr], dim=1) / torch.sqrt(deg_i * deg_j)
        self.edge_attr = self.edge_net(m)
        return self.edge_attr

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.node_net(c)
