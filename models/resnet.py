from __future__ import annotations

import math

from torch import Tensor
from torch.nn import Linear, Module, ModuleList, ReLU, init


class ResNet(Module):
    """Fully connected NN w/ residual connections and Gaussian init
    Args:
    in_dim: input dimension
    out_dim: output dimension
    hidden_dim: width, hidden dimension
    alpha: strength of the residual connection
    L: depth, number of hidden layers
    Cw: variance scale for Gaussian init
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        alpha: float = 0,
        L: int = 2,
        Cw: float = 2,
    ):
        super(ResNet, self).__init__()

        self.layers = ModuleList()
        for layer in range(L + 1):
            self.layers.append(
                Linear(
                    in_dim if (layer == 0) else hidden_dim,
                    out_dim if (layer == L) else hidden_dim,
                    bias=False,
                )
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.alpha = alpha
        self.L = L
        self.Cw = Cw
        self.relu = ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for layer, weights in enumerate(self.layers):
            for p in weights.weight:
                std = math.sqrt(1.0 / self.in_dim)
                if layer > 0:
                    std = math.sqrt(self.Cw / self.hidden_dim)
                init.normal_(p.data, mean=0, std=std)

    def forward(self, x: Tensor):
        for layer, weights in enumerate(self.layers):
            if layer == 0:
                x = weights(x)
            elif (layer > 0) and (layer < self.L):
                x = self.alpha * x + weights(self.relu(x))
            else:
                x = weights(self.relu(x))
        return x
