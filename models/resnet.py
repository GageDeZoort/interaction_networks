import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, ModuleList, init

import torch_geometric
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import to_dense_adj
from matplotlib import pyplot as plt

class ResNet(Module):
    def __init__(self, in_dim, out_dim, hidden_dim, alpha=0, L=2, Cw=2):
        super(ResNet, self).__init__()
        
        self.layers = ModuleList()
        for l in range(L+1):
            self.layers.append(
                Linear(
                    in_dim if (l==0) else hidden_dim,
                    out_dim if (l==L) else hidden_dim,
                    bias = False,
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
        for l, layer in enumerate(self.layers):
            for p in layer.weight:
                std = math.sqrt(1.0 / self.in_dim) 
                if (l>0): std = math.sqrt(self.Cw / self.hidden_dim)
                init.normal_(p.data, mean=0, std=std)
            
    def forward(self, x):
        for l, layer in enumerate(self.layers):
            if l==0:
                x = layer(x)
            elif (l>0) and (l<self.L):
                x = self.alpha*x + layer(self.relu(x))
            else:
                x = layer(self.relu(x))
        return x
