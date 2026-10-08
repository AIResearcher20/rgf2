import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .gate import ResidualGate


class RGFLayer(nn.Module):

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.conv = GCNConv(dim, dim)
        self.gate = ResidualGate(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h_res = x
        h_msg = self.conv(x, edge_index)
        h_msg = F.gelu(h_msg)
        g = self.gate(h_res, h_msg)
        out = (1.0 - g) * h_res + g * h_msg
        out = self.norm(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class RGF(nn.Module):

    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [RGFLayer(hidden, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        h = self.input(x)
        h = F.gelu(h)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h)

    def embed(self, x, edge_index):
        h = self.input(x)
        h = F.gelu(h)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h
