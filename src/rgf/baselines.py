import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):

    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GCNConv(hidden, hidden) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.gelu(self.input(x))
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)
