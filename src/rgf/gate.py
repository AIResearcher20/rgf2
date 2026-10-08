import torch
import torch.nn as nn


class ResidualGate(nn.Module):

    def __init__(self, dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = dim
        self.fc1 = nn.Linear(2 * dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, h_res, h_msg):
        z = torch.cat([h_res, h_msg], dim=-1)
        z = self.act(self.fc1(z))
        g = torch.sigmoid(self.fc2(z))
        return g
