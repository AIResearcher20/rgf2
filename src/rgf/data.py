import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_dataset(name="Cora", root="data"):
    dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
    data = dataset[0]
    return dataset, data


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
