import json
import time
import hashlib
import torch
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def config_hash(config):
    s = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()[:12]


def snapshot(config):
    return {
        "config": config,
        "hash": config_hash(config),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
