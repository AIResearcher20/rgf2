import sys
sys.path.insert(0, "src")

from rgf.data import load_dataset
from rgf.model import RGF
from rgf.train import train
from rgf.eval import evaluate
from rgf.repro import seed_everything, snapshot
import json


def main():
    seed = 0
    seed_everything(seed)

    dataset, data = load_dataset("Cora")

    model = RGF(
        in_dim=dataset.num_features,
        hidden=64,
        out_dim=dataset.num_classes,
        num_layers=2,
        dropout=0.5,
    )

    model, history = train(model, data, epochs=200, seed=seed)
    result = evaluate(model, data)

    print(json.dumps(result, indent=2))

    snap = snapshot({
        "dataset": "Cora",
        "hidden": 64,
        "layers": 2,
        "dropout": 0.5,
        "seed": seed,
    })
    print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()
