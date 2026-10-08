import sys
sys.path.insert(0, "src")

import json
from rgf.data import load_dataset
from rgf.model import RGF
from rgf.baselines import GCN
from rgf.train import train
from rgf.eval import evaluate
from rgf.repro import seed_everything, snapshot


def run(model, data, name, seed):
    model, history = train(model, data, epochs=200, seed=seed)
    result = evaluate(model, data)
    print(f"=== {name} ===")
    print(json.dumps(result, indent=2))
    return result


def main():
    seed = 0
    seed_everything(seed)

    dataset, data = load_dataset("Cora")

    common = {
        "in_dim": dataset.num_features,
        "hidden": 64,
        "out_dim": dataset.num_classes,
        "num_layers": 2,
        "dropout": 0.5,
    }

    rgf = RGF(**common)
    gcn = GCN(**common)

    rgf_result = run(rgf, data, "RGF2", seed)
    gcn_result = run(gcn, data, "GCN", seed)

    print()
    print("=== SUMMARY ===")
    print(f"RGF2 test acc: {rgf_result['test']['acc']:.4f}")
    print(f"GCN  test acc: {gcn_result['test']['acc']:.4f}")

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
