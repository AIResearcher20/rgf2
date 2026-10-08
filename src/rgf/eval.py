import torch
from sklearn.metrics import f1_score


def evaluate(model, data):
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    result = {}
    for split in ["train", "val", "test"]:
        mask = getattr(data, f"{split}_mask")
        result[split] = metrics(logits[mask], data.y[mask])
    return result


def metrics(logits, labels):
    pred = logits.argmax(dim=-1)
    acc = (pred == labels).float().mean().item()
    f1 = f1_score(
        labels.cpu().numpy(),
        pred.cpu().numpy(),
        average="macro",
        zero_division=0,
    )
    return {"acc": acc, "f1": f1}
