import torch
import torch.nn.functional as F
from .data import set_seed


def train(model, data, epochs=200, lr=0.01, weight_decay=5e-4, seed=0):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_acc = accuracy(out[data.val_mask], data.y[data.val_mask])
            train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])

        history.append({
            "epoch": epoch,
            "loss": loss.item(),
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def accuracy(logits, labels):
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()
