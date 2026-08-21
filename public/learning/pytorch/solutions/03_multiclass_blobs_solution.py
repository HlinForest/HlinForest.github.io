"""Reference solution for Lab 03."""

from pathlib import Path

import torch
from torch import nn


SEED = 7
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def make_blobs(points_per_class: int = 160, std: float = 0.65):
    centers = torch.tensor([[-2.5, -2.0], [-2.0, 2.5], [2.5, -2.0], [2.0, 2.5]])
    generator = torch.Generator().manual_seed(SEED)
    features, labels = [], []
    for class_id, center in enumerate(centers):
        cloud = center + std * torch.randn(points_per_class, 2, generator=generator)
        features.append(cloud)
        labels.append(torch.full((points_per_class,), class_id, dtype=torch.long))
    X = torch.cat(features)
    y = torch.cat(labels)
    order = torch.randperm(len(X), generator=generator)
    return X[order], y[order]


X, y = make_blobs()
split = int(0.8 * len(X))
X_train, y_train = X[:split].to(device), y[:split].to(device)
X_test, y_test = X[split:].to(device), y[split:].to(device)


class BlobMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = BlobMLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

for epoch in range(200):
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_loss = loss_fn(test_logits, y_test)
        test_pred = test_logits.argmax(dim=1)

    if epoch % 40 == 0:
        test_acc = (test_pred == y_test).float().mean()
        print(epoch, round(loss.item(), 4), round(test_loss.item(), 4), round(test_acc.item(), 4))

assert logits.shape == (len(X_train), 4)
test_acc = (test_pred == y_test).float().mean().item()
assert test_acc > 0.95

confusion = torch.zeros(4, 4, dtype=torch.long, device=device)
for true, pred in zip(y_test, test_pred):
    confusion[true, pred] += 1
assert confusion.sum().item() == len(y_test)

path = Path(__file__).parent / "artifacts" / "blob_mlp_state.pt"
path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), path)
loaded = BlobMLP().to(device)
try:
    state = torch.load(path, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(path, map_location=device)
loaded.load_state_dict(state, strict=True)
loaded.eval()
with torch.inference_mode():
    loaded_logits = loaded(X_test)
torch.testing.assert_close(test_logits, loaded_logits)

print(f"PASS on {device}: accuracy={test_acc:.3f}")
print("confusion matrix (rows=true, cols=pred):\n", confusion.cpu())
