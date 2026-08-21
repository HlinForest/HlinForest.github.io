"""Lab 03: train a four-class classifier without scikit-learn.

Fill the TODOs. Data are deterministic Gaussian blobs; CPU is sufficient.
"""

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
        # TODO: 2 -> 16 -> 4, with a ReLU between the Linear layers.
        self.net = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ...  # TODO: raw logits [B,4]


model = BlobMLP().to(device)
loss_fn = ...  # TODO: multiclass loss accepting raw logits and long class indices
optimizer = ...  # TODO: AdamW, lr=0.02

for epoch in range(200):
    # TODO: full training step.
    ...

    # TODO: evaluation; compute test_loss and test_pred from test_logits.
    ...

    if epoch % 40 == 0:
        test_acc = (test_pred == y_test).float().mean()
        print(epoch, round(loss.item(), 4), round(test_loss.item(), 4), round(test_acc.item(), 4))

assert logits.shape == (len(X_train), 4)
assert y_train.dtype == torch.long
assert test_pred.shape == y_test.shape
test_acc = (test_pred == y_test).float().mean().item()
assert test_acc > 0.95, f"accuracy too low: {test_acc:.3f}"

# TODO: build a [4,4] confusion matrix where rows=true and columns=predicted.
confusion = ...
assert confusion.shape == (4, 4)
assert confusion.sum().item() == len(y_test)

# TODO: save state_dict, rebuild, load, and prove identical test logits.
path = Path(__file__).parent / "artifacts" / "blob_mlp_state.pt"
path.parent.mkdir(parents=True, exist_ok=True)
...

print(f"PASS on {device}: accuracy={test_acc:.3f}")
print("confusion matrix (rows=true, cols=pred):\n", confusion.cpu())
