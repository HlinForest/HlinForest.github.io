"""Reference solution for Lab 02."""

from pathlib import Path

import torch
from torch import nn


SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.arange(0, 1, 0.02).unsqueeze(1)
generator = torch.Generator().manual_seed(SEED)
noise = 0.01 * torch.randn(X.shape, generator=generator)
y = 0.7 * X + 0.3 + noise
split = int(0.8 * len(X))
X_train, y_train = X[:split].to(device), y[:split].to(device)
X_test, y_test = X[split:].to(device), y[split:].to(device)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


model = LinearModel().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
initial_weight = model.linear.weight.detach().clone()
history = {"train": [], "test": []}

for _ in range(300):
    model.train()
    train_pred = model(X_train)
    train_loss = loss_fn(train_pred, y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    history["train"].append(train_loss.item())
    history["test"].append(test_loss.item())

assert history["train"][-1] < history["train"][0] * 0.05
assert history["test"][-1] < 0.01
assert not torch.equal(initial_weight, model.linear.weight.detach())
assert abs(model.linear.weight.item() - 0.7) < 0.12
assert abs(model.linear.bias.item() - 0.3) < 0.12

checkpoint_path = Path(__file__).parent / "artifacts" / "linear_state.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)

loaded = LinearModel().to(device)
try:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
except TypeError:  # Older PyTorch compatibility.
    state = torch.load(checkpoint_path, map_location=device)
loaded.load_state_dict(state, strict=True)
loaded.eval()

with torch.inference_mode():
    before = model(X_test)
    after = loaded(X_test)
torch.testing.assert_close(before, after)

print(f"PASS on {device}: train={history['train'][-1]:.6f}, "
      f"test={history['test'][-1]:.6f}, "
      f"w={model.linear.weight.item():.3f}, b={model.linear.bias.item():.3f}")
