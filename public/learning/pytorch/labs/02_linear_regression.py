"""Lab 02: complete an end-to-end regression workflow.

Fill the TODOs. The final assertions check optimization, parameter recovery,
evaluation semantics, and checkpoint reload consistency.
"""

from pathlib import Path

import torch
from torch import nn


SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Known data-generating process: y = 0.7x + 0.3 + small noise.
X = torch.arange(0, 1, 0.02).unsqueeze(1)
generator = torch.Generator().manual_seed(SEED)
noise = 0.01 * torch.randn(X.shape, generator=generator)
y = 0.7 * X + 0.3 + noise

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = ...  # TODO: one input feature, one output feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ...  # TODO


model = LinearModel().to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

loss_fn = ...  # TODO: mean squared error
optimizer = ...  # TODO: SGD over model parameters, lr=0.1

initial_weight = model.linear.weight.detach().clone()
history = {"train": [], "test": []}

for epoch in range(300):
    # TODO: training mode, forward, loss, zero, backward, step.
    ...

    # TODO: evaluation mode and inference context; compute test_loss.
    ...

    history["train"].append(train_loss.item())
    history["test"].append(test_loss.item())

assert history["train"][-1] < history["train"][0] * 0.05
assert history["test"][-1] < 0.01
assert not torch.equal(initial_weight, model.linear.weight.detach())
assert abs(model.linear.weight.item() - 0.7) < 0.12
assert abs(model.linear.bias.item() - 0.3) < 0.12

# TODO: save state_dict to this exact path.
checkpoint_path = Path(__file__).parent / "artifacts" / "linear_state.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
...

# TODO: create a new model, load state dict with map_location, switch to eval.
loaded = ...
...

with torch.inference_mode():
    before = model(X_test)
    after = loaded(X_test)
torch.testing.assert_close(before, after)

print(f"PASS on {device}: train={history['train'][-1]:.6f}, "
      f"test={history['test'][-1]:.6f}, "
      f"w={model.linear.weight.item():.3f}, b={model.linear.bias.item():.3f}")
