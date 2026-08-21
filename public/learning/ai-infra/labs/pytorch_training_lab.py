"""Synthetic PyTorch training, checkpoint, memory, and profiler lab."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path


def require_torch():
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("PyTorch is not installed in this Python environment.")
    import torch

    return torch


def main(profile_enabled: bool) -> None:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 4),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()

    features = torch.randn(256, 32)
    labels = torch.randint(0, 4, (256,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size=32,
        shuffle=True,
    )

    def train_step(batch):
        inputs, targets = (item.to(device) for item in batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    losses = []
    if profile_enabled:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for step, batch in enumerate(loader):
                losses.append(float(train_step(batch)))
                if step == 4:
                    break
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(profiler.key_averages().table(sort_by=sort_key, row_limit=10))
    else:
        for step, batch in enumerate(loader):
            losses.append(float(train_step(batch)))
            if step == 9:
                break

    with tempfile.TemporaryDirectory(prefix="aiinfra-lab-") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": len(losses),
            },
            checkpoint,
        )
        clone = type(model)(
            torch.nn.Linear(32, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 4),
        ).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        clone.load_state_dict(saved["model"])
        for expected, actual in zip(model.parameters(), clone.parameters()):
            assert torch.equal(expected, actual)

    model.eval()
    with torch.no_grad():
        sample_output = model(features[:2].to(device))
    assert sample_output.shape == (2, 4)
    print("device:", device)
    print("first/last loss:", f"{losses[0]:.4f}", f"{losses[-1]:.4f}")
    if device.type == "cuda":
        print("peak allocated bytes:", torch.cuda.max_memory_allocated())
    print("PASS: training, checkpoint restore, and evaluation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    arguments = parser.parse_args()
    main(arguments.profile)

# EXERCISES
# 1. Remove zero_grad, run twice, and inspect gradient norms.
# 2. Add Dropout and compare train() with eval().
# 3. Add an intentional .item() every step and inspect a CUDA timeline.
# 4. Compare FP32 and BF16 on CUDA using equal warmup and repeat counts.
