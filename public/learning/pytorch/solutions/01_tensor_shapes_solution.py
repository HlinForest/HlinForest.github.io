"""Reference solution for Lab 01. Read only after a serious attempt."""

import torch


torch.manual_seed(42)
x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
assert x.shape == (2, 3, 4) and x[1, 2, 3].item() == 23

x_flat = x.reshape(6, 4)
assert x_flat.shape == (6, 4)
torch.testing.assert_close(x_flat[3], x[1, 0])

x_bht = x.permute(0, 2, 1)
assert x_bht.shape == (2, 4, 3)

pos = torch.arange(12, dtype=torch.float32).reshape(3, 4)
x_with_pos = x + pos
assert x_with_pos.shape == (2, 3, 4)

projection = torch.randn(4, 7)
y = x @ projection
assert y.shape == (2, 3, 7)

rows = [torch.arange(4), torch.arange(4) + 10, torch.arange(4) + 20]
stacked = torch.stack(rows, dim=0)
concatenated = torch.cat(rows, dim=0)
assert stacked.shape == (3, 4)
assert concatenated.shape == (12,)

logits = torch.randn(1, 1)
safe_logits = logits.squeeze(-1)
assert safe_logits.shape == (1,)

tracked = torch.randn(3, requires_grad=True)
array = tracked.detach().cpu().numpy()
assert array.shape == (3,)

print("PASS: all tensor shape contracts hold")
