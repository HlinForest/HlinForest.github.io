"""Lab 01: tensor shape drills.

Goal: fill every TODO without trial-and-error. Write the expected shape in a
comment first. Run this file; assertions provide immediate feedback.
"""

import torch


torch.manual_seed(42)

# 1) Build a float32 tensor containing 0..23 with semantic shape [B=2,T=3,H=4].
x = ...  # TODO
assert isinstance(x, torch.Tensor)
assert x.shape == (2, 3, 4)
assert x.dtype == torch.float32
assert x[1, 2, 3].item() == 23

# 2) Merge B and T while preserving H: [2,3,4] -> [6,4].
x_flat = ...  # TODO
assert x_flat.shape == (6, 4)
torch.testing.assert_close(x_flat[3], x[1, 0])

# 3) Move H before T: [B,T,H] -> [B,H,T].
x_bht = ...  # TODO
assert x_bht.shape == (2, 4, 3)
assert x_bht[1, 3, 2].item() == x[1, 2, 3].item()

# 4) Add positional encoding [T,H] to token states [B,T,H] by broadcasting.
pos = torch.arange(12, dtype=torch.float32).reshape(3, 4)
x_with_pos = ...  # TODO
assert x_with_pos.shape == (2, 3, 4)
torch.testing.assert_close(x_with_pos[0] - x[0], pos)
torch.testing.assert_close(x_with_pos[1] - x[1], pos)

# 5) Project the hidden axis H=4 to O=7: [B,T,H] @ [H,O] -> [B,T,O].
projection = torch.randn(4, 7)
y = ...  # TODO
assert y.shape == (2, 3, 7)

# 6) Stack three [H] tensors on a NEW batch axis: -> [3,H].
rows = [torch.arange(4), torch.arange(4) + 10, torch.arange(4) + 20]
stacked = ...  # TODO
assert stacked.shape == (3, 4)
assert stacked[2, 0].item() == 20

# 7) Concatenate the same tensors on their EXISTING axis: -> [12].
concatenated = ...  # TODO
assert concatenated.shape == (12,)
assert concatenated[4].item() == 10

# 8) Binary logits [B,1] -> [B] without deleting the batch axis when B=1.
logits = torch.randn(1, 1)
safe_logits = ...  # TODO
assert safe_logits.shape == (1,)

# 9) Convert a gradient-tracking tensor to a CPU NumPy array safely.
tracked = torch.randn(3, requires_grad=True)
array = ...  # TODO
assert array.shape == (3,)
assert not isinstance(array, torch.Tensor)

print("PASS: all tensor shape contracts hold")
