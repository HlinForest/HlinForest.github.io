"""Parameter and KV-cache ledger for a decoder-only Transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 4096
    intermediate: int = 11008
    layers: int = 32
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 32_000


def block_parameters(config: ModelConfig) -> dict[str, int]:
    h = config.hidden
    kv_width = config.kv_heads * config.head_dim
    q_width = config.query_heads * config.head_dim
    return {
        "q_projection": h * q_width,
        "k_projection": h * kv_width,
        "v_projection": h * kv_width,
        "o_projection": q_width * h,
        "swiglu": 3 * h * config.intermediate,
        "norms": 2 * h,
    }


def kv_cache_bytes(
    config: ModelConfig,
    batch: int,
    sequence: int,
    bytes_per_element: int = 2,
) -> int:
    return (
        2
        * config.layers
        * batch
        * sequence
        * config.kv_heads
        * config.head_dim
        * bytes_per_element
    )


def gibibytes(value: int) -> float:
    return value / 1024**3


def main() -> None:
    config = ModelConfig()
    parts = block_parameters(config)
    print("per-block parameters")
    for name, value in parts.items():
        print(f"  {name:16s} {value:>15,d}")
    per_block = sum(parts.values())
    embedding = config.vocab * config.hidden
    total = per_block * config.layers + embedding
    print("per block:", f"{per_block:,}")
    print("rough model total:", f"{total:,}")

    cache = kv_cache_bytes(config, batch=32, sequence=8192)
    print("KV cache (B=32,S=8192,BF16):", f"{gibibytes(cache):.2f} GiB")
    assert config.query_heads * config.head_dim == config.hidden
    print("PASS: configuration is shape-consistent")


if __name__ == "__main__":
    main()

# EXERCISES
# 1. Compare kv_heads = 32, 8, and 1.
# 2. Add tied vs untied output-embedding parameters.
# 3. Add bias terms and explain why many LLMs omit them.
# 4. Extend the ledger with parameter, gradient, and AdamW-state bytes.
