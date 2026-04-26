"""Architecture validation and derived parameter helpers."""

from __future__ import annotations

from dataclasses import replace

from vram_calc.types import ArchitectureConfig


def estimate_params_from_architecture(arch: ArchitectureConfig) -> float:
    """Estimate total parameters in billions from architecture fields."""
    hidden = arch.hidden
    layers = arch.layers
    vocab_size = arch.vocab_size
    intermediate = arch.intermediate_size
    heads = max(1, arch.heads)
    if arch.attention_type == "gqa":
        kv_heads = max(1, arch.kv_heads)
    else:
        kv_heads = heads

    head_dim = hidden // heads if heads > 0 else 128

    embedding_params = vocab_size * hidden * 2
    q_params = hidden * hidden
    k_params = hidden * (head_dim * kv_heads)
    v_params = hidden * (head_dim * kv_heads)
    o_params = hidden * hidden
    attn_params = q_params + k_params + v_params + o_params

    if arch.ffn_type == "moe":
        mlp_params = hidden * intermediate * max(1, arch.num_experts)
        router_params = hidden * max(1, arch.num_experts)
    else:
        mlp_params = hidden * intermediate
        router_params = 0

    ln_params = hidden * 4
    total_per_layer = attn_params + mlp_params + router_params + ln_params
    total_params = embedding_params + (layers * total_per_layer)
    return total_params / 1e9


def finalize_architecture(arch: ArchitectureConfig) -> ArchitectureConfig:
    """Validate fields and derive defaults for attention and MoE settings."""
    if arch.params_b <= 0 or arch.hidden <= 0 or arch.layers <= 0:
        raise ValueError("Parameters (B), hidden size, and layers must be > 0.")

    if arch.heads <= 0:
        raise ValueError("Attention heads must be > 0.")

    if arch.hidden % arch.heads != 0:
        raise ValueError("Hidden size must be divisible by attention heads.")

    attention_type = arch.attention_type.lower()
    if attention_type not in {"mha", "gqa"}:
        raise ValueError("Attention type must be one of: mha, gqa.")

    if attention_type == "mha":
        kv_heads = arch.heads
    else:
        kv_heads = arch.kv_heads

    if kv_heads <= 0 or arch.heads % kv_heads != 0:
        raise ValueError("KV heads must be > 0 and divide attention heads.")

    if arch.intermediate_size <= 0:
        raise ValueError("FFN intermediate size must be > 0.")

    if arch.vocab_size <= 0:
        raise ValueError("Vocab size must be > 0.")

    ffn_type = arch.ffn_type.lower()
    if ffn_type not in {"dense", "moe"}:
        raise ValueError("FFN type must be one of: dense, moe.")

    if ffn_type == "dense":
        num_experts = 1
        experts_per_token = 1
    else:
        num_experts = max(2, arch.num_experts)
        experts_per_token = max(1, arch.experts_per_token)
        if experts_per_token > num_experts:
            raise ValueError("Experts per token must be <= number of experts.")

    if ffn_type == "dense":
        # Dense models always activate the full parameter set.
        active_params_b = arch.params_b
    else:
        active_params_b = arch.active_params_b
        if active_params_b <= 0:
            expert_ratio = experts_per_token / num_experts
            active_params_b = arch.params_b * (0.33 + 0.67 * expert_ratio)
        active_params_b = min(active_params_b, arch.params_b)

    return replace(
        arch,
        attention_type=attention_type,  # type: ignore[arg-type]
        ffn_type=ffn_type,  # type: ignore[arg-type]
        kv_heads=kv_heads,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        active_params_b=active_params_b,
    )
