"""Datatypes for inputs, resolved architecture, and estimation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class VRAMEstimate:
    """Container for VRAM estimation results."""

    model_weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    kv_cache_gb: float
    compile_overhead_gb: float
    cuda_overhead_gb: float
    total_gb: float
    peak_gb: float
    available_gb: float
    fits: bool
    utilization_pct: float
    breakdown: Dict[str, float]
    model_params_b: float
    trainable_params_b: float
    attn_activations_gb: float = 0.0
    ffn_activations_gb: float = 0.0
    other_activations_gb: float = 0.0
    forward_pass_gb: float = 0.0
    backward_pass_gb: float = 0.0
    uses_swiglu: bool = True
    ddp_overhead_gb: float = 0.0
    config_hidden: int = 0
    config_layers: int = 0
    config_heads: int = 0
    config_kv_heads: int = 0
    config_intermediate: int = 0
    config_vocab_size: int = 0
    is_moe: bool = False
    num_experts: int = 1
    experts_per_token: int = 1
    active_params_b: float = 0.0
    memory_per_token_kb: float = 0.0
    kv_cache_per_token_kb: float = 0.0


@dataclass
class ManualModelConfig:
    """Optional overrides when auto-detection is missing or wrong."""

    params_b: float = 0.0
    hidden: int = 0
    layers: int = 0
    heads: int = 0
    kv_heads: int = 0
    intermediate: int = 0
    vocab_size: int = 0
    uses_swiglu: str = "Auto"
    num_experts: int = 0
    experts_per_token: int = 0
    active_params_b: float = 0.0


@dataclass
class VRAMInput:
    """Single object describing one VRAM estimation scenario."""

    model_id: str
    gpu_name: str
    mode: str
    dtype: str
    batch_size: int
    seq_length: int
    gradient_checkpointing: bool
    optimizer: str
    lora_rank: int
    lora_enabled: bool
    use_torch_compile: bool = False
    ddp_enabled: bool = False
    mixed_precision: bool = False
    manual: ManualModelConfig = field(default_factory=ManualModelConfig)

    @classmethod
    def from_gradio(
        cls,
        model_id: str,
        gpu_name: str,
        mode: str,
        dtype: str,
        batch_size: int,
        seq_length: int,
        gradient_checkpointing: bool,
        optimizer: str,
        lora_enabled: bool,
        lora_rank: int,
        use_torch_compile: bool,
        ddp_enabled: bool,
        mixed_precision: bool,
        manual_params_b: Any,
        manual_hidden: Any,
        manual_layers: Any,
        manual_heads: Any,
        manual_kv_heads: Any,
        manual_intermediate: Any,
        manual_vocab_size: Any,
        manual_uses_swiglu: str,
        manual_num_experts: Any,
        manual_experts_per_token: Any,
        manual_active_params_b: Any,
    ) -> VRAMInput:
        def _f(x: Any) -> float:
            return float(x) if x else 0.0

        def _i(x: Any) -> int:
            return int(x) if x else 0

        return cls(
            model_id=model_id.strip(),
            gpu_name=gpu_name,
            mode=mode,
            dtype=dtype,
            batch_size=int(batch_size),
            seq_length=int(seq_length),
            gradient_checkpointing=gradient_checkpointing,
            optimizer=optimizer,
            lora_rank=int(lora_rank),
            lora_enabled=lora_enabled,
            use_torch_compile=use_torch_compile,
            ddp_enabled=ddp_enabled,
            mixed_precision=mixed_precision,
            manual=ManualModelConfig(
                params_b=_f(manual_params_b),
                hidden=_i(manual_hidden),
                layers=_i(manual_layers),
                heads=_i(manual_heads),
                kv_heads=_i(manual_kv_heads),
                intermediate=_i(manual_intermediate),
                vocab_size=_i(manual_vocab_size),
                uses_swiglu=manual_uses_swiglu or "Auto",
                num_experts=_i(manual_num_experts),
                experts_per_token=_i(manual_experts_per_token),
                active_params_b=_f(manual_active_params_b),
            ),
        )


@dataclass
class ResolvedModelConfig:
    """Architecture and scale after merging presets, HF config, and manual fields."""

    model_id: str
    params_b: float
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    vocab_size: int
    intermediate_size: int
    uses_swiglu: bool
    num_experts: int
    experts_per_token: int
    is_moe: bool
    active_params_b: float
