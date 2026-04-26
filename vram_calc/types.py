"""Datatypes for architecture input and VRAM estimation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

AttentionType = Literal["mha", "gqa"]
FFNType = Literal["dense", "moe"]


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
    ddp_overhead_gb: float = 0.0
    config_hidden: int = 0
    config_layers: int = 0
    config_heads: int = 0
    config_kv_heads: int = 0
    is_moe: bool = False
    num_experts: int = 1
    experts_per_token: int = 1
    active_params_b: float = 0.0
    kv_cache_per_token_kb: float = 0.0
    attention_type: str = "mha"
    ffn_type: str = "dense"


@dataclass
class ArchitectureConfig:
    """User-provided architecture definition for dynamic calculations."""

    params_b: float
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    intermediate_size: int
    vocab_size: int
    attention_type: AttentionType
    ffn_type: FFNType
    num_experts: int = 1
    experts_per_token: int = 1
    active_params_b: float = 0.0


@dataclass
class VRAMInput:
    """Single object describing one VRAM estimation scenario."""

    architecture: ArchitectureConfig
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
    @classmethod
    def from_gradio(
        cls,
        params_b: Any,
        hidden: Any,
        layers: Any,
        heads: Any,
        kv_heads: Any,
        intermediate_size: Any,
        vocab_size: Any,
        attention_type: str,
        ffn_type: str,
        num_experts: Any,
        experts_per_token: Any,
        active_params_b: Any,
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
    ) -> VRAMInput:
        dtype_aliases = {
            "bfloat16 (bf16)": "16-bit (BF16/FP16)",
            "float16 (fp16)": "16-bit (BF16/FP16)",
        }

        def _f(x: Any) -> float:
            return float(x) if x else 0.0

        def _i(x: Any) -> int:
            return int(x) if x else 0

        normalized_dtype = dtype_aliases.get(dtype.strip().lower(), dtype)

        architecture = ArchitectureConfig(
            params_b=_f(params_b),
            hidden=_i(hidden),
            layers=_i(layers),
            heads=_i(heads),
            kv_heads=_i(kv_heads),
            intermediate_size=_i(intermediate_size),
            vocab_size=_i(vocab_size),
            attention_type=attention_type.strip().lower(),  # type: ignore[arg-type]
            ffn_type=ffn_type.strip().lower(),  # type: ignore[arg-type]
            num_experts=_i(num_experts),
            experts_per_token=_i(experts_per_token),
            active_params_b=_f(active_params_b),
        )

        return cls(
            architecture=architecture,
            gpu_name=gpu_name,
            mode=mode,
            dtype=normalized_dtype,
            batch_size=int(batch_size),
            seq_length=int(seq_length),
            gradient_checkpointing=gradient_checkpointing,
            optimizer=optimizer,
            lora_rank=int(lora_rank),
            lora_enabled=lora_enabled,
            use_torch_compile=use_torch_compile,
            ddp_enabled=ddp_enabled,
            mixed_precision=mixed_precision,
        )
