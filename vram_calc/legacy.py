"""Backward-compatible ``calculate_vram`` keyword API wrapping :class:`~vram_calc.types.VRAMInput`."""

from __future__ import annotations

from typing import Optional

from vram_calc.engine import estimate_vram
from vram_calc.types import ManualModelConfig, VRAMInput, VRAMEstimate


def calculate_vram(
    model_id: str,
    gpu_name: str,
    mode: str,
    dtype: str,
    batch_size: int,
    seq_length: int,
    gradient_checkpointing: bool,
    optimizer: str,
    lora_rank: int,
    lora_enabled: bool,
    use_torch_compile: bool = False,
    ddp_enabled: bool = False,
    mixed_precision: bool = False,
    manual_params_b: float = 0,
    manual_hidden: int = 0,
    manual_layers: int = 0,
    manual_heads: int = 0,
    manual_kv_heads: int = 0,
    manual_intermediate: int = 0,
    manual_vocab_size: int = 0,
    manual_uses_swiglu: str = "Auto",
    manual_num_experts: int = 0,
    manual_experts_per_token: int = 0,
    manual_active_params_b: float = 0,
) -> Optional[VRAMEstimate]:
    """Calculate VRAM requirements (legacy keyword interface)."""
    inp = VRAMInput(
        model_id=model_id,
        gpu_name=gpu_name,
        mode=mode,
        dtype=dtype,
        batch_size=batch_size,
        seq_length=seq_length,
        gradient_checkpointing=gradient_checkpointing,
        optimizer=optimizer,
        lora_rank=lora_rank,
        lora_enabled=lora_enabled,
        use_torch_compile=use_torch_compile,
        ddp_enabled=ddp_enabled,
        mixed_precision=mixed_precision,
        manual=ManualModelConfig(
            params_b=float(manual_params_b or 0),
            hidden=int(manual_hidden or 0),
            layers=int(manual_layers or 0),
            heads=int(manual_heads or 0),
            kv_heads=int(manual_kv_heads or 0),
            intermediate=int(manual_intermediate or 0),
            vocab_size=int(manual_vocab_size or 0),
            uses_swiglu=manual_uses_swiglu,
            num_experts=int(manual_num_experts or 0),
            experts_per_token=int(manual_experts_per_token or 0),
            active_params_b=float(manual_active_params_b or 0),
        ),
    )
    return estimate_vram(inp)
