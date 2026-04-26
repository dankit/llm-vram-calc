"""Programmatic API wrapper for architecture-first VRAM calculation."""

from __future__ import annotations

from vram_calc.engine import estimate_vram
from vram_calc.types import ArchitectureConfig, VRAMInput, VRAMEstimate


def calculate_vram(
    architecture: ArchitectureConfig,
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
) -> VRAMEstimate:
    """Calculate VRAM requirements from explicit architecture + runtime inputs."""
    inp = VRAMInput(
        architecture=architecture,
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
    )
    return estimate_vram(inp)
