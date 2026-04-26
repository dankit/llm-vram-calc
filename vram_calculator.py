"""Shim module so ``python vram_calculator.py`` and ``from vram_calculator import ...`` keep working."""

from vram_calc import (
    DTYPE_BYTES,
    GPU_SPECS,
    ArchitectureConfig,
    VRAMEstimate,
    VRAMInput,
    calculate_vram,
    compute_vram_estimate,
    estimate_params_from_architecture,
    estimate_vram,
    finalize_architecture,
    launch_app,
)

__all__ = [
    "DTYPE_BYTES",
    "GPU_SPECS",
    "ArchitectureConfig",
    "VRAMEstimate",
    "VRAMInput",
    "calculate_vram",
    "compute_vram_estimate",
    "estimate_params_from_architecture",
    "estimate_vram",
    "finalize_architecture",
    "launch_app",
]


if __name__ == "__main__":
    print("Starting LLM VRAM Calculator")
    print("=" * 50)
    launch_app()
