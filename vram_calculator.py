"""Shim module so ``python vram_calculator.py`` and ``from vram_calculator import ...`` keep working."""

from vram_calc import (
    DTYPE_BYTES,
    GPU_SPECS,
    MODEL_PRESETS,
    ManualModelConfig,
    ResolvedModelConfig,
    VRAMEstimate,
    VRAMInput,
    calculate_vram,
    compute_vram_estimate,
    estimate_params_from_config,
    estimate_vram,
    fetch_model_config,
    launch_app,
    resolve_model_config,
)

__all__ = [
    "DTYPE_BYTES",
    "GPU_SPECS",
    "MODEL_PRESETS",
    "ManualModelConfig",
    "ResolvedModelConfig",
    "VRAMEstimate",
    "VRAMInput",
    "calculate_vram",
    "compute_vram_estimate",
    "estimate_params_from_config",
    "estimate_vram",
    "fetch_model_config",
    "launch_app",
    "resolve_model_config",
]


if __name__ == "__main__":
    print("Starting LLM VRAM Calculator")
    print("=" * 50)
    launch_app()
