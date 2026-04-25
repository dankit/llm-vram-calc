"""LLM VRAM estimation: presets, Hugging Face config resolution, and memory math."""

from vram_calc.constants import DTYPE_BYTES, GPU_SPECS, MODEL_PRESETS
from vram_calc.config import (
    estimate_params_from_config,
    fetch_model_config,
    resolve_model_config,
)
from vram_calc.engine import compute_vram_estimate, estimate_vram
from vram_calc.legacy import calculate_vram
from vram_calc.types import (
    ManualModelConfig,
    ResolvedModelConfig,
    VRAMEstimate,
    VRAMInput,
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


def launch_app(**launch_kwargs):
    """Start the Gradio web UI (imports Gradio on demand)."""
    from vram_calc.ui.gradio_app import launch_app as _launch_app

    _launch_app(**launch_kwargs)
