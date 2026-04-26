"""LLM VRAM estimation from explicit architecture and runtime inputs."""

from vram_calc.constants import DTYPE_BYTES, GPU_SPECS
from vram_calc.config import estimate_params_from_architecture, finalize_architecture
from vram_calc.engine import compute_vram_estimate, estimate_vram
from vram_calc.legacy import calculate_vram
from vram_calc.types import (
    ArchitectureConfig,
    VRAMEstimate,
    VRAMInput,
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


def launch_app(**launch_kwargs):
    """Start the Gradio web UI (imports Gradio on demand)."""
    from vram_calc.ui.gradio_app import launch_app as _launch_app

    _launch_app(**launch_kwargs)
