"""Persistence helpers for user-defined GPU specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from vram_calc.constants import GPU_SPECS

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CUSTOM_GPU_PATH = _DATA_DIR / "custom_gpus.json"


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def _coerce_positive_float(value: float, field_name: str) -> float:
    num = float(value)
    if num <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return num


def _read_custom_raw() -> Dict[str, dict]:
    if not _CUSTOM_GPU_PATH.exists():
        return {}
    try:
        payload = json.loads(_CUSTOM_GPU_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_custom_gpu_specs() -> Dict[str, dict]:
    """Return validated custom GPU specs from disk."""
    raw = _read_custom_raw()
    cleaned: Dict[str, dict] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        try:
            vram_gb = _coerce_positive_float(spec["vram_gb"], "VRAM")
        except (KeyError, TypeError, ValueError):
            continue
        cleaned[name] = {"vram_gb": vram_gb}
    return cleaned


def get_all_gpu_specs() -> Dict[str, dict]:
    """Return built-in specs merged with custom specs."""
    return {**GPU_SPECS, **load_custom_gpu_specs()}


def save_custom_gpu_spec(name: str, vram_gb: float) -> Dict[str, dict]:
    """Persist a custom GPU and return updated custom mapping."""
    display_name = " ".join(name.strip().split())
    if not display_name:
        raise ValueError("GPU name is required.")

    vram_value = _coerce_positive_float(vram_gb, "VRAM")

    if _normalize_name(display_name) in {_normalize_name(n) for n in GPU_SPECS}:
        raise ValueError("This GPU name already exists in built-in presets.")

    custom_specs = load_custom_gpu_specs()
    duplicate_name = next(
        (existing for existing in custom_specs if _normalize_name(existing) == _normalize_name(display_name)),
        None,
    )
    final_name = duplicate_name or display_name
    custom_specs[final_name] = {"vram_gb": vram_value}

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CUSTOM_GPU_PATH.write_text(json.dumps(custom_specs, indent=2), encoding="utf-8")
    return custom_specs
