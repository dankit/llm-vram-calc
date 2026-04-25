"""Hugging Face config fetch, param estimation, and preset / manual resolution."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests

from vram_calc.constants import MODEL_PRESETS
from vram_calc.types import ManualModelConfig, ResolvedModelConfig


def fetch_model_config(model_id: str) -> Optional[dict]:
    """Fetch model configuration from HuggingFace (config.json on main)."""
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None

        config = response.json()

        hidden = config.get("hidden_size") or config.get("d_model")
        layers = config.get("num_hidden_layers") or config.get("n_layer")

        if hidden is None or layers is None:
            return None

        heads = config.get("num_attention_heads") or config.get("n_head")
        kv_heads = config.get("num_key_value_heads") or heads
        vocab_size = config.get("vocab_size") or 32000
        intermediate_size = config.get("intermediate_size")

        num_experts = (
            config.get("num_local_experts")
            or config.get("num_experts")
            or config.get("n_routed_experts")
            or 1
        )
        experts_per_token = (
            config.get("num_experts_per_tok")
            or config.get("num_experts_per_token")
            or config.get("top_k")
            or config.get("topk_group")
            or (2 if num_experts > 1 else 1)
        )

        result: Dict[str, Any] = {
            "params_b": config.get("num_parameters", 0) / 1e9 if "num_parameters" in config else None,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "kv_heads": kv_heads,
            "vocab_size": vocab_size,
            "intermediate_size": intermediate_size,
        }

        if num_experts > 1:
            result["num_experts"] = num_experts
            result["experts_per_token"] = experts_per_token

        return result
    except Exception:
        return None


def estimate_params_from_config(config: dict) -> Optional[float]:
    """Estimate total parameters in billions from architecture fields."""
    hidden = config.get("hidden")
    layers = config.get("layers")

    if hidden is None or layers is None:
        return None

    vocab_size = config.get("vocab_size") or 32000
    intermediate = config.get("intermediate_size") or int(hidden * 3.5)
    heads = config.get("heads") or max(1, hidden // 128)
    kv_heads = config.get("kv_heads") or heads

    head_dim = hidden // heads if heads > 0 else 128

    embedding_params = vocab_size * hidden * 2

    q_params = hidden * hidden
    k_params = hidden * (head_dim * kv_heads)
    v_params = hidden * (head_dim * kv_heads)
    o_params = hidden * hidden
    attn_params = q_params + k_params + v_params + o_params

    mlp_params = hidden * intermediate * 3
    ln_params = hidden * 4

    total_per_layer = attn_params + mlp_params + ln_params
    total_params = embedding_params + (layers * total_per_layer)

    return total_params / 1e9


def _lookup_preset(model_id: str) -> Optional[Tuple[dict, float]]:
    """Return (preset dict copy, params_b) if model_id matches a preset."""
    model_id_lower = model_id.lower().strip()
    for preset_name, preset_config in MODEL_PRESETS.items():
        preset_lower = preset_name.lower()
        if preset_lower == model_id_lower:
            return preset_config.copy(), preset_config["params_b"]
        if "/" in preset_name and "/" in model_id:
            preset_model_name = preset_name.split("/")[1].lower()
            input_model_name = model_id.split("/")[1].lower()
            if preset_model_name == input_model_name:
                return preset_config.copy(), preset_config["params_b"]
    return None


def resolve_model_config(model_id: str, manual: ManualModelConfig) -> Optional[ResolvedModelConfig]:
    """Merge presets, HuggingFace config, and manual overrides into a single resolved config."""
    config: Optional[dict] = None
    params_b: Optional[float] = None

    hit = _lookup_preset(model_id)
    if hit:
        config, params_b = hit

    if config is None:
        config = fetch_model_config(model_id)
        if config:
            raw_pb = config.get("params_b")
            params_b = raw_pb if raw_pb is not None else estimate_params_from_config(config)

    has_manual = (
        manual.params_b > 0
        and manual.hidden > 0
        and manual.layers > 0
    )

    if config is None and not has_manual:
        return None

    if config is None:
        config = {}

    m = manual

    if m.hidden > 0:
        hidden = int(m.hidden)
    elif config.get("hidden") is not None:
        hidden = config["hidden"]
    else:
        return None

    if m.layers > 0:
        layers = int(m.layers)
    elif config.get("layers") is not None:
        layers = config["layers"]
    else:
        return None

    if m.heads > 0:
        heads = int(m.heads)
    elif config.get("heads") is not None:
        heads = config["heads"]
    else:
        heads = max(1, hidden // 128)

    if m.kv_heads > 0:
        kv_heads = int(m.kv_heads)
    elif config.get("kv_heads") is not None:
        kv_heads = config["kv_heads"]
    else:
        kv_heads = heads

    if m.vocab_size > 0:
        vocab_size = int(m.vocab_size)
    elif config.get("vocab_size") is not None:
        vocab_size = config["vocab_size"]
    else:
        vocab_size = 32000

    if m.intermediate > 0:
        intermediate_size = int(m.intermediate)
    elif config.get("intermediate_size") is not None:
        intermediate_size = config["intermediate_size"]
    else:
        intermediate_size = int(hidden * 3.5)

    if params_b is None:
        params_b = estimate_params_from_config(
            {
                "hidden": hidden,
                "layers": layers,
                "heads": heads,
                "kv_heads": kv_heads,
                "vocab_size": vocab_size,
                "intermediate_size": intermediate_size,
            }
        )

    if m.params_b > 0:
        params_b = m.params_b

    if params_b is None:
        return None

    if m.uses_swiglu == "Yes":
        uses_swiglu = True
    elif m.uses_swiglu == "No":
        uses_swiglu = False
    else:
        uses_swiglu = True

    if m.num_experts > 0:
        num_experts = int(m.num_experts)
    elif config.get("num_experts") is not None:
        num_experts = config["num_experts"]
    else:
        num_experts = 1

    if m.experts_per_token > 0:
        experts_per_token = int(m.experts_per_token)
    elif config.get("experts_per_token") is not None:
        experts_per_token = config["experts_per_token"]
    else:
        experts_per_token = 2 if num_experts > 1 else 1

    is_moe = num_experts > 1

    if m.active_params_b > 0:
        active_params_b = m.active_params_b
    elif config.get("active_params_b") is not None:
        active_params_b = config["active_params_b"]
    elif is_moe:
        expert_ratio = experts_per_token / num_experts
        active_params_b = params_b * (0.33 + 0.67 * expert_ratio)
    else:
        active_params_b = params_b

    return ResolvedModelConfig(
        model_id=model_id,
        params_b=params_b,
        hidden=hidden,
        layers=layers,
        heads=heads,
        kv_heads=kv_heads,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        uses_swiglu=uses_swiglu,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        is_moe=is_moe,
        active_params_b=active_params_b,
    )
