"""VRAM math: weights, gradients, optimizer, activations, KV cache, totals."""

from __future__ import annotations

import math
from typing import Optional

from vram_calc.config import resolve_model_config
from vram_calc.constants import DTYPE_BYTES, GPU_SPECS
from vram_calc.types import VRAMEstimate, VRAMInput, ResolvedModelConfig


def _weights_gb(params_b: float, bytes_per_param: float) -> float:
    return (params_b * 1e9 * bytes_per_param) / (1024**3)


def _trainable_params_b(
    params_b: float,
    lora_enabled: bool,
    mode: str,
    lora_rank: int,
    hidden: int,
    layers: int,
) -> float:
    if lora_enabled and mode == "Training":
        num_target_modules = 7
        lora_params = lora_rank * hidden * 2 * num_target_modules * layers
        return lora_params / 1e9
    return params_b


def _optimizer_states_gb(trainable_params_b: float, optimizer: str) -> float:
    if optimizer == "AdamW (32-bit)":
        return (trainable_params_b * 1e9 * 4 * 2) / (1024**3)
    if optimizer == "AdamW (8-bit)":
        return (trainable_params_b * 1e9 * 1 * 2) / (1024**3)
    if optimizer == "SGD":
        return (trainable_params_b * 1e9 * 4) / (1024**3)
    if optimizer == "Adafactor":
        return (trainable_params_b * 1e9 * 4 * 0.5) / (1024**3)
    return (trainable_params_b * 1e9 * 4 * 2) / (1024**3)


def compute_vram_estimate(resolved: ResolvedModelConfig, inp: VRAMInput) -> VRAMEstimate:
    """Compute VRAM estimate from resolved architecture and runtime input."""
    gpu_spec = GPU_SPECS[inp.gpu_name]
    total_vram_gb = gpu_spec["vram_gb"]

    hidden = resolved.hidden
    layers = resolved.layers
    heads = resolved.heads
    kv_heads = resolved.kv_heads
    vocab_size = resolved.vocab_size
    intermediate_size = resolved.intermediate_size
    params_b = resolved.params_b
    uses_swiglu = resolved.uses_swiglu
    num_experts = resolved.num_experts
    experts_per_token = resolved.experts_per_token
    is_moe = resolved.is_moe
    active_params_b = resolved.active_params_b

    head_dim = hidden // heads if heads > 0 else 128
    bytes_per_param = DTYPE_BYTES[inp.dtype]
    is_mixed_precision = inp.mixed_precision and inp.mode == "Training"
    batch_size = inp.batch_size
    seq_length = inp.seq_length
    mode = inp.mode

    model_weights_gb = _weights_gb(params_b, bytes_per_param)

    trainable_params_b = _trainable_params_b(
        params_b, inp.lora_enabled, mode, inp.lora_rank, hidden, layers
    )

    master_weights_gb = 0.0
    if mode == "Training":
        if is_mixed_precision:
            gradients_gb = (trainable_params_b * 1e9 * 2) / (1024**3)
            master_weights_gb = (trainable_params_b * 1e9 * 4) / (1024**3)
        else:
            gradients_gb = (trainable_params_b * 1e9 * bytes_per_param) / (1024**3)
    else:
        gradients_gb = 0.0

    if mode == "Training":
        optimizer_states_gb = _optimizer_states_gb(trainable_params_b, inp.optimizer)
        optimizer_states_gb += master_weights_gb
    else:
        optimizer_states_gb = 0.0

    ddp_overhead_gb = 0.0
    if mode == "Training" and inp.ddp_enabled:
        if is_mixed_precision:
            gradient_buffer_gb = (trainable_params_b * 1e9 * 2) / (1024**3)
        else:
            gradient_buffer_gb = (trainable_params_b * 1e9 * bytes_per_param) / (1024**3)
        bucket_overhead_gb = 0.05 + (gradient_buffer_gb * 0.02)
        ddp_overhead_gb = gradient_buffer_gb + bucket_overhead_gb

    bytes_activation = 2

    if mode == "Training":
        if inp.gradient_checkpointing:
            effective_layers = max(1, int(math.sqrt(layers)))
        else:
            effective_layers = layers

        attn_input_act = batch_size * seq_length * hidden * bytes_activation
        qkv_act = 3 * batch_size * seq_length * hidden * bytes_activation
        attn_scores_act = batch_size * heads * seq_length * seq_length * bytes_activation
        attn_output_act = batch_size * seq_length * hidden * bytes_activation

        per_layer_attn_act = attn_input_act + qkv_act + attn_scores_act + attn_output_act
        attn_activations_gb = (effective_layers * per_layer_attn_act) / (1024**3)

        ffn_input_act = batch_size * seq_length * hidden * bytes_activation
        effective_intermediate = intermediate_size * experts_per_token if is_moe else intermediate_size

        if uses_swiglu:
            ffn_gate_act = batch_size * seq_length * effective_intermediate * bytes_activation
            ffn_up_act = batch_size * seq_length * effective_intermediate * bytes_activation
            ffn_intermediate_act = batch_size * seq_length * effective_intermediate * bytes_activation
            per_layer_ffn_act = ffn_input_act + ffn_gate_act + ffn_up_act + ffn_intermediate_act
        else:
            ffn_up_act = batch_size * seq_length * effective_intermediate * bytes_activation
            per_layer_ffn_act = ffn_input_act + ffn_up_act

        if is_moe:
            router_act = batch_size * seq_length * num_experts * bytes_activation
            per_layer_ffn_act += router_act

        ffn_activations_gb = (effective_layers * per_layer_ffn_act) / (1024**3)

        layernorm_act = 2 * batch_size * seq_length * hidden * bytes_activation
        residual_act = 2 * batch_size * seq_length * hidden * bytes_activation
        per_layer_other_act = layernorm_act + residual_act
        other_activations_gb = (effective_layers * per_layer_other_act) / (1024**3)

        embedding_act = batch_size * seq_length * hidden * bytes_activation
        other_activations_gb += embedding_act / (1024**3)

        activations_gb = attn_activations_gb + ffn_activations_gb + other_activations_gb
        forward_pass_gb = activations_gb
        backward_gradient_temp = (per_layer_attn_act + per_layer_ffn_act + per_layer_other_act) / (1024**3)
        backward_pass_gb = gradients_gb + backward_gradient_temp
    else:
        attn_activations_gb = (batch_size * seq_length * hidden * 2) / (1024**3)
        ffn_activations_gb = (batch_size * seq_length * intermediate_size * 2) / (1024**3)
        other_activations_gb = (batch_size * seq_length * hidden * 2) / (1024**3)
        activations_gb = attn_activations_gb + ffn_activations_gb + other_activations_gb
        forward_pass_gb = activations_gb
        backward_pass_gb = 0.0

    if mode == "Inference":
        kv_cache_bytes = 2 * batch_size * layers * seq_length * kv_heads * head_dim * bytes_per_param
        kv_cache_gb = kv_cache_bytes / (1024**3)
    else:
        kv_cache_gb = 0.0

    kv_cache_per_token_bytes = 2 * batch_size * layers * kv_heads * head_dim * bytes_per_param
    kv_cache_per_token_kb = kv_cache_per_token_bytes / 1024
    memory_per_token_kb = kv_cache_per_token_kb

    if inp.use_torch_compile:
        compile_overhead_gb = model_weights_gb * 0.1
    else:
        compile_overhead_gb = 0.0

    cuda_overhead_gb = 0.5

    total_gb = (
        model_weights_gb
        + gradients_gb
        + optimizer_states_gb
        + activations_gb
        + kv_cache_gb
        + compile_overhead_gb
        + cuda_overhead_gb
        + ddp_overhead_gb
    )
    peak_gb = total_gb
    fits = peak_gb <= total_vram_gb
    utilization_pct = (peak_gb / total_vram_gb) * 100

    breakdown = {
        "Model Weights": model_weights_gb,
        "Gradients": gradients_gb,
        "Optimizer States": optimizer_states_gb,
        "Activations (Attn)": attn_activations_gb,
        "Activations (FFN)": ffn_activations_gb,
        "Activations (Other)": other_activations_gb,
        "KV Cache": kv_cache_gb,
        "DDP Overhead": ddp_overhead_gb,
        "torch.compile": compile_overhead_gb,
        "CUDA Overhead": cuda_overhead_gb,
    }

    return VRAMEstimate(
        model_weights_gb=model_weights_gb,
        gradients_gb=gradients_gb,
        optimizer_states_gb=optimizer_states_gb,
        activations_gb=activations_gb,
        kv_cache_gb=kv_cache_gb,
        compile_overhead_gb=compile_overhead_gb,
        cuda_overhead_gb=cuda_overhead_gb,
        total_gb=total_gb,
        peak_gb=peak_gb,
        available_gb=total_vram_gb,
        fits=fits,
        utilization_pct=utilization_pct,
        breakdown=breakdown,
        model_params_b=params_b,
        trainable_params_b=trainable_params_b,
        attn_activations_gb=attn_activations_gb,
        ffn_activations_gb=ffn_activations_gb,
        other_activations_gb=other_activations_gb,
        forward_pass_gb=forward_pass_gb,
        backward_pass_gb=backward_pass_gb,
        uses_swiglu=uses_swiglu,
        ddp_overhead_gb=ddp_overhead_gb,
        config_hidden=hidden,
        config_layers=layers,
        config_heads=heads,
        config_kv_heads=kv_heads,
        config_intermediate=intermediate_size,
        config_vocab_size=vocab_size,
        is_moe=is_moe,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        active_params_b=active_params_b,
        memory_per_token_kb=memory_per_token_kb,
        kv_cache_per_token_kb=kv_cache_per_token_kb,
    )


def estimate_vram(inp: VRAMInput) -> Optional[VRAMEstimate]:
    """Resolve model architecture from ``inp`` and return a VRAM estimate, or ``None`` if unresolved."""
    resolved = resolve_model_config(inp.model_id, inp.manual)
    if resolved is None:
        return None
    return compute_vram_estimate(resolved, inp)
