"""VRAM math: modular calculators for weights, activations, and runtime overhead."""

from __future__ import annotations

import math

from vram_calc.config import finalize_architecture
from vram_calc.constants import DTYPE_BYTES
from vram_calc.custom_gpu_store import get_all_gpu_specs
from vram_calc.types import ArchitectureConfig, VRAMEstimate, VRAMInput

BYTES_PER_KIB = 1024
BYTES_PER_GIB = 1024**3
FLASH_ATTN_BLOCK_SIZE = 128


def _weights_gb(params_b: float, bytes_per_param: float) -> float:
    return (params_b * 1e9 * bytes_per_param) / BYTES_PER_GIB


def _is_training(mode: str) -> bool:
    return mode.strip().lower() == "training"


def _is_inference(mode: str) -> bool:
    return mode.strip().lower() == "inference"


def _trainable_params_b(
    params_b: float,
    lora_enabled: bool,
    mode: str,
    lora_rank: int,
    hidden: int,
    layers: int,
) -> float:
    if lora_enabled and _is_training(mode):
        num_target_modules = 7
        lora_params = lora_rank * hidden * 2 * num_target_modules * layers
        return lora_params / 1e9
    return params_b


def _optimizer_states_gb(trainable_params_b: float, optimizer: str) -> float:
    if optimizer == "AdamW (32-bit)":
        return (trainable_params_b * 1e9 * 4 * 2) / BYTES_PER_GIB
    if optimizer == "AdamW (8-bit)":
        return (trainable_params_b * 1e9 * 1 * 2) / BYTES_PER_GIB
    if optimizer == "SGD":
        return (trainable_params_b * 1e9 * 4) / BYTES_PER_GIB
    if optimizer == "Adafactor":
        return (trainable_params_b * 1e9 * 4 * 0.5) / BYTES_PER_GIB
    return (trainable_params_b * 1e9 * 4 * 2) / BYTES_PER_GIB


def calc_weights_and_states(
    arch: ArchitectureConfig, inp: VRAMInput, bytes_per_param: float, bytes_activation: int
) -> tuple[float, float, float, float, float]:
    """Calculate model weights, grads, optimizer states, trainable params, and DDP overhead."""
    resident_params_b = arch.params_b
    model_weights_gb = _weights_gb(resident_params_b, bytes_per_param)
    trainable_params_b = _trainable_params_b(
        arch.params_b, inp.lora_enabled, inp.mode, inp.lora_rank, arch.hidden, arch.layers
    )
    is_mixed_precision = inp.mixed_precision and _is_training(inp.mode)

    gradients_gb = 0.0
    optimizer_states_gb = 0.0
    ddp_overhead_gb = 0.0
    if _is_training(inp.mode):
        if is_mixed_precision:
            gradients_gb = (trainable_params_b * 1e9 * 2) / BYTES_PER_GIB
            master_weights_gb = (trainable_params_b * 1e9 * 4) / BYTES_PER_GIB
        else:
            # Training gradients follow compute precision, not quantized weight storage precision.
            gradients_gb = (trainable_params_b * 1e9 * bytes_activation) / BYTES_PER_GIB
            master_weights_gb = 0.0
        optimizer_states_gb = _optimizer_states_gb(trainable_params_b, inp.optimizer) + master_weights_gb
        if inp.ddp_enabled:
            gradient_buffer_gb = gradients_gb
            ddp_overhead_gb = gradient_buffer_gb + 0.05 + (gradient_buffer_gb * 0.02)
    return model_weights_gb, gradients_gb, optimizer_states_gb, trainable_params_b, ddp_overhead_gb


def calc_attention_memory(arch: ArchitectureConfig, inp: VRAMInput, bytes_activation: int) -> tuple[float, float]:
    """Calculate attention activation memory and per-layer temporary attention bytes."""
    batch_size = inp.batch_size
    seq_length = inp.seq_length
    hidden = arch.hidden
    heads = arch.heads
    kv_heads = arch.kv_heads

    q_proj = batch_size * seq_length * hidden * bytes_activation
    kv_proj = 2 * batch_size * seq_length * hidden * (kv_heads / heads) * bytes_activation
    # Attention score tensors scale with query heads, not KV heads.
    # Vanilla attention materializes an SxS score matrix.
    attn_scores = batch_size * heads * seq_length * seq_length * bytes_activation
    if inp.flash_attention:
        # Flash Attention uses block-wise streaming and avoids full score-matrix residency.
        # Approximate peak score workspace as S x block_size instead of S x S.
        tiled_scores = batch_size * heads * seq_length * min(seq_length, FLASH_ATTN_BLOCK_SIZE) * bytes_activation
        # Keep a small extra buffer for numerically stable running softmax stats.
        running_stats = batch_size * heads * seq_length * 2 * bytes_activation
        attn_scores = tiled_scores + running_stats
    attn_output = batch_size * seq_length * hidden * bytes_activation

    per_layer = q_proj + kv_proj + attn_scores + attn_output
    if _is_inference(inp.mode):
        # Inference peak is modeled as a single-layer working set, not layers-summed residency.
        return per_layer / BYTES_PER_GIB, per_layer

    effective_layers = max(1, int(math.sqrt(arch.layers))) if inp.gradient_checkpointing else arch.layers
    return (effective_layers * per_layer) / BYTES_PER_GIB, per_layer


def calc_ffn_memory(arch: ArchitectureConfig, inp: VRAMInput, bytes_activation: int) -> tuple[float, float]:
    """Calculate FFN activation memory and per-layer temporary FFN bytes."""
    batch_size = inp.batch_size
    seq_length = inp.seq_length
    hidden = arch.hidden
    intermediate = arch.intermediate_size

    if arch.ffn_type == "moe":
        active_ratio = arch.active_params_b / arch.params_b if arch.params_b > 0 else 1.0
        active_ratio = max(0.0, min(1.0, active_ratio))
        effective_intermediate = intermediate * active_ratio
    else:
        effective_intermediate = intermediate

    ffn_input = batch_size * seq_length * hidden * bytes_activation
    per_layer = ffn_input + (batch_size * seq_length * effective_intermediate * bytes_activation)

    if arch.ffn_type == "moe":
        # Router keeps top-k expert scores/indices per token, not full expert logits residency.
        per_layer += batch_size * seq_length * arch.experts_per_token * bytes_activation

    if _is_inference(inp.mode):
        # Inference peak is modeled as a single-layer working set.
        return per_layer / BYTES_PER_GIB, per_layer

    effective_layers = max(1, int(math.sqrt(arch.layers))) if inp.gradient_checkpointing else arch.layers
    return (effective_layers * per_layer) / BYTES_PER_GIB, per_layer


def calc_runtime_overheads(model_weights_gb: float, use_torch_compile: bool) -> tuple[float, float]:
    """Calculate compile and CUDA runtime overheads."""
    compile_overhead_gb = model_weights_gb * 0.1 if use_torch_compile else 0.0
    cuda_overhead_gb = 0.5
    return compile_overhead_gb, cuda_overhead_gb


def compute_vram_estimate(arch: ArchitectureConfig, inp: VRAMInput) -> VRAMEstimate:
    """Compute VRAM estimate from architecture and runtime input."""
    gpu_specs = get_all_gpu_specs()
    if inp.gpu_name not in gpu_specs:
        raise KeyError(f"Unknown GPU: {inp.gpu_name}")
    total_vram_gb = gpu_specs[inp.gpu_name]["vram_gb"]

    bytes_per_param = DTYPE_BYTES[inp.dtype]
    # Runtime activations/KV cache are modeled at compute precision:
    # FP32 runs at 4 bytes; quantized weight formats still keep activations/cache at 2 bytes.
    bytes_activation = 4 if bytes_per_param >= 4 else 2

    model_weights_gb, gradients_gb, optimizer_states_gb, trainable_params_b, ddp_overhead_gb = calc_weights_and_states(
        arch, inp, bytes_per_param, bytes_activation
    )
    attn_activations_gb, per_layer_attn = calc_attention_memory(arch, inp, bytes_activation)
    ffn_activations_gb, per_layer_ffn = calc_ffn_memory(arch, inp, bytes_activation)

    layernorm_act = 2 * inp.batch_size * inp.seq_length * arch.hidden * bytes_activation
    residual_act = 2 * inp.batch_size * inp.seq_length * arch.hidden * bytes_activation
    per_layer_other = layernorm_act + residual_act
    if _is_training(inp.mode):
        effective_layers = max(1, int(math.sqrt(arch.layers))) if inp.gradient_checkpointing else arch.layers
        other_activations_gb = (effective_layers * per_layer_other) / BYTES_PER_GIB
        other_activations_gb += (inp.batch_size * inp.seq_length * arch.hidden * bytes_activation) / BYTES_PER_GIB
    else:
        other_activations_gb = per_layer_other / BYTES_PER_GIB

    activations_gb = attn_activations_gb + ffn_activations_gb + other_activations_gb
    forward_pass_gb = activations_gb
    backward_pass_gb = (
        gradients_gb + ((per_layer_attn + per_layer_ffn + per_layer_other) / BYTES_PER_GIB)
        if _is_training(inp.mode)
        else 0.0
    )

    head_dim = arch.hidden // arch.heads if arch.heads > 0 else 128
    kv_cache_per_token_bytes = 2 * inp.batch_size * arch.layers * arch.kv_heads * head_dim * bytes_activation
    kv_cache_per_token_kb = kv_cache_per_token_bytes / BYTES_PER_KIB
    attention_workspace_per_layer_mb = per_layer_attn / (1024**2)
    kv_cache_gb = (
        (kv_cache_per_token_bytes * inp.seq_length) / BYTES_PER_GIB if _is_inference(inp.mode) else 0.0
    )

    compile_overhead_gb, cuda_overhead_gb = calc_runtime_overheads(model_weights_gb, inp.use_torch_compile)

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
    fits = total_gb <= total_vram_gb
    utilization_pct = (total_gb / total_vram_gb) * 100

    is_moe = arch.ffn_type == "moe"
    attn_breakdown_key = "Activations (Attn)"
    breakdown = {
        "Model Weights": model_weights_gb,
        "Gradients": gradients_gb,
        "Optimizer States": optimizer_states_gb,
        attn_breakdown_key: attn_activations_gb,
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
        peak_gb=total_gb,
        available_gb=total_vram_gb,
        fits=fits,
        utilization_pct=utilization_pct,
        breakdown=breakdown,
        model_params_b=arch.params_b,
        trainable_params_b=trainable_params_b,
        attn_activations_gb=attn_activations_gb,
        ffn_activations_gb=ffn_activations_gb,
        other_activations_gb=other_activations_gb,
        forward_pass_gb=forward_pass_gb,
        backward_pass_gb=backward_pass_gb,
        ddp_overhead_gb=ddp_overhead_gb,
        config_hidden=arch.hidden,
        config_layers=arch.layers,
        config_heads=arch.heads,
        config_kv_heads=arch.kv_heads,
        is_moe=is_moe,
        num_experts=arch.num_experts,
        experts_per_token=arch.experts_per_token,
        active_params_b=arch.active_params_b,
        kv_cache_per_token_kb=kv_cache_per_token_kb,
        attention_workspace_per_layer_mb=attention_workspace_per_layer_mb,
        attention_type=arch.attention_type,
        ffn_type=arch.ffn_type,
    )


def estimate_vram(inp: VRAMInput) -> VRAMEstimate:
    """Validate architecture and return VRAM estimate."""
    architecture = finalize_architecture(inp.architecture)
    return compute_vram_estimate(architecture, inp)
