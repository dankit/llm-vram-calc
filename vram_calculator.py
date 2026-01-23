"""
LLM VRAM Calculator

A comprehensive tool for estimating GPU memory requirements for Large Language Model
training and inference. Supports dense transformers and Mixture of Experts (MoE) 
architectures with detailed memory breakdowns.

Features:
- Accurate VRAM estimation for training (full fine-tuning, LoRA) and inference
- Support for MoE models (Mixtral, DeepSeek) with proper active parameter calculations
- Detailed breakdown: model weights, gradients, optimizer states, activations, KV cache
- Mixed precision training support (FP16, BF16, INT8, INT4)
- Gradient checkpointing, DDP overhead, and torch.compile estimation
- Auto-fetch model configs from HuggingFace or use built-in presets
- Interactive Gradio web interface with real-time calculations
"""

import gradio as gr
import math
from dataclasses import dataclass
from typing import Optional
import requests


# ============================================================================
# GPU SPECIFICATIONS DATABASE
# ============================================================================

GPU_SPECS = {
    "NVIDIA RTX 3060 Ti 8GB": {"vram_gb": 8, "bandwidth_gbps": 448},
    "NVIDIA H200 141GB": {"vram_gb": 141, "bandwidth_gbps": 4800},
    "NVIDIA H100 NVL 94GB": {"vram_gb": 94, "bandwidth_gbps": 3958},
    "NVIDIA H100 80GB": {"vram_gb": 80, "bandwidth_gbps": 3350},
    "NVIDIA A100 80GB": {"vram_gb": 80, "bandwidth_gbps": 2039},
    "NVIDIA A100 40GB": {"vram_gb": 40, "bandwidth_gbps": 1555},
    "NVIDIA L40S 48GB": {"vram_gb": 48, "bandwidth_gbps": 864},
    "NVIDIA A6000 48GB": {"vram_gb": 48, "bandwidth_gbps": 768},
    "NVIDIA RTX 4090 24GB": {"vram_gb": 24, "bandwidth_gbps": 1008},
    "NVIDIA RTX 3090 24GB": {"vram_gb": 24, "bandwidth_gbps": 936},
    "NVIDIA V100 32GB": {"vram_gb": 32, "bandwidth_gbps": 900},
    "NVIDIA V100 16GB": {"vram_gb": 16, "bandwidth_gbps": 900},
    "NVIDIA A10G 24GB": {"vram_gb": 24, "bandwidth_gbps": 600},
    "NVIDIA T4 16GB": {"vram_gb": 16, "bandwidth_gbps": 300},
}

# ============================================================================
# DTYPE SPECIFICATIONS
# ============================================================================

DTYPE_BYTES = {
    "bfloat16 (BF16)": 2,      # Recommended for training on modern GPUs
    "float16 (FP16)": 2,       # For older GPUs (V100, T4) without native BF16
    "float32 (FP32)": 4,
    "int8 (8-bit Quantized)": 1,
    "int4 (4-bit Quantized)": 0.5
}

# ============================================================================
# MODEL PRESETS (Common models with known parameters)
# ============================================================================

MODEL_PRESETS = {
    # Llama family
    "meta-llama/Llama-3.2-1B": {"params_b": 1.24, "hidden": 2048, "layers": 16, "heads": 32, "kv_heads": 8},
    "meta-llama/Llama-3.2-3B": {"params_b": 3.21, "hidden": 3072, "layers": 28, "heads": 24, "kv_heads": 8},
    "meta-llama/Llama-3.1-8B": {"params_b": 8.03, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 8},
    "meta-llama/Llama-3.1-70B": {"params_b": 70.6, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 8},
    "meta-llama/Llama-3.3-70B": {"params_b": 70.6, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 8},
    
    # Mistral family
    "mistralai/Mistral-7B-v0.3": {"params_b": 7.25, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 8},
    "mistralai/Mixtral-8x7B-v0.1": {"params_b": 46.7, "active_params_b": 12.9, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 8, "num_experts": 8, "experts_per_token": 2},
    "mistralai/Mixtral-8x22B-v0.1": {"params_b": 141, "active_params_b": 39, "hidden": 6144, "layers": 56, "heads": 48, "kv_heads": 8, "num_experts": 8, "experts_per_token": 2},
    
    # Qwen family
    "Qwen/Qwen2.5-0.5B": {"params_b": 0.49, "hidden": 896, "layers": 24, "heads": 14, "kv_heads": 2},
    "Qwen/Qwen2.5-1.5B": {"params_b": 1.54, "hidden": 1536, "layers": 28, "heads": 12, "kv_heads": 2},
    "Qwen/Qwen2.5-7B": {"params_b": 7.62, "hidden": 3584, "layers": 28, "heads": 28, "kv_heads": 4},
    "Qwen/Qwen2.5-14B": {"params_b": 14.8, "hidden": 5120, "layers": 48, "heads": 40, "kv_heads": 8},
    "Qwen/Qwen2.5-72B": {"params_b": 72.7, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 8},
    
    # Phi family
    "microsoft/phi-3-mini-4k-instruct": {"params_b": 3.82, "hidden": 3072, "layers": 32, "heads": 32, "kv_heads": 32},
    "microsoft/Phi-3.5-mini-instruct": {"params_b": 3.82, "hidden": 3072, "layers": 32, "heads": 32, "kv_heads": 32},
    
    # Gemma family
    "google/gemma-2-2b": {"params_b": 2.61, "hidden": 2304, "layers": 26, "heads": 8, "kv_heads": 4},
    "google/gemma-2-9b": {"params_b": 9.24, "hidden": 3584, "layers": 42, "heads": 16, "kv_heads": 8},
    "google/gemma-2-27b": {"params_b": 27.2, "hidden": 4608, "layers": 46, "heads": 32, "kv_heads": 16},
    
    # DeepSeek (MoE architecture)
    "deepseek-ai/DeepSeek-V2-Lite": {"params_b": 16, "active_params_b": 2.4, "hidden": 2048, "layers": 27, "heads": 16, "kv_heads": 16, "num_experts": 64, "experts_per_token": 6},
    "deepseek-ai/DeepSeek-Coder-V2-Instruct": {"params_b": 236, "active_params_b": 21, "hidden": 5120, "layers": 60, "heads": 128, "kv_heads": 128, "num_experts": 160, "experts_per_token": 6},
}


# ============================================================================
# VRAM CALCULATION ENGINE
# ============================================================================

@dataclass
class VRAMEstimate:
    """Container for VRAM estimation results."""
    model_weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    kv_cache_gb: float
    compile_overhead_gb: float
    cuda_overhead_gb: float
    total_gb: float
    peak_gb: float
    available_gb: float
    fits: bool
    utilization_pct: float
    breakdown: dict
    model_params_b: float
    trainable_params_b: float
    # Detailed activation breakdown
    attn_activations_gb: float = 0.0
    ffn_activations_gb: float = 0.0
    other_activations_gb: float = 0.0
    # Forward vs backward breakdown
    forward_pass_gb: float = 0.0
    backward_pass_gb: float = 0.0
    # Whether model uses SwiGLU
    uses_swiglu: bool = True
    # DDP overhead
    ddp_overhead_gb: float = 0.0
    # Detected/used architecture config
    config_hidden: int = 0
    config_layers: int = 0
    config_heads: int = 0
    config_kv_heads: int = 0
    config_intermediate: int = 0
    config_vocab_size: int = 0
    # MoE (Mixture of Experts) fields
    is_moe: bool = False
    num_experts: int = 1
    experts_per_token: int = 1
    active_params_b: float = 0.0  # Active params per forward pass (for MoE)
    # Memory per token metrics
    memory_per_token_kb: float = 0.0  # KB of memory per token added to sequence
    kv_cache_per_token_kb: float = 0.0  # KV cache memory per token (inference)


def fetch_model_config(model_id: str) -> Optional[dict]:
    """Fetch model configuration from HuggingFace.
    
    Attempts to retrieve config.json from the model repository.
    Returns None if config doesn't exist or is missing essential fields.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B")
    
    Returns:
        Dictionary with model configuration or None if unavailable.
    """
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        config = response.json()
        
        # Extract hidden size - REQUIRED
        hidden = config.get("hidden_size") or config.get("d_model")
        
        # Extract layers - REQUIRED
        layers = config.get("num_hidden_layers") or config.get("n_layer")
        
        # If essential fields are missing, return None entirely
        if hidden is None or layers is None:
            return None
        
        # Extract heads
        heads = config.get("num_attention_heads") or config.get("n_head")
        
        # KV heads can default to heads (MHA) if not specified
        kv_heads = config.get("num_key_value_heads") or heads
        
        # Vocab size and intermediate are less critical
        vocab_size = config.get("vocab_size") or 32000
        intermediate_size = config.get("intermediate_size")
        
        # Detect MoE (Mixture of Experts) configuration
        # Different model families use different config keys
        num_experts = (
            config.get("num_local_experts") or  # Mixtral, Qwen-MoE
            config.get("num_experts") or  # DeepSeek, other MoE
            config.get("n_routed_experts") or  # DeepSeek-V2
            1
        )
        experts_per_token = (
            config.get("num_experts_per_tok") or  # Mixtral
            config.get("num_experts_per_token") or  # Alternative naming
            config.get("top_k") or  # Some MoE configs
            config.get("topk_group") or  # DeepSeek
            (2 if num_experts > 1 else 1)  # Default: 2 for MoE, 1 otherwise
        )
        
        result = {
            "params_b": config.get("num_parameters", 0) / 1e9 if "num_parameters" in config else None,
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "kv_heads": kv_heads,
            "vocab_size": vocab_size,
            "intermediate_size": intermediate_size,
        }
        
        # Add MoE fields if detected
        if num_experts > 1:
            result["num_experts"] = num_experts
            result["experts_per_token"] = experts_per_token
        
        return result
    except Exception:
        return None


def estimate_params_from_config(config: dict) -> Optional[float]:
    """Estimate total parameters from model architecture.
    
    Uses standard transformer parameter formulas to estimate total parameters
    when not explicitly provided in the config.
    
    Args:
        config: Dictionary containing model architecture parameters.
    
    Returns:
        Estimated parameters in billions, or None if essential config is missing.
    """
    hidden = config.get("hidden")
    layers = config.get("layers")
    
    # Can't estimate without hidden dim and layers
    if hidden is None or layers is None:
        return None
    
    vocab_size = config.get("vocab_size") or 32000
    intermediate = config.get("intermediate_size") or int(hidden * 3.5)
    heads = config.get("heads") or max(1, hidden // 128)
    kv_heads = config.get("kv_heads") or heads
    
    head_dim = hidden // heads if heads > 0 else 128
    
    # Embedding + LM head
    embedding_params = vocab_size * hidden * 2
    
    # Per-layer params
    q_params = hidden * hidden
    k_params = hidden * (head_dim * kv_heads)
    v_params = hidden * (head_dim * kv_heads)
    o_params = hidden * hidden
    attn_params = q_params + k_params + v_params + o_params
    
    # MLP (assuming SwiGLU-like: gate, up, down)
    mlp_params = hidden * intermediate * 3
    
    # Layer norms (2 per layer)
    ln_params = hidden * 4
    
    total_per_layer = attn_params + mlp_params + ln_params
    total_params = embedding_params + (layers * total_per_layer)
    
    return total_params / 1e9


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
    # Manual config overrides (0 or None means use auto-detected value)
    manual_params_b: float = 0,
    manual_hidden: int = 0,
    manual_layers: int = 0,
    manual_heads: int = 0,
    manual_kv_heads: int = 0,
    manual_intermediate: int = 0,
    manual_vocab_size: int = 0,
    manual_uses_swiglu: str = "Auto",
    # MoE (Mixture of Experts) manual config
    manual_num_experts: int = 0,
    manual_experts_per_token: int = 0,
    manual_active_params_b: float = 0,
) -> Optional[VRAMEstimate]:
    """Calculate comprehensive VRAM requirements for a single GPU.
    
    This is the core calculation engine. It computes memory requirements for:
    - Model weights (all parameters must be loaded)
    - Gradients (training only, scales with trainable params)
    - Optimizer states (AdamW, SGD, Adafactor with correct state sizes)
    - Activations (attention, FFN, with SwiGLU support)
    - KV cache (inference only)
    - DDP overhead (multi-GPU gradient synchronization)
    - torch.compile overhead (graph compilation memory)
    - Mixed precision training (optional FP32 master weights)
    
    For MoE models, properly handles:
    - Total params (all experts loaded) vs active params (per forward pass)
    - Activation scaling based on experts_per_token
    - Router/gating activation overhead
    
    Args:
        model_id: HuggingFace model identifier or preset name
        gpu_name: GPU model name from GPU_SPECS
        mode: "Training" or "Inference"
        dtype: Data type for model weights
        batch_size: Training/inference batch size
        seq_length: Sequence length in tokens
        gradient_checkpointing: Whether to use activation checkpointing
        optimizer: Optimizer type for training
        lora_rank: LoRA adapter rank (if enabled)
        lora_enabled: Whether LoRA is enabled
        use_torch_compile: Whether torch.compile is used
        ddp_enabled: Whether DistributedDataParallel is used
        mixed_precision: Whether to use FP32 master weights (for FP16/BF16 training)
        manual_*: Manual config overrides (0 means use auto-detected)
    
    Returns:
        VRAMEstimate dataclass with detailed breakdown, or None if config unavailable.
    """
    
    # Get GPU specs
    gpu_spec = GPU_SPECS[gpu_name]
    total_vram_gb = gpu_spec["vram_gb"]
    
    # Get model config
    config = None
    params_b = None
    
    # Check presets first - require exact match or the model name part to match exactly
    model_id_lower = model_id.lower().strip()
    for preset_name, preset_config in MODEL_PRESETS.items():
        preset_lower = preset_name.lower()
        
        # Exact match
        if preset_lower == model_id_lower:
            config = preset_config.copy()
            params_b = preset_config["params_b"]
            break
        
        # Match if the model name part (after /) matches exactly
        if "/" in preset_name and "/" in model_id:
            preset_model_name = preset_name.split("/")[1].lower()
            input_model_name = model_id.split("/")[1].lower()
            if preset_model_name == input_model_name:
                config = preset_config.copy()
                params_b = preset_config["params_b"]
                break
    
    # Try fetching from HuggingFace if not in presets
    if config is None:
        config = fetch_model_config(model_id)
        if config:
            params_b = config.get("params_b") or estimate_params_from_config(config)
    
    # Check if we have enough manual values to proceed without auto-detection
    has_manual_config = (
        manual_params_b and manual_params_b > 0 and
        manual_hidden and manual_hidden > 0 and
        manual_layers and manual_layers > 0
    )
    
    # If no config found and no manual values, return None
    if config is None and not has_manual_config:
        return None
    
    # Use manual values or fall back to config values
    if config is None:
        config = {}
    
    # Apply manual overrides first, then fall back to config values
    if manual_hidden and manual_hidden > 0:
        hidden = int(manual_hidden)
    elif config.get("hidden") is not None:
        hidden = config["hidden"]
    else:
        return None  # Can't proceed without hidden dim
    
    if manual_layers and manual_layers > 0:
        layers = int(manual_layers)
    elif config.get("layers") is not None:
        layers = config["layers"]
    else:
        return None  # Can't proceed without layers
    
    if manual_heads and manual_heads > 0:
        heads = int(manual_heads)
    elif config.get("heads") is not None:
        heads = config["heads"]
    else:
        heads = max(1, hidden // 128)
    
    if manual_kv_heads and manual_kv_heads > 0:
        kv_heads = int(manual_kv_heads)
    elif config.get("kv_heads") is not None:
        kv_heads = config["kv_heads"]
    else:
        kv_heads = heads  # Default to MHA
    
    if manual_vocab_size and manual_vocab_size > 0:
        vocab_size = int(manual_vocab_size)
    elif config.get("vocab_size") is not None:
        vocab_size = config["vocab_size"]
    else:
        vocab_size = 32000
    
    if manual_intermediate and manual_intermediate > 0:
        intermediate_size = int(manual_intermediate)
    elif config.get("intermediate_size") is not None:
        intermediate_size = config["intermediate_size"]
    else:
        intermediate_size = int(hidden * 3.5)
    
    # Estimate params if not provided
    if params_b is None:
        params_b = estimate_params_from_config({
            "hidden": hidden,
            "layers": layers,
            "heads": heads,
            "kv_heads": kv_heads,
            "vocab_size": vocab_size,
            "intermediate_size": intermediate_size,
        })
    
    # Apply manual params override if provided
    if manual_params_b and manual_params_b > 0:
        params_b = manual_params_b
    
    head_dim = hidden // heads if heads > 0 else 128
    
    bytes_per_param = DTYPE_BYTES[dtype]
    
    is_mixed_precision = mixed_precision and mode == "Training"
    
    # Detect SwiGLU (most modern LLMs use it)
    if manual_uses_swiglu == "Yes":
        uses_swiglu = True
    elif manual_uses_swiglu == "No":
        uses_swiglu = False
    else:
        uses_swiglu = True  # Default assumption for modern LLMs
    
    # =========================================================================
    # MoE (Mixture of Experts) Configuration
    # =========================================================================
    if manual_num_experts and manual_num_experts > 0:
        num_experts = int(manual_num_experts)
    elif config.get("num_experts") is not None:
        num_experts = config["num_experts"]
    else:
        num_experts = 1
    
    if manual_experts_per_token and manual_experts_per_token > 0:
        experts_per_token = int(manual_experts_per_token)
    elif config.get("experts_per_token") is not None:
        experts_per_token = config["experts_per_token"]
    else:
        experts_per_token = 2 if num_experts > 1 else 1
    
    is_moe = num_experts > 1
    
    # Calculate active parameters for MoE models
    if manual_active_params_b and manual_active_params_b > 0:
        active_params_b = manual_active_params_b
    elif config.get("active_params_b") is not None:
        active_params_b = config["active_params_b"]
    elif is_moe:
        # Estimate: ~33% non-FFN params + 67% FFN params scaled by expert ratio
        expert_ratio = experts_per_token / num_experts
        active_params_b = params_b * (0.33 + 0.67 * expert_ratio)
    else:
        active_params_b = params_b
    
    # =========================================================================
    # 1. MODEL WEIGHTS
    # =========================================================================
    # All experts need to be loaded (full params_b)
    model_weights_gb = (params_b * 1e9 * bytes_per_param) / (1024**3)
    
    # Determine trainable parameters
    if lora_enabled and mode == "Training":
        # LoRA params: rank * (input_dim + output_dim) * num_target_modules * layers
        num_target_modules = 7  # q, k, v, o, gate, up, down
        lora_params = lora_rank * hidden * 2 * num_target_modules * layers
        trainable_params_b = lora_params / 1e9
    else:
        # Full fine-tuning: gradients and optimizer states need ALL params
        # Even for MoE, all expert weights are updated over a training step
        trainable_params_b = params_b
    
    # =========================================================================
    # 2. GRADIENTS (Training only)
    # =========================================================================
    master_weights_gb = 0.0
    if mode == "Training":
        if is_mixed_precision:
            # Mixed precision: gradients in fp16/bf16, master weights in fp32
            gradients_gb = (trainable_params_b * 1e9 * 2) / (1024**3)
            master_weights_gb = (trainable_params_b * 1e9 * 4) / (1024**3)
        else:
            gradients_gb = (trainable_params_b * 1e9 * bytes_per_param) / (1024**3)
            master_weights_gb = 0.0
    else:
        gradients_gb = 0
    
    # =========================================================================
    # 3. OPTIMIZER STATES (Training only)
    # =========================================================================
    if mode == "Training":
        if optimizer == "AdamW (32-bit)":
            # AdamW: momentum (m) + variance (v) = 2x params in fp32
            optimizer_states_gb = (trainable_params_b * 1e9 * 4 * 2) / (1024**3)
        elif optimizer == "AdamW (8-bit)":
            # 8-bit Adam: m and v in int8
            optimizer_states_gb = (trainable_params_b * 1e9 * 1 * 2) / (1024**3)
        elif optimizer == "SGD":
            # SGD with momentum: just momentum buffer in fp32
            optimizer_states_gb = (trainable_params_b * 1e9 * 4) / (1024**3)
        elif optimizer == "Adafactor":
            # Adafactor: factored second moments ~ 0.5x
            optimizer_states_gb = (trainable_params_b * 1e9 * 4 * 0.5) / (1024**3)
        else:
            optimizer_states_gb = (trainable_params_b * 1e9 * 4 * 2) / (1024**3)
        
        # Add master weights for mixed precision
        optimizer_states_gb += master_weights_gb
    else:
        optimizer_states_gb = 0
    
    # =========================================================================
    # 3.5. DDP OVERHEAD (Training only, multi-GPU)
    # =========================================================================
    ddp_overhead_gb = 0.0
    if mode == "Training" and ddp_enabled:
        # Gradient buffers for all-reduce
        if is_mixed_precision:
            gradient_buffer_gb = (trainable_params_b * 1e9 * 2) / (1024**3)
        else:
            gradient_buffer_gb = (trainable_params_b * 1e9 * bytes_per_param) / (1024**3)
        
        # Bucket overhead (~2% for bucket management)
        bucket_overhead_gb = 0.05 + (gradient_buffer_gb * 0.02)
        
        ddp_overhead_gb = gradient_buffer_gb + bucket_overhead_gb
    
    # =========================================================================
    # 4. ACTIVATIONS
    # =========================================================================
    bytes_activation = 2  # Mixed precision for activations
    
    if mode == "Training":
        if gradient_checkpointing:
            effective_layers = max(1, int(math.sqrt(layers)))
        else:
            effective_layers = layers
        
        # Attention activations
        attn_input_act = batch_size * seq_length * hidden * bytes_activation
        qkv_act = 3 * batch_size * seq_length * hidden * bytes_activation
        attn_scores_act = batch_size * heads * seq_length * seq_length * bytes_activation
        attn_output_act = batch_size * seq_length * hidden * bytes_activation
        
        per_layer_attn_act = attn_input_act + qkv_act + attn_scores_act + attn_output_act
        attn_activations_gb = (effective_layers * per_layer_attn_act) / (1024**3)
        
        # FFN activations
        ffn_input_act = batch_size * seq_length * hidden * bytes_activation
        
        # For MoE, only active experts contribute to activation memory
        effective_intermediate = intermediate_size * experts_per_token if is_moe else intermediate_size
        
        if uses_swiglu:
            # SwiGLU: gate, up, and gate*up intermediate
            ffn_gate_act = batch_size * seq_length * effective_intermediate * bytes_activation
            ffn_up_act = batch_size * seq_length * effective_intermediate * bytes_activation
            ffn_intermediate_act = batch_size * seq_length * effective_intermediate * bytes_activation
            per_layer_ffn_act = ffn_input_act + ffn_gate_act + ffn_up_act + ffn_intermediate_act
        else:
            ffn_up_act = batch_size * seq_length * effective_intermediate * bytes_activation
            per_layer_ffn_act = ffn_input_act + ffn_up_act
        
        # MoE router activations
        if is_moe:
            router_act = batch_size * seq_length * num_experts * bytes_activation
            per_layer_ffn_act += router_act
        
        ffn_activations_gb = (effective_layers * per_layer_ffn_act) / (1024**3)
        
        # Other activations (layernorm, residuals)
        layernorm_act = 2 * batch_size * seq_length * hidden * bytes_activation
        residual_act = 2 * batch_size * seq_length * hidden * bytes_activation
        per_layer_other_act = layernorm_act + residual_act
        other_activations_gb = (effective_layers * per_layer_other_act) / (1024**3)
        
        # Embedding activations
        embedding_act = batch_size * seq_length * hidden * bytes_activation
        other_activations_gb += embedding_act / (1024**3)
        
        activations_gb = attn_activations_gb + ffn_activations_gb + other_activations_gb
        
        # Forward/backward breakdown
        forward_pass_gb = activations_gb
        backward_gradient_temp = (per_layer_attn_act + per_layer_ffn_act + per_layer_other_act) / (1024**3)
        backward_pass_gb = gradients_gb + backward_gradient_temp
        
    else:
        # Inference: minimal activation memory
        attn_activations_gb = (batch_size * seq_length * hidden * 2) / (1024**3)
        ffn_activations_gb = (batch_size * seq_length * intermediate_size * 2) / (1024**3)
        other_activations_gb = (batch_size * seq_length * hidden * 2) / (1024**3)
        activations_gb = attn_activations_gb + ffn_activations_gb + other_activations_gb
        forward_pass_gb = activations_gb
        backward_pass_gb = 0
    
    # =========================================================================
    # 5. KV CACHE (Inference only)
    # =========================================================================
    if mode == "Inference":
        kv_cache_bytes = 2 * batch_size * layers * seq_length * kv_heads * head_dim * bytes_per_param
        kv_cache_gb = kv_cache_bytes / (1024**3)
    else:
        kv_cache_gb = 0
    
    # Memory per token
    kv_cache_per_token_bytes = 2 * batch_size * layers * kv_heads * head_dim * bytes_per_param
    kv_cache_per_token_kb = kv_cache_per_token_bytes / 1024
    memory_per_token_kb = kv_cache_per_token_kb
    
    # =========================================================================
    # 6. TORCH.COMPILE OVERHEAD
    # =========================================================================
    if use_torch_compile:
        compile_overhead_gb = model_weights_gb * 0.1
    else:
        compile_overhead_gb = 0
    
    # =========================================================================
    # 7. CUDA OVERHEAD
    # =========================================================================
    cuda_overhead_gb = 0.5
    
    # =========================================================================
    # TOTALS
    # =========================================================================
    total_gb = (
        model_weights_gb +
        gradients_gb +
        optimizer_states_gb +
        activations_gb +
        kv_cache_gb +
        compile_overhead_gb +
        cuda_overhead_gb +
        ddp_overhead_gb
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


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def _stat_card(label: str, value: str, color: str, subtitle: str = "") -> str:
    """Generate HTML for a stat card in the visualization grid."""
    subtitle_html = f'<div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">{subtitle}</div>' if subtitle else ''
    return f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #1e293b, #0f172a); 
                    border-radius: 14px; text-align: center; border: 1px solid #334155;">
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase; 
                        letter-spacing: 1px; margin-bottom: 8px;">{label}</div>
            <div style="font-size: 22px; font-weight: 700; color: {color}; 
                        font-family: 'JetBrains Mono', monospace;">{value}</div>
            {subtitle_html}
        </div>"""


def _breakdown_bar(name: str, value: float, total: float, color: str) -> str:
    """Generate HTML for a breakdown bar in the visualization."""
    pct = (value / total) * 100 if total > 0 else 0
    return f"""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 500; color: #e2e8f0; font-size: 14px;">
                    <span style="display: inline-block; width: 12px; height: 12px; 
                                 background: {color}; border-radius: 3px; margin-right: 8px;"></span>
                    {name}
                </span>
                <span style="color: #94a3b8; font-size: 14px; font-family: 'JetBrains Mono', monospace;">
                    {value:.2f} GB ({pct:.1f}%)
                </span>
            </div>
            <div style="height: 28px; background: #1e293b; border-radius: 6px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: linear-gradient(90deg, {color}, {color}dd); 
                            border-radius: 6px; transition: width 0.4s ease;"></div>
            </div>
        </div>"""


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_vram_visualization(estimate: VRAMEstimate, mode: str) -> str:
    """Create HTML visualization of VRAM breakdown."""
    
    colors = {
        "Model Weights": "#6366f1",
        "Gradients": "#f59e0b",
        "Optimizer States": "#10b981",
        "Activations (Attn)": "#ec4899",
        "Activations (FFN)": "#f472b6",
        "Activations (Other)": "#fb7185",
        "KV Cache": "#8b5cf6",
        "DDP Overhead": "#06b6d4",
        "torch.compile": "#14b8a6",
        "CUDA Overhead": "#64748b",
    }
    
    status_color = "#22c55e" if estimate.fits else "#ef4444"
    status_text = "FITS IN VRAM" if estimate.fits else "EXCEEDS VRAM"
    status_bg = "rgba(34, 197, 94, 0.1)" if estimate.fits else "rgba(239, 68, 68, 0.1)"
    
    # Build breakdown bars
    total_breakdown = sum(v for v in estimate.breakdown.values() if v > 0)
    breakdown_html = ""
    for name, value in estimate.breakdown.items():
        if value > 0.001:
            color = colors.get(name, "#64748b")
            display_name = "Activations (FFN/SwiGLU)" if name == "Activations (FFN)" and estimate.uses_swiglu else name
            breakdown_html += _breakdown_bar(display_name, value, total_breakdown, color)
    
    # Utilization bar
    util_pct = min(estimate.utilization_pct, 150)
    overflow = estimate.utilization_pct > 100
    
    # Forward/Backward pass breakdown (training only)
    fwd_bwd_html = ""
    if mode == "Training" and estimate.forward_pass_gb > 0:
        fwd_bwd_html = f"""
        <div style="padding: 24px; background: #1e293b; border-radius: 16px; margin-bottom: 24px;">
            <h3 style="margin: 0 0 20px 0; font-size: 18px; font-weight: 600; color: #e2e8f0;">
                Forward vs Backward Pass Memory
            </h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div style="padding: 16px; background: #0f172a; border-radius: 12px; border-left: 4px solid #22c55e;">
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">
                        Forward Pass (Activations)
                    </div>
                    <div style="font-size: 24px; font-weight: 700; color: #22c55e; font-family: 'JetBrains Mono', monospace;">
                        {estimate.forward_pass_gb:.2f} GB
                    </div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
                        Stored for backpropagation
                    </div>
                </div>
                <div style="padding: 16px; background: #0f172a; border-radius: 12px; border-left: 4px solid #f59e0b;">
                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">
                        Backward Pass (Gradients)
                    </div>
                    <div style="font-size: 24px; font-weight: 700; color: #f59e0b; font-family: 'JetBrains Mono', monospace;">
                        {estimate.backward_pass_gb:.2f} GB
                    </div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
                        Gradients + temp computations
                    </div>
                </div>
            </div>
        </div>
        """
    
    # Activation breakdown
    activation_breakdown_html = ""
    total_act = estimate.attn_activations_gb + estimate.ffn_activations_gb + estimate.other_activations_gb
    if total_act > 0.001:
        attn_pct = (estimate.attn_activations_gb / total_act) * 100
        ffn_pct = (estimate.ffn_activations_gb / total_act) * 100
        other_pct = (estimate.other_activations_gb / total_act) * 100
        
        activation_breakdown_html = f"""
        <div style="padding: 24px; background: #1e293b; border-radius: 16px; margin-bottom: 24px;">
            <h3 style="margin: 0 0 20px 0; font-size: 18px; font-weight: 600; color: #e2e8f0;">
                Activation Memory Breakdown
            </h3>
            <div style="display: flex; height: 40px; border-radius: 8px; overflow: hidden; margin-bottom: 16px;">
                <div style="width: {attn_pct}%; background: linear-gradient(90deg, #ec4899, #f472b6); 
                            display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 11px; font-weight: 600; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                        {attn_pct:.0f}%
                    </span>
                </div>
                <div style="width: {ffn_pct}%; background: linear-gradient(90deg, #f472b6, #fb7185); 
                            display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 11px; font-weight: 600; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                        {ffn_pct:.0f}%
                    </span>
                </div>
                <div style="width: {other_pct}%; background: linear-gradient(90deg, #94a3b8, #64748b); 
                            display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 11px; font-weight: 600; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                        {other_pct:.0f}%
                    </span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #ec4899; border-radius: 3px;"></span>
                    <span style="font-size: 13px; color: #e2e8f0;">Attention</span>
                    <span style="font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-left: auto;">
                        {estimate.attn_activations_gb:.2f}GB
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #f472b6; border-radius: 3px;"></span>
                    <span style="font-size: 13px; color: #e2e8f0;">FFN{' (SwiGLU)' if estimate.uses_swiglu else ''}</span>
                    <span style="font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-left: auto;">
                        {estimate.ffn_activations_gb:.2f}GB
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: #64748b; border-radius: 3px;"></span>
                    <span style="font-size: 13px; color: #e2e8f0;">Other</span>
                    <span style="font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-left: auto;">
                        {estimate.other_activations_gb:.2f}GB
                    </span>
                </div>
            </div>
        </div>
        """
    
    html = f"""
    <div style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
                padding: 28px; background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
                border-radius: 20px; color: #f8fafc; border: 1px solid #334155;">
        
        <!-- Status Header -->
        <div style="text-align: center; margin-bottom: 32px; padding: 24px; 
                    background: {status_bg}; border-radius: 16px; border: 1px solid {status_color}33;">
            <div style="font-size: 32px; font-weight: 800; color: {status_color}; 
                        margin-bottom: 8px; letter-spacing: -0.5px;">
                {status_text}
            </div>
            <div style="font-size: 16px; color: #94a3b8;">
                Peak: <span style="color: #f8fafc; font-weight: 600;">{estimate.peak_gb:.2f} GB</span> 
                &nbsp;/&nbsp; Available: <span style="color: #f8fafc; font-weight: 600;">{estimate.available_gb:.1f} GB</span>
            </div>
        </div>
        
        <!-- Main Utilization -->
        <div style="margin-bottom: 32px; padding: 24px; background: #1e293b; border-radius: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <span style="font-size: 18px; font-weight: 600; color: #e2e8f0;">
                    VRAM Utilization
                </span>
                <span style="font-size: 28px; font-weight: 800; color: {status_color}; 
                            font-family: 'JetBrains Mono', monospace;">
                    {estimate.utilization_pct:.1f}%
                </span>
            </div>
            <div style="height: 48px; background: #0f172a; border-radius: 12px; overflow: hidden; 
                        position: relative; border: 2px solid #334155;">
                <div style="position: absolute; height: 100%; width: {min(util_pct, 100)}%; 
                            background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
                            border-radius: 10px; transition: width 0.4s ease;
                            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);"></div>
                {"<div style='position: absolute; height: 100%; left: 66.67%; width: " + str(min(util_pct - 100, 50)) + "%; background: linear-gradient(90deg, #ef4444, #dc2626); opacity: 0.9;'></div>" if overflow else ""}
                <div style="position: absolute; left: 66.67%; top: 0; bottom: 0; width: 3px; 
                            background: #22c55e; box-shadow: 0 0 10px #22c55e;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; 
                        font-size: 12px; color: #64748b;">
                <span>0 GB</span>
                <span style="color: #22c55e; font-weight: 600;">{estimate.available_gb:.0f} GB</span>
            </div>
        </div>
        
        {fwd_bwd_html}
        
        {activation_breakdown_html}
        
        <!-- Memory Breakdown -->
        <div style="padding: 24px; background: #1e293b; border-radius: 16px; margin-bottom: 24px;">
            <h3 style="margin: 0 0 20px 0; font-size: 18px; font-weight: 600; color: #e2e8f0;">
                Full Memory Breakdown
            </h3>
            {breakdown_html}
        </div>
        
        <!-- Stats Grid -->
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px;">
            {_stat_card(
                'Total Params' if estimate.is_moe else 'Model Params',
                f'{estimate.model_params_b:.1f}B',
                '#6366f1',
                f'MoE: {estimate.num_experts} experts' if estimate.is_moe else ''
            )}
            {_stat_card(
                'Active Params' if estimate.is_moe else 'Trainable',
                f'{estimate.active_params_b:.1f}B' if estimate.is_moe else f'{estimate.trainable_params_b:.2f}B',
                '#8b5cf6' if estimate.is_moe else '#10b981',
                f'{estimate.experts_per_token} experts/token' if estimate.is_moe else ''
            )}
            {_stat_card('Memory/Token', f'{estimate.memory_per_token_kb:.1f}KB', '#06b6d4', 'per seq token')}
            {_stat_card('Total VRAM', f'{estimate.total_gb:.1f}GB', '#f59e0b')}
            {_stat_card('Headroom', f'{max(0, estimate.available_gb - estimate.total_gb):.1f}GB', status_color, 'remaining')}
        </div>
        
        <!-- Mode indicator -->
        <div style="margin-top: 20px; text-align: center; padding: 12px; 
                    background: #0f172a; border-radius: 8px; border: 1px solid #334155;">
            <span style="color: #64748b; font-size: 13px;">
                Mode: <span style="color: {'#22c55e' if mode == 'Inference' else '#f59e0b'}; font-weight: 600;">
                    {mode}</span>
            </span>
        </div>
    </div>
    """
    
    return html


def create_na_visualization(model_id: str) -> str:
    """Create visualization when model config cannot be determined."""
    return f"""
    <div style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
                padding: 28px; background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
                border-radius: 20px; color: #f8fafc; border: 1px solid #334155;">
        
        <div style="text-align: center; padding: 48px 24px;">
            <div style="font-size: 64px; margin-bottom: 24px;">&#9881;</div>
            <div style="font-size: 28px; font-weight: 700; color: #f59e0b; margin-bottom: 16px;">
                Could Not Compute
            </div>
            <div style="font-size: 16px; color: #94a3b8; max-width: 500px; margin: 0 auto; line-height: 1.6;">
                Unable to auto-detect configuration for: <code style="background: #1e293b; padding: 4px 8px; 
                border-radius: 4px; color: #e2e8f0;">{model_id}</code>
            </div>
            
            <div style="margin-top: 32px; padding: 24px; background: linear-gradient(135deg, #1e3a5f, #1e293b); border-radius: 12px; 
                        text-align: left; max-width: 520px; margin-left: auto; margin-right: auto; border: 1px solid #3b82f6;">
                <div style="font-size: 16px; font-weight: 600; color: #60a5fa; margin-bottom: 16px;">
                    Please enter model config manually:
                </div>
                <div style="color: #e2e8f0; font-size: 14px; line-height: 1.8;">
                    <p style="margin: 0 0 12px 0;">Enable <strong>Advanced Model Config</strong> on the left and provide at minimum:</p>
                    <ul style="margin: 0; padding-left: 20px; color: #94a3b8;">
                        <li><strong style="color: #e2e8f0;">Parameters (B)</strong> - Total model parameters in billions</li>
                        <li><strong style="color: #e2e8f0;">Hidden Dim</strong> - Hidden dimension size (e.g., 4096)</li>
                        <li><strong style="color: #e2e8f0;">Layers</strong> - Number of transformer layers</li>
                    </ul>
                    <p style="margin: 16px 0 0 0; font-size: 13px; color: #64748b;">
                        Other values will use sensible defaults if not provided.
                    </p>
                </div>
            </div>
            
            <div style="margin-top: 24px; padding: 16px; background: #1e293b; border-radius: 12px; 
                        max-width: 520px; margin-left: auto; margin-right: auto;">
                <div style="font-size: 13px; color: #64748b;">
                    <strong style="color: #94a3b8;">Alternatively:</strong> Select a model from the <strong>Quick Presets</strong> above, 
                    or use the exact HuggingFace model ID (e.g., <code style="background: #0f172a; padding: 2px 6px; border-radius: 4px;">meta-llama/Llama-3.1-8B</code>)
                </div>
            </div>
        </div>
    </div>
    """


def calculate_and_display(
    model_id: str,
    gpu_name: str,
    mode: str,
    dtype: str,
    batch_size: int,
    seq_length: int,
    gradient_checkpointing: bool,
    optimizer: str,
    lora_enabled: bool,
    lora_rank: int,
    use_torch_compile: bool,
    ddp_enabled: bool,
    mixed_precision: bool,
    # Manual config parameters
    manual_params_b: float,
    manual_hidden: int,
    manual_layers: int,
    manual_heads: int,
    manual_kv_heads: int,
    manual_intermediate: int,
    manual_vocab_size: int,
    manual_uses_swiglu: str,
    # MoE manual config
    manual_num_experts: int,
    manual_experts_per_token: int,
    manual_active_params_b: float,
):
    """Main calculation function for Gradio interface."""
    
    if not model_id.strip():
        return ("<div style='padding: 40px; text-align: center; color: #64748b;'>Enter a model ID to calculate VRAM</div>", "",
                gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update())
    
    estimate = calculate_vram(
        model_id=model_id.strip(),
        gpu_name=gpu_name,
        mode=mode,
        dtype=dtype,
        batch_size=int(batch_size),
        seq_length=int(seq_length),
        gradient_checkpointing=gradient_checkpointing,
        optimizer=optimizer,
        lora_rank=int(lora_rank),
        lora_enabled=lora_enabled,
        use_torch_compile=use_torch_compile,
        ddp_enabled=ddp_enabled,
        mixed_precision=mixed_precision,
        manual_params_b=float(manual_params_b) if manual_params_b else 0,
        manual_hidden=int(manual_hidden) if manual_hidden else 0,
        manual_layers=int(manual_layers) if manual_layers else 0,
        manual_heads=int(manual_heads) if manual_heads else 0,
        manual_kv_heads=int(manual_kv_heads) if manual_kv_heads else 0,
        manual_intermediate=int(manual_intermediate) if manual_intermediate else 0,
        manual_vocab_size=int(manual_vocab_size) if manual_vocab_size else 0,
        manual_uses_swiglu=manual_uses_swiglu,
        manual_num_experts=int(manual_num_experts) if manual_num_experts else 0,
        manual_experts_per_token=int(manual_experts_per_token) if manual_experts_per_token else 0,
        manual_active_params_b=float(manual_active_params_b) if manual_active_params_b else 0,
    )
    
    if estimate is None:
        na_viz = create_na_visualization(model_id.strip())
        na_summary = f"""
## Could Not Compute

Unable to auto-detect model configuration for `{model_id.strip()}`.

### To calculate VRAM, please either:

**Option 1: Enter config manually**
1. Enable **Advanced Model Config** on the left
2. Fill in at least: **Parameters (B)**, **Hidden Dim**, and **Layers**
3. Click Calculate again

**Option 2: Use a known model**
- Select from **Quick Presets** above
- Or use exact HuggingFace ID (e.g., `meta-llama/Llama-3.1-8B`)

---

*The model was not found in presets and config.json could not be fetched from HuggingFace.*
"""
        return (na_viz, na_summary,
                0, 0, 0, 0,
                0, 0, 0, "Auto",
                0, 0, 0)
    
    visualization = create_vram_visualization(estimate, mode)
    
    # Recommendations
    recommendations = []
    if not estimate.fits:
        if not lora_enabled and mode == "Training":
            recommendations.append("- Enable **LoRA** to dramatically reduce trainable parameters")
        if "float16" not in dtype.lower() and "int" not in dtype.lower():
            recommendations.append("- Use **FP16/BF16** or quantization to reduce model memory")
        if not gradient_checkpointing and mode == "Training":
            recommendations.append("- Enable **gradient checkpointing** to reduce activation memory")
        if batch_size > 1:
            recommendations.append(f"- Reduce **batch size** (current: {batch_size})")
        if seq_length > 2048:
            recommendations.append(f"- Reduce **sequence length** (current: {seq_length})")
        recommendations.append("- Consider using **multi-GPU** setup with FSDP or DeepSpeed")
    
    reco_text = "\n".join(recommendations) if recommendations else ""
    
    # Build activation breakdown text
    act_breakdown = ""
    if mode == "Training":
        act_breakdown = f"""
| > Attention | {estimate.attn_activations_gb:.2f} |
| > FFN {'(SwiGLU)' if estimate.uses_swiglu else ''} | {estimate.ffn_activations_gb:.2f} |
| > Other | {estimate.other_activations_gb:.2f} |"""

    fwd_bwd_text = ""
    if mode == "Training":
        fwd_bwd_text = f"""
---

### Forward vs Backward Pass

| Pass | Memory (GB) | Description |
|------|-------------|-------------|
| Forward | {estimate.forward_pass_gb:.2f} | Activations stored for backprop |
| Backward | {estimate.backward_pass_gb:.2f} | Gradients + temp computations |
"""

    reco_section = ""
    if reco_text:
        reco_section = f"""
---

### Recommendations
{reco_text}
"""

    summary = f"""
### {'Model FITS!' if estimate.fits else 'Model EXCEEDS available VRAM!'}
{reco_section}
---

## Configuration Summary

| Setting | Value |
|---------|-------|
| **Model** | `{model_id}` |
| **Parameters** | {estimate.model_params_b:.2f}B total, {estimate.trainable_params_b:.3f}B trainable |
| **GPU** | {gpu_name} |
| **Mode** | {mode} |
| **Precision** | {dtype} |
| **Mixed Precision** | {'Yes (FP32 master weights)' if mixed_precision and mode == 'Training' else 'No'} |
| **Batch x Seq** | {int(batch_size)} x {int(seq_length):,} |

---

### Detected Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Dim | {estimate.config_hidden:,} |
| Layers | {estimate.config_layers} |
| Attn Heads | {estimate.config_heads} |
| KV Heads | {estimate.config_kv_heads} |
| FFN Intermediate | {estimate.config_intermediate:,} |
| Vocab Size | {estimate.config_vocab_size:,} |
| Uses SwiGLU | {'Yes' if estimate.uses_swiglu else 'No'} |
| **MoE Architecture** | {'Yes (' + str(estimate.num_experts) + ' experts, ' + str(estimate.experts_per_token) + ' active/token)' if estimate.is_moe else 'No'} |
| **Active Params** | {estimate.active_params_b:.2f}B |

---

### Memory Breakdown

| Component | Size (GB) |
|-----------|-----------|
| Model Weights | {estimate.model_weights_gb:.2f} |
| Gradients | {estimate.gradients_gb:.2f} |
| Optimizer States | {estimate.optimizer_states_gb:.2f} |
| Activations (Total) | {estimate.activations_gb:.2f} |{act_breakdown}
| KV Cache | {estimate.kv_cache_gb:.2f} |
| DDP Overhead | {estimate.ddp_overhead_gb:.2f} |
| torch.compile | {estimate.compile_overhead_gb:.2f} |
| CUDA Overhead | {estimate.cuda_overhead_gb:.2f} |
| **Total** | **{estimate.total_gb:.2f}** |

---

### Memory Per Token

| Metric | Value |
|--------|-------|
| Memory per token | {estimate.memory_per_token_kb:.2f} KB |
| KV cache per token | {estimate.kv_cache_per_token_kb:.2f} KB |
| +1K tokens adds | ~{estimate.memory_per_token_kb * 1024 / 1024:.1f} MB |
| +4K tokens adds | ~{estimate.memory_per_token_kb * 4096 / 1024:.1f} MB |
{fwd_bwd_text}
"""
    
    return (
        visualization, 
        summary,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def build_interface():
    """Build the Gradio interface."""
    
    custom_css = """
    .gradio-container { 
        max-width: 1400px !important; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        transform: translateY(-1px);
    }
    """
    
    with gr.Blocks(
        title="LLM VRAM Calculator",
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as demo:
        
        gr.Markdown("""
        # LLM VRAM Calculator
        **Estimate GPU memory requirements for LLM training and inference**
        """)
        
        # Quick presets
        gr.Markdown("### Quick Model Presets")
        with gr.Row():
            preset_llama8b = gr.Button("Llama 3.1 8B", size="sm")
            preset_llama70b = gr.Button("Llama 3.1 70B", size="sm")
            preset_mistral = gr.Button("Mistral 7B", size="sm")
            preset_mixtral = gr.Button("Mixtral 8x7B (MoE)", size="sm")
            preset_qwen7b = gr.Button("Qwen 2.5 7B", size="sm")
            preset_qwen72b = gr.Button("Qwen 2.5 72B", size="sm")
            preset_gemma = gr.Button("Gemma 2 9B", size="sm")
            preset_phi = gr.Button("Phi-3 Mini", size="sm")
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                
                with gr.Group():
                    gr.Markdown("### Hardware")
                    gpu_dropdown = gr.Dropdown(
                        choices=list(GPU_SPECS.keys()),
                        value="NVIDIA RTX 3060 Ti 8GB",
                        label="GPU Model",
                        info="Single GPU memory estimation"
                    )
                
                with gr.Group():
                    gr.Markdown("### Model")
                    model_id = gr.Textbox(
                        value="meta-llama/Llama-3.1-8B",
                        label="HuggingFace Model ID",
                        placeholder="organization/model-name",
                    )
                    
                    advanced_config = gr.Checkbox(
                        value=False,
                        label="Advanced Model Config",
                    )
                    
                    with gr.Group(visible=False) as advanced_config_group:
                        gr.Markdown("*Manually enter values:*")
                        with gr.Row():
                            manual_params_b = gr.Number(
                                value=None,
                                label="Parameters (B)",
                                minimum=0,
                                precision=2,
                            )
                            manual_hidden = gr.Number(
                                value=None,
                                label="Hidden Dim",
                                minimum=0,
                                precision=0,
                            )
                        with gr.Row():
                            manual_layers = gr.Number(
                                value=None,
                                label="Layers",
                                minimum=0,
                                precision=0,
                            )
                            manual_heads = gr.Number(
                                value=None,
                                label="Attn Heads",
                                minimum=0,
                                precision=0,
                            )
                        with gr.Row():
                            manual_kv_heads = gr.Number(
                                value=None,
                                label="KV Heads (GQA)",
                                minimum=0,
                                precision=0,
                            )
                            manual_intermediate = gr.Number(
                                value=None,
                                label="FFN Intermediate",
                                minimum=0,
                                precision=0,
                            )
                        with gr.Row():
                            manual_vocab_size = gr.Number(
                                value=None,
                                label="Vocab Size",
                                minimum=0,
                                precision=0,
                            )
                            manual_uses_swiglu = gr.Dropdown(
                                choices=["Auto", "Yes", "No"],
                                value="Auto",
                                label="Uses SwiGLU",
                            )
                        
                        gr.Markdown("*MoE (Mixture of Experts) Config:*")
                        with gr.Row():
                            manual_num_experts = gr.Number(
                                value=None,
                                label="Num Experts",
                                info="Total number of experts (e.g., 8)",
                                minimum=0,
                                precision=0,
                            )
                            manual_experts_per_token = gr.Number(
                                value=None,
                                label="Experts per Token",
                                info="Active experts per forward pass (e.g., 2)",
                                minimum=0,
                                precision=0,
                            )
                        with gr.Row():
                            manual_active_params_b = gr.Number(
                                value=None,
                                label="Active Params (B)",
                                info="Params active per forward pass",
                                minimum=0,
                                precision=2,
                            )
                    
                    with gr.Row():
                        mode = gr.Radio(
                            choices=["Training", "Inference"],
                            value="Training",
                            label="Mode",
                        )
                    
                    dtype = gr.Dropdown(
                        choices=list(DTYPE_BYTES.keys()),
                        value="bfloat16 (BF16)",
                        label="Precision",
                        info="BF16 recommended for training on modern GPUs (Ampere+). FP16 for older GPUs (V100, T4).",
                    )
                
                with gr.Group():
                    gr.Markdown("### Runtime")
                    batch_size = gr.Slider(
                        minimum=1, maximum=1024, value=1, step=1,
                        label="Batch Size",
                    )
                    seq_length = gr.Slider(
                        minimum=128, maximum=131072, value=2048, step=128,
                        label="Sequence Length",
                    )
                
                with gr.Group():
                    gr.Markdown("### Training Options")
                    gradient_checkpointing = gr.Checkbox(
                        value=True,
                        label="Gradient Checkpointing",
                    )
                    optimizer = gr.Dropdown(
                        choices=["AdamW (32-bit)", "AdamW (8-bit)", "SGD", "Adafactor"],
                        value="AdamW (32-bit)",
                        label="Optimizer",
                    )
                    mixed_precision = gr.Checkbox(
                        value=False,
                        label="Mixed Precision (FP32 master weights)",
                        info="Keep FP32 copy of weights for stability. Recommended for FP16, optional for BF16.",
                    )
                    lora_enabled = gr.Checkbox(
                        value=True,
                        label="LoRA Enabled",
                    )
                    lora_rank = gr.Slider(
                        minimum=4, maximum=256, value=16, step=4,
                        label="LoRA Rank",
                    )
                    ddp_enabled = gr.Checkbox(
                        value=False,
                        label="DDP (Multi-GPU) - adds gradient buffer for sync",
                    )
                
                with gr.Group():
                    gr.Markdown("### Advanced Options")
                    use_torch_compile = gr.Checkbox(
                        value=False,
                        label="torch.compile - adds ~10% for compiled graphs",
                    )
                
                calculate_btn = gr.Button(
                    "Calculate VRAM",
                    variant="primary",
                    size="lg",
                )
            
            # Right column - Output
            with gr.Column(scale=2):
                visualization = gr.HTML(
                    value="<div style='padding: 60px; text-align: center; color: #64748b; font-size: 16px;'>Configure settings and click Calculate</div>"
                )
                summary = gr.Markdown("")
        
        # Manual config inputs
        manual_config_inputs = [
            manual_params_b, manual_hidden, manual_layers, manual_heads,
            manual_kv_heads, manual_intermediate, manual_vocab_size, manual_uses_swiglu,
            manual_num_experts, manual_experts_per_token, manual_active_params_b
        ]
        
        # All inputs for event handling
        all_inputs = [
            model_id, gpu_dropdown, mode, dtype,
            batch_size, seq_length, gradient_checkpointing,
            optimizer, lora_enabled, lora_rank,
            use_torch_compile, ddp_enabled, mixed_precision
        ] + manual_config_inputs
        all_outputs = [visualization, summary] + manual_config_inputs
        
        # Auto-calculate inputs (excluding manual config)
        auto_calc_inputs = [
            gpu_dropdown, mode, dtype,
            batch_size, seq_length, gradient_checkpointing,
            optimizer, lora_enabled, lora_rank,
            use_torch_compile, ddp_enabled, mixed_precision
        ]
        
        # Calculate button
        calculate_btn.click(
            fn=calculate_and_display,
            inputs=all_inputs,
            outputs=all_outputs,
        )
        
        # Model ID submit on Enter
        model_id.submit(
            fn=calculate_and_display,
            inputs=all_inputs,
            outputs=all_outputs,
        )
        
        # Auto-calculate on input changes
        for component in auto_calc_inputs:
            component.change(
                fn=calculate_and_display,
                inputs=all_inputs,
                outputs=all_outputs,
            )
        
        # Preset button handlers
        def set_preset_and_calc(preset_name, *args):
            return preset_name, *calculate_and_display(preset_name, *args)
        
        preset_inputs = [gpu_dropdown, mode, dtype, batch_size, seq_length, 
                        gradient_checkpointing, optimizer, lora_enabled, lora_rank,
                        use_torch_compile, ddp_enabled, mixed_precision,
                        manual_params_b, manual_hidden, manual_layers, manual_heads,
                        manual_kv_heads, manual_intermediate, manual_vocab_size, manual_uses_swiglu,
                        manual_num_experts, manual_experts_per_token, manual_active_params_b]
        preset_outputs = [model_id, visualization, summary,
                         manual_params_b, manual_hidden, manual_layers, manual_heads,
                         manual_kv_heads, manual_intermediate, manual_vocab_size, manual_uses_swiglu,
                         manual_num_experts, manual_experts_per_token, manual_active_params_b]
        
        preset_llama8b.click(lambda *args: set_preset_and_calc("meta-llama/Llama-3.1-8B", *args), 
                            inputs=preset_inputs, outputs=preset_outputs)
        preset_llama70b.click(lambda *args: set_preset_and_calc("meta-llama/Llama-3.1-70B", *args), 
                             inputs=preset_inputs, outputs=preset_outputs)
        preset_mistral.click(lambda *args: set_preset_and_calc("mistralai/Mistral-7B-v0.3", *args), 
                            inputs=preset_inputs, outputs=preset_outputs)
        preset_mixtral.click(lambda *args: set_preset_and_calc("mistralai/Mixtral-8x7B-v0.1", *args), 
                            inputs=preset_inputs, outputs=preset_outputs)
        preset_qwen7b.click(lambda *args: set_preset_and_calc("Qwen/Qwen2.5-7B", *args), 
                           inputs=preset_inputs, outputs=preset_outputs)
        preset_qwen72b.click(lambda *args: set_preset_and_calc("Qwen/Qwen2.5-72B", *args), 
                            inputs=preset_inputs, outputs=preset_outputs)
        preset_gemma.click(lambda *args: set_preset_and_calc("google/gemma-2-9b", *args), 
                          inputs=preset_inputs, outputs=preset_outputs)
        preset_phi.click(lambda *args: set_preset_and_calc("microsoft/phi-3-mini-4k-instruct", *args), 
                        inputs=preset_inputs, outputs=preset_outputs)
        
        # Toggle advanced config visibility
        advanced_config.change(
            fn=lambda x: gr.update(visible=x),
            inputs=advanced_config,
            outputs=advanced_config_group,
        )
        
        # Toggle LoRA rank visibility
        lora_enabled.change(
            fn=lambda x: gr.update(visible=x),
            inputs=lora_enabled,
            outputs=lora_rank,
        )
        
        # Auto-default mixed precision based on dtype
        # FP16 needs mixed precision for numerical stability, BF16 usually doesn't
        def update_mixed_precision_default(dtype_val):
            if "float16" in dtype_val.lower() or "fp16" in dtype_val.lower():
                return gr.update(value=True)
            elif "bfloat16" in dtype_val.lower() or "bf16" in dtype_val.lower():
                return gr.update(value=False)
            return gr.update()
        
        dtype.change(
            fn=update_mixed_precision_default,
            inputs=dtype,
            outputs=mixed_precision,
        )
        
        # Toggle training options visibility
        def update_training_visibility(mode_val, lora_enabled_val):
            is_training = mode_val == "Training"
            lora_rank_visible = is_training and lora_enabled_val
            return (
                gr.update(visible=is_training),
                gr.update(visible=is_training),
                gr.update(visible=is_training),
                gr.update(visible=is_training),
                gr.update(visible=lora_rank_visible),
                gr.update(visible=is_training),
            )
        
        mode.change(
            fn=update_training_visibility,
            inputs=[mode, lora_enabled],
            outputs=[gradient_checkpointing, optimizer, mixed_precision, lora_enabled, lora_rank, ddp_enabled],
        )
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Starting LLM VRAM Calculator")
    print("=" * 50)
    
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
