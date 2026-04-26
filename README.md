# LLM VRAM Calculator

Architecture-first VRAM estimation for transformer training and inference.

## What Changed

- Model presets were removed.
- Hugging Face model loading was removed.
- Manual model-id entry was removed.
- The UI now lets you build architecture explicitly and calculates memory directly from your inputs.

## Features

- Dynamic architecture builder (no preset dependency)
- Attention mode support: `MHA`, `GQA`, `MQA`
- FFN mode support: `Dense`, `MoE`
- Runtime controls: training/inference, precision, batch size, context length
- Training options: checkpointing, optimizer, LoRA, DDP overhead
- Detailed VRAM breakdown: weights, gradients, optimizer states, activations, KV cache, runtime overhead

## Installation

```bash
git clone https://github.com/dankit/llm-vram-calc.git
cd llm-vram-calc
pip install -r requirements.txt
```

## Web UI

```bash
python vram_calculator.py
```

Open `http://localhost:7860`.

## Programmatic Usage

```python
from vram_calc import ArchitectureConfig, VRAMInput, estimate_vram

arch = ArchitectureConfig(
    params_b=8.0,
    hidden=4096,
    layers=32,
    heads=32,
    kv_heads=8,
    intermediate_size=14336,
    vocab_size=128256,
    uses_swiglu=True,
    attention_type="gqa",
    ffn_type="dense",
)

inp = VRAMInput(
    architecture=arch,
    gpu_name="NVIDIA RTX 4090 24GB",
    mode="Training",
    dtype="16-bit (BF16/FP16)",
    batch_size=1,
    seq_length=2048,
    gradient_checkpointing=True,
    optimizer="AdamW (32-bit)",
    lora_rank=16,
    lora_enabled=True,
)

estimate = estimate_vram(inp)
print(f"Total VRAM: {estimate.total_gb:.2f} GB")
print(f"Fits in GPU: {estimate.fits}")
```

## Architecture Builder Inputs

Required:
- total parameters (`params_b`)
- hidden size
- layers
- heads
- KV heads (used for GQA, auto-derived for MHA/MQA)
- intermediate size
- vocab size
- attention type (`mha`, `gqa`, `mqa`)
- FFN type (`dense`, `moe`)

MoE-only:
- number of experts
- experts per token
- active parameters (optional; auto-derived when zero)

## Validation Rules

- Hidden size must be divisible by heads.
- KV heads must divide heads for GQA.
- MHA forces `kv_heads = heads`.
- MQA forces `kv_heads = 1`.
- For MoE, experts-per-token must be less than or equal to experts.

## Notes

- MLA is intentionally deferred in this refactor.
- Custom GPU entries are persisted in `vram_calc/data/custom_gpus.json`.
