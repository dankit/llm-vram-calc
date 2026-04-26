"""Static lookup tables for GPUs and precision dtypes."""

GPU_SPECS = {
    "NVIDIA GH200 96GB": {"vram_gb": 96},
    "NVIDIA H200 141GB": {"vram_gb": 141},
    "NVIDIA H100 SXM 80GB": {"vram_gb": 80},
    "NVIDIA A100 80GB": {"vram_gb": 80},
    "NVIDIA L40S 48GB": {"vram_gb": 48},
    "NVIDIA RTX 5090 32GB": {"vram_gb": 32},
    "NVIDIA RTX 4090 24GB": {"vram_gb": 24},
    "AMD Instinct MI300X 192GB": {"vram_gb": 192},
}

DTYPE_BYTES = {
    "16-bit (BF16/FP16)": 2,
    "float32 (FP32)": 4,
    "int8 (8-bit Quantized)": 1,
    "int4 (4-bit Quantized)": 0.5,
}

