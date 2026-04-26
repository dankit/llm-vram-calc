"""Unit tests for architecture-driven VRAM estimation."""

from __future__ import annotations

import unittest

from vram_calc.config import finalize_architecture
from vram_calc.engine import estimate_vram
from vram_calc.types import ArchitectureConfig, VRAMInput


def _base_input(architecture: ArchitectureConfig) -> VRAMInput:
    return VRAMInput(
        architecture=architecture,
        gpu_name="NVIDIA H100 SXM 80GB",
        mode="Inference",
        dtype="16-bit (BF16/FP16)",
        batch_size=1,
        seq_length=4096,
        gradient_checkpointing=False,
        optimizer="AdamW (32-bit)",
        lora_rank=16,
        lora_enabled=False,
        use_torch_compile=False,
        ddp_enabled=False,
        mixed_precision=False,
    )


class ArchitectureCalculatorTests(unittest.TestCase):
    def test_mha_forces_kv_heads_equal_heads(self) -> None:
        arch = ArchitectureConfig(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            ffn_multiplier=3.0,
            attention_type="mha",
            ffn_type="dense",
        )
        finalized = finalize_architecture(arch)
        self.assertEqual(finalized.kv_heads, 32)

    def test_mha_ignores_user_kv_heads_in_estimate(self) -> None:
        base = dict(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            intermediate_size=14336,
            vocab_size=128256,
            ffn_multiplier=3.0,
            attention_type="mha",
            ffn_type="dense",
        )
        estimate_a = estimate_vram(_base_input(ArchitectureConfig(kv_heads=1, **base)))
        estimate_b = estimate_vram(_base_input(ArchitectureConfig(kv_heads=16, **base)))
        self.assertAlmostEqual(estimate_a.kv_cache_gb, estimate_b.kv_cache_gb, places=6)

    def test_gqa_uses_provided_kv_heads_for_kv_cache(self) -> None:
        gqa_arch = ArchitectureConfig(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            ffn_multiplier=3.0,
            attention_type="gqa",
            ffn_type="dense",
        )
        mha_arch = ArchitectureConfig(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            ffn_multiplier=3.0,
            attention_type="mha",
            ffn_type="dense",
        )
        gqa_estimate = estimate_vram(_base_input(gqa_arch))
        mha_estimate = estimate_vram(_base_input(mha_arch))
        self.assertLess(gqa_estimate.kv_cache_gb, mha_estimate.kv_cache_gb)

    def test_moe_sets_active_params_below_total(self) -> None:
        arch = ArchitectureConfig(
            params_b=45.0,
            hidden=4096,
            layers=32,
            heads=32,
            kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            ffn_multiplier=3.0,
            attention_type="gqa",
            ffn_type="moe",
            num_experts=8,
            experts_per_token=2,
            active_params_b=0.0,
        )
        estimate = estimate_vram(_base_input(arch))
        self.assertTrue(estimate.is_moe)
        self.assertLess(estimate.active_params_b, estimate.model_params_b)


if __name__ == "__main__":
    unittest.main()
