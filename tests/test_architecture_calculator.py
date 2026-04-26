"""Unit tests for architecture-driven VRAM estimation."""

from __future__ import annotations

import unittest

from vram_calc.config import estimate_params_from_architecture, finalize_architecture
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
    def _dense_arch(self) -> ArchitectureConfig:
        return ArchitectureConfig(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            attention_type="gqa",
            ffn_type="dense",
        )

    def test_mha_forces_kv_heads_equal_heads(self) -> None:
        arch = ArchitectureConfig(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
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
            attention_type="gqa",
            ffn_type="moe",
            num_experts=8,
            experts_per_token=2,
            active_params_b=0.0,
        )
        estimate = estimate_vram(_base_input(arch))
        self.assertTrue(estimate.is_moe)
        self.assertLess(estimate.active_params_b, estimate.model_params_b)

    def test_estimate_inference_other_activations_uses_shared_activation_bytes(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.seq_length = 2048
        inp.batch_size = 2
        estimate = estimate_vram(inp)
        expected = (inp.batch_size * inp.seq_length * arch.hidden * 2 * arch.layers) / (1024**3)
        self.assertAlmostEqual(estimate.other_activations_gb, expected, places=8)

    def test_kv_cache_per_token_metric_is_single_source(self) -> None:
        arch = self._dense_arch()
        estimate = estimate_vram(_base_input(arch))
        expected_kv_cache_gb = estimate.kv_cache_per_token_kb * 1024 * 4096 / (1024**3)
        self.assertAlmostEqual(estimate.kv_cache_gb, expected_kv_cache_gb, places=8)
        self.assertFalse(hasattr(estimate, "memory_per_token_kb"))

    def test_estimate_params_from_architecture_kept_as_public_helper(self) -> None:
        arch = self._dense_arch()
        estimated_params = estimate_params_from_architecture(arch)
        self.assertGreater(estimated_params, 0.0)

    def test_estimate_params_from_architecture_reflects_gqa_kv_heads(self) -> None:
        base = dict(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            intermediate_size=14336,
            vocab_size=128256,
            ffn_type="dense",
        )
        gqa_arch = ArchitectureConfig(attention_type="gqa", kv_heads=8, **base)
        mha_arch = ArchitectureConfig(attention_type="mha", kv_heads=8, **base)
        self.assertLess(estimate_params_from_architecture(gqa_arch), estimate_params_from_architecture(mha_arch))


if __name__ == "__main__":
    unittest.main()
