"""Unit tests for architecture-driven VRAM estimation."""

from __future__ import annotations

import unittest

from vram_calc.config import estimate_params_from_architecture, finalize_architecture
from vram_calc.engine import (
    BYTES_PER_GIB,
    FLASH_ATTN_BLOCK_SIZE,
    calc_attention_memory,
    calc_ffn_memory,
    estimate_vram,
)
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

    def test_moe_inference_model_weights_use_total_params(self) -> None:
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
            active_params_b=10.0,
        )
        inp = _base_input(arch)
        inp.mode = "Inference"
        inp.dtype = "16-bit (BF16/FP16)"
        estimate = estimate_vram(inp)
        expected_weights_gb = (arch.params_b * 1e9 * 2) / BYTES_PER_GIB
        self.assertAlmostEqual(estimate.model_weights_gb, expected_weights_gb, places=8)

    def test_dense_forces_active_params_to_total(self) -> None:
        arch = ArchitectureConfig(
            params_b=8.0,
            hidden=4096,
            layers=32,
            heads=32,
            kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            attention_type="gqa",
            ffn_type="dense",
            active_params_b=1.0,
        )
        finalized = finalize_architecture(arch)
        self.assertEqual(finalized.active_params_b, finalized.params_b)

    def test_moe_active_params_are_capped_by_total(self) -> None:
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
            active_params_b=100.0,
        )
        finalized = finalize_architecture(arch)
        self.assertEqual(finalized.active_params_b, finalized.params_b)

    def test_estimate_inference_other_activations_uses_shared_activation_bytes(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.seq_length = 2048
        inp.batch_size = 2
        estimate = estimate_vram(inp)
        expected = (4 * inp.batch_size * inp.seq_length * arch.hidden * 2) / (1024**3)
        self.assertAlmostEqual(estimate.other_activations_gb, expected, places=8)

    def test_moe_ffn_memory_scales_with_active_param_ratio(self) -> None:
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
            active_params_b=10.0,
        )
        inp = _base_input(arch)
        finalized = finalize_architecture(arch)
        ffn_gb, per_layer = calc_ffn_memory(finalized, inp, bytes_activation=2)
        expected_ratio = finalized.active_params_b / finalized.params_b
        expected_per_layer = (
            inp.batch_size * inp.seq_length * finalized.hidden * 2
            + inp.batch_size * inp.seq_length * finalized.intermediate_size * expected_ratio * 2
            + inp.batch_size * inp.seq_length * finalized.experts_per_token * 2
        )
        self.assertGreater(ffn_gb, 0.0)
        self.assertAlmostEqual(per_layer, expected_per_layer, places=4)

    def test_attention_scores_scale_with_query_heads_not_kv_heads(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.gradient_checkpointing = False
        activation_gb, per_layer = calc_attention_memory(arch, inp, bytes_activation=2)
        expected_attn_scores = inp.batch_size * arch.heads * inp.seq_length * inp.seq_length * 2
        self.assertGreater(per_layer, expected_attn_scores * 0.99)
        self.assertGreater(activation_gb, 0.0)

    def test_flash_attention_reduces_attention_activation_memory(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.gradient_checkpointing = False
        baseline_gb, baseline_per_layer = calc_attention_memory(arch, inp, bytes_activation=2)
        inp.flash_attention = True
        flash_gb, flash_per_layer = calc_attention_memory(arch, inp, bytes_activation=2)
        self.assertLess(flash_gb, baseline_gb)
        self.assertLess(flash_per_layer, baseline_per_layer)

    def test_flash_attention_scales_linearly_with_sequence_length(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.gradient_checkpointing = False
        inp.flash_attention = True
        inp.seq_length = FLASH_ATTN_BLOCK_SIZE * 2
        _, per_layer_small = calc_attention_memory(arch, inp, bytes_activation=2)
        inp.seq_length = FLASH_ATTN_BLOCK_SIZE * 4
        _, per_layer_large = calc_attention_memory(arch, inp, bytes_activation=2)
        ratio = per_layer_large / per_layer_small
        self.assertLess(ratio, 2.5)

    def test_quantized_dtype_keeps_kv_cache_at_activation_precision(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.dtype = "int4 (4-bit Quantized)"
        estimate = estimate_vram(inp)
        head_dim = arch.hidden // arch.heads
        expected_per_token_kb = (
            2 * inp.batch_size * arch.layers * arch.kv_heads * head_dim * 2
        ) / 1024
        self.assertAlmostEqual(estimate.kv_cache_per_token_kb, expected_per_token_kb, places=8)

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

    def test_inference_excludes_training_states_and_includes_kv_cache(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Inference"
        estimate = estimate_vram(inp)
        self.assertEqual(estimate.gradients_gb, 0.0)
        self.assertEqual(estimate.optimizer_states_gb, 0.0)
        self.assertGreater(estimate.kv_cache_gb, 0.0)

    def test_training_includes_training_states_and_excludes_kv_cache(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.lora_enabled = False
        estimate = estimate_vram(inp)
        self.assertGreater(estimate.gradients_gb, 0.0)
        self.assertGreater(estimate.optimizer_states_gb, 0.0)
        self.assertEqual(estimate.kv_cache_gb, 0.0)

    def test_training_breakdown_uses_non_kv_attention_label(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        estimate = estimate_vram(inp)
        self.assertIn("Activations (Attn)", estimate.breakdown)
        self.assertNotIn("Activations (Attn, drives KV cache)", estimate.breakdown)

    def test_int4_training_gradients_use_activation_precision(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.dtype = "int4 (4-bit Quantized)"
        inp.lora_enabled = False
        inp.mixed_precision = False
        estimate = estimate_vram(inp)
        expected = (arch.params_b * 1e9 * 2) / BYTES_PER_GIB
        self.assertAlmostEqual(estimate.gradients_gb, expected, places=8)

    def test_mixed_precision_adds_master_weights(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.lora_enabled = False
        inp.mixed_precision = True
        estimate = estimate_vram(inp)
        params_count = arch.params_b * 1e9
        expected_gradients = (params_count * 2) / BYTES_PER_GIB
        expected_optimizer = ((params_count * 4 * 2) / BYTES_PER_GIB) + ((params_count * 4) / BYTES_PER_GIB)
        self.assertAlmostEqual(estimate.gradients_gb, expected_gradients, places=8)
        self.assertAlmostEqual(estimate.optimizer_states_gb, expected_optimizer, places=8)

    def test_ddp_overhead_formula_matches_gradient_buffer(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.lora_enabled = False
        inp.ddp_enabled = True
        estimate = estimate_vram(inp)
        expected = estimate.gradients_gb + 0.05 + (estimate.gradients_gb * 0.02)
        self.assertAlmostEqual(estimate.ddp_overhead_gb, expected, places=8)

    def test_total_equals_breakdown_sum(self) -> None:
        arch = self._dense_arch()
        inp = _base_input(arch)
        inp.mode = "Training"
        inp.lora_enabled = False
        inp.ddp_enabled = True
        inp.use_torch_compile = True
        estimate = estimate_vram(inp)
        self.assertAlmostEqual(estimate.total_gb, sum(estimate.breakdown.values()), places=8)

    def test_mode_matching_is_case_insensitive(self) -> None:
        arch = self._dense_arch()
        base = _base_input(arch)
        base.mode = "Training"
        base.lora_enabled = False
        ref = estimate_vram(base)

        mixed_case = _base_input(arch)
        mixed_case.mode = "training"
        mixed_case.lora_enabled = False
        estimate = estimate_vram(mixed_case)
        self.assertAlmostEqual(estimate.total_gb, ref.total_gb, places=8)


if __name__ == "__main__":
    unittest.main()
