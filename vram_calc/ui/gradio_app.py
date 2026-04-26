"""Gradio Blocks UI for the VRAM calculator."""

import gradio as gr

from vram_calc.constants import DTYPE_BYTES
from vram_calc.custom_gpu_store import get_all_gpu_specs, save_custom_gpu_spec
from vram_calc.engine import estimate_vram
from vram_calc.types import VRAMInput
from vram_calc.ui.html import create_na_visualization, create_vram_visualization

_APP_CSS = """
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

_APP_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
)


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
    manual_params_b: float,
    manual_hidden: int,
    manual_layers: int,
    manual_heads: int,
    manual_kv_heads: int,
    manual_intermediate: int,
    manual_vocab_size: int,
    manual_uses_swiglu: str,
    manual_num_experts: int,
    manual_experts_per_token: int,
    manual_active_params_b: float,
):
    """Main calculation function for Gradio interface."""

    if not model_id.strip():
        return (
            "<div style='padding: 40px; text-align: center; color: #64748b;'>Enter a model ID to calculate VRAM</div>",
            "Enter a model ID to calculate VRAM.",
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

    inp = VRAMInput.from_gradio(
        model_id,
        gpu_name,
        mode,
        dtype,
        batch_size,
        seq_length,
        gradient_checkpointing,
        optimizer,
        lora_enabled,
        lora_rank,
        use_torch_compile,
        ddp_enabled,
        mixed_precision,
        manual_params_b,
        manual_hidden,
        manual_layers,
        manual_heads,
        manual_kv_heads,
        manual_intermediate,
        manual_vocab_size,
        manual_uses_swiglu,
        manual_num_experts,
        manual_experts_per_token,
        manual_active_params_b,
    )
    estimate = estimate_vram(inp)

    if estimate is None:
        na_viz = create_na_visualization(model_id.strip())
        na_overview = f"""
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
        return (
            na_viz,
            na_overview,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "Auto",
            0,
            0,
            0,
        )

    visualization = create_vram_visualization(estimate, mode)

    recommendations = []
    if not estimate.fits:
        if not lora_enabled and mode == "Training":
            recommendations.append("- Enable **LoRA** to dramatically reduce trainable parameters")
        if "16-bit" not in dtype.lower() and "int" not in dtype.lower():
            recommendations.append("- Use **FP16/BF16** or quantization to reduce model memory")
        if not gradient_checkpointing and mode == "Training":
            recommendations.append("- Enable **gradient checkpointing** to reduce activation memory")
        if batch_size > 1:
            recommendations.append(f"- Reduce **batch size** (current: {batch_size})")
        if seq_length > 2048:
            recommendations.append(f"- Reduce **sequence length** (current: {seq_length})")
        recommendations.append("- Consider using **multi-GPU** setup with FSDP or DeepSpeed")

    reco_text = "\n".join(recommendations) if recommendations else ""

    reco_section = ""
    if reco_text:
        reco_section = f"""
---

### Recommendations
{reco_text}
"""

    token_mb_per_1k = estimate.memory_per_token_kb

    def _fmt_token_delta(ctx_tokens: int) -> str:
        delta_mb = token_mb_per_1k * (ctx_tokens / 1024)
        if delta_mb >= 1024:
            return f"+{(delta_mb / 1024):.2f} GB"
        return f"+{delta_mb:.2f} MB"

    token_scale_cards = "".join(
        [
            (
                f"<div style='padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;'>"
                f"<div style='font-size:11px;color:#94a3b8;'>{ctx:,} tokens</div>"
                f"<div style='font-size:15px;font-weight:700;color:#e2e8f0;'>{_fmt_token_delta(ctx)}</div>"
                f"</div>"
            )
            for ctx in (8192, 32768, 65536, 131072)
        ]
    )

    details_md = f"""
### {'Model FITS' if estimate.fits else 'Model EXCEEDS available VRAM'}
`{model_id}`  |  {mode}
{reco_section}

<details open>
<summary><strong>Core Metrics</strong></summary>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 10px 0;">
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
    <div style="font-size:11px;color:#94a3b8;">Trainable Parameters</div>
    <div style="font-size:16px;font-weight:700;color:#e2e8f0;">{estimate.trainable_params_b:.3f}B</div>
  </div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
    <div style="font-size:11px;color:#94a3b8;">Active Parameters</div>
    <div style="font-size:16px;font-weight:700;color:#e2e8f0;">{estimate.active_params_b:.2f}B</div>
  </div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
    <div style="font-size:11px;color:#94a3b8;">Token Memory</div>
    <div style="font-size:16px;font-weight:700;color:#e2e8f0;">{estimate.memory_per_token_kb:.2f} KB / token</div>
  </div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;">
    <div style="font-size:11px;color:#94a3b8;">Total VRAM</div>
    <div style="font-size:16px;font-weight:700;color:#e2e8f0;">{estimate.total_gb:.2f} GB</div>
  </div>
</div>

</details>

<details>
<summary><strong>Architecture</strong></summary>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 10px 0;">
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;"><div style="font-size:11px;color:#94a3b8;">Hidden Dim</div><div style="font-size:15px;font-weight:700;color:#e2e8f0;">{estimate.config_hidden:,}</div></div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;"><div style="font-size:11px;color:#94a3b8;">Layers</div><div style="font-size:15px;font-weight:700;color:#e2e8f0;">{estimate.config_layers}</div></div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;"><div style="font-size:11px;color:#94a3b8;">Attention Heads</div><div style="font-size:15px;font-weight:700;color:#e2e8f0;">{estimate.config_heads}</div></div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;"><div style="font-size:11px;color:#94a3b8;">KV Heads</div><div style="font-size:15px;font-weight:700;color:#e2e8f0;">{estimate.config_kv_heads}</div></div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;"><div style="font-size:11px;color:#94a3b8;">FFN Intermediate</div><div style="font-size:15px;font-weight:700;color:#e2e8f0;">{estimate.config_intermediate:,}</div></div>
  <div style="padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155;"><div style="font-size:11px;color:#94a3b8;">Vocab Size</div><div style="font-size:15px;font-weight:700;color:#e2e8f0;">{estimate.config_vocab_size:,}</div></div>
</div>

- MoE: {'Yes (' + str(estimate.num_experts) + ' experts, ' + str(estimate.experts_per_token) + ' active/token)' if estimate.is_moe else 'No'}
- SwiGLU: {'Yes' if estimate.uses_swiglu else 'No'}

</details>

<details>
<summary><strong>Token Scaling (Common Context Lengths)</strong></summary>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 10px 0;">
{token_scale_cards}
</div>

<div style="font-size:12px;color:#94a3b8;margin-top:4px;">
In inference mode, this growth is primarily driven by KV cache. Estimated KV-cache growth is
<strong>{estimate.kv_cache_per_token_kb:.2f} KB/token</strong> under the current batch/model setup.
</div>

</details>
"""

    return (
        visualization,
        details_md,
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


def save_custom_gpu_and_refresh(name: str, vram_gb: float):
    """Persist a custom GPU and return dropdown updates + status."""
    try:
        save_custom_gpu_spec(name, vram_gb)
        choices = list(get_all_gpu_specs().keys())
        return (
            gr.update(choices=choices, value=" ".join(name.strip().split())),
            "<span style='color:#22c55e;'>Saved custom GPU.</span>",
        )
    except ValueError as exc:
        return gr.update(), f"<span style='color:#ef4444;'>{exc}</span>"


def build_interface():
    """Build the Gradio interface."""

    with gr.Blocks(title="LLM VRAM Calculator") as demo:
        
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
        
        gpu_choices = list(get_all_gpu_specs().keys())

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                
                with gr.Group():
                    gr.Markdown("### Hardware")
                    gpu_dropdown = gr.Dropdown(
                        choices=gpu_choices,
                        value="NVIDIA H100 SXM 80GB",
                        label="GPU Model",
                        info="Single GPU memory estimation"
                    )
                    with gr.Accordion("Add Custom GPU", open=False):
                        custom_gpu_name = gr.Textbox(label="GPU Name", placeholder="e.g. My Lab GPU 64GB")
                        custom_gpu_vram = gr.Number(label="VRAM (GB)", minimum=1, precision=2)
                        custom_gpu_save_btn = gr.Button("Save Custom GPU", size="sm")
                        custom_gpu_status = gr.HTML("")
                
                with gr.Group():
                    gr.Markdown("### Model")
                    model_id = gr.Textbox(
                        value="meta-llama/Llama-3.1-8B",
                        label="HuggingFace Model ID",
                        placeholder="organization/model-name",
                    )
                    
                    with gr.Accordion("Advanced Model Config", open=False):
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
                        value="16-bit (BF16/FP16)",
                        label="Precision",
                        info="Use 16-bit for most modern training/inference workloads.",
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
                
                with gr.Accordion("Advanced Runtime Options", open=False):
                    use_torch_compile = gr.Checkbox(
                        value=False,
                        label="torch.compile - adds ~10% for compiled graphs",
                    )
                
                calculate_btn = gr.Button(
                    "Calculate VRAM",
                    variant="primary",
                    size="lg",
                )
            
            # Main result area
            with gr.Column(scale=2):
                visualization = gr.HTML(
                    value="<div style='padding: 60px; text-align: center; color: #64748b; font-size: 16px;'>Configure settings and click Calculate</div>"
                )
                details_md = gr.Markdown("")
        
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
        all_outputs = [visualization, details_md] + manual_config_inputs
        
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
        preset_outputs = [model_id, visualization, details_md,
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
        
        # Toggle LoRA rank visibility
        lora_enabled.change(
            fn=lambda x: gr.update(visible=x),
            inputs=lora_enabled,
            outputs=lora_rank,
        )
        
        # Auto-default mixed precision based on dtype
        # FP16 needs mixed precision for numerical stability, BF16 usually doesn't
        def update_mixed_precision_default(dtype_val):
            if "16-bit" in dtype_val.lower():
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

        custom_gpu_save_btn.click(
            fn=save_custom_gpu_and_refresh,
            inputs=[custom_gpu_name, custom_gpu_vram],
            outputs=[gpu_dropdown, custom_gpu_status],
        )
    
    return demo


def launch_app(**launch_kwargs) -> None:
    """Build and launch the Gradio app. Defaults match the CLI entrypoint."""
    demo = build_interface()
    kw = dict(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=_APP_THEME,
        css=_APP_CSS,
    )
    kw.update(launch_kwargs)
    demo.launch(**kw)
