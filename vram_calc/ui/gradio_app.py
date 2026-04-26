"""Gradio Blocks UI for the architecture-driven VRAM calculator."""

import gradio as gr

from vram_calc.constants import DTYPE_BYTES
from vram_calc.custom_gpu_store import get_all_gpu_specs, save_custom_gpu_spec
from vram_calc.engine import estimate_vram
from vram_calc.types import VRAMInput
from vram_calc.ui.html import create_vram_visualization

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

_APP_THEME = gr.themes.Base(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate")


def calculate_and_display(
    params_b: float,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    ffn_multiplier: float,
    attention_type: str,
    ffn_type: str,
    num_experts: int,
    experts_per_token: int,
    active_params_b: float,
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
):
    """Main calculation function for Gradio interface."""
    try:
        inp = VRAMInput.from_gradio(
            params_b,
            hidden,
            layers,
            heads,
            kv_heads,
            intermediate_size,
            vocab_size,
            ffn_multiplier,
            attention_type,
            ffn_type,
            num_experts,
            experts_per_token,
            active_params_b,
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
        )
        estimate = estimate_vram(inp)
    except ValueError as exc:
        return (
            "<div style='padding: 40px; text-align: center; color: #fca5a5;'>Invalid architecture input.</div>",
            f"### Invalid input\n\n{exc}",
        )

    visualization = create_vram_visualization(estimate, mode)
    details_md = f"""
### {'Model FITS' if estimate.fits else 'Model EXCEEDS available VRAM'}
`{mode}` mode with `{estimate.attention_type.upper()}` attention and `{estimate.ffn_type.upper()}` FFN

- Total params: **{estimate.model_params_b:.2f}B**
- Active params: **{estimate.active_params_b:.2f}B**
- Hidden/Layers: **{estimate.config_hidden:,} / {estimate.config_layers}**
- Heads/KV Heads: **{estimate.config_heads} / {estimate.config_kv_heads}**
- Experts (if MoE): **{estimate.num_experts} total, {estimate.experts_per_token} active**
- FFN multiplier: **{estimate.ffn_multiplier:.2f}**
- Total VRAM: **{estimate.total_gb:.2f} GB** ({estimate.utilization_pct:.1f}% of {estimate.available_gb:.1f} GB)
- KV cache growth: **{estimate.kv_cache_per_token_kb:.2f} KB/token**
"""
    return visualization, details_md


def save_custom_gpu_and_refresh(name: str, vram_gb: float):
    """Persist a custom GPU and return dropdown updates + status."""
    try:
        save_custom_gpu_spec(name, vram_gb)
        choices = list(get_all_gpu_specs().keys())
        return gr.update(choices=choices, value=" ".join(name.strip().split())), "<span style='color:#22c55e;'>Saved custom GPU.</span>"
    except ValueError as exc:
        return gr.update(), f"<span style='color:#ef4444;'>{exc}</span>"


def build_interface():
    """Build the Gradio interface."""
    with gr.Blocks(title="LLM VRAM Calculator") as demo:
        gr.Markdown("# LLM VRAM Calculator\nBuild architecture and estimate VRAM dynamically.")
        gpu_choices = list(get_all_gpu_specs().keys())

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Hardware")
                    gpu_dropdown = gr.Dropdown(choices=gpu_choices, value="NVIDIA H100 SXM 80GB", label="GPU Model")
                    with gr.Accordion("Add Custom GPU", open=False):
                        custom_gpu_name = gr.Textbox(label="GPU Name", placeholder="e.g. My Lab GPU 64GB")
                        custom_gpu_vram = gr.Number(label="VRAM (GB)", minimum=1, precision=2)
                        custom_gpu_save_btn = gr.Button("Save Custom GPU", size="sm")
                        custom_gpu_status = gr.HTML("")

                with gr.Group():
                    gr.Markdown("### Transformer Architecture")
                    params_b = gr.Number(value=8.0, label="Total Parameters (B)", minimum=0.01, precision=3)
                    with gr.Row():
                        hidden = gr.Number(value=4096, label="Hidden Size", minimum=1, precision=0)
                        layers = gr.Number(value=32, label="Layers", minimum=1, precision=0)
                    with gr.Row():
                        heads = gr.Number(value=32, label="Attention Heads", minimum=1, precision=0)
                        kv_heads = gr.Number(value=8, label="KV Heads (GQA only)", minimum=1, precision=0, visible=True)
                    with gr.Row():
                        intermediate_size = gr.Number(value=14336, label="FFN Intermediate Size", minimum=1, precision=0)
                        vocab_size = gr.Number(value=128256, label="Vocab Size", minimum=1, precision=0)

                    attention_type = gr.Radio(choices=["mha", "gqa"], value="gqa", label="Attention Type")
                    ffn_type = gr.Radio(choices=["dense", "moe"], value="dense", label="FFN Type")
                    ffn_multiplier = gr.Number(
                        value=3.0,
                        minimum=0.1,
                        precision=2,
                        label="FFN Multiplier",
                        info="Controls FFN expansion/activation shape (higher means more FFN runtime memory).",
                    )

                    with gr.Group(visible=False) as moe_group:
                        gr.Markdown("#### MoE Settings")
                        with gr.Row():
                            num_experts = gr.Number(value=8, label="Number of Experts", minimum=2, precision=0)
                            experts_per_token = gr.Number(value=2, label="Experts per Token", minimum=1, precision=0)
                        active_params_b = gr.Number(
                            value=0,
                            label="Active Parameters (B, optional)",
                            info="0 means auto-derive from total params + routing fraction.",
                            minimum=0,
                            precision=3,
                        )
                with gr.Group():
                    gr.Markdown("### Runtime")
                    mode = gr.Radio(choices=["Training", "Inference"], value="Training", label="Mode")
                    dtype = gr.Dropdown(choices=list(DTYPE_BYTES.keys()), value="16-bit (BF16/FP16)", label="Precision")
                    batch_size = gr.Slider(minimum=1, maximum=1024, value=1, step=1, label="Batch Size")
                    seq_length = gr.Slider(minimum=128, maximum=131072, value=2048, step=128, label="Sequence Length")

                with gr.Group():
                    gr.Markdown("### Training Options")
                    gradient_checkpointing = gr.Checkbox(value=True, label="Gradient Checkpointing")
                    optimizer = gr.Dropdown(
                        choices=["AdamW (32-bit)", "AdamW (8-bit)", "SGD", "Adafactor"],
                        value="AdamW (32-bit)",
                        label="Optimizer",
                    )
                    mixed_precision = gr.Checkbox(value=False, label="Mixed Precision (FP32 master weights)")
                    lora_enabled = gr.Checkbox(value=True, label="LoRA Enabled")
                    lora_rank = gr.Slider(minimum=4, maximum=256, value=16, step=4, label="LoRA Rank")
                    ddp_enabled = gr.Checkbox(value=False, label="DDP overhead")

                with gr.Accordion("Advanced Runtime Options", open=False):
                    use_torch_compile = gr.Checkbox(value=False, label="torch.compile overhead")

                calculate_btn = gr.Button("Calculate VRAM", variant="primary", size="lg")

            with gr.Column(scale=2):
                visualization = gr.HTML(
                    value="<div style='padding: 60px; text-align: center; color: #64748b;'>Configure architecture and click Calculate</div>"
                )
                details_md = gr.Markdown("")

        all_inputs = [
            params_b,
            hidden,
            layers,
            heads,
            kv_heads,
            intermediate_size,
            vocab_size,
            ffn_multiplier,
            attention_type,
            ffn_type,
            num_experts,
            experts_per_token,
            active_params_b,
            gpu_dropdown,
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
        ]
        all_outputs = [visualization, details_md]

        calculate_btn.click(fn=calculate_and_display, inputs=all_inputs, outputs=all_outputs)

        for component in [gpu_dropdown, mode, dtype, batch_size, seq_length, attention_type, ffn_type]:
            component.change(fn=calculate_and_display, inputs=all_inputs, outputs=all_outputs)

        ffn_type.change(fn=lambda v: gr.update(visible=v == "moe"), inputs=ffn_type, outputs=moe_group)
        attention_type.change(fn=lambda v: gr.update(visible=v == "gqa"), inputs=attention_type, outputs=kv_heads)
        lora_enabled.change(fn=lambda x: gr.update(visible=x), inputs=lora_enabled, outputs=lora_rank)
        mode.change(
            fn=lambda m: (
                gr.update(visible=m == "Training"),
                gr.update(visible=m == "Training"),
                gr.update(visible=m == "Training"),
                gr.update(visible=m == "Training"),
                gr.update(visible=m == "Training"),
            ),
            inputs=mode,
            outputs=[gradient_checkpointing, optimizer, mixed_precision, lora_enabled, ddp_enabled],
        )

        custom_gpu_save_btn.click(
            fn=save_custom_gpu_and_refresh,
            inputs=[custom_gpu_name, custom_gpu_vram],
            outputs=[gpu_dropdown, custom_gpu_status],
        )

    return demo


def launch_app(**launch_kwargs) -> None:
    """Build and launch the Gradio app."""
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
