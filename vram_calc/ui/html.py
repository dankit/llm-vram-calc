"""HTML visualization for VRAM estimates."""

from vram_calc.types import VRAMEstimate


def _stat_card(label: str, value: str, color: str, subtitle: str = "") -> str:
    """Generate HTML for a stat card in the visualization grid."""
    subtitle_html = f'<div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">{subtitle}</div>' if subtitle else ''
    return f"""
        <div style="padding: 14px; background: linear-gradient(135deg, #1e293b, #0f172a); 
                    border-radius: 10px; text-align: center; border: 1px solid #334155;">
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
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 500; color: #e2e8f0; font-size: 14px;">
                    <span style="display: inline-block; width: 12px; height: 12px; 
                                 background: {color}; border-radius: 3px; margin-right: 8px;"></span>
                    {name}
                </span>
                <span style="color: #94a3b8; font-size: 14px; font-family: 'JetBrains Mono', monospace;">
                    {value:.2f} GB ({pct:.1f}%)
                </span>
            </div>
            <div style="height: 20px; background: #1e293b; border-radius: 6px; overflow: hidden;">
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
            display_name = "Activations (FFN)" if name == "Activations (FFN)" else name
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
        <div style="padding: 14px; background: #1e293b; border-radius: 16px; margin-bottom: 14px;">
            <h3 style="margin: 0 0 12px 0; font-size: 18px; font-weight: 600; color: #e2e8f0;">
                Activation Memory Breakdown
            </h3>
            <div style="display: flex; height: 34px; border-radius: 8px; overflow: hidden; margin-bottom: 12px; border: 1px solid #334155;">
                <div style="width: {attn_pct}%; background: #ec4899; border-right: 2px solid #0f172a;
                            display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 11px; font-weight: 600; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                        {attn_pct:.0f}%
                    </span>
                </div>
                <div style="width: {ffn_pct}%; background: #f97316; border-right: 2px solid #0f172a;
                            display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 11px; font-weight: 600; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                        {ffn_pct:.0f}%
                    </span>
                </div>
                <div style="width: {other_pct}%; background: #64748b;
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
                    <span style="display: inline-block; width: 12px; height: 12px; background: #f97316; border-radius: 3px;"></span>
                    <span style="font-size: 13px; color: #e2e8f0;">FFN</span>
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
                padding: 18px; background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
                border-radius: 20px; color: #f8fafc; border: 1px solid #334155;">
        
        <!-- Status Header -->
        <div style="text-align: center; margin-bottom: 18px; padding: 14px; 
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
        <div style="margin-bottom: 18px; padding: 14px; background: #1e293b; border-radius: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <span style="font-size: 18px; font-weight: 600; color: #e2e8f0;">
                    VRAM Utilization
                </span>
                <span style="font-size: 28px; font-weight: 800; color: {status_color}; 
                            font-family: 'JetBrains Mono', monospace;">
                    {estimate.utilization_pct:.1f}%
                </span>
            </div>
            <div style="height: 36px; background: #0f172a; border-radius: 12px; overflow: hidden; 
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
        <div style="padding: 14px; background: #1e293b; border-radius: 16px; margin-bottom: 14px;">
            <h3 style="margin: 0 0 20px 0; font-size: 18px; font-weight: 600; color: #e2e8f0;">
                Full Memory Breakdown
            </h3>
            {breakdown_html}
        </div>
        
        <!-- Stats Grid -->
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
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


def create_na_visualization(error_message: str = "") -> str:
    """Create visualization when architecture input is invalid."""
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
                Architecture input is incomplete or invalid.
            </div>
            
            <div style="margin-top: 32px; padding: 24px; background: linear-gradient(135deg, #1e3a5f, #1e293b); border-radius: 12px; 
                        text-align: left; max-width: 520px; margin-left: auto; margin-right: auto; border: 1px solid #3b82f6;">
                <div style="font-size: 16px; font-weight: 600; color: #60a5fa; margin-bottom: 16px;">
                    Please provide valid architecture values:
                </div>
                <div style="color: #e2e8f0; font-size: 14px; line-height: 1.8;">
                    <p style="margin: 0 0 12px 0;">Provide at minimum:</p>
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
            
            {"<div style='margin-top: 24px; color: #94a3b8; font-size: 13px;'>" + error_message + "</div>" if error_message else ""}
        </div>
    </div>
    """