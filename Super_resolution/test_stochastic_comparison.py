"""
STOCHASTIC vs DETERMINISTIC COMPARISON

This demonstrates the difference between:
1. Deterministic Flow Matching (linear interpolation, ODE inference)
2. Stochastic Flow Matching (stochastic interpolation, ODE inference) 
3. Full Stochastic (stochastic interpolation, SDE inference)

Also explains bicubic upsampling as baseline.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from flow_matching import (
    Config, set_seed, get_device, InterpolantSchedule,
    VelocityUNet, prepare_data_loaders, train_flow_matching,
    flow_matching_inference, compute_ssim_scores, compute_psnr
)


def explain_interpolation():
    """Visual explanation of interpolation types."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    UNDERSTANDING INTERPOLATION                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  BICUBIC UPSAMPLING (Baseline - No Learning):                           ║
║  ─────────────────────────────────────────────                          ║
║  LR (32×32) ──[cubic interpolation]──> HR (128×128)                     ║
║                                                                          ║
║  • Uses 16 nearest neighbors with cubic polynomial                       ║
║  • No neural network, just math                                         ║
║  • CANNOT add new details - only smoothly enlarges                      ║
║  • Fast but blurry                                                      ║
║                                                                          ║
║  ═══════════════════════════════════════════════════════════════════    ║
║                                                                          ║
║  FLOW MATCHING - DETERMINISTIC (Linear Interpolation):                  ║
║  ────────────────────────────────────────────────────                   ║
║                                                                          ║
║  Training path:  x_t = (1-t)·x₀ + t·x₁                                  ║
║                                                                          ║
║  t=0 ────────────────────────────────────────> t=1                      ║
║   x₀                    x_t                      x₁                      ║
║  (LR)                                           (HR)                     ║
║                                                                          ║
║  • Straight line between source and target                              ║
║  • Target velocity: v* = x₁ - x₀ (constant!)                           ║
║  • ODE inference: deterministic output                                  ║
║                                                                          ║
║  ═══════════════════════════════════════════════════════════════════    ║
║                                                                          ║
║  FLOW MATCHING - STOCHASTIC (Stochastic Interpolation):                 ║
║  ─────────────────────────────────────────────────────                  ║
║                                                                          ║
║  Training path:  x_t = (1-t)·x₀ + t·x₁ + σ(t)·ε                        ║
║                                                                          ║
║  t=0 ─────────╔═══════════╗────────────────────> t=1                    ║
║   x₀          ║ + noise   ║                        x₁                    ║
║  (LR)         ║  σ(t)·ε   ║                       (HR)                   ║
║               ╚═══════════╝                                              ║
║                     ↑                                                    ║
║               σ(t) = σ_max · sin(πt)                                    ║
║               (zero at boundaries, max at t=0.5)                        ║
║                                                                          ║
║  • Noise added during training path                                     ║
║  • σ(0) = σ(1) = 0 (starts at x₀, ends at x₁)                         ║
║  • Regularization effect - better generalization                        ║
║  • Can use ODE (deterministic) or SDE (stochastic) inference           ║
║                                                                          ║
║  ═══════════════════════════════════════════════════════════════════    ║
║                                                                          ║
║  INFERENCE MODES:                                                        ║
║  ────────────────                                                        ║
║                                                                          ║
║  ODE: dx/dt = v(x,t)                                                    ║
║       → Deterministic: same input → same output                         ║
║                                                                          ║
║  SDE: dx = v(x,t)dt + σ'(t)dW                                          ║
║       → Stochastic: same input → different plausible outputs           ║
║       → Good for uncertainty estimation & diverse samples               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def run_comparison():
    """Compare all interpolation/inference combinations."""
    
    explain_interpolation()
    
    device = get_device()
    
    # Base config
    base_config = {
        'n_images': 300,
        'epochs': 30,
        'batch_size': 8,
        'base_channels': 32,
        'inference_steps': 50,
        'seed': 42
    }
    
    results = []
    
    # =========================================================================
    # 1. DETERMINISTIC (Linear interpolation + ODE)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. DETERMINISTIC FLOW MATCHING")
    print("   Training: Linear interpolation x_t = (1-t)x₀ + tx₁")
    print("   Inference: ODE (deterministic)")
    print("=" * 70)
    
    set_seed(42)
    config_det = Config(
        **base_config,
        fm_type='deterministic',  # Linear interpolation, no noise
        sigma_max=0.0,
        inference_mode='ode',
        coupling_mode='paired'
    )
    
    train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config_det, device)
    model_det = VelocityUNet(base_channels=config_det.base_channels).to(device)
    
    schedule_det = InterpolantSchedule('linear', sigma_max=0.0)
    model_det = train_flow_matching(model_det, train_loader, config_det, device)
    
    sr_det = flow_matching_inference(model_det, lr_val, config_det, device, schedule_det)
    ssim_det = np.mean(compute_ssim_scores(sr_det, x1_val))
    psnr_det = compute_psnr(sr_det, x1_val)
    
    results.append(('Deterministic (ODE)', ssim_det, psnr_det, 'Linear', 'ODE'))
    print(f"Results: SSIM={ssim_det:.4f}, PSNR={psnr_det:.2f} dB")
    
    # =========================================================================
    # 2. STOCHASTIC TRAINING + ODE INFERENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. STOCHASTIC TRAINING + ODE INFERENCE")
    print("   Training: Stochastic interpolation x_t = (1-t)x₀ + tx₁ + σ(t)ε")
    print("   Inference: ODE (deterministic output)")
    print("=" * 70)
    
    set_seed(42)
    config_stoch_ode = Config(
        **base_config,
        fm_type='stochastic',     # Stochastic interpolation with noise
        sigma_max=0.1,            # Noise level
        inference_mode='ode',     # But deterministic inference
        coupling_mode='paired'
    )
    
    train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config_stoch_ode, device)
    model_stoch = VelocityUNet(base_channels=config_stoch_ode.base_channels).to(device)
    
    schedule_stoch = InterpolantSchedule('stochastic', sigma_max=0.1)
    model_stoch = train_flow_matching(model_stoch, train_loader, config_stoch_ode, device)
    
    sr_stoch_ode = flow_matching_inference(model_stoch, lr_val, config_stoch_ode, device, schedule_stoch)
    ssim_stoch_ode = np.mean(compute_ssim_scores(sr_stoch_ode, x1_val))
    psnr_stoch_ode = compute_psnr(sr_stoch_ode, x1_val)
    
    results.append(('Stochastic + ODE', ssim_stoch_ode, psnr_stoch_ode, 'Stochastic', 'ODE'))
    print(f"Results: SSIM={ssim_stoch_ode:.4f}, PSNR={psnr_stoch_ode:.2f} dB")
    
    # =========================================================================
    # 3. STOCHASTIC TRAINING + SDE INFERENCE (Full Stochastic)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. FULL STOCHASTIC (SDE INFERENCE)")
    print("   Training: Stochastic interpolation")
    print("   Inference: SDE (stochastic - diverse outputs)")
    print("=" * 70)
    
    # Use the same trained model but with SDE inference
    config_stoch_sde = Config(
        **base_config,
        fm_type='stochastic',
        sigma_max=0.1,
        inference_mode='sde',      # Stochastic inference!
        sde_noise_scale=0.3,       # Moderate noise for diversity
        coupling_mode='paired'
    )
    
    # Run SDE inference multiple times to show diversity
    print("\nRunning SDE inference 3 times on the same input...")
    sr_sde_runs = []
    ssim_sde_runs = []
    
    for run in range(3):
        sr_sde = flow_matching_inference(model_stoch, lr_val[:5], config_stoch_sde, device, schedule_stoch)
        ssim_run = np.mean(compute_ssim_scores(sr_sde, x1_val[:5]))
        sr_sde_runs.append(sr_sde)
        ssim_sde_runs.append(ssim_run)
        print(f"  Run {run+1}: SSIM={ssim_run:.4f}")
    
    ssim_sde_mean = np.mean(ssim_sde_runs)
    ssim_sde_std = np.std(ssim_sde_runs)
    print(f"\nSDE Results: SSIM={ssim_sde_mean:.4f} ± {ssim_sde_std:.4f}")
    
    results.append(('Full Stochastic (SDE)', ssim_sde_mean, 0, 'Stochastic', 'SDE'))
    
    # =========================================================================
    # 4. BICUBIC BASELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. BICUBIC UPSAMPLING (Baseline)")
    print("   No learning - just mathematical interpolation")
    print("=" * 70)
    
    bicubic_up = F.interpolate(lr_val, scale_factor=4, mode='bicubic', align_corners=False)
    ssim_bicubic = np.mean(compute_ssim_scores(bicubic_up, x1_val))
    psnr_bicubic = compute_psnr(bicubic_up, x1_val)
    
    results.append(('Bicubic (No Learning)', ssim_bicubic, psnr_bicubic, 'N/A', 'N/A'))
    print(f"Results: SSIM={ssim_bicubic:.4f}, PSNR={psnr_bicubic:.2f} dB")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'SSIM':>10} {'PSNR':>10} {'Training':>12} {'Inference':>10}")
    print("-" * 72)
    for name, ssim_val, psnr_val, train_type, infer_type in results:
        psnr_str = f"{psnr_val:.2f}" if psnr_val > 0 else "N/A"
        print(f"{name:<30} {ssim_val:>10.4f} {psnr_str:>10} {train_type:>12} {infer_type:>10}")
    
    # =========================================================================
    # Visualize SDE Diversity
    # =========================================================================
    print("\n" + "=" * 70)
    print("VISUALIZING SDE DIVERSITY")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("SDE Inference: Same Input → Different Outputs", fontsize=14, fontweight='bold')
    
    # Row 1: First sample
    idx = 0
    lr_up = F.interpolate(lr_val[idx:idx+1], scale_factor=4, mode='nearest')[0, 0]
    axes[0, 0].imshow(lr_up.cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title("LR Input")
    axes[0, 0].axis('off')
    
    for i, sr_run in enumerate(sr_sde_runs):
        axes[0, i+1].imshow(sr_run[idx, 0].cpu().numpy(), cmap='viridis')
        axes[0, i+1].set_title(f"SDE Run {i+1}")
        axes[0, i+1].axis('off')
    
    axes[0, 4].imshow(x1_val[idx, 0].cpu().numpy(), cmap='viridis')
    axes[0, 4].set_title("HR Ground Truth")
    axes[0, 4].axis('off')
    
    # Row 2: Second sample
    idx = 1
    lr_up = F.interpolate(lr_val[idx:idx+1], scale_factor=4, mode='nearest')[0, 0]
    axes[1, 0].imshow(lr_up.cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title("LR Input")
    axes[1, 0].axis('off')
    
    for i, sr_run in enumerate(sr_sde_runs):
        axes[1, i+1].imshow(sr_run[idx, 0].cpu().numpy(), cmap='viridis')
        axes[1, i+1].set_title(f"SDE Run {i+1}")
        axes[1, i+1].axis('off')
    
    axes[1, 4].imshow(x1_val[idx, 0].cpu().numpy(), cmap='viridis')
    axes[1, 4].set_title("HR Ground Truth")
    axes[1, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig("results_stochastic_comparison.png", dpi=150, bbox_inches='tight')
    print("Figure saved to results_stochastic_comparison.png")
    plt.show()
    
    # =========================================================================
    # Key Insights
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    1. BICUBIC UPSAMPLING:
       • No learning - uses cubic polynomial interpolation
       • High SSIM because it preserves structure well
       • But CANNOT add details lost in degradation
       • Just "smooth zoom" - no intelligence
    
    2. DETERMINISTIC FLOW MATCHING:
       • Linear interpolation: x_t = (1-t)x₀ + tx₁
       • Straight path from LR to HR
       • ODE inference gives single deterministic output
       • Good for reproducibility
    
    3. STOCHASTIC TRAINING + ODE INFERENCE:
       • Training path has noise: x_t = (1-t)x₀ + tx₁ + σ(t)ε
       • Acts as regularization - often better generalization
       • Still deterministic output at inference
       • Best of both worlds for many applications
    
    4. FULL STOCHASTIC (SDE INFERENCE):
       • Same training as above
       • SDE inference adds noise: dx = v(x,t)dt + σ'(t)dW
       • Each run gives DIFFERENT plausible output
       • Useful for:
         - Uncertainty estimation
         - Generating diverse samples
         - When multiple valid solutions exist
    
    WHY USE STOCHASTIC?
    ─────────────────────
    • Super-resolution is ILL-POSED: many HR images can produce the same LR
    • Stochastic methods capture this uncertainty
    • Multiple valid reconstructions are possible!
    """)


if __name__ == "__main__":
    run_comparison()
