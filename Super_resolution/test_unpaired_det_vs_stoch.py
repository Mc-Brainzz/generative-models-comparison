"""
TRULY UNPAIRED: Deterministic vs Stochastic Training

Based on our comparison, deterministic (linear interpolation) worked
much better than stochastic. Let's test both on the truly unpaired case.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from flow_matching import (
    Config, set_seed, get_device, InterpolantSchedule,
    VelocityUNet, sample_ot_coupling, compute_ssim_scores,
    compute_psnr, flow_matching_inference
)
from test_truly_unpaired import (
    create_truly_unpaired_dataset, evaluate_unpaired_model
)


def train_truly_unpaired_configurable(
    model: nn.Module,
    x0_train: torch.Tensor,
    x1_train: torch.Tensor,
    config: Config,
    schedule: InterpolantSchedule,
    device: torch.device
) -> nn.Module:
    """Train with configurable interpolation (deterministic or stochastic)."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    mse_loss = nn.MSELoss()
    
    n_samples = x0_train.size(0)
    model.train()
    
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        perm0 = torch.randperm(n_samples, device=device)
        perm1 = torch.randperm(n_samples, device=device)
        
        for i in range(0, n_samples - config.batch_size + 1, config.batch_size):
            x0_batch = x0_train[perm0[i:i+config.batch_size]]
            x1_batch = x1_train[perm1[i:i+config.batch_size]]
            
            # OT coupling
            x0_batch, x1_batch = sample_ot_coupling(x0_batch, x1_batch, reg=config.ot_reg)
            
            # Sample time
            t = torch.rand(config.batch_size, 1, 1, 1, device=device)
            
            # Sample noise (used only if stochastic)
            noise = torch.randn_like(x0_batch) if config.fm_type == 'stochastic' else None
            
            # Interpolate using schedule
            x_t = schedule.interpolate(x0_batch, x1_batch, t, noise)
            
            # Target velocity
            v_target = schedule.velocity_target(x0_batch, x1_batch, t, noise)
            
            # Predict
            v_pred = model(x_t, t.squeeze())
            
            # Loss
            loss = mse_loss(v_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch:02d}/{config.epochs} | Loss: {avg_loss:.5f}")
    
    return model


def main():
    print("=" * 70)
    print("TRULY UNPAIRED: Deterministic vs Stochastic Training")
    print("=" * 70)
    
    device = get_device()
    
    base_config = {
        'n_images': 300,
        'epochs': 40,
        'batch_size': 8,
        'base_channels': 32,
        'coupling_mode': 'unpaired',
        'ot_reg': 0.01,
        'inference_mode': 'ode',
        'inference_steps': 50,
        'seed': 42
    }
    
    results = []
    
    # =========================================================================
    # 1. DETERMINISTIC (Linear interpolation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. TRULY UNPAIRED + DETERMINISTIC TRAINING")
    print("   x_t = (1-t)·x₀ + t·x₁  (no noise)")
    print("=" * 70)
    
    set_seed(42)
    config_det = Config(**base_config, fm_type='deterministic', sigma_max=0.0)
    
    HR_train, LR_up_train, HR_test, LR_test = create_truly_unpaired_dataset(config_det, device)
    
    model_det = VelocityUNet(base_channels=config_det.base_channels).to(device)
    schedule_det = InterpolantSchedule('linear', sigma_max=0.0)
    
    start = time.time()
    model_det = train_truly_unpaired_configurable(
        model_det, LR_up_train, HR_train, config_det, schedule_det, device
    )
    train_time_det = time.time() - start
    
    sr_det, ssim_det, psnr_det = evaluate_unpaired_model(
        model_det, LR_test, HR_test, config_det, device
    )
    
    results.append(('Deterministic', ssim_det, psnr_det, train_time_det))
    print(f"\nResults: SSIM={ssim_det:.4f}, PSNR={psnr_det:.2f} dB, Time={train_time_det:.1f}s")
    
    # =========================================================================
    # 2. STOCHASTIC (with noise)
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. TRULY UNPAIRED + STOCHASTIC TRAINING")
    print("   x_t = (1-t)·x₀ + t·x₁ + σ(t)·ε  (with noise)")
    print("=" * 70)
    
    set_seed(42)
    config_stoch = Config(**base_config, fm_type='stochastic', sigma_max=0.1)
    
    HR_train, LR_up_train, HR_test, LR_test = create_truly_unpaired_dataset(config_stoch, device)
    
    model_stoch = VelocityUNet(base_channels=config_stoch.base_channels).to(device)
    schedule_stoch = InterpolantSchedule('stochastic', sigma_max=0.1)
    
    start = time.time()
    model_stoch = train_truly_unpaired_configurable(
        model_stoch, LR_up_train, HR_train, config_stoch, schedule_stoch, device
    )
    train_time_stoch = time.time() - start
    
    sr_stoch, ssim_stoch, psnr_stoch = evaluate_unpaired_model(
        model_stoch, LR_test, HR_test, config_stoch, device
    )
    
    results.append(('Stochastic', ssim_stoch, psnr_stoch, train_time_stoch))
    print(f"\nResults: SSIM={ssim_stoch:.4f}, PSNR={psnr_stoch:.2f} dB, Time={train_time_stoch:.1f}s")
    
    # =========================================================================
    # 3. Baseline
    # =========================================================================
    bicubic_up = F.interpolate(LR_test, scale_factor=4, mode='bicubic', align_corners=False)
    ssim_bicubic = np.mean(compute_ssim_scores(bicubic_up, HR_test))
    psnr_bicubic = compute_psnr(bicubic_up, HR_test)
    results.append(('Bicubic', ssim_bicubic, psnr_bicubic, 0))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (Truly Unpaired)")
    print("=" * 70)
    print(f"{'Method':<25} {'SSIM':>10} {'PSNR (dB)':>12} {'Train Time':>12}")
    print("-" * 60)
    for name, ssim_val, psnr_val, t_time in results:
        print(f"{name:<25} {ssim_val:>10.4f} {psnr_val:>12.2f} {t_time:>12.1f}s")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
    Previous truly unpaired result (stochastic): SSIM={ssim_stoch:.4f}
    New deterministic training result:           SSIM={ssim_det:.4f}
    
    Difference: {ssim_det - ssim_stoch:+.4f}
    
    {'✓ Deterministic is BETTER!' if ssim_det > ssim_stoch else '✗ Stochastic was better'}
    
    Why?
    - Linear interpolation has simpler target: v* = x₁ - x₀ (constant)
    - Stochastic adds noise making training harder
    - For sparse Van der Pol trajectories, deterministic works better
    """)
    
    # =========================================================================
    # Visualization
    # =========================================================================
    import matplotlib.pyplot as plt
    
    n_samples = 4
    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3 * n_samples))
    fig.suptitle("Truly Unpaired: Deterministic vs Stochastic Training", fontsize=14, fontweight='bold')
    
    # Column headers
    col_titles = ["LR Input", "Bicubic", "Deterministic SR", "Stochastic SR", "HR Ground Truth"]
    
    for i in range(n_samples):
        # LR Input (upsampled for display)
        lr_up = F.interpolate(LR_test[i:i+1], scale_factor=4, mode='nearest')[0, 0]
        axes[i, 0].imshow(lr_up.cpu().numpy(), cmap='viridis')
        if i == 0:
            axes[i, 0].set_title(col_titles[0])
        axes[i, 0].axis('off')
        
        # Bicubic
        axes[i, 1].imshow(bicubic_up[i, 0].cpu().numpy(), cmap='viridis')
        if i == 0:
            axes[i, 1].set_title(f"{col_titles[1]}\nSSIM={ssim_bicubic:.3f}")
        axes[i, 1].axis('off')
        
        # Deterministic SR
        ssim_det_i = compute_ssim_scores(sr_det[i:i+1], HR_test[i:i+1])[0]
        axes[i, 2].imshow(sr_det[i, 0].cpu().numpy(), cmap='viridis')
        if i == 0:
            axes[i, 2].set_title(f"{col_titles[2]}\nSSIM={ssim_det:.3f}")
        else:
            axes[i, 2].set_ylabel(f"SSIM={ssim_det_i:.3f}", rotation=0, ha='right', va='center')
        axes[i, 2].axis('off')
        
        # Stochastic SR
        ssim_stoch_i = compute_ssim_scores(sr_stoch[i:i+1], HR_test[i:i+1])[0]
        axes[i, 3].imshow(sr_stoch[i, 0].cpu().numpy(), cmap='viridis')
        if i == 0:
            axes[i, 3].set_title(f"{col_titles[3]}\nSSIM={ssim_stoch:.3f}")
        axes[i, 3].axis('off')
        
        # HR Ground Truth
        axes[i, 4].imshow(HR_test[i, 0].cpu().numpy(), cmap='viridis')
        if i == 0:
            axes[i, 4].set_title(col_titles[4])
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig("results_unpaired_det_vs_stoch.png", dpi=150, bbox_inches='tight')
    print("\nFigure saved to results_unpaired_det_vs_stoch.png")
    plt.show()


if __name__ == "__main__":
    main()
