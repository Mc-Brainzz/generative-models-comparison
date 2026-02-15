"""
PAIRED vs TRULY UNPAIRED COMPARISON

This directly compares:
1. Paired training (LR-HR from same trajectory)
2. Truly unpaired training (LR and HR from different trajectories)

Both tested on the same paired test set.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from flow_matching import (
    Config, set_seed, get_device, InterpolantSchedule,
    VelocityUNet, van_der_pol_ode, points_to_image,
    degrade_operator, sample_ot_coupling, compute_ssim_scores,
    compute_psnr, flow_matching_inference, train_flow_matching,
    prepare_data_loaders
)
from test_truly_unpaired import (
    generate_trajectory_images, create_truly_unpaired_dataset,
    train_truly_unpaired, evaluate_unpaired_model
)


def main():
    print("=" * 70)
    print("PAIRED vs TRULY UNPAIRED COMPARISON")
    print("=" * 70)
    
    config = Config(
        n_images=300,
        epochs=40,
        batch_size=8,
        base_channels=32,
        fm_type='stochastic',
        inference_mode='ode',
        inference_steps=50,
        seed=42
    )
    
    device = get_device()
    
    # =========================================================================
    # Experiment 1: Paired Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: PAIRED TRAINING")
    print("=" * 70)
    print("LR and HR come from the SAME trajectories (normal supervised learning)")
    
    set_seed(config.seed)
    config_paired = Config(**{**config.__dict__, 'coupling_mode': 'paired'})
    
    train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config_paired, device)
    
    model_paired = VelocityUNet(base_channels=config.base_channels).to(device)
    model_paired = train_flow_matching(model_paired, train_loader, config_paired, device)
    
    # Evaluate paired model
    schedule = InterpolantSchedule('stochastic', sigma_max=config.sigma_max)
    sr_paired = flow_matching_inference(model_paired, lr_val, config_paired, device, schedule)
    ssim_paired = np.mean(compute_ssim_scores(sr_paired, x1_val))
    psnr_paired = compute_psnr(sr_paired, x1_val)
    
    print(f"\nPaired Training Results:")
    print(f"  SSIM: {ssim_paired:.4f}")
    print(f"  PSNR: {psnr_paired:.2f} dB")
    
    # =========================================================================
    # Experiment 2: Truly Unpaired Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: TRULY UNPAIRED TRAINING")
    print("=" * 70)
    print("LR and HR come from DIFFERENT trajectories (no correspondence)")
    
    set_seed(config.seed)
    config_unpaired = Config(**{**config.__dict__, 'coupling_mode': 'unpaired', 'ot_reg': 0.01})
    
    HR_train, LR_up_train, HR_test, LR_test = create_truly_unpaired_dataset(config_unpaired, device)
    
    model_unpaired = VelocityUNet(base_channels=config.base_channels).to(device)
    model_unpaired = train_truly_unpaired(model_unpaired, LR_up_train, HR_train, config_unpaired, device)
    
    # Evaluate unpaired model on the same test set structure
    sr_unpaired, ssim_unpaired, psnr_unpaired = evaluate_unpaired_model(
        model_unpaired, LR_test, HR_test, config_unpaired, device
    )
    
    print(f"\nTruly Unpaired Training Results:")
    print(f"  SSIM: {ssim_unpaired:.4f}")
    print(f"  PSNR: {psnr_unpaired:.2f} dB")
    
    # =========================================================================
    # Baseline
    # =========================================================================
    bicubic_up = F.interpolate(LR_test, scale_factor=4, mode='bicubic', align_corners=False)
    ssim_bicubic = np.mean(compute_ssim_scores(bicubic_up, HR_test))
    psnr_bicubic = compute_psnr(bicubic_up, HR_test)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'SSIM':>10} {'PSNR (dB)':>12}")
    print("-" * 52)
    print(f"{'Bicubic Upsampling':<30} {ssim_bicubic:>10.4f} {psnr_bicubic:>12.2f}")
    print(f"{'Truly Unpaired (OT)':<30} {ssim_unpaired:>10.4f} {psnr_unpaired:>12.2f}")
    print(f"{'Paired Training':<30} {ssim_paired:>10.4f} {psnr_paired:>12.2f}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print(f"""
    1. PAIRED TRAINING (SSIM={ssim_paired:.4f}):
       - Has direct correspondence between LR and HR
       - Model learns exact mapping from degraded → original
       - Best performance as expected
    
    2. TRULY UNPAIRED (SSIM={ssim_unpaired:.4f}):
       - NO correspondence between training LR and HR
       - OT coupling tries to find best matches within mini-batches
       - Learns distribution-to-distribution mapping
       - Performance gap shows the value of paired data
    
    3. THE UNPAIRED CHALLENGE:
       - Without pairs, the model learns to map LR distribution → HR distribution
       - But specific details may not match (different trajectories!)
       - OT helps but can't fully compensate for missing supervision
       
    WHEN UNPAIRED IS USEFUL:
       - When paired data is impossible to obtain
       - Domain adaptation (e.g., simulation → real)
       - Style transfer where exact correspondence doesn't matter
    """)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Paired vs Truly Unpaired Training Comparison", fontsize=14, fontweight='bold')
    
    # Row 1: Paired model results (using lr_val)
    axes[0, 0].set_ylabel("Paired\nTraining", fontsize=12, rotation=0, ha='right', va='center')
    for i in range(4):
        if i < lr_val.size(0):
            if i == 0:
                lr_up = F.interpolate(lr_val[i:i+1], scale_factor=4, mode='nearest')[0, 0]
                axes[0, 0].imshow(lr_up.cpu().numpy(), cmap='viridis')
                axes[0, 0].set_title("LR Input")
            elif i == 1:
                axes[0, 1].imshow(sr_paired[0, 0].cpu().numpy(), cmap='viridis')
                ssim_val = compute_ssim_scores(sr_paired[0:1], x1_val[0:1])[0]
                axes[0, 1].set_title(f"SR (SSIM={ssim_val:.3f})")
            elif i == 2:
                axes[0, 2].imshow(x1_val[0, 0].cpu().numpy(), cmap='viridis')
                axes[0, 2].set_title("HR Ground Truth")
            elif i == 3:
                axes[0, 3].imshow(sr_paired[1, 0].cpu().numpy(), cmap='viridis')
                ssim_val = compute_ssim_scores(sr_paired[1:2], x1_val[1:2])[0]
                axes[0, 3].set_title(f"SR #2 (SSIM={ssim_val:.3f})")
        axes[0, i].axis('off')
    
    # Row 2: Unpaired model results
    axes[1, 0].set_ylabel("Unpaired\nTraining", fontsize=12, rotation=0, ha='right', va='center')
    for i in range(4):
        if i < LR_test.size(0):
            if i == 0:
                lr_up = F.interpolate(LR_test[i:i+1], scale_factor=4, mode='nearest')[0, 0]
                axes[1, 0].imshow(lr_up.cpu().numpy(), cmap='viridis')
                axes[1, 0].set_title("LR Input")
            elif i == 1:
                axes[1, 1].imshow(sr_unpaired[0, 0].cpu().numpy(), cmap='viridis')
                ssim_val = compute_ssim_scores(sr_unpaired[0:1], HR_test[0:1])[0]
                axes[1, 1].set_title(f"SR (SSIM={ssim_val:.3f})")
            elif i == 2:
                axes[1, 2].imshow(HR_test[0, 0].cpu().numpy(), cmap='viridis')
                axes[1, 2].set_title("HR Ground Truth")
            elif i == 3:
                axes[1, 3].imshow(sr_unpaired[1, 0].cpu().numpy(), cmap='viridis')
                ssim_val = compute_ssim_scores(sr_unpaired[1:2], HR_test[1:2])[0]
                axes[1, 3].set_title(f"SR #2 (SSIM={ssim_val:.3f})")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results_paired_vs_unpaired.png", dpi=150, bbox_inches='tight')
    print("\nFigure saved to results_paired_vs_unpaired.png")
    plt.show()


if __name__ == "__main__":
    main()
