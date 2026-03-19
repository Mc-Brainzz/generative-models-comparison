"""
TRUE UNPAIRED DATA EXPERIMENT

This tests the real unpaired scenario:
- Training: LR images from one set of trajectories, HR images from DIFFERENT trajectories
- No correspondence between training LR and HR!
- Test: Can the model learn the general LR→HR mapping?

This is the real test of whether OT coupling can learn from unpaired distributions.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from flow_matching import (
    Config, set_seed, get_device, InterpolantSchedule,
    VelocityUNet, van_der_pol_ode, points_to_image,
    degrade_operator, sample_ot_coupling, compute_ssim_scores,
    compute_psnr, flow_matching_inference, _train_cyclegan_coupler
)
from scipy.integrate import solve_ivp


def generate_trajectory_images(
    n_images: int,
    config: Config,
    device: torch.device,
    mu_range: tuple = (0.5, 2.0),
    seed_offset: int = 0
):
    """Generate Van der Pol trajectory images with random parameters."""
    np.random.seed(config.seed + seed_offset)
    
    HR_list = []
    for i in range(n_images):
        # Random Van der Pol parameter
        mu = np.random.uniform(mu_range[0], mu_range[1])
        
        # Random initial conditions
        x0_init = np.random.uniform(-2.5, 2.5)
        v0_init = np.random.uniform(-2.5, 2.5)
        
        # Random time span
        t_start = np.random.uniform(0, 5)
        t_end = t_start + np.random.uniform(15, 25)
        
        # Solve ODE
        sol = solve_ivp(
            van_der_pol_ode,
            [t_start, t_end],
            [x0_init, v0_init],
            args=(mu,),
            dense_output=True
        )
        
        # Sample points
        t_eval = np.linspace(t_start, t_end, config.points_per_image)
        traj = sol.sol(t_eval).T
        
        # Convert to image
        hr_img = points_to_image(traj, resolution=config.hr_resolution)
        HR_list.append(hr_img)
    
    HR = torch.from_numpy(np.stack(HR_list)).unsqueeze(1).float().to(device)
    return HR


def create_truly_unpaired_dataset(config: Config, device: torch.device):
    """
    Create truly unpaired dataset with proper GAN pretraining split:
    - HR images from one set of trajectories (Dataset A)
    - LR images from DIFFERENT trajectories (Dataset B)
    - Split: First half for GAN pretraining, second half for FM training
    - Test set with paired data to evaluate reconstruction
    """
    print("\n--- Creating TRULY UNPAIRED Dataset (with GAN Pretraining Split) ---")
    
    # Dataset A: HR images (for training target distribution)
    print("Generating HR images (Dataset A)...")
    HR_train_full = generate_trajectory_images(
        n_images=config.n_images,
        config=config,
        device=device,
        mu_range=(0.5, 1.5),  # One parameter range
        seed_offset=0
    )
    
    # Dataset B: Different trajectories, then degraded to LR
    print("Generating LR images from DIFFERENT trajectories (Dataset B)...")
    HR_for_LR = generate_trajectory_images(
        n_images=config.n_images,
        config=config,
        device=device,
        mu_range=(1.0, 2.0),  # Different/overlapping parameter range
        seed_offset=1000  # Different seed!
    )
    
    # Degrade Dataset B to get LR
    LR_train_full = degrade_operator(
        HR_for_LR,
        blur_sigma=config.blur_sigma,
        blur_radius=config.blur_radius,
        down_factor=config.downsample_factor
    )
    
    # Upsample LR for flow matching input
    LR_up_train_full = F.interpolate(
        LR_train_full,
        scale_factor=config.downsample_factor,
        mode='bilinear',
        align_corners=False
    )
    
    # SPLIT: First half for GAN pretraining, second half for FM training
    split_idx = config.n_images // 2
    HR_pretrain = HR_train_full[:split_idx]  # For GAN pretraining
    LR_up_pretrain = LR_up_train_full[:split_idx]
    
    HR_fm_train = HR_train_full[split_idx:]  # For FM training (unseen by GAN)
    LR_up_fm_train = LR_up_train_full[split_idx:]
    
    # Test set: PAIRED data to properly evaluate reconstruction
    print("Generating PAIRED test set for evaluation...")
    HR_test = generate_trajectory_images(
        n_images=60,
        config=config,
        device=device,
        mu_range=(0.7, 1.8),  # Mix of both ranges
        seed_offset=5000  # Completely different seed
    )
    
    LR_test = degrade_operator(
        HR_test,
        blur_sigma=config.blur_sigma,
        blur_radius=config.blur_radius,
        down_factor=config.downsample_factor
    )
    
    print(f"\nDataset Summary:")
    print(f"  Pretraining HR (Dataset A, first 50%): {HR_pretrain.shape} - from trajectories with μ ∈ [0.5, 1.5]")
    print(f"  Pretraining LR (Dataset B, first 50%): {LR_up_pretrain.shape} - from trajectories with μ ∈ [1.0, 2.0]")
    print(f"  FM Training HR (Dataset A, second 50%): {HR_fm_train.shape} - from trajectories with μ ∈ [0.5, 1.5]")
    print(f"  FM Training LR (Dataset B, second 50%): {LR_up_fm_train.shape} - from trajectories with μ ∈ [1.0, 2.0]")
    print(f"  Test (Paired): {HR_test.shape} HR, {LR_test.shape} LR - from trajectories with μ ∈ [0.7, 1.8]")
    print(f"\n  ⚠️  GAN sees first 50% of unpaired data during pretraining")
    print(f"  ⚠️  FM training uses second 50% (UNSEEN by GAN)")
    print(f"  ⚠️  Training LR and HR have NO correspondence!")
    print(f"  ✓  Test set is paired for proper evaluation")
    
    return HR_pretrain, LR_up_pretrain, HR_fm_train, LR_up_fm_train, HR_test, LR_test


def train_truly_unpaired(
    model: nn.Module,
    x0_pretrain: torch.Tensor,  # LR for GAN pretraining
    x1_pretrain: torch.Tensor,  # HR for GAN pretraining
    x0_train: torch.Tensor,     # LR for FM training (UNSEEN by GAN)
    x1_train: torch.Tensor,     # HR for FM training (UNSEEN by GAN)
    config: Config,
    device: torch.device
) -> nn.Module:
    """
    Train with truly unpaired data using OT or CycleGAN coupling.
    
    Key difference: x0 and x1 come from completely different trajectories!
    - x0_pretrain, x1_pretrain: Used to pretrain CycleGAN
    - x0_train, x1_train: Used for FM training (NOT seen by GAN)
    - OT mode: best matching within each mini-batch.
    - CycleGAN mode: pseudo-target coupling via pretrained G(LR).
    """
    print(f"\n--- Training with TRULY UNPAIRED Data ---")
    if config.coupling_mode == 'cyclegan':
        print("Using CycleGAN pseudo-target coupling to map LR→HR")
        print(f"  → GAN pretrained on {x0_pretrain.size(0)} unpaired samples")
        print(f"  → FM trained on {x0_train.size(0)} DIFFERENT unpaired samples (unseen by GAN)")
    else:
        print("Using mini-batch Optimal Transport to couple LR↔HR")
    
    schedule = InterpolantSchedule('stochastic', sigma_max=config.sigma_max)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    mse_loss = nn.MSELoss()

    cyclegan_generator = None
    if config.coupling_mode == 'cyclegan':
        pretrain_loader = DataLoader(
            TensorDataset(x0_pretrain, x1_pretrain),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print(f"\n[GAN Pretraining Phase] Training CycleGAN on {x0_pretrain.size(0)} samples...")
        cyclegan_generator = _train_cyclegan_coupler(pretrain_loader, config, device)
        print(f"[GAN Pretraining Complete] Now using G for FM training on UNSEEN data...\n")
    
    n_samples = x0_train.size(0)
    model.train()
    
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle both datasets independently each epoch
        perm0 = torch.randperm(n_samples, device=device)
        perm1 = torch.randperm(n_samples, device=device)
        
        for i in range(0, n_samples - config.batch_size + 1, config.batch_size):
            # Get batches from DIFFERENT permutations (truly unpaired)
            x0_batch = x0_train[perm0[i:i+config.batch_size]]
            x1_batch = x1_train[perm1[i:i+config.batch_size]]
            
            if config.coupling_mode == 'cyclegan':
                with torch.no_grad():
                    lr_batch = F.interpolate(
                        x0_batch,
                        scale_factor=1.0 / config.downsample_factor,
                        mode='bilinear',
                        align_corners=False,
                    )
                    x1_batch = cyclegan_generator(lr_batch)
            else:
                x0_batch, x1_batch = sample_ot_coupling(
                    x0_batch, x1_batch, reg=config.ot_reg
                )
            
            # Sample time
            t = torch.rand(config.batch_size, 1, 1, 1, device=device)
            
            # Sample noise for stochastic interpolant
            noise = torch.randn_like(x0_batch)
            
            # Interpolate
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
        
        if epoch % 5 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch:02d}/{config.epochs} | Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return model


def evaluate_unpaired_model(
    model: nn.Module,
    lr_test: torch.Tensor,
    hr_test: torch.Tensor,
    config: Config,
    device: torch.device
):
    """Evaluate on paired test set."""
    schedule = InterpolantSchedule('stochastic', sigma_max=config.sigma_max)
    
    # Inference
    sr_images = flow_matching_inference(model, lr_test, config, device, schedule)
    
    # Compute metrics
    ssim_scores = compute_ssim_scores(sr_images, hr_test)
    psnr = compute_psnr(sr_images, hr_test)
    
    return sr_images, np.mean(ssim_scores), psnr


def visualize_unpaired_results(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    title: str,
    save_path: str
):
    """Visualize results."""
    n_samples = min(4, lr_images.size(0))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i in range(n_samples):
        # LR (upsampled for display)
        lr_up = F.interpolate(lr_images[i:i+1], scale_factor=4, mode='nearest')[0, 0]
        axes[i, 0].imshow(lr_up.cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title("LR Input" if i == 0 else "")
        axes[i, 0].axis('off')
        
        # SR
        axes[i, 1].imshow(sr_images[i, 0].cpu().numpy(), cmap='viridis')
        ssim_val = compute_ssim_scores(sr_images[i:i+1], hr_images[i:i+1])[0]
        axes[i, 1].set_title(f"SR Output (SSIM={ssim_val:.3f})" if i == 0 else f"SSIM={ssim_val:.3f}")
        axes[i, 1].axis('off')
        
        # HR
        axes[i, 2].imshow(hr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title("HR Ground Truth" if i == 0 else "")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()


def main():
    print("=" * 70)
    print("TRUE UNPAIRED DATA EXPERIMENT")
    print("=" * 70)
    print("""
    This experiment tests REAL unpaired learning:
    
    TRAINING:
      - LR images: Generated from Van der Pol trajectories (Dataset B)
      - HR images: Generated from DIFFERENT trajectories (Dataset A)
      - NO correspondence between LR and HR samples!
    - Coupling can use OT matching or CycleGAN pseudo-targets
    
    TESTING:
      - Paired LR-HR data (different from training)
      - Tests if model learned general LR→HR mapping
    """)
    
    config = Config(
        n_images=300,
        epochs=40,
        batch_size=8,
        base_channels=32,
        fm_type='stochastic',
        coupling_mode='cyclegan',
        ot_reg=0.01,
        cyclegan_pretrain_epochs=6,
        inference_mode='ode',
        inference_steps=50,
        seed=42
    )
    
    set_seed(config.seed)
    device = get_device()
    
    # Create truly unpaired dataset with split for GAN pretraining
    HR_pretrain, LR_up_pretrain, HR_fm_train, LR_up_fm_train, HR_test, LR_test = create_truly_unpaired_dataset(config, device)
    
    # Create and train model
    model = VelocityUNet(base_channels=config.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    start_time = time.time()
    model = train_truly_unpaired(
        model, 
        LR_up_pretrain, HR_pretrain,  # Data for GAN pretraining
        LR_up_fm_train, HR_fm_train,  # Data for FM training (unseen by GAN)
        config, device
    )
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")
    
    # Evaluate
    print("\n--- Evaluating on PAIRED Test Set ---")
    torch.cuda.empty_cache()
    
    sr_images, mean_ssim, mean_psnr = evaluate_unpaired_model(
        model, LR_test, HR_test, config, device
    )
    
    print(f"\n{'='*50}")
    print(f"RESULTS (Truly Unpaired Training - {config.coupling_mode.upper()} coupling)")
    print(f"  ✓ GAN pretrained on 50% of unpaired data")
    print(f"  ✓ FM trained on DIFFERENT 50% (unseen by GAN)")
    print(f"  ✓ Evaluated on completely separate paired test set")
    print(f"{'='*50}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"{'='*50}")
    
    # Visualize
    visualize_unpaired_results(
        LR_test[:4], sr_images[:4], HR_test[:4],
        f"Truly Unpaired Training Results\nSSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f}dB",
        "results_truly_unpaired.png"
    )
    
    # Compare with baseline (bicubic upsampling)
    print("\n--- Baseline Comparison ---")
    bicubic_up = F.interpolate(LR_test, scale_factor=4, mode='bicubic', align_corners=False)
    bicubic_ssim = np.mean(compute_ssim_scores(bicubic_up, HR_test))
    bicubic_psnr = compute_psnr(bicubic_up, HR_test)
    
    print(f"Bicubic Upsampling:  SSIM={bicubic_ssim:.4f}, PSNR={bicubic_psnr:.2f} dB")
    print(f"Unpaired FM Model:   SSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f} dB")
    print(f"Improvement:         ΔSSIM={mean_ssim - bicubic_ssim:+.4f}, ΔPSNR={mean_psnr - bicubic_psnr:+.2f} dB")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if config.coupling_mode == 'cyclegan':
        coupling_line = "✓ CycleGAN pseudo-target coupling replaced OT matching"
    else:
        coupling_line = "✓ OT coupling successfully matched unpaired LR↔HR distributions"

    print(f"""
    If the model improves over bicubic baseline, it means:
    {coupling_line}
    ✓ Model learned the general degradation→restoration mapping
    ✓ This works even without explicit paired training data!
    
    This is useful when:
    - You have HR images from one domain (e.g., clean microscopy)
    - You have LR images from another domain (e.g., noisy captures)
    - No direct correspondence exists between them
    """)


if __name__ == "__main__":
    main()
