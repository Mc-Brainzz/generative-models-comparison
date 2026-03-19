"""
Ablation: Test if longer CycleGAN pretraining improves FM results (with proper held-out split)

Setup:
- GAN pretrains on 150 samples (from unpaired distributions)
- FM trains on DIFFERENT 150 samples (unseen by GAN)
- Test on separate 60 paired samples
- Vary CycleGAN pretraining epochs: 6, 15, 20, 30
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm
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


def create_dataset_with_split(config: Config, device: torch.device):
    """Create truly unpaired dataset with GAN/FM pretraining/training split."""
    print("\n--- Creating TRULY UNPAIRED Dataset (GAN/FM Split) ---")
    
    # Dataset A: HR images
    print("Generating HR images (Dataset A)...")
    HR_full = generate_trajectory_images(
        n_images=config.n_images,
        config=config,
        device=device,
        mu_range=(0.5, 1.5),
        seed_offset=0
    )
    
    # Dataset B: LR (from different trajectories)
    print("Generating LR images (Dataset B)...")
    HR_for_LR = generate_trajectory_images(
        n_images=config.n_images,
        config=config,
        device=device,
        mu_range=(1.0, 2.0),
        seed_offset=1000
    )
    
    LR_full = degrade_operator(
        HR_for_LR,
        blur_sigma=config.blur_sigma,
        blur_radius=config.blur_radius,
        down_factor=config.downsample_factor
    )
    
    LR_up_full = F.interpolate(
        LR_full,
        scale_factor=config.downsample_factor,
        mode='bilinear',
        align_corners=False
    )
    
    # Split: First 150 for GAN, second 150 for FM
    split = config.n_images // 2
    HR_pretrain = HR_full[:split]
    LR_up_pretrain = LR_up_full[:split]
    HR_fm_train = HR_full[split:]
    LR_up_fm_train = LR_up_full[split:]
    
    # Test set: Paired data
    print("Generating test set...")
    HR_test = generate_trajectory_images(
        n_images=60,
        config=config,
        device=device,
        mu_range=(0.7, 1.8),
        seed_offset=5000
    )
    
    LR_test = degrade_operator(
        HR_test,
        blur_sigma=config.blur_sigma,
        blur_radius=config.blur_radius,
        down_factor=config.downsample_factor
    )
    
    print(f"Data splits:")
    print(f"  GAN Pretrain: {HR_pretrain.shape[0]} samples")
    print(f"  FM Train: {HR_fm_train.shape[0]} samples (unseen by GAN)")
    print(f"  Test: {HR_test.shape[0]} paired samples")
    
    return HR_pretrain, LR_up_pretrain, HR_fm_train, LR_up_fm_train, HR_test, LR_test


def train_fm_with_cyclegan(
    model: nn.Module,
    lr_pretrain: torch.Tensor,
    hr_pretrain: torch.Tensor,
    lr_fm_train: torch.Tensor,
    hr_fm_train: torch.Tensor,
    config: Config,
    device: torch.device,
    cyclegan_epochs: int = 6
) -> nn.Module:
    """Train FM with CycleGAN coupling, varying pretraining epochs."""
    
    schedule = InterpolantSchedule('stochastic', sigma_max=config.sigma_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    mse_loss = nn.MSELoss()
    
    # Train CycleGAN with specified epochs
    print(f"\n  [CycleGAN Pretraining: {cyclegan_epochs} epochs]")
    pretrain_loader = DataLoader(
        TensorDataset(lr_pretrain, hr_pretrain),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # Temporarily override config for cyclegan pretraining
    orig_epochs = config.cyclegan_pretrain_epochs
    config.cyclegan_pretrain_epochs = cyclegan_epochs
    cyclegan_generator = _train_cyclegan_coupler(pretrain_loader, config, device)
    config.cyclegan_pretrain_epochs = orig_epochs
    
    model.train()
    n_samples = lr_fm_train.size(0)
    
    # Train FM on unseen data
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        perm0 = torch.randperm(n_samples, device=device)
        perm1 = torch.randperm(n_samples, device=device)
        
        for i in range(0, n_samples - config.batch_size + 1, config.batch_size):
            x0_batch = lr_fm_train[perm0[i:i+config.batch_size]]
            x1_batch = hr_fm_train[perm1[i:i+config.batch_size]]
            
            # Use CycleGAN to generate pseudo-targets
            with torch.no_grad():
                lr_batch = F.interpolate(
                    x0_batch,
                    scale_factor=1.0 / config.downsample_factor,
                    mode='bilinear',
                    align_corners=False,
                )
                x1_batch = cyclegan_generator(lr_batch)
            
            t = torch.rand(config.batch_size, 1, 1, 1, device=device)
            noise = torch.randn_like(x0_batch)
            
            x_t = schedule.interpolate(x0_batch, x1_batch, t, noise)
            v_target = schedule.velocity_target(x0_batch, x1_batch, t, noise)
            v_pred = model(x_t, t)
            
            loss = mse_loss(v_pred, v_target)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
    
    return model


def evaluate_model(model, lr_test, hr_test, config, device):
    """Evaluate on test set."""
    model.eval()
    with torch.no_grad():
        sr_images = flow_matching_inference(
            model, lr_test, config, device
        )
    
    ssim_scores = compute_ssim_scores(sr_images, hr_test)
    psnr = compute_psnr(sr_images, hr_test)
    
    return float(np.mean(ssim_scores)), float(psnr)


def main():
    print("=" * 70)
    print("CYCLEGAN PRETRAINING ABLATION (Proper Held-Out Split)")
    print("=" * 70)
    
    config = Config(
        n_images=300,
        epochs=30,
        batch_size=8,
        base_channels=32,
        fm_type='stochastic',
        coupling_mode='cyclegan',
        ot_reg=0.01,
        cyclegan_pretrain_epochs=6,  # Will be overridden
        inference_mode='ode',
        inference_steps=50,
        seed=42
    )
    
    set_seed(config.seed)
    device = get_device()
    
    # Create data with proper split
    HR_pretrain, LR_up_pretrain, HR_fm_train, LR_up_fm_train, HR_test, LR_test = \
        create_dataset_with_split(config, device)
    
    # Test different CycleGAN pretraining epochs
    cyclegan_epochs_list = [6, 15, 20, 30]
    results = []
    
    print("\n" + "=" * 70)
    print("ABLATION: Testing CycleGAN Pretraining Epochs")
    print("=" * 70)
    
    for cyclegan_epochs in cyclegan_epochs_list:
        print(f"\n--- Testing CycleGAN with {cyclegan_epochs} pretraining epochs ---")
        
        # Fresh model for each run
        model = VelocityUNet(base_channels=config.base_channels).to(device)
        
        start = time.time()
        model = train_fm_with_cyclegan(
            model,
            LR_up_pretrain, HR_pretrain,
            LR_up_fm_train, HR_fm_train,
            config, device,
            cyclegan_epochs=cyclegan_epochs
        )
        train_time = time.time() - start
        
        # Evaluate
        torch.cuda.empty_cache()
        ssim, psnr = evaluate_model(model, LR_test, HR_test, config, device)
        
        results.append({
            'cyclegan_epochs': cyclegan_epochs,
            'ssim': ssim,
            'psnr': psnr,
            'train_time': train_time
        })
        
        print(f"  SSIM: {ssim:.4f}, PSNR: {psnr:.2f} dB, Time: {train_time:.1f}s")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'GAN Epochs':<12} {'SSIM':<12} {'PSNR (dB)':<12} {'Time (s)':<12}")
    print("-" * 70)
    for res in results:
        print(f"{res['cyclegan_epochs']:<12} {res['ssim']:<12.4f} {res['psnr']:<12.2f} {res['train_time']:<12.1f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    best = max(results, key=lambda x: x['ssim'])
    worst = min(results, key=lambda x: x['ssim'])
    
    print(f"Best SSIM: {best['cyclegan_epochs']} epochs → SSIM={best['ssim']:.4f}")
    print(f"Worst SSIM: {worst['cyclegan_epochs']} epochs → SSIM={worst['ssim']:.4f}")
    print(f"ΔSSIM (best-worst): {best['ssim'] - worst['ssim']:+.4f}")
    
    if best['ssim'] - worst['ssim'] < 0.02:
        print("\n⚠️  Minimal difference across pretraining epochs")
        print("    → CycleGAN may not generalize well to unseen data regardless of pretraining length")
    else:
        print(f"\n✓ More pretraining helps: +{best['ssim'] - worst['ssim']:.4f} SSIM improvement")


if __name__ == "__main__":
    main()
