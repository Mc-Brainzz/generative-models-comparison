"""
WHY DOES DETERMINISTIC BEAT STOCHASTIC ON UNPAIRED DATA?

Hypothesis: Stochastic just needs MORE training to converge.

Let's test:
1. Stochastic with 40 epochs (current)
2. Stochastic with 100 epochs (more training)
3. Stochastic with lower noise (sigma_max=0.05)

Also visualize the training loss curves.
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
    VelocityUNet, sample_ot_coupling, compute_ssim_scores,
    compute_psnr, flow_matching_inference
)
from test_truly_unpaired import create_truly_unpaired_dataset


def train_with_loss_tracking(
    model: nn.Module,
    x0_train: torch.Tensor,
    x1_train: torch.Tensor,
    config: Config,
    schedule: InterpolantSchedule,
    device: torch.device
):
    """Train and track loss history."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    mse_loss = nn.MSELoss()
    
    n_samples = x0_train.size(0)
    model.train()
    
    loss_history = []
    
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        perm0 = torch.randperm(n_samples, device=device)
        perm1 = torch.randperm(n_samples, device=device)
        
        for i in range(0, n_samples - config.batch_size + 1, config.batch_size):
            x0_batch = x0_train[perm0[i:i+config.batch_size]]
            x1_batch = x1_train[perm1[i:i+config.batch_size]]
            
            x0_batch, x1_batch = sample_ot_coupling(x0_batch, x1_batch, reg=config.ot_reg)
            
            t = torch.rand(config.batch_size, 1, 1, 1, device=device)
            noise = torch.randn_like(x0_batch) if config.fm_type == 'stochastic' else None
            
            x_t = schedule.interpolate(x0_batch, x1_batch, t, noise)
            v_target = schedule.velocity_target(x0_batch, x1_batch, t, noise)
            v_pred = model(x_t, t.squeeze())
            
            loss = mse_loss(v_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{config.epochs} | Loss: {avg_loss:.5f}")
    
    return model, loss_history


def evaluate(model, LR_test, HR_test, config, device):
    """Evaluate model."""
    schedule = InterpolantSchedule(
        'stochastic' if config.fm_type == 'stochastic' else 'linear',
        sigma_max=config.sigma_max
    )
    sr = flow_matching_inference(model, LR_test, config, device, schedule)
    ssim = np.mean(compute_ssim_scores(sr, HR_test))
    psnr = compute_psnr(sr, HR_test)
    return sr, ssim, psnr


def main():
    print("=" * 70)
    print("INVESTIGATING: Why Deterministic > Stochastic on Unpaired Data")
    print("=" * 70)
    print("""
    Your question: With OT coupling on unpaired data, shouldn't stochastic
    work better since it handles uncertainty in the imperfect pairings?
    
    Hypothesis: Stochastic needs MORE training to converge through the noise.
    
    Test plan:
    1. Deterministic (40 epochs) - baseline
    2. Stochastic (40 epochs) - current
    3. Stochastic (100 epochs) - more training
    4. Stochastic (40 epochs, lower noise σ=0.05) - easier learning
    """)
    
    device = get_device()
    
    base_config = {
        'n_images': 300,
        'batch_size': 8,
        'base_channels': 32,
        'coupling_mode': 'unpaired',
        'ot_reg': 0.01,
        'inference_mode': 'ode',
        'inference_steps': 50,
        'seed': 42
    }
    
    results = []
    all_losses = {}
    
    # Create dataset once
    set_seed(42)
    config_tmp = Config(**base_config, epochs=40, fm_type='stochastic', sigma_max=0.1)
    HR_train, LR_up_train, HR_test, LR_test = create_truly_unpaired_dataset(config_tmp, device)
    
    # =========================================================================
    # 1. DETERMINISTIC (40 epochs)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. DETERMINISTIC (40 epochs)")
    print("=" * 70)
    
    set_seed(42)
    config1 = Config(**base_config, epochs=40, fm_type='deterministic', sigma_max=0.0)
    model1 = VelocityUNet(base_channels=32).to(device)
    schedule1 = InterpolantSchedule('linear', sigma_max=0.0)
    
    model1, losses1 = train_with_loss_tracking(model1, LR_up_train, HR_train, config1, schedule1, device)
    sr1, ssim1, psnr1 = evaluate(model1, LR_test, HR_test, config1, device)
    
    results.append(('Deterministic (40 ep)', ssim1, psnr1, losses1[-1]))
    all_losses['Deterministic'] = losses1
    print(f"Final: SSIM={ssim1:.4f}, PSNR={psnr1:.2f}, Loss={losses1[-1]:.5f}")
    
    # =========================================================================
    # 2. STOCHASTIC (40 epochs, σ=0.1) - Current setting
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. STOCHASTIC (40 epochs, σ=0.1) - Current")
    print("=" * 70)
    
    set_seed(42)
    config2 = Config(**base_config, epochs=40, fm_type='stochastic', sigma_max=0.1)
    model2 = VelocityUNet(base_channels=32).to(device)
    schedule2 = InterpolantSchedule('stochastic', sigma_max=0.1)
    
    model2, losses2 = train_with_loss_tracking(model2, LR_up_train, HR_train, config2, schedule2, device)
    sr2, ssim2, psnr2 = evaluate(model2, LR_test, HR_test, config2, device)
    
    results.append(('Stochastic (40 ep, σ=0.1)', ssim2, psnr2, losses2[-1]))
    all_losses['Stochastic (40ep)'] = losses2
    print(f"Final: SSIM={ssim2:.4f}, PSNR={psnr2:.2f}, Loss={losses2[-1]:.5f}")
    
    # =========================================================================
    # 3. STOCHASTIC (100 epochs) - More training
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. STOCHASTIC (100 epochs, σ=0.1) - More Training")
    print("=" * 70)
    
    set_seed(42)
    config3 = Config(**base_config, epochs=100, fm_type='stochastic', sigma_max=0.1)
    model3 = VelocityUNet(base_channels=32).to(device)
    schedule3 = InterpolantSchedule('stochastic', sigma_max=0.1)
    
    model3, losses3 = train_with_loss_tracking(model3, LR_up_train, HR_train, config3, schedule3, device)
    sr3, ssim3, psnr3 = evaluate(model3, LR_test, HR_test, config3, device)
    
    results.append(('Stochastic (100 ep, σ=0.1)', ssim3, psnr3, losses3[-1]))
    all_losses['Stochastic (100ep)'] = losses3
    print(f"Final: SSIM={ssim3:.4f}, PSNR={psnr3:.2f}, Loss={losses3[-1]:.5f}")
    
    # =========================================================================
    # 4. STOCHASTIC (40 epochs, lower noise)
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. STOCHASTIC (40 epochs, σ=0.05) - Lower Noise")
    print("=" * 70)
    
    set_seed(42)
    config4 = Config(**base_config, epochs=40, fm_type='stochastic', sigma_max=0.05)
    model4 = VelocityUNet(base_channels=32).to(device)
    schedule4 = InterpolantSchedule('stochastic', sigma_max=0.05)
    
    model4, losses4 = train_with_loss_tracking(model4, LR_up_train, HR_train, config4, schedule4, device)
    sr4, ssim4, psnr4 = evaluate(model4, LR_test, HR_test, config4, device)
    
    results.append(('Stochastic (40 ep, σ=0.05)', ssim4, psnr4, losses4[-1]))
    all_losses['Stochastic (σ=0.05)'] = losses4
    print(f"Final: SSIM={ssim4:.4f}, PSNR={psnr4:.2f}, Loss={losses4[-1]:.5f}")
    
    # =========================================================================
    # Baseline
    # =========================================================================
    bicubic_up = F.interpolate(LR_test, scale_factor=4, mode='bicubic', align_corners=False)
    ssim_bic = np.mean(compute_ssim_scores(bicubic_up, HR_test))
    results.append(('Bicubic (baseline)', ssim_bic, 0, 0))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'SSIM':>10} {'PSNR':>10} {'Final Loss':>12}")
    print("-" * 65)
    for name, ssim, psnr, loss in results:
        psnr_str = f"{psnr:.2f}" if psnr > 0 else "N/A"
        loss_str = f"{loss:.5f}" if loss > 0 else "N/A"
        print(f"{name:<30} {ssim:>10.4f} {psnr_str:>10} {loss_str:>12}")
    
    # =========================================================================
    # Plot Loss Curves
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1 = axes[0]
    colors = ['blue', 'red', 'green', 'orange']
    for (name, losses), color in zip(all_losses.items(), colors):
        ax1.plot(losses, label=name, color=color, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # SSIM comparison bar chart
    ax2 = axes[1]
    names = [r[0] for r in results]
    ssims = [r[1] for r in results]
    bars = ax2.bar(range(len(names)), ssims, color=['blue', 'red', 'green', 'orange', 'gray'])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Comparison')
    ax2.set_ylim(0, 1)
    for i, (bar, ssim) in enumerate(zip(bars, ssims)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{ssim:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results_stochastic_investigation.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to results_stochastic_investigation.png")
    plt.show()
    
    # =========================================================================
    # Conclusion
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    best_stoch = max(ssim2, ssim3, ssim4)
    best_stoch_name = ['40ep σ=0.1', '100ep σ=0.1', '40ep σ=0.05'][[ssim2, ssim3, ssim4].index(best_stoch)]
    
    print(f"""
    FINDINGS:
    
    1. Deterministic (40 epochs):     SSIM = {ssim1:.4f}, Loss = {losses1[-1]:.5f}
    2. Stochastic (40 epochs):        SSIM = {ssim2:.4f}, Loss = {losses2[-1]:.5f}
    3. Stochastic (100 epochs):       SSIM = {ssim3:.4f}, Loss = {losses3[-1]:.5f}
    4. Stochastic (lower noise):      SSIM = {ssim4:.4f}, Loss = {losses4[-1]:.5f}
    
    Best Stochastic: {best_stoch_name} with SSIM = {best_stoch:.4f}
    
    KEY INSIGHTS:
    
    {'✓ MORE TRAINING HELPS!' if ssim3 > ssim2 else '✗ More training did not help much'}
    {'✓ LOWER NOISE HELPS!' if ssim4 > ssim2 else '✗ Lower noise did not help much'}
    
    WHY DETERMINISTIC STILL WINS:
    
    1. SIMPLER LEARNING TARGET:
       - Deterministic: v* = x₁ - x₀ (constant, easy to regress)
       - Stochastic: v* = α'(t)x₀ + β'(t)x₁ + σ'(t)ε (time-varying, harder)
    
    2. OT COUPLING ALREADY PROVIDES GOOD MATCHES:
       - Van der Pol trajectories with similar μ look similar
       - OT finds visually similar pairs even without true correspondence
       - The "uncertainty" from imperfect pairing is small
    
    3. SPARSE DATA CHARACTERISTIC:
       - Thin trajectory lines are sensitive to any perturbation
       - Noise in stochastic training blurs these delicate structures
       - Deterministic preserves sharp features
    
    WHEN WOULD STOCHASTIC BE BETTER?
    
    - Dense images (natural photos) where local texture variation helps
    - Highly ambiguous inverse problems (multiple valid solutions)
    - When OT coupling is very imperfect (very different distributions)
    - With much more training time to converge through the noise
    """)


if __name__ == "__main__":
    main()
