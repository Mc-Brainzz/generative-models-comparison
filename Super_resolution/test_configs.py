"""
Test script for comparing different Flow Matching configurations:
1. Best Quality: More data, epochs, channels
2. Diverse Outputs: Stochastic FM with SDE inference
3. Unpaired Data: Using Optimal Transport coupling
"""

import sys
sys.path.insert(0, '.')

from flow_matching import (
    Config, set_seed, get_device, InterpolantSchedule,
    prepare_data_loaders, VelocityUNet, train_flow_matching,
    evaluate_on_validation, flow_matching_inference, compute_ssim_scores,
    compute_psnr, visualize_results
)
import torch
import time


def run_experiment(name: str, config: Config):
    """Run a single experiment with given config."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print(f"Config: n_images={config.n_images}, epochs={config.epochs}, "
          f"base_channels={config.base_channels}")
    print(f"        fm_type={config.fm_type}, coupling_mode={config.coupling_mode}, "
          f"inference_mode={config.inference_mode}")
    print(f"        inference_steps={config.inference_steps}, sde_noise_scale={config.sde_noise_scale}")
    
    set_seed(config.seed)
    device = get_device()
    
    # Setup interpolant schedule
    schedule = InterpolantSchedule(
        'stochastic' if config.fm_type == 'stochastic' else 'linear',
        sigma_max=config.sigma_max
    )
    
    # Prepare data
    train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config, device)
    
    # Create model
    model = VelocityUNet(base_channels=config.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Train
    start_time = time.time()
    model = train_flow_matching(model, train_loader, config, device)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f}min)")
    
    # Clear GPU cache before evaluation
    torch.cuda.empty_cache()
    
    # Evaluate with smaller batch size to avoid OOM
    # Note: flow_matching_inference expects LR (32x32) and upsamples it internally
    # So we pass lr_val, not x0_val (which is already upsampled)
    start_time = time.time()
    mean_ssim, mean_psnr = evaluate_on_validation(
        model, lr_val, x1_val, config, device, schedule, batch_size=4
    )
    infer_time = time.time() - start_time
    
    print(f"\n--- Results for {name} ---")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"Inference time: {infer_time:.2f}s")
    
    # Visualize a few samples
    torch.cuda.empty_cache()
    n_vis = min(3, lr_val.size(0))
    sr_samples = flow_matching_inference(model, lr_val[:n_vis], config, device, schedule)
    ssim_scores = compute_ssim_scores(sr_samples, x1_val[:n_vis])
    
    visualize_results(
        lr_val[:n_vis], sr_samples, x1_val[:n_vis], ssim_scores,
        title=f"{name}\nSSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f}dB",
        save_path=f"results_{name.lower().replace(' ', '_')}.png"
    )
    
    return {
        'name': name,
        'mean_ssim': mean_ssim,
        'mean_psnr': mean_psnr,
        'train_time': train_time,
        'infer_time': infer_time,
        'n_params': n_params
    }


def main():
    print("=" * 70)
    print("FLOW MATCHING CONFIGURATION COMPARISON")
    print("=" * 70)
    
    results = []
    
    # =========================================================================
    # Experiment 1: Best Quality (more data, epochs, channels)
    # =========================================================================
    # NOTE: Using reduced settings to fit in 6GB GPU
    # Full config would be: n_images=1000, epochs=100, base_channels=64
    config_best = Config(
        n_images=300,          # Reduced for memory
        epochs=30,             # Reduced from 100 for faster testing  
        inference_steps=50,    # Reduced from 100 for faster testing
        base_channels=48,      # Larger model (reduced from 64 to avoid OOM)
        fm_type='stochastic',
        coupling_mode='paired',
        inference_mode='ode',
        batch_size=8,          # Smaller batch to fit in memory
        seed=123
    )
    results.append(run_experiment("Best Quality", config_best))
    
    # =========================================================================
    # Experiment 2: Diverse Outputs (stochastic with SDE)
    # =========================================================================
    config_diverse = Config(
        n_images=300,
        epochs=30,
        inference_steps=50,
        base_channels=32,
        fm_type='stochastic',
        coupling_mode='paired',
        inference_mode='sde',       # SDE for stochastic inference
        sde_noise_scale=0.3,        # Reduced from 1.5 - more reasonable for diversity
        sigma_max=0.1,              # Reduced from 0.15
        batch_size=8,
        seed=123
    )
    results.append(run_experiment("Diverse Outputs (SDE)", config_diverse))
    
    # =========================================================================
    # Experiment 3: Unpaired Data with OT coupling
    # =========================================================================
    config_unpaired = Config(
        n_images=300,
        epochs=30,
        inference_steps=50,
        base_channels=32,
        fm_type='stochastic',
        coupling_mode='unpaired',   # Use Optimal Transport coupling
        ot_reg=0.01,                # Sinkhorn regularization
        inference_mode='ode',
        batch_size=8,
        seed=123
    )
    results.append(run_experiment("Unpaired Data (OT)", config_unpaired))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<25} {'SSIM':>10} {'PSNR (dB)':>12} {'Train (s)':>12} {'Params':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['mean_ssim']:>10.4f} {r['mean_psnr']:>12.2f} "
              f"{r['train_time']:>12.1f} {r['n_params']:>12,}")
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print("""
1. BEST QUALITY (larger model, more training):
   - Highest SSIM/PSNR expected due to more capacity
   - Longer training time due to larger model
   
2. DIVERSE OUTPUTS (SDE inference):
   - May have slightly lower SSIM (stochastic outputs)
   - Each run produces different plausible outputs
   - Good for uncertainty estimation
   
3. UNPAIRED DATA (OT coupling):
   - Tests mini-batch Optimal Transport
   - Useful when paired data unavailable
   - Usually slightly lower than paired training
""")


if __name__ == "__main__":
    main()
