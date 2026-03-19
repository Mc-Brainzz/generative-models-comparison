import time
from dataclasses import replace

import numpy as np
import torch

from Super_resolution.flow_matching import (
    Config,
    InterpolantSchedule,
    VelocityUNet,
    compute_ssim_scores,
    evaluate_on_validation,
    flow_matching_inference,
    get_device,
    prepare_data_loaders,
    set_seed,
    train_flow_matching,
)


def train_model(config: Config, device: torch.device):
    schedule = InterpolantSchedule(
        "linear" if config.fm_type == "deterministic" else "stochastic",
        sigma_max=config.sigma_max,
    )
    train_loader, hr_val, _, lr_val = prepare_data_loaders(config, device)
    model = VelocityUNet(base_channels=config.base_channels).to(device)

    t0 = time.time()
    model = train_flow_matching(model, train_loader, config, device)
    train_sec = time.time() - t0

    return model, schedule, hr_val, lr_val, train_sec


def stochastic_best_of_k_ssim(
    model,
    hr_val,
    lr_val,
    base_config: Config,
    schedule,
    device,
    k: int = 6,
    batch_size: int = 6,
    noise_scale: float = 0.2,
):
    cfg = replace(base_config, inference_mode="sde", sde_noise_scale=noise_scale)

    per_image_best = []
    per_image_mean = []

    for i in range(0, lr_val.size(0), batch_size):
        lr_batch = lr_val[i : i + batch_size]
        hr_batch = hr_val[i : i + batch_size]

        run_scores = []
        for _ in range(k):
            sr = flow_matching_inference(model, lr_batch, cfg, device, schedule)
            run_scores.append(np.array(compute_ssim_scores(sr, hr_batch)))

        stacked = np.stack(run_scores, axis=0)
        per_image_best.extend(stacked.max(axis=0).tolist())
        per_image_mean.extend(stacked.mean(axis=0).tolist())

    return float(np.mean(per_image_best)), float(np.mean(per_image_mean))


def stochastic_diversity(
    model,
    lr_val,
    base_config: Config,
    schedule,
    device,
    k: int = 6,
    n_eval: int = 8,
    noise_scale: float = 0.2,
):
    cfg = replace(base_config, inference_mode="sde", sde_noise_scale=noise_scale)
    lr_batch = lr_val[:n_eval]

    outputs = []
    for _ in range(k):
        sr = flow_matching_inference(model, lr_batch, cfg, device, schedule)
        outputs.append(sr)

    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            mse = torch.mean((outputs[i] - outputs[j]) ** 2).item()
            pairwise.append(mse)

    return float(np.mean(pairwise)) if pairwise else 0.0


def main():
    print("=" * 78)
    print("COMPLEX REGIME SHOWCASE: WHERE STOCHASTIC CAN HELP")
    print("=" * 78)

    set_seed(123)
    device = get_device()

    # Harder regime: stronger degradation + larger ambiguity
    base = Config(
        n_images=180,
        epochs=8,
        batch_size=8,
        base_channels=16,
        learning_rate=1e-4,
        hr_resolution=128,
        downsample_factor=8,
        blur_sigma=1.8,
        blur_radius=4,
        coupling_mode="unpaired",
        ot_reg=0.01,
        inference_steps=24,
        sde_noise_scale=1.0,
        seed=123,
    )

    det_cfg = replace(base, fm_type="deterministic", sigma_max=0.0, inference_mode="ode")
    stoch_cfg = replace(base, fm_type="stochastic", sigma_max=0.1, inference_mode="ode")

    print("\n[1/2] Train deterministic model...")
    det_model, det_schedule, hr_val_det, lr_val_det, det_train_sec = train_model(det_cfg, device)

    print("\n[2/2] Train stochastic model...")
    stoch_model, stoch_schedule, hr_val_st, lr_val_st, stoch_train_sec = train_model(stoch_cfg, device)

    print("\nEvaluating single-sample quality (ODE for both)...")
    det_ssim_ode, det_psnr_ode = evaluate_on_validation(
        det_model, lr_val_det, hr_val_det, det_cfg, device, det_schedule, batch_size=6
    )
    stoch_ssim_ode, stoch_psnr_ode = evaluate_on_validation(
        stoch_model, lr_val_st, hr_val_st, stoch_cfg, device, stoch_schedule, batch_size=6
    )

    print("\nEvaluating stochastic multi-sample capability (SDE noise sweep)...")
    sweep_scales = [0.05, 0.1, 0.15, 0.2]
    sweep_results = []
    for noise_scale in sweep_scales:
        best_k_ssim, mean_k_ssim = stochastic_best_of_k_ssim(
            stoch_model,
            hr_val_st,
            lr_val_st,
            stoch_cfg,
            stoch_schedule,
            device,
            k=6,
            batch_size=6,
            noise_scale=noise_scale,
        )
        diversity = stochastic_diversity(
            stoch_model,
            lr_val_st,
            stoch_cfg,
            stoch_schedule,
            device,
            k=6,
            n_eval=8,
            noise_scale=noise_scale,
        )
        sweep_results.append((noise_scale, best_k_ssim, mean_k_ssim, diversity))

    best_noise_scale, best_k_ssim, mean_k_ssim, diversity = max(
        sweep_results, key=lambda x: x[1]
    )

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"Deterministic ODE  -> SSIM={det_ssim_ode:.4f}, PSNR={det_psnr_ode:.2f}, train={det_train_sec:.1f}s")
    print(f"Stochastic ODE     -> SSIM={stoch_ssim_ode:.4f}, PSNR={stoch_psnr_ode:.2f}, train={stoch_train_sec:.1f}s")
    print(f"Best stochastic SDE noise scale: {best_noise_scale}")
    print(f"Stochastic SDE mean-of-6 SSIM : {mean_k_ssim:.4f}")
    print(f"Stochastic SDE best-of-6 SSIM : {best_k_ssim:.4f}")
    print(f"Stochastic diversity (pairwise MSE over samples): {diversity:.6f}")
    print("\nSDE sweep details (noise_scale, best_of_6, mean_of_6, diversity):")
    for noise_scale, best_val, mean_val, div in sweep_results:
        print(f"  {noise_scale:>4.2f} -> best={best_val:.4f}, mean={mean_val:.4f}, div={div:.6f}")

    print("\nInterpretation:")
    print("- ODE scores compare single deterministic outputs.")
    if best_k_ssim > stoch_ssim_ode:
        print("- Stochastic sampling helps this model via best-of-k (better than its own ODE output).")
    else:
        print("- In this run, stochastic sampling did not beat stochastic ODE on SSIM.")
    if best_k_ssim > det_ssim_ode:
        print("- In this ambiguity regime, stochastic best-of-k exceeded deterministic single-output SSIM.")
    else:
        print("- Deterministic still wins single-output quality in this run.")
    print("- Diversity > 0 confirms stochastic model generates distinct outputs for same LR input.")


if __name__ == "__main__":
    main()
