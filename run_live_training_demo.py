import time

from Super_resolution.flow_matching import (
    Config,
    InterpolantSchedule,
    VelocityUNet,
    evaluate_on_validation,
    get_device,
    prepare_data_loaders,
    set_seed,
    train_flow_matching,
)


def run_experiment(name: str, fm_type: str, sigma_max: float):
    config = Config(
        n_images=120,
        epochs=6,
        batch_size=8,
        learning_rate=1e-4,
        base_channels=16,
        fm_type=fm_type,
        coupling_mode="unpaired",
        sigma_max=sigma_max,
        inference_mode="ode",
        inference_steps=20,
        ot_reg=0.01,
        seed=42,
    )

    set_seed(config.seed)
    device = get_device()
    print(f"\n[{name}] Starting setup...")

    train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config, device)
    model = VelocityUNet(base_channels=config.base_channels).to(device)

    schedule = InterpolantSchedule(
        "linear" if fm_type == "deterministic" else "stochastic",
        sigma_max=sigma_max,
    )

    print(f"[{name}] Training now...")
    t0 = time.time()
    model = train_flow_matching(model, train_loader, config, device)
    train_seconds = time.time() - t0

    print(f"[{name}] Evaluating...")
    ssim, psnr = evaluate_on_validation(
        model, lr_val, x1_val, config, device, schedule, batch_size=4
    )

    return {
        "name": name,
        "fm_type": fm_type,
        "sigma_max": sigma_max,
        "ssim": ssim,
        "psnr": psnr,
        "train_seconds": train_seconds,
    }


def main():
    print("=" * 72)
    print("LIVE TRAINING DEMO (UNPAIRED): DETERMINISTIC VS STOCHASTIC")
    print("=" * 72)

    det = run_experiment("Deterministic", "deterministic", 0.0)
    stoch = run_experiment("Stochastic", "stochastic", 0.1)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Deterministic  -> SSIM={det['ssim']:.4f}, PSNR={det['psnr']:.2f}, train={det['train_seconds']:.1f}s")
    print(f"Stochastic     -> SSIM={stoch['ssim']:.4f}, PSNR={stoch['psnr']:.2f}, train={stoch['train_seconds']:.1f}s")
    print(f"Delta (Det-Stoch): SSIM={det['ssim'] - stoch['ssim']:+.4f}, PSNR={det['psnr'] - stoch['psnr']:+.2f}")


if __name__ == "__main__":
    main()
