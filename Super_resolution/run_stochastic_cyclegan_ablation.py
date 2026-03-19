import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_matching import (
    Config,
    InterpolantSchedule,
    VelocityUNet,
    compute_psnr,
    compute_ssim_scores,
    evaluate_on_validation,
    flow_matching_inference,
    get_device,
    prepare_data_loaders,
    set_seed,
    train_flow_matching,
)


def train_once(config: Config, device: torch.device):
    set_seed(config.seed)
    train_loader, hr_val, _x0_val, lr_val = prepare_data_loaders(config, device)

    model = VelocityUNet(base_channels=config.base_channels).to(device)
    schedule = InterpolantSchedule(
        "stochastic" if config.fm_type == "stochastic" else "linear",
        sigma_max=config.sigma_max,
    )

    t0 = time.time()
    model = train_flow_matching(model, train_loader, config, device)
    train_sec = time.time() - t0

    return model, schedule, hr_val, lr_val, train_sec


def best_of_k_metrics(
    model,
    hr_val,
    lr_val,
    base_cfg: Config,
    schedule,
    device,
    noise_scale: float,
    k: int = 6,
    batch_size: int = 6,
):
    cfg = replace(base_cfg, inference_mode="sde", sde_noise_scale=noise_scale)

    all_best_ssim = []
    all_best_psnr = []

    for i in range(0, lr_val.size(0), batch_size):
        lr_batch = lr_val[i : i + batch_size]
        hr_batch = hr_val[i : i + batch_size]

        runs = []
        run_ssim = []
        for _ in range(k):
            sr = flow_matching_inference(model, lr_batch, cfg, device, schedule)
            runs.append(sr)
            run_ssim.append(np.array(compute_ssim_scores(sr, hr_batch)))

        run_ssim = np.stack(run_ssim, axis=0)
        best_idx = run_ssim.argmax(axis=0)

        for sample_idx in range(lr_batch.size(0)):
            best_sr = runs[int(best_idx[sample_idx])][sample_idx : sample_idx + 1]
            best_hr = hr_batch[sample_idx : sample_idx + 1]
            all_best_ssim.append(float(run_ssim[:, sample_idx].max()))
            all_best_psnr.append(float(compute_psnr(best_sr, best_hr)))

    return float(np.mean(all_best_ssim)), float(np.mean(all_best_psnr))


def diversity_score(
    model,
    lr_val,
    base_cfg: Config,
    schedule,
    device,
    noise_scale: float,
    k: int = 6,
    n_eval: int = 8,
):
    cfg = replace(base_cfg, inference_mode="sde", sde_noise_scale=noise_scale)
    lr_batch = lr_val[:n_eval]

    outputs = []
    for _ in range(k):
        outputs.append(flow_matching_inference(model, lr_batch, cfg, device, schedule))

    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            pairwise.append(torch.mean((outputs[i] - outputs[j]) ** 2).item())

    return float(np.mean(pairwise)) if pairwise else 0.0


def save_visuals(
    det_model,
    det_schedule,
    stoch_model,
    stoch_schedule,
    hr_val,
    lr_val,
    base_cfg: Config,
    best_sigma: float,
    best_noise: float,
    out_path: Path,
):
    n = min(4, lr_val.size(0))
    lr = lr_val[:n]
    hr = hr_val[:n]

    det_cfg = replace(base_cfg, fm_type="deterministic", coupling_mode="cyclegan", sigma_max=0.0, inference_mode="ode")
    stoch_ode_cfg = replace(base_cfg, fm_type="stochastic", coupling_mode="cyclegan", sigma_max=best_sigma, inference_mode="ode")
    stoch_sde_cfg = replace(
        base_cfg,
        fm_type="stochastic",
        coupling_mode="cyclegan",
        sigma_max=best_sigma,
        inference_mode="sde",
        sde_noise_scale=best_noise,
    )

    with torch.no_grad():
        det_sr = flow_matching_inference(det_model, lr, det_cfg, device, det_schedule)
        stoch_ode_sr = flow_matching_inference(stoch_model, lr, stoch_ode_cfg, device, stoch_schedule)
        stoch_sde_runs = [flow_matching_inference(stoch_model, lr, stoch_sde_cfg, device, stoch_schedule) for _ in range(6)]

    best_sde = []
    for idx in range(n):
        scores = [
            compute_ssim_scores(run[idx : idx + 1], hr[idx : idx + 1])[0]
            for run in stoch_sde_runs
        ]
        best_run_idx = int(np.argmax(scores))
        best_sde.append(stoch_sde_runs[best_run_idx][idx : idx + 1])
    best_sde = torch.cat(best_sde, dim=0)

    fig, axes = plt.subplots(5, n, figsize=(3.0 * n, 12))
    row_names = [
        "LR Input",
        "Det FM (CycleGAN)",
        "Stoch FM ODE",
        f"Stoch FM SDE best-of-6\n(noise={best_noise:.2f})",
        "HR Target",
    ]

    for c in range(n):
        axes[0, c].imshow(lr[c, 0].cpu().numpy(), cmap="viridis")
        axes[0, c].axis("off")
        axes[0, c].set_title(f"Sample {c+1}")

        axes[1, c].imshow(det_sr[c, 0].cpu().numpy(), cmap="viridis")
        axes[1, c].axis("off")

        axes[2, c].imshow(stoch_ode_sr[c, 0].cpu().numpy(), cmap="viridis")
        axes[2, c].axis("off")

        axes[3, c].imshow(best_sde[c, 0].cpu().numpy(), cmap="viridis")
        axes[3, c].axis("off")

        axes[4, c].imshow(hr[c, 0].cpu().numpy(), cmap="viridis")
        axes[4, c].axis("off")

    for r in range(5):
        axes[r, 0].text(
            -0.08,
            0.5,
            row_names[r],
            transform=axes[r, 0].transAxes,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Stochastic CycleGAN-Coupled SF²M Ablation (30 epochs)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.06, 0.02, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(rows, det_ref, out_md: Path, best_sigma: float, best_noise: float, best_bestk_ssim: float):
    lines = []
    lines.append("# Stochastic CycleGAN-Coupled SF²M Ablation")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Coupling: CycleGAN pseudo-targets (OT replaced)")
    lines.append("- Regime: 300 images, 30 epochs, batch=8, base_channels=32")
    lines.append("- Evaluations: ODE single-output, SDE single-output, SDE best-of-6")
    lines.append("")
    lines.append("## Deterministic Reference")
    lines.append(f"- Deterministic FM + CycleGAN coupling (ODE): SSIM={det_ref['ssim']:.4f}, PSNR={det_ref['psnr']:.2f}")
    lines.append("")
    lines.append("## Stochastic Sweep")
    lines.append("")
    lines.append("| sigma_max | best noise_scale | ODE SSIM | ODE PSNR | best SDE SSIM | best SDE PSNR | best-of-6 SSIM | best-of-6 PSNR | diversity |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['sigma_max']:.2f} | {row['best_noise']:.2f} | {row['ode_ssim']:.4f} | {row['ode_psnr']:.2f} | "
            f"{row['best_sde_ssim']:.4f} | {row['best_sde_psnr']:.2f} | {row['bestk_ssim']:.4f} | {row['bestk_psnr']:.2f} | {row['diversity']:.6f} |"
        )

    lines.append("")
    lines.append("## Takeaway")
    lines.append(f"- Best stochastic configuration: sigma_max={best_sigma:.2f}, noise_scale={best_noise:.2f}, best-of-6 SSIM={best_bestk_ssim:.4f}")
    lines.append("- If stochastic is still below deterministic, main reason is target/pseudo-target ambiguity + noise-sensitive objective under single-target SSIM.")
    lines.append("- Visuals: `STOCHASTIC_CYCLEGAN_ABLATION_VISUALS.png`")

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    set_seed(123)
    device = get_device()

    base = Config(
        seed=123,
        n_images=300,
        epochs=30,
        batch_size=8,
        base_channels=32,
        train_ratio=0.85,
        downsample_factor=4,
        blur_sigma=1.0,
        blur_radius=3,
        inference_steps=50,
        coupling_mode="cyclegan",
        ot_reg=0.01,
        cyclegan_pretrain_epochs=30,
        cyclegan_gan_start_epoch=2,
    )

    noise_scales = [0.02, 0.05, 0.08]
    sigma_sweep = [0.02, 0.05, 0.10]

    print("=" * 78)
    print("STOCHASTIC CYCLEGAN-COUPLED ABLATION (30-EPOCH REGIME)")
    print("=" * 78)

    det_cfg = replace(base, fm_type="deterministic", sigma_max=0.0, inference_mode="ode")
    det_model, det_schedule, det_hr_val, det_lr_val, det_train_sec = train_once(det_cfg, device)
    det_ssim, det_psnr = evaluate_on_validation(det_model, det_lr_val, det_hr_val, det_cfg, device, det_schedule)
    det_ref = {"ssim": det_ssim, "psnr": det_psnr, "train_s": det_train_sec}

    rows = []
    best_global = None
    best_model_pack = None

    for sigma_max in sigma_sweep:
        print("\n" + "-" * 78)
        print(f"Training stochastic model with sigma_max={sigma_max:.2f}")
        print("-" * 78)

        stoch_cfg = replace(base, fm_type="stochastic", sigma_max=sigma_max, inference_mode="ode")
        model, schedule, hr_val, lr_val, train_sec = train_once(stoch_cfg, device)

        ode_ssim, ode_psnr = evaluate_on_validation(model, lr_val, hr_val, stoch_cfg, device, schedule)

        best_sde_ssim = -1.0
        best_sde_psnr = -1.0
        bestk_ssim = -1.0
        bestk_psnr = -1.0
        best_noise = noise_scales[0]
        best_div = 0.0

        for noise_scale in noise_scales:
            sde_cfg = replace(stoch_cfg, inference_mode="sde", sde_noise_scale=noise_scale)
            sde_ssim, sde_psnr = evaluate_on_validation(model, lr_val, hr_val, sde_cfg, device, schedule)
            bo_ssim, bo_psnr = best_of_k_metrics(
                model,
                hr_val,
                lr_val,
                stoch_cfg,
                schedule,
                device,
                noise_scale=noise_scale,
                k=6,
                batch_size=6,
            )
            div = diversity_score(
                model,
                lr_val,
                stoch_cfg,
                schedule,
                device,
                noise_scale=noise_scale,
                k=6,
                n_eval=8,
            )

            if bo_ssim > bestk_ssim:
                bestk_ssim = bo_ssim
                bestk_psnr = bo_psnr
                best_noise = noise_scale
                best_div = div

            if sde_ssim > best_sde_ssim:
                best_sde_ssim = sde_ssim
                best_sde_psnr = sde_psnr

        row = {
            "sigma_max": sigma_max,
            "ode_ssim": ode_ssim,
            "ode_psnr": ode_psnr,
            "best_sde_ssim": best_sde_ssim,
            "best_sde_psnr": best_sde_psnr,
            "bestk_ssim": bestk_ssim,
            "bestk_psnr": bestk_psnr,
            "best_noise": best_noise,
            "diversity": best_div,
            "train_s": train_sec,
        }
        rows.append(row)

        if best_global is None or row["bestk_ssim"] > best_global["bestk_ssim"]:
            best_global = row
            best_model_pack = (model, schedule, hr_val, lr_val)

    out_md = Path(__file__).with_name("STOCHASTIC_CYCLEGAN_ABLATION_RESULTS.md")
    write_report(
        rows=rows,
        det_ref=det_ref,
        out_md=out_md,
        best_sigma=best_global["sigma_max"],
        best_noise=best_global["best_noise"],
        best_bestk_ssim=best_global["bestk_ssim"],
    )

    out_img = Path(__file__).with_name("STOCHASTIC_CYCLEGAN_ABLATION_VISUALS.png")
    save_visuals(
        det_model=det_model,
        det_schedule=det_schedule,
        stoch_model=best_model_pack[0],
        stoch_schedule=best_model_pack[1],
        hr_val=best_model_pack[2],
        lr_val=best_model_pack[3],
        base_cfg=base,
        best_sigma=best_global["sigma_max"],
        best_noise=best_global["best_noise"],
        out_path=out_img,
    )

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Deterministic ref SSIM: {det_ref['ssim']:.4f}")
    print(f"Best stochastic best-of-6 SSIM: {best_global['bestk_ssim']:.4f}")
    print(f"Best stochastic sigma_max={best_global['sigma_max']:.2f}, noise={best_global['best_noise']:.2f}")
    print(f"Report: {out_md}")
    print(f"Visuals: {out_img}")
