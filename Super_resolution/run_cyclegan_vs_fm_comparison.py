"""
CycleGAN-Coupled SF²M Comparison

Compares Flow Matching variants under three coupling strategies:
1) paired      : oracle paired supervision
2) unpaired    : mini-batch OT coupling
3) cyclegan    : CycleGAN pseudo-target coupling (replaces OT)

Also compares deterministic vs stochastic FM under CycleGAN coupling.
This version uses the same 30-epoch regime as your other test cases
and saves image-based visual comparisons.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_matching import (
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


def run_single_experiment(base: Config, coupling_mode: str, fm_type: str, device: torch.device):
    exp_cfg = Config(**{**base.__dict__, "coupling_mode": coupling_mode, "fm_type": fm_type})

    set_seed(exp_cfg.seed)
    train_loader, x1_val, _x0_val, lr_val = prepare_data_loaders(exp_cfg, device)

    model = VelocityUNet(base_channels=exp_cfg.base_channels).to(device)

    start_train = time.time()
    model = train_flow_matching(model, train_loader, exp_cfg, device)
    train_time = time.time() - start_train

    schedule = InterpolantSchedule(
        "stochastic" if exp_cfg.fm_type == "stochastic" else "linear",
        sigma_max=exp_cfg.sigma_max,
    )

    start_eval = time.time()
    mean_ssim, mean_psnr = evaluate_on_validation(model, lr_val, x1_val, exp_cfg, device, schedule)
    eval_time = time.time() - start_eval

    preview_n = min(4, lr_val.size(0))
    with torch.no_grad():
        sr_preview = flow_matching_inference(
            model,
            lr_val[:preview_n],
            exp_cfg,
            device,
            schedule,
        )
    preview_ssim = compute_ssim_scores(sr_preview, x1_val[:preview_n])

    params = sum(p.numel() for p in model.parameters())

    return {
        "coupling": coupling_mode,
        "fm_type": fm_type,
        "ssim": float(mean_ssim),
        "psnr": float(mean_psnr),
        "train_time_s": float(train_time),
        "eval_time_s": float(eval_time),
        "params": int(params),
        "preview_lr": lr_val[:preview_n].detach().cpu(),
        "preview_hr": x1_val[:preview_n].detach().cpu(),
        "preview_sr": sr_preview.detach().cpu(),
        "preview_ssim": [float(x) for x in preview_ssim],
    }


def save_visual_comparison(results, out_img: Path):
    method_order = [
        ("paired", "deterministic", "Det FM (Paired)"),
        ("unpaired", "deterministic", "Det FM (OT)"),
        ("cyclegan", "deterministic", "Det FM (CycleGAN)"),
        ("cyclegan", "stochastic", "Stoch FM (CycleGAN)"),
    ]

    selected = []
    for coupling, fm_type, label in method_order:
        result = next(
            (r for r in results if r["coupling"] == coupling and r["fm_type"] == fm_type),
            None,
        )
        if result is not None:
            selected.append((label, result))

    if not selected:
        return

    n_samples = selected[0][1]["preview_lr"].size(0)
    n_rows = len(selected) + 2  # LR + method rows + HR

    fig, axes = plt.subplots(n_rows, n_samples, figsize=(3.1 * n_samples, 2.6 * n_rows))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    lr_ref = selected[0][1]["preview_lr"]
    hr_ref = selected[0][1]["preview_hr"]

    for col in range(n_samples):
        axes[0, col].imshow(lr_ref[col, 0].numpy(), cmap="viridis")
        axes[0, col].set_title(f"Sample {col + 1}")
        axes[0, col].axis("off")

    for row_idx, (label, result) in enumerate(selected, start=1):
        for col in range(n_samples):
            axes[row_idx, col].imshow(result["preview_sr"][col, 0].numpy(), cmap="viridis")
            axes[row_idx, col].set_title(f"SSIM {result['preview_ssim'][col]:.3f}")
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].text(
            -0.09,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
        )

    for col in range(n_samples):
        axes[-1, col].imshow(hr_ref[col, 0].numpy(), cmap="viridis")
        axes[-1, col].axis("off")

    axes[0, 0].text(
        -0.09,
        0.5,
        "LR Input",
        transform=axes[0, 0].transAxes,
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
    )
    axes[-1, 0].text(
        -0.09,
        0.5,
        "HR Target",
        transform=axes[-1, 0].transAxes,
        va="center",
        ha="right",
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle("CycleGAN-Coupled SF²M: Visual Method Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.05, 0.02, 1, 0.97])
    plt.savefig(out_img, dpi=150)
    plt.close(fig)


def write_report(results, out_md: Path):
    lines = []
    lines.append("# CycleGAN Integration Report (SF²M)")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Task: Unpaired super-resolution on Van der Pol trajectory images")
    lines.append("- Goal: Replace OT coupling with CycleGAN pseudo-target coupling in SF²M")
    lines.append("- Epochs: 30 (same regime as your other test cases)")
    lines.append("")
    lines.append("## Quantitative Results")
    lines.append("")
    lines.append("| Method | Coupling | SSIM | PSNR (dB) | Train (s) | Eval (s) | Params |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for r in results:
        method = "Deterministic FM" if r["fm_type"] == "deterministic" else "Stochastic FM"
        lines.append(
            f"| {method} | {r['coupling']} | {r['ssim']:.4f} | {r['psnr']:.2f} | "
            f"{r['train_time_s']:.1f} | {r['eval_time_s']:.1f} | {r['params']:,} |"
        )

    lines.append("")
    lines.append("## Key Comparison")

    det_ot = next((x for x in results if x["coupling"] == "unpaired" and x["fm_type"] == "deterministic"), None)
    det_cyc = next((x for x in results if x["coupling"] == "cyclegan" and x["fm_type"] == "deterministic"), None)
    stoch_cyc = next((x for x in results if x["coupling"] == "cyclegan" and x["fm_type"] == "stochastic"), None)

    if det_ot and det_cyc:
        lines.append(f"- Deterministic FM (CycleGAN - OT) SSIM delta: {det_cyc['ssim'] - det_ot['ssim']:+.4f}")
    if det_cyc and stoch_cyc:
        lines.append(f"- Under CycleGAN coupling, (Det - Stoch) SSIM delta: {det_cyc['ssim'] - stoch_cyc['ssim']:+.4f}")

    lines.append("")
    lines.append("## Visual Comparison")
    lines.append("- See `CYCLEGAN_INTEGRATION_VISUALS.png` for side-by-side LR/output/HR samples.")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
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
        inference_mode="ode",
        ot_reg=0.01,
        sigma_max=0.1,
        cyclegan_pretrain_epochs=30,
        cyclegan_gan_start_epoch=2,
    )

    experiments = [
        ("paired", "deterministic"),
        ("unpaired", "deterministic"),
        ("cyclegan", "deterministic"),
        ("cyclegan", "stochastic"),
    ]

    results = []
    for coupling, fm_type in experiments:
        print("\n" + "=" * 72)
        print(f"Running: coupling={coupling}, fm_type={fm_type}")
        print("=" * 72)
        results.append(run_single_experiment(base, coupling, fm_type, device))

    out_md = Path(__file__).with_name("CYCLEGAN_INTEGRATION_RESULTS.md")
    write_report(results, out_md)

    out_img = Path(__file__).with_name("CYCLEGAN_INTEGRATION_VISUALS.png")
    save_visual_comparison(results, out_img)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Report: {out_md}")
    print(f"Visuals: {out_img}")


if __name__ == "__main__":
    main()
