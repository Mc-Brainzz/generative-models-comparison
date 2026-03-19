# Stochastic CycleGAN-Coupled SF²M Ablation

## Setup
- Coupling: CycleGAN pseudo-targets (OT replaced)
- Regime: 300 images, 30 epochs, batch=8, base_channels=32
- Evaluations: ODE single-output, SDE single-output, SDE best-of-6

## Deterministic Reference
- Deterministic FM + CycleGAN coupling (ODE): SSIM=0.8617, PSNR=28.07

## Stochastic Sweep

| sigma_max | best noise_scale | ODE SSIM | ODE PSNR | best SDE SSIM | best SDE PSNR | best-of-6 SSIM | best-of-6 PSNR | diversity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.02 | 0.8770 | 28.25 | 0.8706 | 28.25 | 0.8708 | 28.49 | 0.000001 |
| 0.05 | 0.02 | 0.4685 | 26.32 | 0.4663 | 26.62 | 0.4688 | 26.83 | 0.000112 |
| 0.10 | 0.02 | 0.1269 | 21.64 | 0.1152 | 23.16 | 0.1182 | 23.18 | 0.001045 |

## Takeaway
- Best stochastic configuration: sigma_max=0.02, noise_scale=0.02, best-of-6 SSIM=0.8708
- If stochastic is still below deterministic, main reason is target/pseudo-target ambiguity + noise-sensitive objective under single-target SSIM.
- Visuals: `STOCHASTIC_CYCLEGAN_ABLATION_VISUALS.png`