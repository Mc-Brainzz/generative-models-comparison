# Live Demo Results (Executed in Current Environment)

Date: 2026-03-16
Script: [run_live_training_demo.py](run_live_training_demo.py)

## Setup
- Dataset size: `n_images=120`
- Epochs: `6`
- Batch size: `8`
- Base channels: `16`
- Coupling mode: `unpaired`
- Inference mode: `ode`
- Inference steps: `20`
- Device: NVIDIA GeForce RTX 4050 Laptop GPU

## Results

| Method | SSIM | PSNR | Train Time |
|---|---:|---:|---:|
| Deterministic (unpaired FM, sigma=0.0) | 0.3371 | 23.51 | 3.9s |
| Stochastic (unpaired FM, sigma=0.1) | 0.2762 | 21.68 | 1.3s |

Delta (Det - Stoch):
- SSIM: `+0.0608`
- PSNR: `+1.84`

## Training-Loss Behavior Observed
- Deterministic FM loss decreased quickly: `0.107 -> 0.00534`
- Stochastic FM loss stayed higher and noisier: `0.133 -> 0.06072`

## Interpretation (Small-Scale Demo)
- In this quick unpaired run, deterministic interpolation gave better reconstruction metrics than stochastic interpolation.
- This supports your broader repository finding direction, but this is still a small budget run and should be presented as a demo result, not a final claim.
