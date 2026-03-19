# CycleGAN Integration Report (SF²M)

## Setup
- Task: Unpaired super-resolution on Van der Pol trajectory images
- Goal: Replace OT coupling with CycleGAN pseudo-target coupling in SF²M
- Epochs: 30 (same regime as your other test cases)

## Quantitative Results

| Method | Coupling | SSIM | PSNR (dB) | Train (s) | Eval (s) | Params |
|---|---|---:|---:|---:|---:|---:|
| Deterministic FM | paired | 0.8733 | 28.25 | 37.5 | 6.6 | 1,992,129 |
| Deterministic FM | unpaired | 0.8510 | 27.99 | 35.7 | 6.6 | 1,992,129 |
| Deterministic FM | cyclegan | 0.8617 | 28.07 | 87.5 | 6.6 | 1,992,129 |
| Stochastic FM | cyclegan | 0.1269 | 21.64 | 87.8 | 6.6 | 1,992,129 |

## Key Comparison
- Deterministic FM (CycleGAN - OT) SSIM delta: +0.0108
- Under CycleGAN coupling, (Det - Stoch) SSIM delta: +0.7348

## Visual Comparison
- See `CYCLEGAN_INTEGRATION_VISUALS.png` for side-by-side LR/output/HR samples.