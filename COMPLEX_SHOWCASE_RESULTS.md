# Complex Regime Showcase Results

Script: [run_complex_stochastic_showcase.py](run_complex_stochastic_showcase.py)
Date: 2026-03-17

## Hard setup used
- Unpaired coupling
- Strong degradation: `downsample_factor=8`, `blur_sigma=1.8`
- `n_images=180`, `epochs=8`, `base_channels=16`

## Main results
- Deterministic ODE: SSIM `0.7315`, PSNR `24.91`
- Stochastic ODE: SSIM `0.6188`, PSNR `24.91`

Stochastic SDE sweep (best-of-6 SSIM):
- noise `0.05` -> `0.3554`
- noise `0.10` -> `0.1277`
- noise `0.15` -> `0.0671`
- noise `0.20` -> `0.0395`

Best SDE setting from sweep: `noise_scale=0.05`.

## Takeaway
- In this run, deterministic still wins single-output quality.
- Stochastic produces diverse outputs (non-zero pairwise sample MSE), but that diversity did not translate into higher SSIM under this metric.
- This indicates the benchmark is still largely single-target/structure-alignment dominated.

## What to change if we want stochastic to win fairly
1. Use a genuinely one-to-many target setup (multiple plausible HR per LR).
2. Evaluate with best-of-k against a set of plausible targets or downstream uncertainty-aware metrics.
3. Keep SDE noise moderate (`<=0.05`) and train stochastic model longer.
