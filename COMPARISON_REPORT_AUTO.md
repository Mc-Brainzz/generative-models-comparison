# Auto Comparison Report (Generated)

Generated on: 2026-03-16 23:11:09

## Sources Parsed
- Super_resolution/FINAL_README.md
- Super_resolution/TRULY_UNPAIRED_README.md
- Super_resolution/FLOW_MATCHING_README.md

## Quantitative Comparison

| Method | SSIM | PSNR | Train Time | Infer Time | Notes | Source |
|---|---:|---:|---:|---:|---|---|
| Bicubic | 0.87 | 28.24 | N/A |  | Baseline | Super_resolution/TRULY_UNPAIRED_README.md |
| SDEdit | 0.09 | 20 | Medium | Slow |  | Super_resolution/FLOW_MATCHING_README.md |
| DDIB | 0.10 | 21 | Medium | Medium |  | Super_resolution/FLOW_MATCHING_README.md |
| CycleGAN SR |  |  |  |  | pending run |  |
| Flow Matching Stochastic (Unpaired) | 0.5161 | 25.91 |  |  |  | Super_resolution/FINAL_README.md |
| Flow Matching Deterministic (Unpaired) | 0.7772 | 27.93 |  |  |  | Super_resolution/FINAL_README.md |

## Key Extracted Findings
- Deterministic vs stochastic SSIM delta: +0.2611
- Deterministic vs bicubic SSIM gap: -0.0928
- Use this as a controlled benchmark claim, then validate on an external dataset.

## Available Visuals
- Super_resolution/results_best_quality.png
- Super_resolution/results_diverse_outputs_(sde).png
- Super_resolution/results_paired_vs_unpaired.png
- Super_resolution/results_stochastic_comparison.png
- Super_resolution/results_stochastic_investigation.png
- Super_resolution/results_truly_unpaired.png
- Super_resolution/results_unpaired_data_(ot).png
- Super_resolution/results_unpaired_det_vs_stoch.png