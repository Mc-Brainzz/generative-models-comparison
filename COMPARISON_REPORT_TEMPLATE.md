# Usual Methods vs My Method: Comparison Document

Use this file as your final comparison write-up for reports, posts, and threads.

---

## 1) Comparison Goal

Problem statement:
- Recover useful high-resolution thin-structure details from low-resolution inputs.
- Focus on the realistic case where paired high-resolution labels are limited or unavailable.

Primary question:
- Does my unpaired flow-based approach outperform standard baselines under the same setup?

---

## 2) Methods Compared

### Usual methods (baselines)
1. Bicubic upsampling (non-learning baseline)
2. SDEdit-style SR
3. DDIB-style SR
4. CycleGAN-style unpaired SR
5. Flow Matching (stochastic, unpaired)

### My method
6. Flow Matching (deterministic, unpaired OT-coupled)
7. Flow Matching (deterministic/stochastic, unpaired CycleGAN-coupled)

Optional future method:
8. Ambiguity-aware adaptive interpolation (proposed extension)

---

## 3) Fairness Protocol (must be identical unless noted)

Data and split:
- Same train/validation/test split for all methods
- Same random seeds (report seeds)
- Same degradation operator and resolution settings

Training budget:
- Equal epochs or equal wall-clock budget
- Comparable model capacity where possible
- Report parameter count

Inference budget:
- Report inference steps
- Report deterministic/stochastic sampling mode
- Report average runtime per image

Metrics:
- SSIM (higher is better)
- PSNR (higher is better)
- Training time (lower is better)
- Inference time (lower is better)

---

## 4) Exact Setup Used (fill this)

Dataset:
- Source type:
- Target type:
- Resolution:
- Downsample factor:
- Blur settings:

Training:
- Number of images:
- Batch size:
- Epochs:
- Learning rate:
- Seed list:

Unpaired coupling:
- Coupling mode:
- OT regularization:
- Batch OT solver:

Flow settings:
- Flow type:
- sigma max:
- Inference mode:
- Inference steps:

---

## 5) Quantitative Results Table (fill this)

| Method | Pairing | SSIM | PSNR | Train Time | Infer Time | Params | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Bicubic | N/A |  |  | N/A |  | N/A | interpolation only |
| SDEdit | Paired |  |  |  |  |  |  |
| DDIB | Paired/Unpaired |  |  |  |  |  |  |
| CycleGAN SR | Unpaired |  |  |  |  |  |  |
| FM stochastic | Unpaired OT |  |  |  |  |  |  |
| FM deterministic (my method) | Unpaired OT |  |  |  |  |  |  |
| FM deterministic | Unpaired CycleGAN |  |  |  |  |  | OT replaced by pseudo-target coupling |
| FM stochastic | Unpaired CycleGAN |  |  |  |  |  | OT replaced by pseudo-target coupling |

Add mean ± std over multiple seeds if possible.

---

## 6) Qualitative Comparison (what people can see)

For each method, show 2 to 4 examples with this layout:
- LR input
- Bicubic output
- Method output
- Ground truth

Comment on:
- edge sharpness,
- continuity of thin structures,
- artifact level,
- obvious hallucinations.

---

## 7) Key Findings (write in this style)

1. Under truly unpaired training, deterministic flow matching achieved better structural recovery than stochastic flow matching in this benchmark.
2. More stochastic training did not reliably close the gap in the tested setup.
3. Performance depends on coupling uncertainty and data sparsity, not only model family.

---

## 8) Failure Cases and Limits (mandatory)

Document where your method fails:
- very large domain gap,
- severe noise,
- extreme sparsity,
- unstable OT pairing at small batch size,
- weak CycleGAN pretraining quality causing poor pseudo-targets.

This section increases credibility and helps avoid over-claiming.

---

## 9) Reproducibility Checklist

- Scripts used:
  - Super_resolution/test_configs.py
  - Super_resolution/test_stochastic_comparison.py
  - Super_resolution/test_truly_unpaired.py
  - Super_resolution/test_paired_vs_unpaired.py
  - Super_resolution/test_unpaired_det_vs_stoch.py
  - Super_resolution/test_stochastic_investigation.py
  - Super_resolution/cyclegan_sr.py
- Environment:
  - Python version:
  - Torch version:
  - GPU/CPU:
- Runtime date:
- Commit hash:

---

## 10) Copy-Ready Public Summary

Short version:
- Real bottleneck: paired high-quality labels are scarce.
- We tested usual SR methods vs unpaired deterministic flow matching.
- In our controlled benchmark, deterministic unpaired flow recovered thin structures better than stochastic alternatives.

One-line claim (safe):
- In this defined unpaired setting, a simpler deterministic approach gave stronger structure recovery than expected.

---

## 11) Slide and Twitter Packaging

For a 1-slide comparison:
- Left: problem and bottleneck
- Middle: table with SSIM/PSNR/time
- Right: 2 visual examples and one takeaway

For a thread:
1. Bottleneck statement
2. Setup fairness statement
3. Results table snapshot
4. Visual comparisons
5. Failure case + next step (adaptive interpolation)

---

## 12) What to do next if reviewers ask “what is novel?”

Answer:
- Not a new base model family.
- Novelty is the controlled unpaired comparison protocol and the deterministic advantage regime finding.
- Next novelty extension is ambiguity-aware interpolation driven by OT uncertainty.
