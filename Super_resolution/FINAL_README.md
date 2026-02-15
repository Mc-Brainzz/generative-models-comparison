# Flow Matching for Super-Resolution: Complete Guide

## Everything We Learned About Paired, Unpaired, Deterministic & Stochastic Training

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What is Flow Matching?](#2-what-is-flow-matching)
3. [The Super-Resolution Problem](#3-the-super-resolution-problem)
4. [Paired vs Truly Unpaired Training](#4-paired-vs-truly-unpaired-training)
5. [Optimal Transport Coupling](#5-optimal-transport-coupling)
6. [Deterministic vs Stochastic Interpolation](#6-deterministic-vs-stochastic-interpolation)
7. [The Counter-Intuitive Finding](#7-the-counter-intuitive-finding)
8. [Why Deterministic Beats Stochastic (Analysis)](#8-why-deterministic-beats-stochastic-analysis)
9. [How to Fix It & When to Use What](#9-how-to-fix-it--when-to-use-what)
10. [Experimental Results Summary](#10-experimental-results-summary)
11. [Errors Encountered & Solutions](#11-errors-encountered--solutions)
12. [Configuration Reference](#12-configuration-reference)
13. [Key Takeaways](#13-key-takeaways)
14. [Files in This Project](#14-files-in-this-project)

---

## 1. Executive Summary

### What We Built
A Flow Matching model for image super-resolution that can be trained with:
- **Paired data**: LR and HR from the same image (traditional supervised learning)
- **Truly unpaired data**: LR from one dataset, HR from a completely different dataset

### The Counter-Intuitive Discovery
When training on **truly unpaired data** with Optimal Transport coupling:

| Training Method | Expected | Actual Result |
|-----------------|----------|---------------|
| Stochastic interpolation | Better (handles uncertainty) | SSIM = 0.27 |
| Deterministic interpolation | Worse | **SSIM = 0.78** ✓ |

**Deterministic training was 3× better than stochastic!**

### Why This Happened
1. OT coupling found good matches (Van der Pol trajectories are similar)
2. Sparse data (thin lines) is sensitive to noise
3. Simpler learning target (constant velocity) converges faster

### Key Lesson
**Match your training method to your data characteristics**, not just theoretical expectations.

---

## 2. What is Flow Matching?

### The Core Idea
Instead of learning a probability distribution directly, learn a **velocity field** that transports samples from one distribution to another.

```
Source Distribution (LR images)  →  Target Distribution (HR images)
         p₀                    v(x,t)                    p₁
         
At t=0: x₀ ~ p₀ (blurry image)
At t=1: x₁ ~ p₁ (sharp image)

The velocity field v(x,t) tells us how to move from x₀ to x₁
```

### Training Objective
```
Loss = E_{t, x₀, x₁} || v_θ(x_t, t) - v*(t) ||²

Where:
- v_θ is the neural network (we train this)
- v* is the target velocity (we compute this from data)
- x_t is the interpolated point at time t
```

### Inference
Solve the ODE from t=0 to t=1:
```
dx/dt = v_θ(x, t)
Starting from x₀ (upsampled LR image)
Ending at x₁ (super-resolved HR image)
```

---

## 3. The Super-Resolution Problem

### The Challenge
```
HR Image (128×128) → [Blur] → [Downsample 4×] → LR Image (32×32)
                                                      ↓
                              We want to reverse this! (But it's ambiguous)
```

Super-resolution is **ill-posed**: many HR images can produce the same LR image.

### Our Test Data: Van der Pol Oscillator
We use trajectories from the Van der Pol oscillator:
```python
dx/dt = v
dv/dt = μ(1 - x²)v - x
```

These create characteristic limit-cycle patterns rendered as images.

**Why this data?**
- Structured, non-Gaussian distributions
- Controllable complexity
- Ground truth available
- Physics-based (deterministic ODE)

### Baseline: Bicubic Upsampling
Simple mathematical interpolation (no learning):
- Uses 16 neighbors with cubic polynomial
- **Cannot add missing details** - just smooth enlargement
- Our baseline to beat: SSIM ≈ 0.87

---

## 4. Paired vs Truly Unpaired Training

### Paired Training (Traditional)
```
LR_1 ←→ HR_1  (same image, degraded vs original)
LR_2 ←→ HR_2
LR_3 ←→ HR_3

Direct supervision: "This LR → This HR"
```

### Truly Unpaired Training (Our Innovation)
```
LR images: [A, B, C, D, ...]  ← From Dataset 1 (e.g., noisy captures)
HR images: [X, Y, Z, W, ...]  ← From Dataset 2 (e.g., clean references)

NO relationship between A and X, B and Y, etc.!
```

### Why Unpaired Matters
Real-world scenarios where paired data is impossible:
- HR: Clean simulation renders / LR: Noisy real captures
- HR: High-quality microscopy dataset A / LR: Low-quality dataset B
- Medical imaging with privacy constraints

### How We Make It Work: Optimal Transport
OT finds the **best matching** within each mini-batch, even without true pairs.

---

## 5. Optimal Transport Coupling

### The Problem
Given unpaired batches:
```
Source (LR): [s₁, s₂, s₃, s₄]
Target (HR): [t₁, t₂, t₃, t₄]
```

Find the optimal pairing that minimizes total "transport cost".

### Cost Matrix
```python
Cost[i,j] = ||s_i - t_j||²  # L2 distance between flattened images
```

### Solving OT

#### Exact OT (Hungarian Algorithm)
```python
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost_matrix)
# col_ind gives the optimal permutation
```
- O(n³) complexity
- Optimal solution
- Use when `ot_reg = 0`

#### Sinkhorn OT (Entropy-Regularized)
```python
K = exp(-cost_matrix / reg)
for _ in range(n_iters):
    u = 1 / (K @ v)
    v = 1 / (K.T @ u)
P = diag(u) @ K @ diag(v)
```
- Faster, differentiable
- Approximate solution
- Use when `ot_reg > 0` (e.g., 0.01)

### Visualization
```
Before OT:              After OT:
LR: [A, B, C, D]       LR: [A, B, C, D]
HR: [W, X, Y, Z]       HR: [X, Z, W, Y]  ← Reordered!
     ↓                       ↓
No correspondence      Optimal matching:
                       A↔X, B↔Z, C↔W, D↔Y
```

---

## 6. Deterministic vs Stochastic Interpolation

### Deterministic (Linear Interpolation)
```
x_t = (1-t)·x₀ + t·x₁

Target velocity: v* = x₁ - x₀  (CONSTANT!)
```

```
t=0 ─────────────────────────────────────> t=1
 x₀                                          x₁
(LR)          straight line                 (HR)
```

**Properties:**
- Simple learning target (constant velocity)
- Fast convergence
- Deterministic output

### Stochastic Interpolation
```
x_t = (1-t)·x₀ + t·x₁ + σ(t)·ε

Where: σ(t) = σ_max · sin(πt)  (zero at boundaries, max at t=0.5)
       ε ~ N(0, I)             (random noise)

Target velocity: v* = -x₀ + x₁ + σ'(t)·ε  (TIME-VARYING!)
```

```
t=0 ─────────╔═══════════╗─────────────> t=1
 x₀          ║  + noise  ║                x₁
(LR)         ║   σ(t)ε   ║               (HR)
             ╚═══════════╝
```

**Properties:**
- Noise acts as regularization
- Can use ODE (deterministic) or SDE (stochastic) inference
- Harder to train (time-varying target)

### Inference Modes

| Mode | Equation | Output |
|------|----------|--------|
| ODE | dx/dt = v(x,t) | Deterministic (same every time) |
| SDE | dx = v(x,t)dt + σ'(t)dW | Stochastic (different each run) |

---

## 7. The Counter-Intuitive Finding

### Our Expectation
For **truly unpaired data** with OT coupling:
- OT gives imperfect matches (best within batch, not true pairs)
- Stochastic interpolation should handle this uncertainty better
- Adding noise should provide robustness to imperfect pairings

**Expected: Stochastic > Deterministic**

### What Actually Happened

| Method | SSIM | PSNR | Final Loss |
|--------|------|------|------------|
| **Deterministic (40 ep)** | **0.7772** | **27.93** | **0.00037** |
| Stochastic (40 ep, σ=0.1) | 0.2686 | 24.36 | 0.00668 |
| Stochastic (100 ep, σ=0.1) | 0.2502 | 18.75 | 0.00270 |
| Stochastic (40 ep, σ=0.05) | 0.5161 | 25.91 | 0.00258 |
| Bicubic (baseline) | 0.8698 | 28.24 | - |

**Actual: Deterministic >> Stochastic (3× better!)**

### The Surprise
- More training (100 epochs) made stochastic **worse**!
- Lower noise (σ=0.05) helped but still didn't match deterministic

---

## 8. Why Deterministic Beats Stochastic (Analysis)

### Reason 1: OT Coupling Works Better Than Expected

Van der Pol trajectories with similar μ parameters **look visually similar**:

```
μ = 0.8: ╭──────╮    μ = 1.2: ╭──────╮
         │      │             │      │
         ╰──────╯             ╰──────╯
         
These look similar! OT finds good matches.
```

The "uncertainty" from imperfect pairing is **smaller than expected**.

### Reason 2: Sparse Data Characteristic

Van der Pol trajectories are **thin lines** on mostly black background:

```
┌────────────────┐
│                │
│    ╭────╮      │  ← Thin lines (sparse)
│    │    │      │
│    ╰────╯      │
│                │
└────────────────┘
```

**Why this matters:**
- Thin lines are extremely sensitive to spatial perturbation
- Any noise during training **blurs** these delicate structures
- SSIM heavily penalizes spatial misalignment
- Deterministic preserves sharp line positions

### Reason 3: Learning Target Complexity

```
Deterministic:
  v* = x₁ - x₀
  
  → CONSTANT in time
  → Simple MSE regression
  → Easy to learn
  → Loss converges to 0.0003

Stochastic:
  v* = α'(t)·x₀ + β'(t)·x₁ + σ'(t)·ε
  
  → VARIES with time t
  → Different target at each t
  → Must learn time-dependent function
  → Loss stuck at 0.006 (20× higher!)
```

### Reason 4: Training Dynamics

```
Training Loss Curves:

Deterministic:  ████████▓▓▓░░░░░░  → 0.0003 (converged!)
Stochastic:     ████████████████▓▓ → 0.0066 (not converged)

Stochastic needs MUCH more training to converge through the noise.
```

---

## 9. How to Fix It & When to Use What

### For Sparse/Thin Structure Data (like Van der Pol)

**Use: Deterministic Training**
```python
Config(
    fm_type='deterministic',
    sigma_max=0.0,
    inference_mode='ode'
)
```

### For Dense Natural Images (photos, textures)

**Use: Stochastic with Low Noise**
```python
Config(
    fm_type='stochastic',
    sigma_max=0.02,  # Very low noise
    inference_mode='ode'  # Still deterministic inference
)
```

### For Diverse Output Generation

**Use: Stochastic Training + SDE Inference**
```python
Config(
    fm_type='stochastic',
    sigma_max=0.05,
    inference_mode='sde',
    sde_noise_scale=0.3
)
```

### For Truly Unpaired Data

**Start with: Deterministic**
```python
Config(
    coupling_mode='unpaired',
    ot_reg=0.01,
    fm_type='deterministic'  # Start here!
)
```

If results are blurry, try stochastic with very low noise.

### Decision Flowchart

```
                    ┌─────────────────────┐
                    │ What's your data?   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
      ┌───────▼───────┐               ┌───────▼───────┐
      │ Sparse/Lines  │               │ Dense/Texture │
      │ (Van der Pol) │               │ (Natural img) │
      └───────┬───────┘               └───────┬───────┘
              │                               │
              ▼                               ▼
      ┌───────────────┐               ┌───────────────┐
      │ DETERMINISTIC │               │  STOCHASTIC   │
      │  fm_type=det  │               │   σ=0.02-0.05 │
      └───────────────┘               └───────────────┘
              │                               │
              │         ┌─────────────────────┘
              │         │
              ▼         ▼
      ┌─────────────────────────────────────────┐
      │ Do you need diverse outputs?            │
      └───────────────────┬─────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌─────▼─────┐           ┌─────▼─────┐
        │    NO     │           │    YES    │
        │           │           │           │
        │ ODE infer │           │ SDE infer │
        └───────────┘           └───────────┘
```

---

## 10. Experimental Results Summary

### Experiment 1: Paired vs Unpaired (Stochastic Training)

| Method | SSIM | Notes |
|--------|------|-------|
| Paired Training | 0.25 | Direct LR↔HR correspondence |
| Truly Unpaired (OT) | 0.27 | Different trajectories, OT coupling |
| Bicubic | 0.87 | Baseline |

**Finding:** Unpaired ≈ Paired (with stochastic training)

### Experiment 2: Deterministic vs Stochastic (Paired)

| Method | SSIM | Loss |
|--------|------|------|
| Deterministic | 0.88 | 0.0003 |
| Stochastic | 0.39 | 0.008 |
| Bicubic | 0.87 | - |

**Finding:** Deterministic >> Stochastic for this data

### Experiment 3: Truly Unpaired - Det vs Stoch

| Method | SSIM | Loss |
|--------|------|------|
| Deterministic | **0.78** | 0.0003 |
| Stochastic (σ=0.1) | 0.27 | 0.007 |
| Stochastic (σ=0.05) | 0.52 | 0.003 |
| Bicubic | 0.87 | - |

**Finding:** Deterministic is 3× better; lower noise helps stochastic

### Experiment 4: More Training for Stochastic

| Epochs | SSIM | Loss |
|--------|------|------|
| 40 | 0.27 | 0.007 |
| 100 | 0.25 | 0.003 |

**Finding:** More training didn't help (and slightly hurt!)

---

## 11. Errors Encountered & Solutions

### Error 1: CUDA Out of Memory

**When:** `base_channels=64, batch_size=16` on 6GB GPU

**Message:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
```

**Solution:**
```python
# Option 1: Smaller model
Config(base_channels=48)

# Option 2: Smaller inference batch
evaluate_on_validation(model, ..., batch_size=4)

# Option 3: Clear cache
torch.cuda.empty_cache()
```

### Error 2: Tensor Contiguity

**When:** Using `view()` after tensor indexing

**Message:**
```
RuntimeError: view size is not compatible with input tensor's size and stride
```

**Solution:**
```python
# Replace view with reshape
x_flat = x.reshape(x.size(0), -1)  # Not x.view(...)
```

### Error 3: Sinkhorn Numerical Instability

**When:** `torch.multinomial(P, ...)` with Sinkhorn output

**Message:**
```
RuntimeError: CUDA error: device-side assert triggered
Assertion `input[0] != 0` failed
```

**Solution:**
```python
P = solve_ot_sinkhorn(cost, reg=reg)
P = P + 1e-8  # Add epsilon to avoid zeros
P = P / P.sum(dim=1, keepdim=True)  # Renormalize
```

### Error 4: SSIM Dimension Mismatch

**When:** Passing already-upsampled images to `flow_matching_inference`

**Message:**
```
ValueError: Input images must have the same dimensions.
```

**Solution:**
```python
# Pass original LR (32×32), not upsampled (128×128)
sr = flow_matching_inference(model, lr_val, ...)  # Not x0_val
```

### Error 5: SDE Noise Too High

**When:** `sde_noise_scale=1.5`

**Result:** SSIM dropped to 0.003, outputs were pure noise

**Solution:**
```python
Config(sde_noise_scale=0.3)  # Much lower
```

---

## 12. Configuration Reference

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_images` | 300-600 | Training set size |
| `epochs` | 30-50 | Training epochs |
| `batch_size` | 8-16 | Mini-batch size |
| `base_channels` | 32-48 | Model capacity |
| `learning_rate` | 1e-4 | Adam learning rate |

### Flow Matching Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fm_type` | 'deterministic' | 'deterministic' or 'stochastic' |
| `sigma_max` | 0.0-0.1 | Max noise (stochastic only) |
| `inference_mode` | 'ode' | 'ode' or 'sde' |
| `inference_steps` | 50 | Integration steps |
| `sde_noise_scale` | 0.3 | SDE noise scale |

### Unpaired/OT Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coupling_mode` | 'paired' | 'paired' or 'unpaired' |
| `ot_reg` | 0.01 | Sinkhorn regularization (0=exact) |

### Recommended Configurations

**Sparse data (thin lines), paired:**
```python
Config(fm_type='deterministic', sigma_max=0.0, inference_mode='ode')
```

**Sparse data, unpaired:**
```python
Config(fm_type='deterministic', coupling_mode='unpaired', ot_reg=0.01)
```

**Dense images, paired:**
```python
Config(fm_type='stochastic', sigma_max=0.05, inference_mode='ode')
```

**Diverse outputs needed:**
```python
Config(fm_type='stochastic', sigma_max=0.05, inference_mode='sde', sde_noise_scale=0.3)
```

---

## 13. Key Takeaways

### 1. Theory vs Practice
> **Theoretical expectations don't always match empirical results.**
> 
> We expected stochastic to handle unpaired uncertainty better, but deterministic won decisively.

### 2. Data Characteristics Matter
> **Match your method to your data, not just the problem.**
> 
> Sparse data (thin lines) → Deterministic
> Dense data (textures) → Stochastic may help

### 3. OT Coupling Works!
> **Truly unpaired training is feasible with Optimal Transport.**
> 
> Even without any paired data, we achieved SSIM=0.78 (vs 0.87 bicubic baseline).

### 4. Simple Can Be Better
> **Simpler learning targets converge faster and more reliably.**
> 
> Deterministic: v* = x₁ - x₀ (constant) → Loss = 0.0003
> Stochastic: v* = f(t, x₀, x₁, ε) (complex) → Loss = 0.007

### 5. Noise Isn't Always Helpful
> **Regularization through noise can hurt sparse data.**
> 
> Lower noise (σ=0.05) significantly improved stochastic: 0.27 → 0.52

### 6. More Training Isn't Always Better
> **Training longer with the wrong method won't fix fundamental issues.**
> 
> 100 epochs of stochastic was actually worse than 40 epochs!

### 7. Debugging Deep Learning
> **Common errors have common solutions:**
> - OOM → Reduce batch/model size
> - Contiguity → Use reshape not view
> - Numerical issues → Add epsilon, renormalize

---

## 14. Files in This Project

| File | Description |
|------|-------------|
| `flow_matching.py` | Core implementation (model, training, inference) |
| `test_configs.py` | Configuration comparison (quality, SDE, OT) |
| `test_truly_unpaired.py` | True unpaired data experiment |
| `test_paired_vs_unpaired.py` | Paired vs unpaired comparison |
| `test_stochastic_comparison.py` | Det vs stoch interpolation |
| `test_unpaired_det_vs_stoch.py` | Unpaired: det vs stoch |
| `test_stochastic_investigation.py` | Why stochastic underperforms |
| `FLOW_MATCHING_README.md` | General Flow Matching docs |
| `TRULY_UNPAIRED_README.md` | Unpaired training docs |
| `FINAL_README.md` | **This comprehensive guide** |

### Generated Figures
- `results_best_quality.png` - Best quality config results
- `results_diverse_outputs_(sde).png` - SDE diverse outputs
- `results_unpaired_data_(ot).png` - OT coupling results
- `results_truly_unpaired.png` - True unpaired results
- `results_paired_vs_unpaired.png` - Comparison visualization
- `results_stochastic_comparison.png` - Det vs stoch comparison
- `results_unpaired_det_vs_stoch.png` - Unpaired det vs stoch
- `results_stochastic_investigation.png` - Loss curves analysis

---

## References

1. **Flow Matching for Generative Modeling** - Lipman et al. (2023)
2. **Building Normalizing Flows with Stochastic Interpolants** - Albergo & Vanden-Eijnden (2023)
3. **Computational Optimal Transport** - Peyré & Cuturi (2019)
4. **Sinkhorn Distances** - Cuturi (2013)

---

*Last updated: January 2026*
