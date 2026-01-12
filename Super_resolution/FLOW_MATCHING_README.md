# [SF]²M: Stochastic Flow Matching for Super-Resolution

## A Comprehensive Guide to Understanding, Using, and Extending This Implementation

---

## Table of Contents

1. [Introduction: What Problem Are We Solving?](#1-introduction-what-problem-are-we-solving)
2. [Why Flow Matching? Comparison with Other Methods](#2-why-flow-matching-comparison-with-other-methods)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Code Walkthrough: Understanding Each Component](#5-code-walkthrough-understanding-each-component)
6. [Configuration Parameters: What They Do](#6-configuration-parameters-what-they-do)
7. [Experiments and Ablations](#7-experiments-and-ablations)
8. [How to Improve Results](#8-how-to-improve-results)
9. [Frequently Asked Questions](#9-frequently-asked-questions)
10. [References](#10-references)

---

## 1. Introduction: What Problem Are We Solving?

### The Super-Resolution Challenge

**Super-resolution (SR)** is the task of recovering a high-resolution (HR) image from a low-resolution (LR) observation. This is an **ill-posed inverse problem** because:

- Multiple HR images can produce the same LR image when degraded
- Information is fundamentally lost during degradation (blur + downsampling)
- We need to "hallucinate" plausible high-frequency details

```
HR Image (128×128) → [Blur] → [Downsample 4×] → LR Image (32×32)
                                                      ↓
                              We want to reverse this! (But it's ambiguous)
```

### Our Test Case: Van der Pol Oscillator Trajectories

Instead of natural images, we use **Van der Pol oscillator trajectories** as our data:

```python
dx/dt = v
dv/dt = μ(1 - x²)v - x
```

**Why this choice?**
1. **Structured, non-Gaussian distributions** - Real-world data isn't Gaussian
2. **Physics-based** - Trajectories follow deterministic ODEs (useful for EDDIB comparison)
3. **Controllable complexity** - Easy to generate, analyze, and visualize
4. **Ground truth available** - We know the exact HR trajectory

The trajectories form characteristic **limit cycles** that look like distorted figure-8s when rendered as images.

---

## 2. Why Flow Matching? Comparison with Other Methods

### The Generative Model Landscape

| Method | Training | Inference | Pros | Cons |
|--------|----------|-----------|------|------|
| **VAE** | Single forward pass | Single forward pass | Fast, stable | Blurry outputs |
| **GAN** | Adversarial (unstable) | Single forward pass | Sharp outputs | Mode collapse, training instability |
| **Diffusion/DDPM** | Noise prediction | 100-1000 steps | High quality | Slow inference, complex schedules |
| **DDIB** | Two denoisers | Encode-bridge-decode | Good for translation | Needs two models |
| **Flow Matching** | Velocity prediction | 20-50 steps | Fast, stable, simple | Newer, less studied |

### Why Flow Matching Wins for Super-Resolution

#### 1. **Simpler Training Objective**
```
Diffusion: Predict noise ε from x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
Flow Matching: Predict velocity v from x_t = (1-t)·x₀ + t·x₁
```
Flow Matching has a **constant target** along each path - no complex noise schedules!

#### 2. **Faster Inference**
- Diffusion models: Typically need 100-1000 denoising steps
- Flow Matching: Works well with **20-50 ODE integration steps**
- Our implementation: ~0.1 seconds per image on RTX 4050

#### 3. **Natural Handling of Paired Data**
- In SR, we have natural pairs: (degraded LR, original HR)
- Flow Matching directly models the transport from LR → HR
- No need for unconditional generation + conditioning tricks

#### 4. **Deterministic or Stochastic**
- **ODE inference**: Deterministic, reproducible results
- **SDE inference**: Stochastic, can generate diverse outputs
- User chooses based on application needs

### Comparison Results on Van der Pol Data

| Method | SSIM | PSNR | Training Time | Inference Time |
|--------|------|------|---------------|----------------|
| Bicubic Upsampling | ~0.05 | ~18 dB | N/A | Instant |
| VAE | ~0.07 | ~19 dB | Fast | Fast |
| SDEdit | ~0.09 | ~20 dB | Medium | Slow |
| DDIB | ~0.10 | ~21 dB | Medium | Medium |
| **Flow Matching** | **~0.11** | **~22 dB** | Medium | **Fast** |

*Note: Results depend heavily on hyperparameters and training duration.*

---

## 3. Mathematical Foundation

### 3.1 The Core Idea: Probability Flow

We want to transform samples from distribution p₀ (blurry LR) to distribution p₁ (sharp HR).

**Key insight**: Instead of learning p₁ directly, learn a **velocity field** v(x, t) that transports particles from p₀ to p₁.

```
t=0: x₀ ~ p₀ (LR images)
     ↓ Follow velocity field v(x,t)
t=1: x₁ ~ p₁ (HR images)
```

### 3.2 Interpolation Paths

Given a paired sample (x₀, x₁), we define an interpolation path:

#### Linear Interpolant (Deterministic)
```
x_t = (1 - t)·x₀ + t·x₁
```

The velocity along this path is **constant**:
```
v*(t) = dx_t/dt = x₁ - x₀
```

#### Stochastic Interpolant (SF²M)
```
x_t = α(t)·x₀ + β(t)·x₁ + σ(t)·ε,  where ε ~ N(0, I)
```

With boundary conditions:
- α(0) = 1, α(1) = 0  (starts at x₀)
- β(0) = 0, β(1) = 1  (ends at x₁)
- σ(0) = σ(1) = 0    (no noise at boundaries)

**Our implementation uses**: `σ(t) = σ_max · sin(πt)`
- Zero at t=0 and t=1 (boundaries)
- Maximum at t=0.5 (midpoint)

The target velocity becomes:
```
v*(t) = α'(t)·x₀ + β'(t)·x₁ + σ'(t)·ε
```

### 3.3 Training Objective

We train a neural network v_θ(x_t, t) to predict the velocity:

```
L = E_{t~U[0,1], (x₀,x₁)~data} || v_θ(x_t, t) - v*(t) ||²
```

**Why this works**:
- At each time t, we sample a random point on the interpolation path
- The network learns to predict which direction leads to the target
- No adversarial training, no complex schedules - just MSE regression!

### 3.4 Inference: ODE vs SDE

#### ODE (Deterministic)
```
dx/dt = v_θ(x, t)
```
Integrate from t=0 to t=1 using numerical methods (we use Heun's method).

**Properties**:
- Deterministic: Same input → same output
- Faster convergence
- Good for applications requiring reproducibility

#### SDE (Stochastic)
```
dx = v_θ(x, t)dt + σ'(t)dW
```
where dW is Brownian motion.

**Properties**:
- Stochastic: Same input → different outputs
- Can correct errors in learned velocity field
- Good for generating diverse plausible SR images

### 3.5 Optimal Transport Coupling (Unpaired Data)

When we don't have paired data, we use **mini-batch Optimal Transport**:

```
Given: Batch of x₀ samples, Batch of x₁ samples (unpaired)
Goal: Find optimal pairing that minimizes transport cost
```

We solve:
```
min_{P} Σᵢⱼ Pᵢⱼ · ||x₀ⁱ - x₁ʲ||²
subject to: P1 = 1/n, P^T1 = 1/n, Pᵢⱼ ≥ 0
```

**Methods**:
1. **Exact OT** (Hungarian algorithm): Optimal but O(n³)
2. **Sinkhorn OT**: Approximate but faster, differentiable

---

## 4. Implementation Architecture

### 4.1 Overall Structure

```
flow_matching.py
├── Configuration (Config dataclass)
├── Data Generation (Van der Pol → Images)
├── Optimal Transport Coupling
├── Interpolant Schedule (α, β, σ coefficients)
├── Neural Network (VelocityUNet)
├── Training Loop
├── Inference (ODE/SDE solvers)
├── Evaluation (SSIM, PSNR)
└── Experiments (Ablations)
```

### 4.2 Neural Network: VelocityUNet

```
Input: x_t (noisy/interpolated image) + t (time)
       ↓
[Time Embedding] ← t (sinusoidal positional encoding)
       ↓
[Encoder]  ──┬── Conv → Pool → Conv → Pool → Conv → Pool
             │   (32)         (64)         (128)
             │
[Bottleneck] │   Conv (256 channels)
             │
[Decoder]  ──┴── Up → Conv → Up → Conv → Up → Conv
                 (+skip)    (+skip)    (+skip)
       ↓
Output: v (predicted velocity, same size as input)
```

**Key features**:
- **U-Net architecture**: Skip connections preserve spatial details
- **FiLM conditioning**: Time embedding modulates features via `x * (1 + γ(t))`
- **GroupNorm + SiLU**: Stable training, smooth gradients

### 4.3 Time Embedding

```python
# Sinusoidal embedding (from Transformers)
freqs = exp(-log(10000) · [0, 1, ..., d/2] / (d/2))
embedding = [sin(t · freqs), cos(t · freqs)]

# MLP to learn useful representation
embedding = MLP(embedding)  # d → 4d → d
```

**Why sinusoidal?**
- Different frequencies capture different time scales
- Smooth interpolation between time points
- Proven effective in Transformers and diffusion models

---

## 5. Code Walkthrough: Understanding Each Component

### 5.1 Configuration

```python
@dataclass
class Config:
    # Data
    n_images: int = 600        # Dataset size
    hr_resolution: int = 128   # Output resolution
    downsample_factor: int = 4 # 128 → 32 pixels
    
    # Training
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    
    # Flow Matching Type
    fm_type: Literal['deterministic', 'stochastic'] = 'stochastic'
    sigma_max: float = 0.1     # Noise level for stochastic
    
    # Data Coupling
    coupling_mode: Literal['paired', 'unpaired'] = 'paired'
    
    # Inference
    inference_steps: int = 50
    inference_mode: Literal['ode', 'sde'] = 'ode'
```

### 5.2 Interpolant Schedule

```python
class InterpolantSchedule:
    def interpolate(self, x0, x1, t, noise=None):
        """
        x_t = α(t)·x₀ + β(t)·x₁ + σ(t)·ε
        """
        alpha_t = self.alpha(t)   # Coefficient for source
        beta_t = self.beta(t)     # Coefficient for target
        sigma_t = self.sigma(t)   # Noise coefficient
        
        return alpha_t * x0 + beta_t * x1 + sigma_t * noise
    
    def velocity_target(self, x0, x1, t, noise=None):
        """
        v* = α'(t)·x₀ + β'(t)·x₁ + σ'(t)·ε
        """
        # Derivatives of coefficients
        return alpha_prime * x0 + beta_prime * x1 + sigma_prime * noise
```

### 5.3 Training Loop

```python
for epoch in range(epochs):
    for x0_batch, x1_batch in train_loader:
        # 1. Optional: OT coupling for unpaired data
        if coupling_mode == 'unpaired':
            x0_batch, x1_batch = sample_ot_coupling(x0_batch, x1_batch)
        
        # 2. Sample random time
        t = torch.rand(batch_size, 1, 1, 1)  # U[0, 1]
        
        # 3. Sample noise (for stochastic interpolant)
        noise = torch.randn_like(x0_batch)
        
        # 4. Construct interpolated sample
        x_t = schedule.interpolate(x0_batch, x1_batch, t, noise)
        
        # 5. Compute target velocity
        v_target = schedule.velocity_target(x0_batch, x1_batch, t, noise)
        
        # 6. Predict velocity
        v_pred = model(x_t, t)
        
        # 7. MSE loss
        loss = MSE(v_pred, v_target)
        
        # 8. Backprop
        loss.backward()
        optimizer.step()
```

### 5.4 ODE Inference (Heun's Method)

```python
def heun_integrate(model, x0, steps):
    x = x0.clone()
    dt = 1.0 / steps
    
    for k in range(steps):
        t_cur = k * dt
        t_next = (k + 1) * dt
        
        # Heun's method (2nd order Runge-Kutta)
        v1 = model(x, t_cur)           # Slope at current point
        x_pred = x + dt * v1           # Euler prediction
        v2 = model(x_pred, t_next)     # Slope at predicted point
        x = x + 0.5 * dt * (v1 + v2)   # Average slopes
    
    return x
```

**Why Heun over Euler?**
- Euler: First-order, accumulates error quickly
- Heun: Second-order, much better accuracy with same step count
- Cost: 2× model evaluations, but worth it for stability

### 5.5 SDE Inference (Euler-Maruyama)

```python
def sde_integrate(model, x0, steps, schedule, noise_scale):
    x = x0.clone()
    dt = 1.0 / steps
    sqrt_dt = sqrt(dt)
    
    for k in range(steps):
        t_cur = k * dt
        
        v = model(x, t_cur)                    # Drift
        sigma_prime = schedule.sigma_derivative(t_cur)  # Diffusion
        z = torch.randn_like(x)                # Random noise
        
        x = x + v * dt + noise_scale * sigma_prime * sqrt_dt * z
    
    return x
```

---

## 6. Configuration Parameters: What They Do

### 6.1 Data Parameters

| Parameter | Default | Effect of Increasing | Effect of Decreasing |
|-----------|---------|---------------------|---------------------|
| `n_images` | 600 | Better generalization, longer training | Faster training, may overfit |
| `hr_resolution` | 128 | More details, more compute | Faster, less detail |
| `downsample_factor` | 4 | Harder problem, more ambiguity | Easier problem |
| `blur_sigma` | 1.0 | More information loss | Easier reconstruction |

### 6.2 Training Parameters

| Parameter | Default | Effect of Increasing | Effect of Decreasing |
|-----------|---------|---------------------|---------------------|
| `epochs` | 30 | Better convergence, may overfit | Underfit |
| `batch_size` | 16 | Stable gradients, more memory | Noisy gradients, less memory |
| `learning_rate` | 1e-4 | Faster learning, may diverge | Slower, more stable |
| `base_channels` | 32 | More capacity, more compute | Less capacity, faster |

### 6.3 Flow Matching Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `fm_type` | 'stochastic' | 'deterministic': No noise in training path<br>'stochastic': Adds σ(t)·ε term |
| `sigma_max` | 0.1 | Maximum noise at t=0.5. Higher = more regularization |
| `coupling_mode` | 'paired' | 'paired': Use direct pairs<br>'unpaired': Use OT matching |
| `ot_reg` | 0.01 | Sinkhorn regularization. 0 = exact OT |

### 6.4 Inference Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `inference_steps` | 50 | More steps = better accuracy, slower |
| `inference_mode` | 'ode' | 'ode': Deterministic<br>'sde': Stochastic |
| `sde_noise_scale` | 1.0 | Scale of stochastic noise. Higher = more diverse |

### 6.5 Recommended Settings for Different Scenarios

**Fast prototyping**:
```python
Config(n_images=200, epochs=10, inference_steps=20)
```

**Best quality**:
```python
Config(n_images=1000, epochs=100, inference_steps=100, base_channels=64)
```

**Diverse outputs**:
```python
Config(fm_type='stochastic', inference_mode='sde', sde_noise_scale=1.5)
```

**Unpaired data**:
```python
Config(coupling_mode='unpaired', ot_reg=0.01)
```

---

## 7. Experiments and Ablations

### 7.1 Dataset Size Experiment

```python
experiment_dataset_size(config, device, 
    sizes=[100, 300, 600, 1000],
    epochs_list=[10, 20, 30, 50]
)
```

**Expected results**:
| Size | Epochs | SSIM | Notes |
|------|--------|------|-------|
| 100 | 10 | ~0.06 | Severely underfit |
| 100 | 50 | ~0.08 | Overfit to small data |
| 600 | 30 | ~0.11 | Good balance |
| 1000 | 50 | ~0.13 | Best generalization |

**Takeaway**: More data helps more than more epochs (up to a point).

### 7.2 Paired vs Unpaired Experiment

```python
experiment_paired_vs_unpaired(config, device)
```

**Expected results**:
| Mode | SSIM | Notes |
|------|------|-------|
| Paired | ~0.11 | Direct supervision |
| Unpaired (OT) | ~0.09 | OT finds good matches, but not perfect |

**Takeaway**: Paired data is better when available, but OT coupling works surprisingly well for unpaired data.

### 7.3 Deterministic vs Stochastic Training

| fm_type | Training | SSIM | Notes |
|---------|----------|------|-------|
| deterministic | Simpler targets | ~0.10 | Cleaner optimization |
| stochastic | Noisy targets | ~0.11 | Better generalization |

**Takeaway**: Stochastic training often generalizes better due to implicit regularization.

### 7.4 ODE vs SDE Inference

| inference_mode | SSIM | Diversity | Notes |
|----------------|------|-----------|-------|
| ODE | Higher | None | Same output every time |
| SDE | Lower | High | Different plausible outputs |

**Takeaway**: Use ODE for best single output, SDE for diverse samples.

---

## 8. How to Improve Results

### 8.1 Quick Wins

1. **Increase training time**:
   ```python
   Config(epochs=100)  # Instead of 30
   ```

2. **More data**:
   ```python
   Config(n_images=1000)  # Instead of 600
   ```

3. **More inference steps**:
   ```python
   Config(inference_steps=100)  # Instead of 50
   ```

### 8.2 Architecture Improvements

1. **Larger model**:
   ```python
   Config(base_channels=64)  # Instead of 32
   ```

2. **Add attention layers** (modify VelocityUNet):
   ```python
   # Add self-attention in bottleneck
   self.attention = nn.MultiheadAttention(c * 8, num_heads=8)
   ```

3. **Use pretrained encoder** (for natural images):
   ```python
   # Replace encoder with pretrained ResNet/EfficientNet features
   ```

### 8.3 Training Improvements

1. **Learning rate scheduling**:
   ```python
   # Already implemented: CosineAnnealingLR
   # Could try: OneCycleLR, WarmupCosine
   ```

2. **Gradient clipping** (already implemented):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```

3. **EMA (Exponential Moving Average)**:
   ```python
   # Use EMA of model weights for inference
   ema_model = EMA(model, decay=0.999)
   ```

4. **Data augmentation**:
   ```python
   # Random flips, rotations (if applicable to your data)
   ```

### 8.4 Advanced Techniques

1. **Classifier-free guidance** (for conditional generation):
   ```python
   # Train with random condition dropout
   # Inference: v_guided = v_uncond + scale * (v_cond - v_uncond)
   ```

2. **Progressive training**:
   ```python
   # Start with low resolution, gradually increase
   ```

3. **Rectified Flow** (straighten paths):
   ```python
   # After training, use generated pairs to retrain with straighter paths
   ```

4. **Consistency models**:
   ```python
   # Distill to single-step model
   ```

### 8.5 For This Specific Dataset

1. **Increase trajectory density**:
   ```python
   Config(points_per_image=280)  # More points = denser images
   ```

2. **Tune blur parameters**:
   ```python
   Config(blur_sigma=0.5)  # Less blur = easier problem
   ```

3. **Multi-scale training**:
   ```python
   # Train on multiple downsample factors simultaneously
   ```

---

## 9. Frequently Asked Questions

### Q: Why is my SSIM low (~0.1)?

**A**: Several reasons:
1. **Sparse data**: Van der Pol trajectories are thin lines on mostly black background
2. **Limited training**: 30 epochs may not be enough
3. **High ambiguity**: 4× downsampling loses significant information

**Solutions**: More epochs, more data, smaller downsample factor.

### Q: ODE vs SDE - which should I use?

**A**: 
- **ODE**: When you need reproducible, single best output
- **SDE**: When you want diverse plausible outputs, or for uncertainty quantification

### Q: Why does SDE give lower SSIM?

**A**: SDE adds stochasticity, so outputs vary. While each individual output may have lower SSIM, the **distribution** of outputs better captures the true posterior. For ill-posed problems, this is actually desirable!

### Q: Can I use this for real images?

**A**: Yes! Modifications needed:
1. Replace Van der Pol data with your image dataset
2. Use 3 channels (RGB) instead of 1
3. Increase model capacity (base_channels=64 or 128)
4. Consider pretrained encoders

### Q: How does this compare to Stable Diffusion for SR?

**A**: 
- **Stable Diffusion**: Works in latent space, massive model, general purpose
- **Flow Matching**: Works in pixel space, smaller model, task-specific
- For domain-specific SR, Flow Matching can be more efficient and controllable

### Q: What if I don't have paired data?

**A**: Use `coupling_mode='unpaired'`. The mini-batch OT will find good matchings. Quality will be slightly lower than paired, but often acceptable.

### Q: Memory issues?

**A**: Try:
```python
Config(batch_size=8, base_channels=16)
```
Or use gradient checkpointing.

---

## 10. References

### Papers

1. **Flow Matching for Generative Modeling** (Lipman et al., 2023)
   - Original Flow Matching formulation
   - [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

2. **Building Normalizing Flows with Stochastic Interpolants** (Albergo & Vanden-Eijnden, 2023)
   - Stochastic interpolant theory
   - [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)

3. **Improving and Generalizing Flow-Based Generative Models** (Tong et al., 2023)
   - Mini-batch OT coupling
   - [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)

4. **Dual Diffusion Implicit Bridges** (Su et al., 2022)
   - DDIB for image-to-image translation
   - [arXiv:2203.08382](https://arxiv.org/abs/2203.08382)

5. **[SF]²M: A Stochastic Flow Matching Framework** 
   - Super-resolution specific formulation
   - Combines OT coupling with stochastic interpolants

### Related Implementations

- [TorchCFM](https://github.com/atong01/conditional-flow-matching): General Flow Matching library
- [flow-matching](https://github.com/facebookresearch/flow_matching): Meta's implementation

### This Project

- **Repository**: [Mc-Brainzz/generative-models-comparison](https://github.com/Mc-Brainzz/generative-models-comparison)
- **Related files**:
  - `ddib.py`: DDIB implementation for comparison
  - `sdedit.py`: SDEdit implementation for comparison
  - `vae.py`, `gan.py`: Baseline models on 2D data

---

## Summary: The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLOW MATCHING FOR SR                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Low-Resolution Image (32×32)                           │
│           ↓                                                     │
│  [Bilinear Upsample to 128×128] ← Starting point x₀            │
│           ↓                                                     │
│  ┌─────────────────────────────────────────────┐               │
│  │  ODE Integration: dx/dt = v_θ(x, t)         │               │
│  │                                             │               │
│  │  t=0.0 ──→ t=0.2 ──→ t=0.5 ──→ t=0.8 ──→ t=1.0 │           │
│  │   x₀        ↓         ↓         ↓        x₁   │           │
│  │ (blurry)  (less    (halfway)  (mostly  (sharp)│           │
│  │           blurry)             sharp)          │           │
│  └─────────────────────────────────────────────┘               │
│           ↓                                                     │
│  OUTPUT: High-Resolution Image (128×128)                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  WHY IT WORKS:                                                  │
│  • Neural network learns the "direction" to sharp images       │
│  • ODE integration smoothly follows this learned direction     │
│  • Stochastic training provides regularization                 │
│  • OT coupling handles unpaired data                           │
├─────────────────────────────────────────────────────────────────┤
│  ADVANTAGES OVER ALTERNATIVES:                                  │
│  • Simpler than diffusion (no noise schedules)                 │
│  • Faster than diffusion (fewer steps needed)                  │
│  • More stable than GANs (no adversarial training)             │
│  • Sharper than VAEs (direct transport, not reconstruction)    │
└─────────────────────────────────────────────────────────────────┘
```

---

*This implementation is part of a comparative study of generative models for super-resolution. For questions or contributions, please open an issue on GitHub.*
