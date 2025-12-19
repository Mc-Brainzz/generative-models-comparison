# Generative Models Comparison

A comprehensive comparison of modern generative models on two controlled tasks: **2D distribution learning** (Two Moons) and **image super-resolution** (Van der Pol density maps).

This repository provides clean, modular, research-quality implementations designed for ML researchers, PhD students, and engineers who want to understand the core differences between generative modeling paradigms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Overview

### Why This Repository?

Understanding generative models requires seeing them side-by-side on identical tasks with identical evaluation. This repository provides:

- **Fair comparison**: Same data, same architecture backbone, same evaluation metrics
- **Clean code**: Modular, well-documented, research-reproducible
- **Two complementary tasks**: Simple 2D distributions and image super-resolution
- **Educational**: Extensive comments explaining *why*, not just *what*

### Models Implemented

| Model | Task | Key Idea |
|-------|------|----------|
| **VAE** | 2D Distribution | Latent space learning with ELBO objective |
| **GAN** | 2D Distribution | Adversarial training (Generator vs Discriminator) |
| **SDEdit** | Super-Resolution | Iterative denoising with data consistency |
| **DDIB** | Super-Resolution | Diffusion bridge between distributions |
| **Flow Matching** | Super-Resolution | Velocity field learning along linear paths |

---

## 📁 Repository Structure

```
Model_Comparision/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── Two_moon/                    # 2D Distribution Learning
│   ├── vae.py                   # Variational Autoencoder
│   └── gan.py                   # Generative Adversarial Network
│
└── Super_resolution/            # Image Super-Resolution
    ├── sdedit.py                # SDEdit (Stochastic Differential Edit)
    ├── ddib.py                  # Denoising Diffusion Implicit Bridge
    └── flow_matching.py         # Flow Matching ([SF]²M-style)
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/generative-models-comparison.git
cd generative-models-comparison
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

**For GPU (recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 4. Run Any Model

```bash
# 2D Distribution Learning
python Two_moon/vae.py
python Two_moon/gan.py

# Super-Resolution
python Super_resolution/sdedit.py
python Super_resolution/ddib.py
python Super_resolution/flow_matching.py
```

---

## 📊 Tasks & Datasets

### Task 1: Two Moons Distribution (2D)

A classic ML benchmark for testing generative models on non-convex distributions.

<table>
<tr>
<td width="50%">

**Dataset**: Scikit-learn's `make_moons`
- 2D points forming two interleaving half-circles
- Tests ability to learn multi-modal distributions
- Fast iteration for experimentation

</td>
<td width="50%">

**Evaluation**:
- Visual inspection of generated samples
- Distribution coverage
- Mode collapse detection (GAN)

</td>
</tr>
</table>

### Task 2: Van der Pol Super-Resolution (Images)

Controlled super-resolution task using synthetic density images from Van der Pol oscillator trajectories.

<table>
<tr>
<td width="50%">

**Why Van der Pol?**
- Deterministic, reproducible data generation
- No licensing/copyright concerns
- Rich geometric structure (limit cycles)
- Perfect ground truth available

</td>
<td width="50%">

**Setup**:
- HR: 128×128 density images
- LR: 32×32 (blur + 4× downsample)
- Metric: SSIM (Structural Similarity)

</td>
</tr>
</table>

---

## 🔬 Model Details

### VAE (Variational Autoencoder)

```
Two_moon/vae.py
```

**Key Concepts**:
- Encoder maps data → latent distribution (μ, σ)
- Reparameterization trick: z = μ + σ ⊙ ε
- ELBO loss = Reconstruction + β·KL-divergence
- Beta-annealing for stable training

**Architecture**: MLP with 2→128→64→2 (encoder) and reverse (decoder)

---

### GAN (Generative Adversarial Network)

```
Two_moon/gan.py
```

**Key Concepts**:
- Generator: noise → fake samples
- Discriminator: samples → real/fake probability
- Adversarial game: G tries to fool D, D tries to catch G
- Label smoothing for training stability

**Includes**: 4 ablation experiments (Stable, High LR, No BatchNorm, ReLU activation)

---

### SDEdit (Stochastic Differential Edit)

```
Super_resolution/sdedit.py
```

**Key Concepts**:
- Train denoiser on HR images (noise prediction)
- Inference: Add noise to upscaled LR, iteratively denoise
- **Data consistency**: After each step, ensure output matches LR when degraded

**Unique Feature**: Feedback loop maintains fidelity to input

---

### DDIB (Denoising Diffusion Implicit Bridge)

```
Super_resolution/ddib.py
```

**Key Concepts**:
- Two denoisers: q₀ (LR distribution) and q₁ (HR distribution)
- Bridge: Encode LR → latent → Decode to HR
- Optional energy guidance (EDDIB) using physics priors

**Process**: encode_q0 → bridge_forward → decode_q1

---

### Flow Matching ([SF]²M)

```
Super_resolution/flow_matching.py
```

**Key Concepts**:
- Learn velocity field v_θ(x,t) along straight path
- Linear interpolation: x_t = (1-t)·x₀ + t·x₁
- Target velocity: v* = x₁ - x₀ (constant, easy to learn)
- Inference: ODE integration dx/dt = v_θ(x,t)

**Advantage**: No noise schedule to tune, deterministic sampling

---

## 📈 Method Comparison

| Aspect | VAE | GAN | SDEdit | DDIB | Flow Matching |
|--------|-----|-----|--------|------|---------------|
| **Training** | Stable | Tricky | Stable | Stable | Stable |
| **Objective** | ELBO | Adversarial | Noise pred. | Noise pred. | Velocity pred. |
| **Sampling** | Single pass | Single pass | Iterative | Iterative | ODE solve |
| **Mode collapse** | Rare | Common | Rare | Rare | Rare |
| **Inference cost** | Low | Low | High | High | Medium |

---

## ⚙️ Configuration

Each model uses a `Config` dataclass for centralized hyperparameter management:

```python
@dataclass
class Config:
    seed: int = 42
    n_samples: int = 2000
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 100
    # ... model-specific params
```

Modify these to experiment with different settings.

---

## 🖥️ Hardware Requirements

| Setup | Minimum | Recommended |
|-------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **GPU** | None (CPU works) | NVIDIA GTX 1060+ |
| **VRAM** | - | 4 GB+ |
| **Storage** | 1 GB | 2 GB |

**Tested on**: NVIDIA RTX 4050, RTX 3080, CPU (slower but works)

---

## 📚 References

### Papers

- **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
- **GAN**: Goodfellow et al., "Generative Adversarial Networks", NeurIPS 2014
- **SDEdit**: Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", ICLR 2022
- **DDIB**: Su et al., "Dual Diffusion Implicit Bridges for Image-to-Image Translation", ICLR 2023
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023

### Useful Resources

- [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Understanding VAEs](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- [GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Scikit-learn for the Two Moons dataset
- The authors of the referenced papers for their groundbreaking work

---

<p align="center">
  <i>If you find this useful, please ⭐ star the repository!</i>
</p>
