"""
[SF]²M: Stochastic Flow Matching for Super-Resolution

A comprehensive implementation of Stochastic Flow Matching (SF²M) for image
super-resolution, covering all key aspects of the methodology:

1. **Deterministic vs Stochastic Flow Matching**
   - ODE-based (deterministic) inference
   - SDE-based (stochastic) inference with diffusion term

2. **Paired vs Unpaired Data Training**
   - Paired: Direct (x0, x1) coupling from degradation
   - Unpaired: Mini-batch Optimal Transport (OT) coupling

3. **Interpolant Types**
   - Linear interpolant: x_t = (1-t)x0 + tx1
   - Stochastic interpolant: x_t = (1-t)x0 + tx1 + σ(t)ε

4. **Gaussian Probability Paths**
   - Time-dependent mean: μ_t(x0, x1)
   - Time-dependent variance: σ²_t

Key Mathematical Formulation (SF²M):
    Given source x0 ~ p0 and target x1 ~ p1, we define:
    
    Stochastic Interpolant:
        x_t = α_t x0 + β_t x1 + σ_t ε,  ε ~ N(0, I)
        
    where α_t, β_t, σ_t are time-dependent coefficients satisfying:
        α_0 = 1, α_1 = 0 (starts at x0)
        β_0 = 0, β_1 = 1 (ends at x1)
        σ_0 = σ_1 = 0 (no noise at boundaries)
    
    The velocity field v(x, t) satisfies:
        v(x_t, t) = E[α'_t x0 + β'_t x1 | x_t]
    
    For inference, we can use either:
        ODE: dx = v(x, t) dt                     (deterministic)
        SDE: dx = [v(x, t) + σ'_t/σ_t · s(x,t)] dt + σ'_t dW  (stochastic)

References:
    - Lipman et al. (2023): Flow Matching for Generative Modeling
    - Albergo & Vanden-Eijnden (2023): Building Normalizing Flows with Stochastic Interpolants
    - Liu et al. (2023): Flow Matching for Scalable Simulation-Based Inference
    - Tong et al. (2023): Improving and Generalizing Flow-Based Generative Models

Author: [Your Name]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal
import time
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """
    Comprehensive configuration for SF²M Flow Matching.
    
    Attributes are grouped by category for clarity.
    """
    # Reproducibility
    seed: int = 123
    
    # Data generation
    n_images: int = 600
    points_per_image: int = 140
    hr_resolution: int = 128
    downsample_factor: int = 4
    
    # Degradation
    blur_sigma: float = 1.0
    blur_radius: int = 3
    
    # Dataset split
    train_ratio: float = 0.85
    
    # Training
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    base_channels: int = 32
    
    # Flow Matching Type
    # 'deterministic': Linear interpolation, ODE inference
    # 'stochastic': Stochastic interpolant with noise, SDE inference option
    fm_type: Literal['deterministic', 'stochastic'] = 'stochastic'
    
    # Data Coupling Mode
    # 'paired': Direct (x0, x1) pairs from degradation
    # 'unpaired': Mini-batch OT coupling
    coupling_mode: Literal['paired', 'unpaired'] = 'paired'
    
    # Stochastic Interpolant Parameters
    # Controls the amount of noise injected during training
    # σ(t) = σ_max * sin(π * t) - zero at boundaries, max at t=0.5
    sigma_max: float = 0.1  # Maximum noise level at t=0.5
    
    # Inference
    inference_steps: int = 50
    inference_mode: Literal['ode', 'sde'] = 'ode'  # ODE or SDE sampler
    sde_noise_scale: float = 1.0  # Scale for SDE noise term
    
    # Mini-batch OT
    ot_reg: float = 0.01  # Entropic regularization (0 = exact OT)
    
    @property
    def lr_resolution(self) -> int:
        return self.hr_resolution // self.downsample_factor


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# =============================================================================
# Data Generation: Van der Pol Oscillator
# =============================================================================

def van_der_pol_ode(t: float, y: np.ndarray, mu: float = 1.0) -> list:
    """
    Van der Pol oscillator ODE.
    
    This nonlinear system produces limit-cycle trajectories that serve
    as realistic test data - structured, non-Gaussian distributions.
    
    Equations:
        dx/dt = v
        dv/dt = μ(1 - x²)v - x
    """
    x, v = y
    return [v, mu * (1 - x**2) * v - x]


def points_to_image(
    points: np.ndarray,
    resolution: int = 128,
    bounds: tuple = (-4, 4)
) -> np.ndarray:
    """Convert 2D trajectory points to grayscale image via 2D histogram."""
    heat, _, _ = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=resolution,
        range=[[bounds[0], bounds[1]], [bounds[0], bounds[1]]]
    )
    heat = heat.T.astype(np.float32)
    
    if heat.max() > 0:
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    
    return heat


def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Create 1D Gaussian kernel for separable convolution."""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_torch(x: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
    """Apply separable Gaussian blur to 4D tensor [B, C, H, W]."""
    assert x.dim() == 4
    
    kernel_1d = gaussian_kernel_1d(sigma, radius)
    kernel = torch.as_tensor(kernel_1d, device=x.device, dtype=x.dtype)
    
    k_h = kernel.view(1, 1, 1, -1)
    k_v = kernel.view(1, 1, -1, 1)
    
    x = F.conv2d(x, k_h, padding=(0, radius), groups=1)
    x = F.conv2d(x, k_v, padding=(radius, 0), groups=1)
    
    return x


def degrade_operator(
    hr_img: torch.Tensor,
    blur_sigma: float,
    blur_radius: int,
    down_factor: int
) -> torch.Tensor:
    """
    Degradation operator D(x): blur + downsample.
    
    Models typical SR degradation: blur loses high-frequency info,
    downsampling reduces spatial resolution.
    """
    x = gaussian_blur_torch(hr_img, sigma=blur_sigma, radius=blur_radius)
    x = F.interpolate(x, scale_factor=1.0/down_factor, mode='bilinear', align_corners=False)
    return x


def create_dataset(
    config: Config,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate Van der Pol trajectory dataset.
    
    Returns:
        HR images (target), LR images (degraded), upsampled LR images (source)
    """
    HR_list, LR_list = [], []
    
    for _ in tqdm(range(config.n_images), desc="Generating Van der Pol data"):
        t_eval = 5.0 + np.random.rand() * 5.0
        init = (np.random.rand(2) - 0.5) * 6.0
        
        sol = solve_ivp(
            van_der_pol_ode, [0, t_eval], init,
            t_eval=np.linspace(0, t_eval, config.points_per_image)
        )
        points = sol.y.T
        
        if points.shape[0] < 2:
            continue
        
        hr_np = points_to_image(points, resolution=config.hr_resolution)
        hr_t = torch.from_numpy(hr_np[None, None, :, :]).float()
        
        with torch.no_grad():
            lr_t = degrade_operator(
                hr_t,
                blur_sigma=config.blur_sigma,
                blur_radius=config.blur_radius,
                down_factor=config.downsample_factor
            )
        
        HR_list.append(hr_np)
        LR_list.append(lr_t.squeeze().numpy())
    
    HR = torch.from_numpy(np.stack(HR_list)).unsqueeze(1).float().to(device)
    LR = torch.from_numpy(np.stack(LR_list)).unsqueeze(1).float().to(device)
    
    # Create upsampled LR as source distribution (x0)
    LR_up = F.interpolate(
        LR,
        scale_factor=config.downsample_factor,
        mode='bilinear',
        align_corners=False
    )
    
    return HR, LR, LR_up


# =============================================================================
# Mini-batch Optimal Transport Coupling
# =============================================================================

def compute_cost_matrix(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L2 cost matrix between two batches.
    
    Args:
        x0: Source batch [B, C, H, W]
        x1: Target batch [B, C, H, W]
    
    Returns:
        Cost matrix [B, B] where C[i,j] = ||x0[i] - x1[j]||²
    """
    x0_flat = x0.view(x0.size(0), -1)  # [B, D]
    x1_flat = x1.view(x1.size(0), -1)  # [B, D]
    
    # ||x0[i] - x1[j]||² = ||x0[i]||² + ||x1[j]||² - 2<x0[i], x1[j]>
    x0_sq = (x0_flat ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    x1_sq = (x1_flat ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    cross = x0_flat @ x1_flat.T  # [B, B]
    
    cost = x0_sq + x1_sq.T - 2 * cross
    return cost


def solve_ot_exact(cost_matrix: torch.Tensor) -> torch.Tensor:
    """
    Solve exact Optimal Transport using Hungarian algorithm.
    
    Returns permutation indices that minimize total transport cost.
    """
    cost_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return torch.tensor(col_ind, device=cost_matrix.device, dtype=torch.long)


def solve_ot_sinkhorn(
    cost_matrix: torch.Tensor,
    reg: float = 0.01,
    n_iters: int = 50
) -> torch.Tensor:
    """
    Solve entropy-regularized OT using Sinkhorn algorithm.
    
    For reg → 0, approaches exact OT.
    Larger reg gives smoother but less optimal coupling.
    
    Returns:
        Soft coupling matrix P where P[i,j] ≈ probability of matching i to j
    """
    B = cost_matrix.size(0)
    
    # Initialize with uniform marginals
    K = torch.exp(-cost_matrix / reg)
    u = torch.ones(B, device=cost_matrix.device)
    v = torch.ones(B, device=cost_matrix.device)
    
    for _ in range(n_iters):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.T @ u + 1e-8)
    
    P = torch.diag(u) @ K @ torch.diag(v)
    return P


def sample_ot_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create OT-coupled pairs from two batches.
    
    Mini-batch OT: Find optimal pairing within each mini-batch.
    This provides a principled way to couple unpaired data.
    
    Args:
        x0: Source batch [B, C, H, W]
        x1: Target batch [B, C, H, W]
        reg: Entropic regularization (0 = exact OT)
    
    Returns:
        Reordered (x0, x1) according to OT coupling
    """
    cost = compute_cost_matrix(x0, x1)
    
    if reg <= 0:
        # Exact OT
        perm = solve_ot_exact(cost)
        return x0, x1[perm]
    else:
        # Sinkhorn OT - sample from coupling
        P = solve_ot_sinkhorn(cost, reg=reg)
        # Sample indices according to coupling
        perm = torch.multinomial(P, num_samples=1).squeeze(-1)
        return x0, x1[perm]


# =============================================================================
# Stochastic Interpolant Coefficients
# =============================================================================

class InterpolantSchedule:
    """
    Defines the interpolant path coefficients.
    
    For stochastic interpolant:
        x_t = α(t) x0 + β(t) x1 + σ(t) ε
    
    Boundary conditions:
        α(0) = 1, α(1) = 0
        β(0) = 0, β(1) = 1
        σ(0) = σ(1) = 0
    
    This class implements several choices:
    
    1. Linear (deterministic):
        α(t) = 1 - t, β(t) = t, σ(t) = 0
        
    2. Stochastic (sinusoidal noise):
        α(t) = 1 - t, β(t) = t, σ(t) = σ_max * sin(πt)
        
    3. VP-style (variance preserving):
        α(t) = √(1-t), β(t) = √t, σ(t) = √(2t(1-t)) * σ_max
    """
    
    def __init__(
        self,
        schedule_type: Literal['linear', 'stochastic', 'vp'] = 'stochastic',
        sigma_max: float = 0.1
    ):
        self.schedule_type = schedule_type
        self.sigma_max = sigma_max
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient for source x0."""
        if self.schedule_type == 'vp':
            return torch.sqrt(1 - t)
        return 1 - t
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient for target x1."""
        if self.schedule_type == 'vp':
            return torch.sqrt(t)
        return t
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient."""
        if self.schedule_type == 'linear':
            return torch.zeros_like(t)
        elif self.schedule_type == 'stochastic':
            # Sinusoidal: max at t=0.5, zero at boundaries
            return self.sigma_max * torch.sin(np.pi * t)
        elif self.schedule_type == 'vp':
            # Variance preserving
            return self.sigma_max * torch.sqrt(2 * t * (1 - t))
        else:
            return torch.zeros_like(t)
    
    def alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """d α(t) / dt."""
        if self.schedule_type == 'vp':
            return -0.5 / torch.sqrt(1 - t + 1e-8)
        return -torch.ones_like(t)
    
    def beta_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """d β(t) / dt."""
        if self.schedule_type == 'vp':
            return 0.5 / torch.sqrt(t + 1e-8)
        return torch.ones_like(t)
    
    def sigma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """d σ(t) / dt."""
        if self.schedule_type == 'linear':
            return torch.zeros_like(t)
        elif self.schedule_type == 'stochastic':
            return self.sigma_max * np.pi * torch.cos(np.pi * t)
        elif self.schedule_type == 'vp':
            # d/dt sqrt(2t(1-t)) = (1-2t) / sqrt(2t(1-t))
            denom = torch.sqrt(2 * t * (1 - t) + 1e-8)
            return self.sigma_max * (1 - 2*t) / denom
        else:
            return torch.zeros_like(t)
    
    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute x_t = α(t)x0 + β(t)x1 + σ(t)ε.
        
        Args:
            x0: Source samples [B, C, H, W]
            x1: Target samples [B, C, H, W]
            t: Time values [B, 1, 1, 1]
            noise: Optional pre-sampled noise [B, C, H, W]
        
        Returns:
            Interpolated samples x_t
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        sigma_t = self.sigma(t)
        
        x_t = alpha_t * x0 + beta_t * x1 + sigma_t * noise
        return x_t
    
    def velocity_target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute target velocity for Flow Matching loss.
        
        For stochastic interpolant:
            v* = α'(t)x0 + β'(t)x1 + σ'(t)ε
            
        This is the conditional expectation E[dx_t/dt | x_t, x0, x1].
        """
        alpha_prime = self.alpha_derivative(t)
        beta_prime = self.beta_derivative(t)
        sigma_prime = self.sigma_derivative(t)
        
        if noise is None:
            # For deterministic interpolant (σ=0), noise term vanishes
            noise = torch.zeros_like(x0)
        
        v_target = alpha_prime * x0 + beta_prime * x1 + sigma_prime * noise
        return v_target


# =============================================================================
# Model Architecture
# =============================================================================

class ConvBlock(nn.Module):
    """Double convolution block with GroupNorm and SiLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding (from Transformer positional encoding).
    
    Maps scalar time t to high-dimensional embedding for conditioning.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] or [B, 1, 1, 1]
        
        Returns:
            Time embedding [B, dim]
        """
        t = t.view(-1)  # Flatten to [B]
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return self.mlp(embedding)


class VelocityUNet(nn.Module):
    """
    U-Net architecture for velocity prediction.
    
    Takes noisy input x_t and time t, predicts velocity v(x_t, t).
    Uses time embedding for conditioning via FiLM (Feature-wise Linear Modulation).
    """
    
    def __init__(self, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        
        c = base_channels
        self.time_embed = TimeEmbedding(time_dim)
        
        # Encoder
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        
        # Time projection layers (for FiLM)
        self.time_proj1 = nn.Linear(time_dim, c)
        self.time_proj2 = nn.Linear(time_dim, c * 2)
        self.time_proj3 = nn.Linear(time_dim, c * 4)
        self.time_proj_mid = nn.Linear(time_dim, c * 8)
        
        # Downsampling
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.mid = ConvBlock(c * 4, c * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec3 = ConvBlock(c * 8, c * 4)  # +skip
        
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec2 = ConvBlock(c * 4, c * 2)  # +skip
        
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec1 = ConvBlock(c * 2, c)  # +skip
        
        # Output: predict velocity
        self.out = nn.Conv2d(c, 1, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input x_t [B, 1, H, W]
            t: Time [B, 1, 1, 1] or [B]
        
        Returns:
            Predicted velocity [B, 1, H, W]
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [B, time_dim]
        
        # Encoder with time conditioning (FiLM)
        e1 = self.enc1(x)
        e1 = e1 * (1 + self.time_proj1(t_emb)[:, :, None, None])
        
        e2 = self.enc2(self.pool(e1))
        e2 = e2 * (1 + self.time_proj2(t_emb)[:, :, None, None])
        
        e3 = self.enc3(self.pool(e2))
        e3 = e3 * (1 + self.time_proj3(t_emb)[:, :, None, None])
        
        # Bottleneck
        m = self.mid(self.pool(e3))
        m = m * (1 + self.time_proj_mid(t_emb)[:, :, None, None])
        
        # Decoder with skip connections
        d3 = self.up3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


class ScoreNet(nn.Module):
    """
    Optional score network for SDE sampling.
    
    For stochastic interpolant, we may also need s(x, t) = ∇_x log p(x_t | x1).
    This can be derived from velocity via:
        s(x_t, t) = (x_t - α_t x0_pred) / σ_t²
    
    Or trained separately. For simplicity, we share weights with velocity net.
    """
    
    def __init__(self, velocity_net: VelocityUNet):
        super().__init__()
        self.velocity_net = velocity_net
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        schedule: InterpolantSchedule
    ) -> torch.Tensor:
        """
        Estimate score from velocity prediction.
        
        For linear interpolant with σ(t) > 0:
            v = β'(t) (x1 - x_t) / β(t) + α'(t) x0 / α(t)  (approximately)
            
        We use a heuristic based on the relation between score and velocity.
        """
        v = self.velocity_net(x, t)
        sigma_t = schedule.sigma(t)
        
        # For small σ, score contribution is negligible
        # For larger σ, use: s ≈ -v / σ (rough approximation)
        score = -v / (sigma_t + 1e-6)
        return score


# =============================================================================
# Data Loaders
# =============================================================================

def prepare_data_loaders(
    config: Config,
    device: torch.device
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare training and validation data.
    
    Returns:
        - train_loader: DataLoader for training
        - x1_val: Validation HR images (target)
        - x0_val: Validation upsampled LR images (source)
        - lr_val: Validation LR images (for display)
    """
    print("\n--- Preparing Dataset ---")
    
    HR, LR, LR_up = create_dataset(config, device)
    
    # Train/val split
    N = HR.size(0)
    perm = torch.randperm(N, device=device)
    cut = int(config.train_ratio * N)
    train_idx, val_idx = perm[:cut], perm[cut:]
    
    x1_train = HR[train_idx]  # Target (HR)
    x0_train = LR_up[train_idx]  # Source (upsampled LR)
    x1_val = HR[val_idx]
    x0_val = LR_up[val_idx]
    lr_val = LR[val_idx]
    
    print(f"Training samples: {x1_train.size(0)}")
    print(f"Validation samples: {x1_val.size(0)}")
    print(f"Image resolution: {config.hr_resolution}x{config.hr_resolution}")
    print(f"Coupling mode: {config.coupling_mode}")
    print(f"Flow Matching type: {config.fm_type}")
    
    # Create DataLoader
    train_dataset = TensorDataset(x0_train, x1_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return train_loader, x1_val, x0_val, lr_val


# =============================================================================
# Training
# =============================================================================

def train_flow_matching(
    model: VelocityUNet,
    train_loader: DataLoader,
    config: Config,
    device: torch.device
) -> nn.Module:
    """
    Train velocity network using Flow Matching objective.
    
    Supports both deterministic and stochastic flow matching.
    
    Flow Matching Loss:
        L = E_{t, x0, x1} || v_θ(x_t, t) - v*(t; x0, x1, ε) ||²
        
    where:
        - Deterministic: v* = x1 - x0 (constant along linear path)
        - Stochastic: v* = α'(t)x0 + β'(t)x1 + σ'(t)ε
    """
    print(f"\n--- Training {'Stochastic' if config.fm_type == 'stochastic' else 'Deterministic'} Flow Matching ---")
    
    # Setup interpolant schedule
    if config.fm_type == 'stochastic':
        schedule = InterpolantSchedule('stochastic', sigma_max=config.sigma_max)
    else:
        schedule = InterpolantSchedule('linear', sigma_max=0.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    mse_loss = nn.MSELoss()
    
    model.train()
    
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        for x0_batch, x1_batch in train_loader:
            batch_size = x0_batch.size(0)
            
            # Optional: Mini-batch OT coupling for unpaired data
            if config.coupling_mode == 'unpaired':
                x0_batch, x1_batch = sample_ot_coupling(
                    x0_batch, x1_batch, reg=config.ot_reg
                )
            
            # Sample random time t ~ U[0, 1]
            t = torch.rand(batch_size, 1, 1, 1, device=device)
            
            # Sample noise (for stochastic interpolant)
            noise = torch.randn_like(x0_batch)
            
            # Construct interpolated sample x_t
            x_t = schedule.interpolate(x0_batch, x1_batch, t, noise)
            
            # Target velocity
            v_target = schedule.velocity_target(x0_batch, x1_batch, t, noise)
            
            # Predict velocity
            v_pred = model(x_t, t)
            
            # Flow Matching loss
            loss = mse_loss(v_pred, v_target)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        
        print(f"Epoch {epoch:02d}/{config.epochs} | FM Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return model


# =============================================================================
# Inference: ODE and SDE Samplers
# =============================================================================

@torch.no_grad()
def ode_integrate_heun(
    model: nn.Module,
    x0: torch.Tensor,
    steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Deterministic ODE integration using Heun's method.
    
    Solves: dx/dt = v_θ(x, t) from t=0 to t=1.
    
    Heun's method (2nd-order):
        1. k1 = v(x_n, t_n)
        2. x_pred = x_n + dt * k1
        3. k2 = v(x_pred, t_{n+1})
        4. x_{n+1} = x_n + dt/2 * (k1 + k2)
    """
    model.eval()
    
    x = x0.clone()
    batch_size = x.size(0)
    dt = 1.0 / steps
    
    for k in tqdm(range(steps), desc="ODE Integration"):
        t_cur = torch.full((batch_size, 1, 1, 1), k * dt, device=device, dtype=x.dtype)
        t_next = torch.full((batch_size, 1, 1, 1), (k + 1) * dt, device=device, dtype=x.dtype)
        
        # Heun predictor
        v1 = model(x, t_cur)
        x_pred = x + dt * v1
        
        # Heun corrector
        v2 = model(x_pred, t_next)
        x = x + 0.5 * dt * (v1 + v2)
        
        x = x.clamp(0.0, 1.0)
    
    return x


@torch.no_grad()
def sde_integrate_euler_maruyama(
    model: nn.Module,
    x0: torch.Tensor,
    steps: int,
    schedule: InterpolantSchedule,
    noise_scale: float,
    device: torch.device
) -> torch.Tensor:
    """
    Stochastic SDE integration using Euler-Maruyama method.
    
    Solves the SDE:
        dx = v_θ(x, t) dt + g(t) dW
        
    where g(t) = σ'(t) is the diffusion coefficient.
    
    Euler-Maruyama:
        x_{n+1} = x_n + v(x_n, t_n) * dt + g(t_n) * √dt * z
        
    where z ~ N(0, I).
    
    Benefits of SDE over ODE:
        - Can improve sample diversity
        - May correct errors in learned velocity field
        - More faithful to stochastic interpolant formulation
    """
    model.eval()
    
    x = x0.clone()
    batch_size = x.size(0)
    dt = 1.0 / steps
    sqrt_dt = np.sqrt(dt)
    
    for k in tqdm(range(steps), desc="SDE Integration"):
        t_cur = torch.full((batch_size, 1, 1, 1), k * dt, device=device, dtype=x.dtype)
        
        # Drift term: v(x, t)
        v = model(x, t_cur)
        
        # Diffusion coefficient: σ'(t)
        sigma_prime = schedule.sigma_derivative(t_cur)
        
        # Stochastic noise
        z = torch.randn_like(x)
        
        # Euler-Maruyama step
        x = x + v * dt + noise_scale * sigma_prime * sqrt_dt * z
        x = x.clamp(0.0, 1.0)
    
    return x


@torch.no_grad()
def flow_matching_inference(
    model: nn.Module,
    lr_images: torch.Tensor,
    config: Config,
    device: torch.device,
    schedule: Optional[InterpolantSchedule] = None
) -> torch.Tensor:
    """
    Super-resolve images using trained Flow Matching model.
    
    Supports both ODE (deterministic) and SDE (stochastic) inference.
    
    Args:
        model: Trained velocity network
        lr_images: Low-resolution inputs [B, 1, h, w]
        config: Configuration
        device: Compute device
        schedule: Interpolant schedule (for SDE mode)
    
    Returns:
        Super-resolved images [B, 1, H, W]
    """
    # Upsample LR to HR resolution (starting point x0)
    x0 = F.interpolate(
        lr_images,
        scale_factor=config.downsample_factor,
        mode='bilinear',
        align_corners=False
    )
    
    if config.inference_mode == 'sde' and schedule is not None:
        # Stochastic SDE inference
        sr_images = sde_integrate_euler_maruyama(
            model, x0, config.inference_steps, schedule,
            config.sde_noise_scale, device
        )
    else:
        # Deterministic ODE inference
        sr_images = ode_integrate_heun(model, x0, config.inference_steps, device)
    
    return sr_images


# =============================================================================
# Evaluation and Visualization
# =============================================================================

def compute_ssim_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> List[float]:
    """Compute SSIM for each image pair."""
    scores = []
    for i in range(predictions.size(0)):
        pred_np = predictions[i, 0].cpu().numpy()
        target_np = targets[i, 0].cpu().numpy()
        score = ssim(target_np, pred_np, data_range=1.0)
        scores.append(score)
    return scores


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def evaluate_on_validation(
    model: nn.Module,
    lr_val: torch.Tensor,
    hr_val: torch.Tensor,
    config: Config,
    device: torch.device,
    schedule: Optional[InterpolantSchedule] = None,
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Evaluate on full validation set.
    
    Returns:
        Mean SSIM and PSNR
    """
    ssim_scores = []
    psnr_scores = []
    
    for i in range(0, lr_val.size(0), batch_size):
        lr_batch = lr_val[i:i+batch_size].to(device)
        hr_batch = hr_val[i:i+batch_size].to(device)
        
        sr_batch = flow_matching_inference(model, lr_batch, config, device, schedule)
        
        ssim_scores.extend(compute_ssim_scores(sr_batch, hr_batch))
        psnr_scores.append(compute_psnr(sr_batch, hr_batch))
    
    return float(np.mean(ssim_scores)), float(np.mean(psnr_scores))


def visualize_results(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    ssim_scores: List[float],
    title: str = "SF²M Flow Matching Super-Resolution",
    save_path: Optional[str] = None
) -> None:
    """Create visualization comparing LR, SR, and HR images."""
    n_samples = sr_images.size(0)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.6 * n_samples))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        axes[i, 0].imshow(lr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title("Low-Resolution Input", fontsize=11)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f"FM Output\nSSIM: {ssim_scores[i]:.3f}", fontsize=11)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(hr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title("Ground Truth (HR)", fontsize=11)
        axes[i, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def compare_ode_vs_sde(
    model: nn.Module,
    lr_sample: torch.Tensor,
    hr_sample: torch.Tensor,
    config: Config,
    schedule: InterpolantSchedule,
    device: torch.device
) -> None:
    """
    Compare ODE vs SDE inference on the same sample.
    
    Demonstrates the difference between deterministic and stochastic sampling.
    """
    print("\n--- Comparing ODE vs SDE Inference ---")
    
    # ODE inference
    config_ode = Config(**{**config.__dict__, 'inference_mode': 'ode'})
    sr_ode = flow_matching_inference(model, lr_sample, config_ode, device, schedule)
    ssim_ode = compute_ssim_scores(sr_ode, hr_sample)[0]
    
    # SDE inference
    config_sde = Config(**{**config.__dict__, 'inference_mode': 'sde'})
    sr_sde = flow_matching_inference(model, lr_sample, config_sde, device, schedule)
    ssim_sde = compute_ssim_scores(sr_sde, hr_sample)[0]
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle("ODE vs SDE Inference Comparison", fontsize=14, fontweight='bold')
    
    axes[0].imshow(lr_sample[0, 0].cpu().numpy(), cmap='viridis')
    axes[0].set_title("LR Input")
    axes[0].axis('off')
    
    axes[1].imshow(sr_ode[0, 0].cpu().numpy(), cmap='viridis')
    axes[1].set_title(f"ODE (SSIM: {ssim_ode:.3f})")
    axes[1].axis('off')
    
    axes[2].imshow(sr_sde[0, 0].cpu().numpy(), cmap='viridis')
    axes[2].set_title(f"SDE (SSIM: {ssim_sde:.3f})")
    axes[2].axis('off')
    
    axes[3].imshow(hr_sample[0, 0].cpu().numpy(), cmap='viridis')
    axes[3].set_title("Ground Truth")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"ODE SSIM: {ssim_ode:.4f}")
    print(f"SDE SSIM: {ssim_sde:.4f}")


def evaluate_model(
    model: nn.Module,
    x1_val: torch.Tensor,
    x0_val: torch.Tensor,
    lr_val: torch.Tensor,
    config: Config,
    device: torch.device,
    schedule: Optional[InterpolantSchedule] = None,
    n_samples: int = 3
) -> None:
    """Full evaluation with visualization."""
    print("\n--- Evaluating Model ---")
    
    # Select random samples
    indices = torch.randperm(x1_val.size(0), device=device)[:n_samples]
    lr_samples = lr_val[indices]
    hr_samples = x1_val[indices]
    
    # Run inference
    start_time = time.time()
    sr_samples = flow_matching_inference(model, lr_samples, config, device, schedule)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.2f}s for {n_samples} samples")
    print(f"Average time per sample: {inference_time/n_samples:.2f}s")
    print(f"Inference mode: {config.inference_mode.upper()}")
    
    # Compute metrics
    ssim_scores = compute_ssim_scores(sr_samples, hr_samples)
    psnr = compute_psnr(sr_samples, hr_samples)
    
    print(f"SSIM scores: {[f'{s:.3f}' for s in ssim_scores]}")
    print(f"Mean SSIM: {np.mean(ssim_scores):.3f}")
    print(f"Mean PSNR: {psnr:.2f} dB")
    
    # Visualize
    title = f"[SF]²M {'Stochastic' if config.fm_type == 'stochastic' else 'Deterministic'} Flow Matching"
    visualize_results(lr_samples, sr_samples, hr_samples, ssim_scores, title=title)
    
    # Full validation evaluation
    print("\n--- Full Validation Set Evaluation ---")
    mean_ssim, mean_psnr = evaluate_on_validation(model, lr_val, x1_val, config, device, schedule)
    print(f"Validation SSIM: {mean_ssim:.3f}")
    print(f"Validation PSNR: {mean_psnr:.2f} dB")


# =============================================================================
# Experiment: Effect of Dataset Size and Epochs
# =============================================================================

def experiment_dataset_size(
    base_config: Config,
    device: torch.device,
    sizes: List[int] = [100, 300, 600],
    epochs_list: List[int] = [10, 20, 30]
) -> dict:
    """
    Study effect of dataset size and training epochs on performance.
    
    This experiment helps understand:
    1. How much data is needed for good generalization
    2. Trade-off between training time and quality
    
    Returns:
        Dictionary of results for each (size, epochs) combination
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Effect of Dataset Size and Epochs")
    print("="*60)
    
    results = {}
    
    for n_images in sizes:
        for epochs in epochs_list:
            print(f"\n--- Dataset Size: {n_images}, Epochs: {epochs} ---")
            
            # Create config for this experiment
            config = Config(
                n_images=n_images,
                epochs=epochs,
                seed=base_config.seed,
                fm_type=base_config.fm_type
            )
            
            # Prepare data
            set_seed(config.seed)
            train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config, device)
            
            # Train model
            model = VelocityUNet(base_channels=config.base_channels).to(device)
            model = train_flow_matching(model, train_loader, config, device)
            
            # Evaluate
            schedule = InterpolantSchedule(
                'stochastic' if config.fm_type == 'stochastic' else 'linear',
                sigma_max=config.sigma_max
            )
            mean_ssim, mean_psnr = evaluate_on_validation(
                model, lr_val, x1_val, config, device, schedule
            )
            
            results[(n_images, epochs)] = {
                'ssim': mean_ssim,
                'psnr': mean_psnr
            }
            print(f"Results: SSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f}dB")
    
    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Size':>8} {'Epochs':>8} {'SSIM':>10} {'PSNR (dB)':>12}")
    print("-"*40)
    for (size, epochs), metrics in sorted(results.items()):
        print(f"{size:>8} {epochs:>8} {metrics['ssim']:>10.4f} {metrics['psnr']:>12.2f}")
    
    return results


# =============================================================================
# Experiment: Paired vs Unpaired Training
# =============================================================================

def experiment_paired_vs_unpaired(
    config: Config,
    device: torch.device
) -> dict:
    """
    Compare paired vs unpaired (OT-coupled) training.
    
    This demonstrates:
    1. Paired training uses direct (degraded, original) pairs
    2. Unpaired training uses mini-batch OT to find good couplings
    
    Unpaired setting is important when:
    - Ground truth pairs are not available
    - We only have samples from source and target distributions
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Paired vs Unpaired Training")
    print("="*60)
    
    results = {}
    
    for coupling_mode in ['paired', 'unpaired']:
        print(f"\n--- Coupling Mode: {coupling_mode.upper()} ---")
        
        exp_config = Config(
            **{**config.__dict__, 'coupling_mode': coupling_mode}
        )
        
        set_seed(exp_config.seed)
        train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(exp_config, device)
        
        model = VelocityUNet(base_channels=exp_config.base_channels).to(device)
        model = train_flow_matching(model, train_loader, exp_config, device)
        
        schedule = InterpolantSchedule(
            'stochastic' if exp_config.fm_type == 'stochastic' else 'linear',
            sigma_max=exp_config.sigma_max
        )
        mean_ssim, mean_psnr = evaluate_on_validation(
            model, lr_val, x1_val, exp_config, device, schedule
        )
        
        results[coupling_mode] = {
            'ssim': mean_ssim,
            'psnr': mean_psnr
        }
        print(f"Results: SSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f}dB")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Paired:   SSIM={results['paired']['ssim']:.4f}, PSNR={results['paired']['psnr']:.2f}dB")
    print(f"Unpaired: SSIM={results['unpaired']['ssim']:.4f}, PSNR={results['unpaired']['psnr']:.2f}dB")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point demonstrating all SF²M capabilities.
    
    This script showcases:
    1. Stochastic Flow Matching training
    2. Both ODE and SDE inference
    3. Paired and unpaired data handling
    4. Ablation experiments
    """
    # Configuration
    config = Config(
        n_images=600,
        epochs=30,
        fm_type='stochastic',  # 'stochastic' or 'deterministic'
        coupling_mode='paired',  # 'paired' or 'unpaired'
        sigma_max=0.1,
        inference_mode='ode',  # 'ode' or 'sde'
        inference_steps=50
    )
    
    set_seed(config.seed)
    device = get_device()
    
    # Setup interpolant schedule
    schedule = InterpolantSchedule(
        'stochastic' if config.fm_type == 'stochastic' else 'linear',
        sigma_max=config.sigma_max
    )
    
    # Prepare data
    train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config, device)
    
    # Create and train model
    print("\n--- Model Architecture ---")
    model = VelocityUNet(base_channels=config.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    model = train_flow_matching(model, train_loader, config, device)
    
    # Evaluate
    evaluate_model(model, x1_val, x0_val, lr_val, config, device, schedule)
    
    # Optional: Compare ODE vs SDE
    if config.fm_type == 'stochastic':
        idx = torch.randint(0, lr_val.size(0), (1,)).item()
        compare_ode_vs_sde(
            model,
            lr_val[idx:idx+1].to(device),
            x1_val[idx:idx+1].to(device),
            config,
            schedule,
            device
        )
    
    # Summary
    print("\n" + "="*60)
    print("SF²M FLOW MATCHING - SUMMARY")
    print("="*60)
    print(f"""
Key Implementation Features:
    
1. INTERPOLANT TYPES:
   • Linear: x_t = (1-t)x0 + tx1  [Deterministic]
   • Stochastic: x_t = (1-t)x0 + tx1 + σ(t)ε  [With noise]
   
2. TRAINING OBJECTIVES:
   • Velocity matching: v_θ(x_t, t) ≈ α'(t)x0 + β'(t)x1 + σ'(t)ε
   • Works for both deterministic and stochastic cases
   
3. DATA COUPLING:
   • Paired: Direct (degraded, clean) pairs
   • Unpaired: Mini-batch Optimal Transport coupling
   
4. INFERENCE METHODS:
   • ODE: Deterministic, integrates dx = v(x,t)dt
   • SDE: Stochastic, adds diffusion term for diversity
   
5. NUMERICAL SOLVERS:
   • Heun's method (2nd order) for ODE
   • Euler-Maruyama for SDE

Configuration used:
   • FM Type: {config.fm_type}
   • Coupling: {config.coupling_mode}
   • Inference: {config.inference_mode}
   • σ_max: {config.sigma_max}
   • Steps: {config.inference_steps}
""")


if __name__ == "__main__":
    main()
