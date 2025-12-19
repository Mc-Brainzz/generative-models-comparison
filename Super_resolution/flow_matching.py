"""
Flow Matching ([SF]²M-style) Super-Resolution

This module implements Flow Matching for image super-resolution, providing an
alternative to denoising-based methods like SDEdit and DDIB. Instead of learning
to predict noise, Flow Matching learns a velocity field that transports samples
along a deterministic path from low-resolution to high-resolution images.

Key Concepts:
    1. Linear Interpolation Path: x_t = (1-t)*x0 + t*x1
    2. Constant Velocity Target: v* = x1 - x0 (closed-form, stable to learn)
    3. ODE Integration: dx/dt = v_θ(x,t) from t=0 to t=1
    4. Heun's Method: 2nd-order Runge-Kutta for stable integration

Why Flow Matching vs Denoising?
    - Different training objective: velocity prediction vs noise prediction
    - No noise schedule to tune; linear path is simple and effective
    - ODE integration is deterministic (vs stochastic sampling in diffusion)
    - Comparable quality with potentially fewer function evaluations

Reference:
    - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
    - https://arxiv.org/abs/2210.02747

Usage:
    python flow_matching.py

Requirements:
    torch, numpy, matplotlib, scipy, scikit-image, tqdm
"""

import random
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Central configuration for all hyperparameters and settings."""
    
    # Reproducibility
    seed: int = 123
    
    # Data generation
    n_images: int = 600
    n_points_per_image: int = 140
    hr_resolution: int = 128
    downsample_factor: int = 4
    blur_sigma: float = 1.0
    blur_radius: int = 3
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 30
    train_split: float = 0.85
    
    # Inference
    inference_steps: int = 60
    
    # Model
    base_channels: int = 32


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select the best available device (CUDA GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# =============================================================================
# Data Generation
# =============================================================================

def van_der_pol_ode(t: float, y: np.ndarray, mu: float = 1.0) -> List[float]:
    """
    Van der Pol oscillator ODE system.
    
    This non-linear oscillator produces a stable limit cycle, creating complex
    but structured 2D trajectories ideal for testing generative models.
    
    Args:
        t: Time (unused but required by ODE solver interface)
        y: State vector [x, v] (position, velocity)
        mu: Nonlinearity parameter controlling limit cycle shape
    
    Returns:
        Derivatives [dx/dt, dv/dt]
    """
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]


def points_to_image(
    points: np.ndarray,
    resolution: int = 128,
    bounds: Tuple[float, float] = (-4.0, 4.0)
) -> np.ndarray:
    """
    Convert a 2D point cloud to a grayscale density image.
    
    Args:
        points: Array of shape (N, 2) containing 2D coordinates
        resolution: Output image resolution (resolution x resolution)
        bounds: Spatial extent (min, max) for both x and y axes
    
    Returns:
        Normalized grayscale image of shape (resolution, resolution)
    """
    histogram, _, _ = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=resolution,
        range=[[bounds[0], bounds[1]], [bounds[0], bounds[1]]]
    )
    
    # Transpose to align with image conventions
    image = histogram.T
    
    # Normalize to [0, 1] range
    if image.max() > 0:
        image = (image - image.min()) / (image.max() - image.min())
    
    return image.astype(np.float32)


def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """
    Create a 1D Gaussian kernel for convolution-based blurring.
    
    Args:
        sigma: Standard deviation of the Gaussian
        radius: Kernel extends from -radius to +radius
    
    Returns:
        Normalized 1D kernel array
    """
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_torch(
    x: torch.Tensor,
    sigma: float = 1.0,
    radius: int = 3
) -> torch.Tensor:
    """
    Apply Gaussian blur using separable convolution.
    
    Separable convolution is more efficient than 2D convolution:
    O(n*r) vs O(n*r²) where r is kernel radius.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        sigma: Gaussian standard deviation
        radius: Kernel radius
    
    Returns:
        Blurred tensor of same shape
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {x.shape}")
    
    B, C, H, W = x.shape
    
    kernel_np = gaussian_kernel_1d(sigma, radius)
    kernel = torch.as_tensor(kernel_np, device=x.device, dtype=x.dtype)
    
    # Reshape for horizontal and vertical convolution
    kernel_h = kernel.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    kernel_v = kernel.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    
    # Apply separable blur (depthwise convolution)
    x = F.conv2d(x, kernel_h, padding=(0, radius), groups=C)
    x = F.conv2d(x, kernel_v, padding=(radius, 0), groups=C)
    
    return x


def degrade_operator(
    hr_image: torch.Tensor,
    blur_sigma: float = 1.0,
    blur_radius: int = 3,
    downsample_factor: int = 4
) -> torch.Tensor:
    """
    Apply the degradation operator D(x) to simulate low-resolution imaging.
    
    Standard SR degradation model: LR = (HR * G) ↓ s
    where G is Gaussian blur and ↓s is downsampling by factor s.
    
    Args:
        hr_image: High-resolution image tensor (B, C, H, W)
        blur_sigma: Gaussian blur standard deviation
        blur_radius: Blur kernel radius
        downsample_factor: Spatial downsampling ratio
    
    Returns:
        Degraded low-resolution image
    """
    blurred = gaussian_blur_torch(hr_image, sigma=blur_sigma, radius=blur_radius)
    scale = 1.0 / downsample_factor
    lr_image = F.interpolate(blurred, scale_factor=scale, mode='bilinear', align_corners=False)
    return lr_image


def create_dataset(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate paired dataset of HR and LR images from Van der Pol trajectories.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (HR_images, LR_images) as numpy arrays
    """
    hr_list, lr_list = [], []
    
    for _ in tqdm(range(config.n_images), desc="Generating Van der Pol dataset"):
        # Randomize simulation for dataset diversity
        t_final = 5.0 + np.random.rand() * 5.0
        initial_state = (np.random.rand(2) - 0.5) * 6.0
        
        # Integrate ODE
        solution = solve_ivp(
            van_der_pol_ode,
            t_span=[0, t_final],
            y0=initial_state,
            t_eval=np.linspace(0, t_final, config.n_points_per_image)
        )
        points = solution.y.T
        
        if points.shape[0] < 2:
            continue
        
        # Create HR image
        hr_np = points_to_image(points, resolution=config.hr_resolution)
        hr_tensor = torch.from_numpy(hr_np[None, None, :, :])
        
        # Apply degradation
        with torch.no_grad():
            lr_tensor = degrade_operator(
                hr_tensor,
                blur_sigma=config.blur_sigma,
                blur_radius=config.blur_radius,
                downsample_factor=config.downsample_factor
            )
        
        hr_list.append(hr_tensor.squeeze(0).numpy())
        lr_list.append(lr_tensor.squeeze(0).numpy())
    
    return np.stack(hr_list, axis=0), np.stack(lr_list, axis=0)


def prepare_data_loaders(
    config: Config,
    device: torch.device
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate dataset and prepare data loaders for training.
    
    For Flow Matching, the training pairs are:
        - x0: Blurry upsampled LR (starting point at t=0)
        - x1: Sharp HR (target point at t=1)
    
    Returns:
        train_loader: DataLoader yielding (x0, x1) pairs
        x1_val: Validation HR images (ground truth)
        x0_val: Validation upsampled LR images
        lr_val: Validation LR images (for display)
    """
    print("\n--- Generating Dataset ---")
    hr_np, lr_np = create_dataset(config)
    
    # Move to device
    hr = torch.from_numpy(hr_np).to(device)  # x1: target
    lr = torch.from_numpy(lr_np).to(device)
    
    # x0: Blurry bilinear-upsampled LR (FM starting point at t=0)
    # Using the SAME upsampling as SDEdit/DDIB ensures fair comparison
    x0_up = F.interpolate(lr, scale_factor=config.downsample_factor, 
                          mode='bilinear', align_corners=False)
    
    # Train/validation split
    n_samples = hr.shape[0]
    perm = torch.randperm(n_samples, device=device)
    split_idx = int(config.train_split * n_samples)
    
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]
    
    x1_train, x0_train = hr[train_idx], x0_up[train_idx]
    x1_val, x0_val = hr[val_idx], x0_up[val_idx]
    lr_val = lr[val_idx]
    
    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Create data loader with (x0, x1) pairs
    train_dataset = TensorDataset(x0_train, x1_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return train_loader, x1_val, x0_val, lr_val


# =============================================================================
# Model Architecture
# =============================================================================

class ConvBlock(nn.Module):
    """Basic convolutional block: two 3x3 convolutions with ReLU activations."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SmallCondDenoiser(nn.Module):
    """
    U-Net backbone shared with SDEdit implementation.
    
    Architecture kept identical to enable fair comparison between methods.
    The backbone expects 3 input channels: [x_in, cond, aux_map].
    
    For Flow Matching, we wrap this to expose forward(x, t) interface.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 1
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                       kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                       kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels)
        
        # Output projection
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(
        self,
        x_in: torch.Tensor,
        cond: torch.Tensor,
        aux_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with 3-channel input structure.
        
        Args:
            x_in: Current image/state (B, 1, H, W)
            cond: Conditioning map (B, 1, H, W)
            aux_map: Auxiliary scalar field like time (B, 1, H, W)
        
        Returns:
            Output tensor (B, 1, H, W)
        """
        # Concatenate inputs
        x = torch.cat([x_in, cond, aux_map], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        return self.out_conv(d2)


class FlowMatchingVelocityNet(nn.Module):
    """
    Velocity field network for Flow Matching with forward(x, t) interface.
    
    This wrapper maintains architectural parity with SDEdit while conforming
    to the Flow Matching API. The backbone is identical; only the interface
    differs.
    
    Key Design Decisions:
        - Pass zeros for 'cond' to match backbone capacity exactly
        - Use time t as the 'aux_map' input
        - No leaking of extra conditioning beyond t
    
    If you want conditional FM (conditioning on LR), replace cond_zeros
    with the upsampled LR image in the forward pass.
    
    Args:
        base_channels: Number of channels in first layer
    """
    
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.backbone = SmallCondDenoiser(
            in_channels=3,
            base_channels=base_channels,
            out_channels=1
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field v_θ(x, t).
        
        Args:
            x: Current state x_t (B, 1, H, W)
            t: Time in [0, 1], scalar or (B, 1, 1, 1) tensor
        
        Returns:
            Predicted velocity dx/dt (B, 1, H, W)
        """
        # Handle various time input formats
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        
        if t.dim() == 0:
            t = t.view(1, 1, 1, 1).expand(x.size(0), 1, 1, 1)
        elif t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        # Broadcast t to spatial map
        t_map = t.expand(-1, 1, x.size(2), x.size(3))
        
        # Use zeros for conditioning to maintain architectural parity
        cond_zeros = torch.zeros_like(x)
        
        return self.backbone(x, cond_zeros, t_map)


# =============================================================================
# Training
# =============================================================================

def train_flow_matching(
    model: nn.Module,
    train_loader: DataLoader,
    config: Config,
    device: torch.device
) -> nn.Module:
    """
    Train velocity network using Flow Matching objective.
    
    Flow Matching Training Objective:
        Given paired (x0, x1), we define a straight interpolation path:
            x_t = (1 - t) * x0 + t * x1
        
        The true velocity along this path is constant:
            v* = d/dt x_t = x1 - x0
        
        We train v_θ(x_t, t) to predict v* with MSE loss:
            L = E_{t~U[0,1]} || v_θ(x_t, t) - v* ||²
    
    Why This Works:
        - v* is constant along the path → simple, stable target
        - x0 already encodes LR structure → meaningful starting point
        - No noise schedule to tune (vs diffusion models)
        - Linear path is the simplest choice; more complex paths exist
    
    Args:
        model: FlowMatchingVelocityNet
        train_loader: DataLoader yielding (x0, x1) batches
        config: Configuration object
        device: Compute device
    
    Returns:
        Trained model
    """
    print("\n--- Training Flow Matching ---")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse_loss = nn.MSELoss()
    
    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        
        for x0_batch, x1_batch in train_loader:
            batch_size = x0_batch.size(0)
            
            # Sample random time t ~ U[0, 1] for each sample
            t = torch.rand(batch_size, 1, 1, 1, device=device)
            
            # Construct point on linear interpolation path
            # x_t = (1 - t) * x0 + t * x1
            x_t = (1.0 - t) * x0_batch + t * x1_batch
            
            # Target velocity is constant: v* = x1 - x0
            v_target = x1_batch - x0_batch
            
            # Predict velocity
            v_pred = model(x_t, t)
            
            # FM loss: MSE between predicted and true velocity
            loss = mse_loss(v_pred, v_target)
            
            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:02d}/{config.epochs} | FM Loss: {avg_loss:.5f}")
    
    return model


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def heun_integrate(
    model: nn.Module,
    x0: torch.Tensor,
    steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Integrate ODE dx/dt = v_θ(x, t) from t=0 to t=1 using Heun's method.
    
    Heun's Method (2nd-order Runge-Kutta):
        For each step from t_k to t_{k+1}:
        1. Predictor (Euler): x_pred = x_k + dt * v_θ(x_k, t_k)
        2. Corrector: x_{k+1} = x_k + 0.5 * dt * (v_θ(x_k, t_k) + v_θ(x_pred, t_{k+1}))
    
    Why Heun over Euler?
        - Euler can be biased/unstable with larger steps
        - Heun adds one extra slope evaluation for much better accuracy
        - Only ~2x compute cost, but significantly more stable
    
    Args:
        model: Velocity network v_θ
        x0: Starting point (B, 1, H, W) - blurry upsampled LR
        steps: Number of integration steps
        device: Compute device
    
    Returns:
        Final state x(t=1) - super-resolved image
    """
    model.eval()
    
    x = x0.clone()
    batch_size = x.size(0)
    dt = 1.0 / steps
    
    for k in tqdm(range(steps), desc="Flow Matching Inference"):
        # Current and next time as tensors
        t_cur = torch.full((batch_size, 1, 1, 1), k * dt, device=device, dtype=x.dtype)
        t_next = torch.full((batch_size, 1, 1, 1), (k + 1) * dt, device=device, dtype=x.dtype)
        
        # Heun predictor: Euler step
        v1 = model(x, t_cur)
        x_pred = x + dt * v1
        
        # Heun corrector: average of two slopes
        v2 = model(x_pred, t_next)
        x = x + 0.5 * dt * (v1 + v2)
        
        # Clamp to valid image range (prevents numerical drift)
        x = x.clamp(0.0, 1.0)
    
    return x


@torch.no_grad()
def flow_matching_inference(
    model: nn.Module,
    lr_images: torch.Tensor,
    config: Config,
    device: torch.device
) -> torch.Tensor:
    """
    Super-resolve low-resolution images using Flow Matching.
    
    Process:
        1. Upsample LR to HR resolution (bilinear) → x0 at t=0
        2. Integrate ODE dx/dt = v_θ(x, t) from t=0 to t=1
        3. Return final state x(t=1) as super-resolved image
    
    Args:
        model: Trained velocity network
        lr_images: Low-resolution inputs (B, 1, h, w)
        config: Configuration object
        device: Compute device
    
    Returns:
        Super-resolved images (B, 1, H, W)
    """
    # Upsample LR to HR resolution (starting point x0)
    x0 = F.interpolate(
        lr_images,
        scale_factor=config.downsample_factor,
        mode='bilinear',
        align_corners=False
    )
    
    # Integrate ODE to t=1
    sr_images = heun_integrate(model, x0, config.inference_steps, device)
    
    return sr_images


# =============================================================================
# Evaluation and Visualization
# =============================================================================

def compute_ssim_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> List[float]:
    """
    Compute SSIM (Structural Similarity Index) for each image pair.
    
    SSIM is preferred over MSE/PSNR for super-resolution because it
    focuses on perceptual quality: luminance, contrast, and structure.
    
    Args:
        predictions: Predicted images (B, 1, H, W)
        targets: Ground truth images (B, 1, H, W)
    
    Returns:
        List of SSIM scores for each image
    """
    scores = []
    for i in range(predictions.size(0)):
        pred_np = predictions[i, 0].cpu().numpy()
        target_np = targets[i, 0].cpu().numpy()
        score = ssim(target_np, pred_np, data_range=1.0)
        scores.append(score)
    return scores


def evaluate_on_validation(
    model: nn.Module,
    lr_val: torch.Tensor,
    hr_val: torch.Tensor,
    config: Config,
    device: torch.device,
    batch_size: int = 32
) -> float:
    """
    Compute mean SSIM on full validation set.
    
    Args:
        model: Trained velocity network
        lr_val: All validation LR images
        hr_val: All validation HR images (ground truth)
        config: Configuration object
        device: Compute device
        batch_size: Batch size for inference
    
    Returns:
        Mean SSIM score across validation set
    """
    ssim_scores = []
    
    for i in range(0, lr_val.size(0), batch_size):
        lr_batch = lr_val[i:i+batch_size].to(device)
        hr_batch = hr_val[i:i+batch_size].to(device)
        
        sr_batch = flow_matching_inference(model, lr_batch, config, device)
        
        batch_scores = compute_ssim_scores(sr_batch, hr_batch)
        ssim_scores.extend(batch_scores)
    
    return float(np.mean(ssim_scores))


def visualize_results(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    ssim_scores: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization comparing LR input, FM output, and HR ground truth.
    
    Args:
        lr_images: Low-resolution inputs
        sr_images: Super-resolved outputs
        hr_images: High-resolution ground truth
        ssim_scores: SSIM scores for each sample
        save_path: If provided, save figure to this path
    """
    n_samples = sr_images.size(0)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.6 * n_samples))
    fig.suptitle("[SF]²M Flow Matching Super-Resolution Results", fontsize=16, fontweight='bold')
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Low-resolution input
        axes[i, 0].imshow(lr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title("Low-Resolution Input", fontsize=11)
        axes[i, 0].axis('off')
        
        # Flow Matching output
        axes[i, 1].imshow(sr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f"FM Output\nSSIM: {ssim_scores[i]:.3f}", fontsize=11)
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(hr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title("Ground Truth (HR)", fontsize=11)
        axes[i, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def evaluate_model(
    model: nn.Module,
    x1_val: torch.Tensor,
    x0_val: torch.Tensor,
    lr_val: torch.Tensor,
    config: Config,
    device: torch.device,
    n_samples: int = 3
) -> None:
    """
    Evaluate trained model on validation samples.
    
    Args:
        model: Trained velocity network
        x1_val: Validation HR images (ground truth)
        x0_val: Validation upsampled LR images
        lr_val: Validation LR images (for display)
        config: Configuration object
        device: Compute device
        n_samples: Number of samples to visualize
    """
    print("\n--- Evaluating Model ---")
    
    # Select random validation samples
    indices = torch.randperm(x1_val.size(0), device=device)[:n_samples]
    lr_samples = lr_val[indices]
    hr_samples = x1_val[indices]
    
    # Run inference
    start_time = time.time()
    sr_samples = flow_matching_inference(model, lr_samples, config, device)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.2f}s for {n_samples} samples")
    print(f"Average time per sample: {inference_time/n_samples:.2f}s")
    
    # Compute metrics
    ssim_scores = compute_ssim_scores(sr_samples, hr_samples)
    print(f"SSIM scores: {[f'{s:.3f}' for s in ssim_scores]}")
    print(f"Mean SSIM (shown samples): {np.mean(ssim_scores):.3f}")
    
    # Visualize
    visualize_results(lr_samples, sr_samples, hr_samples, ssim_scores)
    
    # Full validation set evaluation
    print("\n--- Full Validation Set Evaluation ---")
    mean_ssim = evaluate_on_validation(model, lr_val, x1_val, config, device)
    print(f"Mean SSIM on validation set: {mean_ssim:.3f}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main entry point for training and evaluating Flow Matching super-resolution."""
    
    # Initialize
    config = Config()
    set_seed(config.seed)
    device = get_device()
    
    # Prepare data
    train_loader, x1_val, x0_val, lr_val = prepare_data_loaders(config, device)
    
    # Create model
    print("\n--- Model Architecture ---")
    model = FlowMatchingVelocityNet(base_channels=config.base_channels).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Train
    model = train_flow_matching(model, train_loader, config, device)
    
    # Evaluate
    evaluate_model(model, x1_val, x0_val, lr_val, config, device)
    
    print("\n--- Training Complete ---")
    print("\nFlow Matching Key Points:")
    print("  • Training: Learn velocity v* = x1 - x0 along linear path")
    print("  • Inference: Integrate ODE dx/dt = v_θ(x,t) with Heun's method")
    print("  • No noise schedule to tune (vs diffusion models)")
    print("  • Deterministic sampling via ODE integration")


if __name__ == "__main__":
    main()
