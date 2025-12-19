"""
SDEdit-Style Conditional Super-Resolution

This module implements a super-resolution pipeline inspired by the SDEdit and DCSR
papers. The approach trains a conditional denoising U-Net on high-resolution images,
then uses iterative denoising with data consistency feedback to restore sharp details
from low-resolution inputs.

Key Components:
    1. Data Generation: Van der Pol oscillator trajectories converted to density images
    2. Degradation Model: Gaussian blur + downsampling to simulate realistic LR images
    3. Conditional U-Net: Denoiser conditioned on LR guidance and noise level
    4. SDEdit Inference: Iterative denoising with data consistency correction

Reference:
    - Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic
      Differential Equations", ICLR 2022
    - https://arxiv.org/abs/2108.01073

Usage:
    python sdedit.py

Requirements:
    torch, numpy, matplotlib, scipy, scikit-image, tqdm
"""

import random
import time
from dataclasses import dataclass
from typing import Tuple, List

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
    epochs: int = 25
    train_split: float = 0.85
    
    # Noise schedule for training
    sigma_min: float = 0.02
    sigma_max: float = 0.30
    
    # Inference
    inference_sigma: float = 0.18
    inference_steps: int = 25
    data_consistency_weight: float = 0.8
    
    # Model
    base_channels: int = 32


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: For full determinism, also set torch.backends.cudnn.deterministic = True


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
    
    The system is defined as:
        dx/dt = v
        dv/dt = mu * (1 - x^2) * v - x
    
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
    
    Creates a 2D histogram where each pixel value represents the density
    of points in that spatial region. This transforms abstract trajectory
    data into a format suitable for convolutional neural networks.
    
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
    
    # Transpose to align with image conventions (y increases downward)
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
        radius: Kernel extends from -radius to +radius (size = 2*radius + 1)
    
    Returns:
        Normalized 1D kernel array
    """
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize to sum to 1
    return kernel


def gaussian_blur_torch(
    x: torch.Tensor,
    sigma: float = 1.0,
    radius: int = 3
) -> torch.Tensor:
    """
    Apply Gaussian blur using separable convolution.
    
    Separable convolution applies 1D blur horizontally then vertically,
    which is mathematically equivalent to 2D Gaussian blur but more efficient
    (O(n*r) vs O(n*r^2) where r is kernel radius).
    
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
    
    # Create 1D kernel
    kernel_np = gaussian_kernel_1d(sigma, radius)
    kernel = torch.as_tensor(kernel_np, device=x.device, dtype=x.dtype)
    
    # Reshape for horizontal and vertical convolution
    # groups=C applies same kernel to each channel independently (depthwise)
    kernel_h = kernel.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    kernel_v = kernel.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    
    # Apply separable blur
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
    
    This models the forward process of image degradation commonly seen in
    real imaging systems: optical blur (lens imperfections, diffraction)
    followed by sensor downsampling.
    
    Args:
        hr_image: High-resolution image tensor (B, C, H, W)
        blur_sigma: Gaussian blur standard deviation
        blur_radius: Blur kernel radius
        downsample_factor: Spatial downsampling ratio
    
    Returns:
        Degraded low-resolution image
    """
    # Apply blur first (models optical PSF)
    blurred = gaussian_blur_torch(hr_image, sigma=blur_sigma, radius=blur_radius)
    
    # Then downsample (models sensor resolution)
    scale = 1.0 / downsample_factor
    lr_image = F.interpolate(blurred, scale_factor=scale, mode='bilinear', align_corners=False)
    
    return lr_image


def generate_single_sample(
    n_points: int,
    hr_resolution: int,
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single HR/LR image pair from Van der Pol dynamics.
    
    Args:
        n_points: Number of trajectory points to sample
        hr_resolution: High-resolution image size
        config: Configuration with blur/downsample settings
    
    Returns:
        Tuple of (hr_image, lr_image) as numpy arrays
    """
    # Randomize simulation parameters for variety
    t_final = 5.0 + np.random.rand() * 5.0
    initial_state = (np.random.rand(2) - 0.5) * 6.0
    
    # Integrate ODE
    solution = solve_ivp(
        van_der_pol_ode,
        t_span=[0, t_final],
        y0=initial_state,
        t_eval=np.linspace(0, t_final, n_points)
    )
    points = solution.y.T  # Shape: (n_points, 2)
    
    if points.shape[0] < 2:
        return None, None
    
    # Convert to image
    hr_np = points_to_image(points, resolution=hr_resolution)
    hr_tensor = torch.from_numpy(hr_np[None, None, :, :])  # Add batch and channel dims
    
    # Apply degradation
    with torch.no_grad():
        lr_tensor = degrade_operator(
            hr_tensor,
            blur_sigma=config.blur_sigma,
            blur_radius=config.blur_radius,
            downsample_factor=config.downsample_factor
        )
    
    return hr_tensor.squeeze(0).numpy(), lr_tensor.squeeze(0).numpy()


def create_dataset(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the complete paired dataset of HR and LR images.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (HR_images, LR_images) as numpy arrays
    """
    hr_list, lr_list = [], []
    
    for _ in tqdm(range(config.n_images), desc="Generating Van der Pol dataset"):
        hr, lr = generate_single_sample(
            config.n_points_per_image,
            config.hr_resolution,
            config
        )
        if hr is not None:
            hr_list.append(hr)
            lr_list.append(lr)
    
    return np.stack(hr_list, axis=0), np.stack(lr_list, axis=0)


def prepare_data_loaders(
    config: Config,
    device: torch.device
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate dataset and prepare data loaders for training.
    
    Returns:
        train_loader: DataLoader for training
        hr_val: Validation HR images
        cond_val: Validation condition images (upscaled LR)
        lr_val: Validation LR images (for inference)
        cond_train: Training condition images (for reference)
    """
    print("\n--- Generating Dataset ---")
    hr_np, lr_np = create_dataset(config)
    
    # Move to device
    hr = torch.from_numpy(hr_np).to(device)
    lr = torch.from_numpy(lr_np).to(device)
    
    # Create condition tensor: LR upscaled to HR resolution via bilinear interpolation
    # This serves as the "guidance" input to the conditional model
    cond = F.interpolate(lr, scale_factor=config.downsample_factor, 
                         mode='bilinear', align_corners=False)
    
    # Train/validation split
    n_samples = hr.shape[0]
    perm = torch.randperm(n_samples, device=device)
    split_idx = int(config.train_split * n_samples)
    
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]
    
    hr_train, cond_train = hr[train_idx], cond[train_idx]
    hr_val, cond_val, lr_val = hr[val_idx], cond[val_idx], lr[val_idx]
    
    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Create data loader
    train_dataset = TensorDataset(cond_train, hr_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return train_loader, hr_val, cond_val, lr_val, cond_train


# =============================================================================
# Model Architecture
# =============================================================================

class ConvBlock(nn.Module):
    """
    Basic convolutional block: two 3x3 convolutions with ReLU activations.
    
    This is the fundamental building block of the U-Net architecture.
    Each block refines features while maintaining spatial resolution.
    """
    
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


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net denoiser for super-resolution.
    
    This network predicts the noise component in a noisy image, conditioned on:
    1. The noisy HR image estimate
    2. The upscaled LR image (provides structural guidance)
    3. A noise level map (tells the network the current noise scale)
    
    Architecture:
        - Encoder: Two downsampling stages with skip connections
        - Bottleneck: Deep feature processing at lowest resolution
        - Decoder: Two upsampling stages with skip connection fusion
    
    The U-Net's multi-scale processing and skip connections make it ideal for
    image restoration, preserving both fine details and global structure.
    
    Args:
        in_channels: Number of input channels (noisy + cond + sigma_map = 3)
        base_channels: Number of channels in first layer (doubled at each level)
        out_channels: Number of output channels (1 for grayscale noise prediction)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 1
    ):
        super().__init__()
        
        # Encoder path
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        
        # Decoder path with skip connections
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                                       kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 4, base_channels * 2)  # *4 due to concat
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                       kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels)  # *2 due to concat
        
        # Output projection
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        cond: torch.Tensor,
        sigma_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass predicting noise from noisy image.
        
        Args:
            x_noisy: Noisy HR image estimate (B, 1, H, W)
            cond: Condition image - upscaled LR (B, 1, H, W)
            sigma_map: Noise level map (B, 1, H, W)
        
        Returns:
            Predicted noise tensor (B, 1, H, W)
        """
        # Concatenate inputs along channel dimension
        x = torch.cat([x_noisy, cond, sigma_map], dim=1)
        
        # Encoder with skip connection storage
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder with skip connection fusion
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)  # Skip connection from enc2
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection from enc1
        d2 = self.dec2(d2)
        
        return self.out_conv(d2)


# =============================================================================
# Training
# =============================================================================

def sample_noise_level(batch_size: int, config: Config, device: torch.device) -> torch.Tensor:
    """
    Sample random noise levels uniformly from [sigma_min, sigma_max].
    
    During training, we expose the model to various noise levels so it learns
    to denoise at any point in the diffusion process.
    """
    sigma = torch.rand(batch_size, device=device)
    sigma = sigma * (config.sigma_max - config.sigma_min) + config.sigma_min
    return sigma.view(batch_size, 1, 1, 1)


def train_denoiser(
    model: nn.Module,
    train_loader: DataLoader,
    config: Config,
    device: torch.device
) -> nn.Module:
    """
    Train the conditional denoiser using noise prediction loss.
    
    Training Objective:
        Given a clean image x0, we:
        1. Sample a random noise level sigma
        2. Add Gaussian noise: x_noisy = x0 + sigma * epsilon
        3. Train the model to predict epsilon from (x_noisy, cond, sigma)
        4. Minimize MSE between predicted and actual noise
    
    This denoising score matching objective teaches the model the gradient
    of the data distribution, enabling iterative refinement during inference.
    
    Args:
        model: ConditionalUNet model
        train_loader: DataLoader yielding (condition, hr_target) batches
        config: Configuration object
        device: Compute device
    
    Returns:
        Trained model
    """
    print("\n--- Training Denoiser ---")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse_loss = nn.MSELoss()
    
    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        
        for cond_batch, hr_batch in train_loader:
            batch_size = hr_batch.size(0)
            
            # Sample noise level for each image in batch
            sigma = sample_noise_level(batch_size, config, device)
            
            # Generate random noise
            noise = torch.randn_like(hr_batch)
            
            # Create noisy image: x_noisy = x0 + sigma * noise
            x_noisy = (hr_batch + sigma * noise).clamp(0.0, 1.0)
            
            # Expand sigma to spatial map for conditioning
            sigma_map = sigma.expand(-1, 1, hr_batch.size(2), hr_batch.size(3))
            
            # Predict noise
            noise_pred = model(x_noisy, cond_batch, sigma_map)
            
            # Compute loss
            loss = mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:02d}/{config.epochs} | MSE Loss: {avg_loss:.5f}")
    
    return model


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def data_consistency_step(
    hr_estimate: torch.Tensor,
    lr_reference: torch.Tensor,
    weight: float,
    config: Config
) -> torch.Tensor:
    """
    Apply data consistency correction to ensure fidelity to LR input.
    
    This is the key innovation of SDEdit-style super-resolution. After each
    denoising step, we check if the current HR estimate, when degraded, matches
    the original LR input. Any discrepancy is fed back to correct the estimate.
    
    Mathematically:
        residual = lr_reference - D(hr_estimate)
        hr_corrected = hr_estimate + weight * upsample(residual)
    
    This ensures the output respects the data constraint while still benefiting
    from the model's learned prior over sharp images.
    
    Args:
        hr_estimate: Current HR image estimate (B, 1, H, W)
        lr_reference: Original LR input (B, 1, h, w)
        weight: Strength of correction (lambda in paper)
        config: Configuration for degradation operator
    
    Returns:
        Corrected HR estimate
    """
    # Degrade current estimate to LR space
    lr_estimate = degrade_operator(
        hr_estimate,
        blur_sigma=config.blur_sigma,
        blur_radius=config.blur_radius,
        downsample_factor=config.downsample_factor
    )
    
    # Compute residual in LR space
    residual = lr_reference - lr_estimate
    
    # Upsample residual and apply correction
    residual_upsampled = F.interpolate(
        residual, 
        scale_factor=config.downsample_factor,
        mode='bilinear', 
        align_corners=False
    )
    
    corrected = hr_estimate + weight * residual_upsampled
    return corrected


@torch.no_grad()
def sdedit_inference(
    model: nn.Module,
    lr_images: torch.Tensor,
    config: Config,
    device: torch.device
) -> torch.Tensor:
    """
    SDEdit-style iterative super-resolution inference.
    
    Process:
        1. Start with upscaled LR + initial noise (perturbs to diffusion path)
        2. Iteratively denoise with decreasing noise levels
        3. Apply data consistency after each step
        4. Return final refined HR estimate
    
    The key insight is that adding noise and denoising allows the model to
    "explore" the space of possible HR images consistent with the LR input,
    while data consistency keeps the output faithful to observations.
    
    Args:
        model: Trained ConditionalUNet denoiser
        lr_images: Low-resolution input images (B, 1, h, w)
        config: Configuration object
        device: Compute device
    
    Returns:
        Super-resolved images (B, 1, H, W)
    """
    model.eval()
    
    # Create condition: bilinear upscale of LR (constant throughout inference)
    cond = F.interpolate(
        lr_images, 
        scale_factor=config.downsample_factor,
        mode='bilinear', 
        align_corners=False
    )
    
    # Initialize with condition + noise (places us on diffusion path)
    x = cond + config.inference_sigma * torch.randn_like(cond)
    x = x.clamp(0.0, 1.0)
    
    # Create decreasing noise schedule
    sigmas = np.linspace(config.inference_sigma, 0.01, config.inference_steps, dtype=np.float32)
    
    for sigma_val in tqdm(sigmas, desc="SDEdit Inference"):
        batch_size = x.size(0)
        
        # Create sigma tensor and map
        sigma = torch.full(
            (batch_size, 1, 1, 1), 
            float(sigma_val), 
            device=device, 
            dtype=x.dtype
        )
        sigma_map = sigma.expand(-1, 1, x.size(2), x.size(3))
        
        # Denoising step: predict noise and subtract scaled version
        noise_pred = model(x, cond, sigma_map)
        step_size = 0.5 * sigma_val  # Conservative step size
        x = (x - step_size * noise_pred).clamp(0.0, 1.0)
        
        # Data consistency correction
        x = data_consistency_step(x, lr_images, config.data_consistency_weight, config)
        x = x.clamp(0.0, 1.0)
    
    return x


# =============================================================================
# Evaluation and Visualization
# =============================================================================

def compute_ssim_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> List[float]:
    """
    Compute SSIM (Structural Similarity Index) for each image pair.
    
    SSIM measures perceptual similarity considering luminance, contrast,
    and structure. It's more aligned with human perception than MSE/PSNR.
    
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


def visualize_results(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    ssim_scores: List[float],
    save_path: str = None
) -> None:
    """
    Create visualization comparing LR input, SR output, and HR ground truth.
    
    Args:
        lr_images: Low-resolution inputs
        sr_images: Super-resolved outputs
        hr_images: High-resolution ground truth
        ssim_scores: SSIM scores for each sample
        save_path: If provided, save figure to this path
    """
    n_samples = sr_images.size(0)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.6 * n_samples))
    fig.suptitle("SDEdit Super-Resolution Results", fontsize=16, fontweight='bold')
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Low-resolution input
        axes[i, 0].imshow(lr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title("Low-Resolution Input", fontsize=11)
        axes[i, 0].axis('off')
        
        # Super-resolved output
        axes[i, 1].imshow(sr_images[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f"SDEdit Output\nSSIM: {ssim_scores[i]:.3f}", fontsize=11)
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
    hr_val: torch.Tensor,
    lr_val: torch.Tensor,
    config: Config,
    device: torch.device,
    n_samples: int = 3
) -> None:
    """
    Evaluate trained model on validation samples.
    
    Args:
        model: Trained denoiser model
        hr_val: Validation HR images
        lr_val: Validation LR images
        config: Configuration object
        device: Compute device
        n_samples: Number of samples to visualize
    """
    print("\n--- Evaluating Model ---")
    
    # Select random validation samples
    indices = torch.randperm(hr_val.size(0))[:n_samples]
    lr_samples = lr_val[indices]
    hr_samples = hr_val[indices]
    
    # Run inference
    start_time = time.time()
    sr_samples = sdedit_inference(model, lr_samples, config, device)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.2f}s for {n_samples} samples")
    print(f"Average time per sample: {inference_time/n_samples:.2f}s")
    
    # Compute metrics
    ssim_scores = compute_ssim_scores(sr_samples, hr_samples)
    print(f"SSIM scores: {[f'{s:.3f}' for s in ssim_scores]}")
    print(f"Mean SSIM: {np.mean(ssim_scores):.3f}")
    
    # Visualize
    visualize_results(lr_samples, sr_samples, hr_samples, ssim_scores)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main entry point for training and evaluating SDEdit super-resolution."""
    
    # Initialize
    config = Config()
    set_seed(config.seed)
    device = get_device()
    
    # Prepare data
    train_loader, hr_val, cond_val, lr_val, _ = prepare_data_loaders(config, device)
    
    # Create model
    print("\n--- Model Architecture ---")
    model = ConditionalUNet(
        in_channels=3,  # noisy_image + condition + sigma_map
        base_channels=config.base_channels,
        out_channels=1
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Train
    model = train_denoiser(model, train_loader, config, device)
    
    # Evaluate
    evaluate_model(model, hr_val, lr_val, config, device)
    
    print("\n--- Training Complete ---")


if __name__ == "__main__":
    main()
