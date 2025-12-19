"""
Denoising Diffusion Implicit Bridge (DDIB) for Super-Resolution

A clean implementation of DDIB for image super-resolution on toy data (Van der Pol oscillator).
Uses the "Encode-Bridge-Decode" paradigm with two separate denoisers for LR and HR domains.

Key concepts:
    - Two denoisers: q0 (LR expert) and q1 (HR expert), each trained on their respective domain
    - Inference: Encode LR → latent using q0, bridge (upscale), decode to HR using q1
    - Optional energy guidance for physics-informed generation (EDDIB extension)

References:
    - Su et al. (2022): Dual Diffusion Implicit Bridges for Image-to-Image Translation
    - EDDIB: Energy-guided extensions for physics-constrained generation

Author: [Your Name]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.integrate import solve_ivp
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Hyperparameters and settings for DDIB training and inference."""
    
    # Reproducibility
    seed: int = 123
    
    # Data generation
    n_images: int = 600
    points_per_image: int = 140
    hr_resolution: int = 128
    upscale_factor: int = 4
    
    # Degradation (blur + downsample)
    blur_sigma_min: float = 0.6
    blur_sigma_max: float = 1.8
    randomize_blur: bool = True  # Helps avoid deterministic blur bias
    
    # Dataset mode
    paired_data: bool = True  # True: paired HR/LR, False: unpaired marginals
    
    # Model architecture
    base_channels: int = 32
    
    # Training
    learning_rate: float = 1e-4
    n_epochs: int = 20
    batch_size: int = 16
    
    # Noise schedule for diffusion
    sigma_min: float = 0.02
    sigma_max: float = 0.30
    
    # Inference
    inference_steps: int = 25
    
    # Energy guidance (EDDIB extension)
    use_energy: bool = False  # Enable physics-based energy guidance
    energy_scale: float = 0.1
    energy_directions: int = 12  # Number of finite-difference directions
    energy_eps: float = 1e-3
    
    # Train q0 on upscaled LR (single-scale option)
    train_on_upscaled_lr: bool = False
    
    @property
    def lr_resolution(self) -> int:
        return self.hr_resolution // self.upscale_factor


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For full determinism (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Data Generation: Van der Pol Oscillator
# =============================================================================

def van_der_pol_ode(t: float, y: np.ndarray, mu: float = 1.0) -> list:
    """
    Van der Pol oscillator ODE.
    
    This nonlinear oscillator produces limit cycle trajectories that serve
    as a good test case for generative models (non-Gaussian, structured).
    
    dx/dt = v
    dv/dt = mu * (1 - x^2) * v - x
    """
    x, v = y
    return [v, mu * (1 - x**2) * v - x]


def points_to_image(
    points: np.ndarray,
    resolution: int = 128,
    bounds: tuple = (-4, 4)
) -> np.ndarray:
    """
    Convert 2D trajectory points to a grayscale image via 2D histogram.
    
    This creates a "heatmap" where each pixel intensity corresponds to
    the density of trajectory points passing through that region.
    """
    heat, _, _ = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=resolution,
        range=[[bounds[0], bounds[1]], [bounds[0], bounds[1]]]
    )
    heat = heat.T.astype(np.float32)
    
    # Normalize to [0, 1]
    if heat.max() > 0:
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    
    return heat


def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Create a 1D Gaussian kernel for separable convolution."""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0, radius: int = 3) -> torch.Tensor:
    """
    Apply efficient separable Gaussian blur.
    
    Uses depthwise-separable convolution for O(n) instead of O(n^2) complexity.
    """
    assert x.dim() == 4, f"Expected 4D tensor [B,C,H,W], got {x.shape}"
    
    sigma = max(1e-6, float(sigma))  # Avoid degenerate sigma
    kernel_1d = gaussian_kernel_1d(sigma, radius)
    kernel = torch.as_tensor(kernel_1d, device=x.device, dtype=x.dtype)
    
    # Horizontal and vertical kernels
    k_h = kernel.view(1, 1, 1, -1)
    k_v = kernel.view(1, 1, -1, 1)
    
    # Separable convolution
    x = F.conv2d(x, k_h, padding=(0, radius), groups=1)
    x = F.conv2d(x, k_v, padding=(radius, 0), groups=1)
    
    return x


def degrade_operator(
    hr_img: torch.Tensor,
    blur_sigma: float = 1.0,
    blur_radius: int = 3,
    down_factor: int = 4
) -> torch.Tensor:
    """
    Degradation operator D(x): blur followed by downsampling.
    
    This models a typical image degradation pipeline where high-frequency
    details are lost through blur, then spatial resolution is reduced.
    """
    x = gaussian_blur(hr_img, sigma=blur_sigma, radius=blur_radius)
    x = F.interpolate(x, scale_factor=1.0 / down_factor, mode='bilinear', align_corners=False)
    return x


def create_dataset(config: Config, device: torch.device) -> tuple:
    """
    Generate paired or unpaired Van der Pol trajectory images.
    
    Paired mode: Generate HR images, degrade to LR (supervised SR setting)
    Unpaired mode: Generate HR and LR from independent trajectories (unsupervised)
    
    Returns:
        HR tensor, LR tensor, train/val indices
    """
    HR_list, LR_list = [], []
    
    if config.paired_data:
        # Paired: HR and LR are from the same trajectory
        for _ in tqdm(range(config.n_images), desc="Generating paired data"):
            # Random initial conditions and integration time
            t_eval = 5.0 + np.random.rand() * 5.0
            init = (np.random.rand(2) - 0.5) * 6.0
            
            sol = solve_ivp(
                van_der_pol_ode, [0, t_eval], init,
                t_eval=np.linspace(0, t_eval, config.points_per_image)
            )
            points = sol.y.T
            
            if points.shape[0] < 2:
                continue
            
            # Generate HR image
            hr_np = points_to_image(points, resolution=config.hr_resolution)
            hr_t = torch.from_numpy(hr_np[None, None, :, :]).float()
            
            # Degrade to LR with optional blur randomization
            if config.randomize_blur:
                sigma = float(np.random.uniform(config.blur_sigma_min, config.blur_sigma_max))
            else:
                sigma = (config.blur_sigma_min + config.blur_sigma_max) / 2
            
            with torch.no_grad():
                lr_t = degrade_operator(hr_t, blur_sigma=sigma, down_factor=config.upscale_factor)
            
            HR_list.append(hr_np)
            LR_list.append(lr_t.squeeze().numpy())
    else:
        # Unpaired: HR and LR from independent trajectories
        n_per_set = config.n_images // 2
        
        # HR marginal: dense trajectories
        for _ in tqdm(range(n_per_set), desc="Generating HR marginal"):
            t_eval = 5.0 + np.random.rand() * 5.0
            init = (np.random.rand(2) - 0.5) * 6.0
            
            sol = solve_ivp(
                van_der_pol_ode, [0, t_eval], init,
                t_eval=np.linspace(0, t_eval, config.points_per_image * 2)  # Denser for HR
            )
            points = sol.y.T
            
            if points.shape[0] < 2:
                continue
            
            HR_list.append(points_to_image(points, resolution=config.hr_resolution))
        
        # LR marginal: coarser trajectories + degradation
        for _ in tqdm(range(n_per_set), desc="Generating LR marginal"):
            t_eval = 5.0 + np.random.rand() * 5.0
            init = (np.random.rand(2) - 0.5) * 6.0
            
            sol = solve_ivp(
                van_der_pol_ode, [0, t_eval], init,
                t_eval=np.linspace(0, t_eval, config.points_per_image // 2)  # Coarser
            )
            points = sol.y.T
            
            if points.shape[0] < 2:
                continue
            
            lr_np = points_to_image(points, resolution=config.lr_resolution)
            lr_t = torch.from_numpy(lr_np[None, None, :, :]).float()
            
            if config.randomize_blur:
                sigma = float(np.random.uniform(config.blur_sigma_min, config.blur_sigma_max))
            else:
                sigma = (config.blur_sigma_min + config.blur_sigma_max) / 2
            
            with torch.no_grad():
                lr_t = degrade_operator(lr_t, blur_sigma=sigma, down_factor=config.upscale_factor)
            
            LR_list.append(lr_t.squeeze().numpy())
    
    # Convert to tensors
    HR = torch.from_numpy(np.stack(HR_list)).unsqueeze(1).float().to(device)
    LR = torch.from_numpy(np.stack(LR_list)).unsqueeze(1).float().to(device)
    
    # Train/val split
    N = HR.shape[0]
    perm = torch.randperm(N, device=device)
    cut = int(0.85 * N)
    train_idx, val_idx = perm[:cut], perm[cut:]
    
    return HR, LR, train_idx, val_idx


# =============================================================================
# Model Architecture: Small U-Net Denoiser
# =============================================================================

class ConvBlock(nn.Module):
    """Basic double-convolution block with ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    """
    Small U-Net for noise prediction (denoising).
    
    Architecture:
        Input: [x_noisy, sigma_map] concatenated (2 channels)
        Encoder: 2 downsampling stages
        Decoder: 2 upsampling stages with skip connections
        Output: Predicted noise (1 channel)
    
    This is an unconditional denoiser - it receives only the noisy input
    and a sigma map indicating the noise level at each spatial location.
    """
    
    def __init__(self, base_channels: int = 32):
        super().__init__()
        
        # Input: noisy image (1) + sigma map (1) = 2 channels
        in_ch = 2
        base = base_channels
        
        # Encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base * 2, base * 4)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec1 = ConvBlock(base * 4, base * 2)  # +skip = base*4 input
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec2 = ConvBlock(base * 2, base)  # +skip = base*2 input
        
        # Output: predict noise
        self.out = nn.Conv2d(base, 1, 1)
    
    def forward(self, x_noisy: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict noise given noisy input and noise level.
        
        Args:
            x_noisy: Noisy image [B, 1, H, W]
            sigma_map: Noise level map [B, 1, H, W]
        
        Returns:
            Predicted noise [B, 1, H, W]
        """
        x = torch.cat([x_noisy, sigma_map], dim=1)
        
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
        
        return self.out(d2)


# =============================================================================
# Training
# =============================================================================

def sample_sigma(batch_size: int, config: Config, device: torch.device) -> torch.Tensor:
    """Sample random noise levels uniformly from [sigma_min, sigma_max]."""
    return torch.rand(batch_size, device=device) * (config.sigma_max - config.sigma_min) + config.sigma_min


def train_denoiser(
    model: SmallUNet,
    dataloader: DataLoader,
    config: Config,
    device: torch.device,
    name: str = "model"
) -> list:
    """
    Train a denoiser using denoising score matching.
    
    Objective: Given x_noisy = x_0 + sigma * eps, predict eps.
    This trains the model to denoise at various noise levels.
    
    Args:
        model: U-Net denoiser to train
        dataloader: DataLoader with clean images
        config: Training configuration
        device: torch device
        name: Model name for logging
    
    Returns:
        List of loss values per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    losses = []
    
    model.train()
    
    for epoch in range(1, config.n_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)
            B, _, H, W = x_batch.shape
            
            # Sample noise level and noise
            sigma = sample_sigma(B, config, device).view(B, 1, 1, 1)
            eps = torch.randn_like(x_batch)
            
            # Create noisy image
            x_noisy = (x_batch + sigma * eps).clamp(0., 1.)
            
            # Sigma map (broadcast to spatial dimensions)
            sigma_map = sigma.expand(B, 1, H, W)
            
            # Predict noise
            eps_pred = model(x_noisy, sigma_map)
            
            # Loss: MSE between predicted and true noise
            loss = criterion(eps_pred, eps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {name} Epoch {epoch:02d}/{config.n_epochs} | MSE: {avg_loss:.5f}")
    
    return losses


def train_ddib(
    model_q0: SmallUNet,
    model_q1: SmallUNet,
    hr_train: torch.Tensor,
    lr_train: torch.Tensor,
    config: Config,
    device: torch.device
) -> dict:
    """
    Train both DDIB denoisers.
    
    - model_q0: Trained on LR domain (becomes LR expert)
    - model_q1: Trained on HR domain (becomes HR expert)
    """
    # Prepare LR data (optionally upscale for single-scale training)
    if config.train_on_upscaled_lr:
        lr_input = F.interpolate(lr_train, scale_factor=config.upscale_factor, 
                                  mode='bilinear', align_corners=False)
    else:
        lr_input = lr_train
    
    # Create dataloaders
    lr_loader = DataLoader(
        TensorDataset(lr_input),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    hr_loader = DataLoader(
        TensorDataset(hr_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Train q0 on LR domain
    print("\n[Training] q0 (LR domain expert)")
    losses_q0 = train_denoiser(model_q0, lr_loader, config, device, name="q0")
    
    # Train q1 on HR domain
    print("\n[Training] q1 (HR domain expert)")
    losses_q1 = train_denoiser(model_q1, hr_loader, config, device, name="q1")
    
    return {'q0': losses_q0, 'q1': losses_q1}


# =============================================================================
# Physics-Based Energy Guidance (EDDIB Extension)
# =============================================================================

def image_to_points(img_tensor: torch.Tensor, n_samples: int = 140) -> list:
    """
    Extract trajectory points from image by thresholding.
    
    This is a simple inverse of points_to_image for computing physics residuals.
    """
    B, _, H, W = img_tensor.shape
    points_list = []
    arr = img_tensor.detach().cpu().numpy()
    
    for b in range(B):
        img = arr[b, 0]
        ys, xs = np.where(img > 0.1)  # Threshold
        
        if len(xs) > 1:
            # Sample points
            idx = np.random.choice(len(xs), min(n_samples, len(xs)), replace=False)
            # Scale to original coordinate bounds
            pts_x = -4 + (xs[idx] / W) * 8
            pts_y = -4 + (ys[idx] / H) * 8
            points_list.append(np.stack([pts_x, pts_y], axis=1))
        else:
            points_list.append(np.zeros((n_samples, 2), dtype=np.float32))
    
    return points_list


def compute_physics_residual(img_tensor: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Compute Van der Pol ODE residual for physics-informed guidance.
    
    Measures how well the generated trajectory satisfies the ODE.
    Lower residual = more physically consistent.
    """
    points_list = image_to_points(img_tensor)
    residuals = []
    
    for pts in points_list:
        if pts.shape[0] < 2:
            residuals.append(0.0)
            continue
        
        res = 0.0
        for i in range(1, pts.shape[0]):
            y = pts[i - 1]
            dy = np.array(van_der_pol_ode(0, y)) * dt
            pred = pts[i - 1] + dy
            res += np.mean((pts[i] - pred) ** 2)
        
        residuals.append(res / max(1, pts.shape[0] - 1))
    
    return torch.tensor(residuals, device=img_tensor.device, dtype=torch.float32)


def approx_energy_gradient(
    x: torch.Tensor,
    residual_fn,
    n_directions: int = 12,
    eps: float = 1e-3
) -> torch.Tensor:
    """
    Approximate gradient of energy function using finite differences.
    
    Uses random direction sampling for efficient gradient estimation
    in high-dimensional spaces (SPSA-like approach).
    """
    B = x.size(0)
    base_res = residual_fn(x).detach()
    grad_est = torch.zeros_like(x)
    
    for _ in range(n_directions):
        # Random unit direction
        u = torch.randn_like(x)
        u_norm = u.view(B, -1).norm(dim=1).view(B, 1, 1, 1)
        u = u / (u_norm + 1e-12)
        
        # Finite difference
        x_perturbed = (x + eps * u).clamp(0., 1.)
        res_perturbed = residual_fn(x_perturbed).detach()
        
        diff = (res_perturbed - base_res) / eps
        
        # Broadcast to image dimensions
        while diff.ndim < x.ndim:
            diff = diff.unsqueeze(-1)
        diff = diff.expand_as(u)
        
        grad_est += diff * u
    
    grad_est /= float(n_directions)
    return grad_est


# =============================================================================
# Inference: DDIB Encode-Bridge-Decode
# =============================================================================

@torch.no_grad()
def ddib_inference(
    model_q0: SmallUNet,
    model_q1: SmallUNet,
    lr_images: torch.Tensor,
    config: Config,
    device: torch.device
) -> torch.Tensor:
    """
    DDIB inference: Encode-Bridge-Decode super-resolution.
    
    Three stages:
    1. ENCODE: Transform LR image to latent space using q0 (add noise gradually)
    2. BRIDGE: Upscale the latent representation to HR resolution
    3. DECODE: Denoise using q1 to get final HR image (remove noise gradually)
    
    This is similar to solving a probability flow ODE in two steps.
    """
    model_q0.eval()
    model_q1.eval()
    
    # Ensure correct input format
    lr_images = lr_images.to(device)
    if lr_images.dim() == 3:
        lr_images = lr_images.unsqueeze(1)
    
    # Sigma schedules
    sigmas_encode = np.linspace(config.sigma_min, config.sigma_max, config.inference_steps).astype(np.float32)
    sigmas_decode = np.linspace(config.sigma_max, config.sigma_min, config.inference_steps).astype(np.float32)
    
    # Optionally upscale LR input for encoding
    if config.train_on_upscaled_lr:
        x = F.interpolate(lr_images, scale_factor=config.upscale_factor, 
                          mode='bilinear', align_corners=False)
    else:
        x = lr_images.clone()
    
    # Stage 1: ENCODE (gradually add noise using q0)
    for i in range(len(sigmas_encode) - 1):
        s_cur = float(sigmas_encode[i])
        s_next = float(sigmas_encode[i + 1])
        
        sigma_map = torch.full_like(x, s_cur)
        eps_pred = model_q0(x, sigma_map)
        
        # Euler step in probability flow ODE
        x = (x + (s_next - s_cur) * eps_pred).clamp(0., 1.)
    
    z_latent = x
    
    # Stage 2: BRIDGE (upscale to HR resolution)
    if config.train_on_upscaled_lr:
        z_hr = z_latent  # Already at HR resolution
    else:
        z_hr = F.interpolate(z_latent, scale_factor=config.upscale_factor,
                             mode='bilinear', align_corners=False)
    
    x_hr = z_hr.clone()
    
    # Stage 3: DECODE (gradually denoise using q1)
    for i in range(len(sigmas_decode) - 1):
        s_cur = float(sigmas_decode[i])
        s_next = float(sigmas_decode[i + 1])
        
        sigma_map = torch.full_like(x_hr, s_cur)
        eps_pred = model_q1(x_hr, sigma_map)
        
        # Optional: Energy guidance (EDDIB)
        if config.use_energy:
            grad_energy = approx_energy_gradient(
                x_hr, compute_physics_residual,
                n_directions=config.energy_directions,
                eps=config.energy_eps
            )
            grad_energy = torch.clamp(grad_energy, -1.0, 1.0)
            eps_pred = eps_pred - config.energy_scale * grad_energy
        
        # Euler step
        x_hr = (x_hr + (s_next - s_cur) * eps_pred).clamp(0., 1.)
    
    return x_hr


# =============================================================================
# Evaluation & Visualization
# =============================================================================

def evaluate_results(
    sr_images: torch.Tensor,
    hr_ground_truth: torch.Tensor,
    config: Config
) -> dict:
    """Compute evaluation metrics: SSIM and physics residual."""
    
    def to_np(t):
        return t.detach().cpu().numpy()
    
    # SSIM scores
    ssim_scores = []
    for i in range(sr_images.size(0)):
        hr_np = to_np(hr_ground_truth[i, 0])
        sr_np = to_np(sr_images[i, 0])
        
        # Resize if needed
        if hr_np.shape != sr_np.shape:
            sr_resized = F.interpolate(
                sr_images[i:i+1], 
                size=hr_np.shape, 
                mode='bilinear', 
                align_corners=False
            )
            sr_np = to_np(sr_resized[0, 0])
        
        try:
            score = ssim(hr_np, sr_np, data_range=1.0)
        except Exception as e:
            warnings.warn(f"SSIM failed for sample {i}: {e}")
            score = float('nan')
        
        ssim_scores.append(score)
    
    # Physics residuals
    physics_residuals = compute_physics_residual(sr_images).cpu().numpy()
    
    return {
        'ssim': ssim_scores,
        'ssim_mean': np.nanmean(ssim_scores),
        'physics_residual': physics_residuals,
        'physics_mean': np.mean(physics_residuals)
    }


def plot_training_curves(history: dict, save_path: str = None) -> None:
    """Plot training loss curves for both models."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['q0'], label='q0 (LR expert)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('LR Denoiser Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['q1'], label='q1 (HR expert)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('HR Denoiser Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_results(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_ground_truth: torch.Tensor,
    metrics: dict,
    n_samples: int = 3,
    save_path: str = None
) -> None:
    """Visualize super-resolution results."""
    n = min(n_samples, sr_images.size(0))
    
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    fig.suptitle('DDIB Super-Resolution Results', fontsize=14)
    
    for i in range(n):
        # LR input
        axes[i, 0].imshow(lr_images[i, 0].cpu(), cmap='viridis')
        axes[i, 0].set_title('LR Input')
        axes[i, 0].axis('off')
        
        # SR output
        ssim_str = f"{metrics['ssim'][i]:.3f}" if not np.isnan(metrics['ssim'][i]) else "N/A"
        phys_str = f"{metrics['physics_residual'][i]:.3f}"
        axes[i, 1].imshow(sr_images[i, 0].cpu(), cmap='viridis')
        axes[i, 1].set_title(f'DDIB Output\nSSIM: {ssim_str}, Phys: {phys_str}')
        axes[i, 1].axis('off')
        
        # HR ground truth
        axes[i, 2].imshow(hr_ground_truth[i, 0].cpu(), cmap='viridis')
        axes[i, 2].set_title('HR Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training and evaluation pipeline."""
    
    # Setup
    config = Config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data mode: {'Paired' if config.paired_data else 'Unpaired'}")
    print(f"Energy guidance: {'Enabled' if config.use_energy else 'Disabled'}")
    
    # Generate dataset
    print("\n[1/4] Generating dataset...")
    HR, LR, train_idx, val_idx = create_dataset(config, device)
    print(f"  HR shape: {HR.shape}, LR shape: {LR.shape}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Split data
    hr_train, hr_val = HR[train_idx], HR[val_idx]
    lr_train, lr_val = LR[train_idx], LR[val_idx]
    
    # Upscale LR for visualization
    lr_val_up = F.interpolate(lr_val, scale_factor=config.upscale_factor,
                               mode='bilinear', align_corners=False)
    
    # Initialize models
    print("\n[2/4] Initializing models...")
    model_q0 = SmallUNet(config.base_channels).to(device)
    model_q1 = SmallUNet(config.base_channels).to(device)
    
    n_params = sum(p.numel() for p in model_q0.parameters())
    print(f"  Parameters per model: {n_params / 1e6:.3f}M")
    
    # Train
    print("\n[3/4] Training DDIB models...")
    history = train_ddib(model_q0, model_q1, hr_train, lr_train, config, device)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Inference and evaluation
    print("\n[4/4] Running inference on validation set...")
    n_eval = min(6, hr_val.size(0))
    idx = torch.randperm(hr_val.size(0))[:n_eval]
    
    lr_samples = lr_val_up[idx] if config.train_on_upscaled_lr else lr_val[idx]
    hr_samples = hr_val[idx]
    
    sr_output = ddib_inference(model_q0, model_q1, lr_samples, config, device)
    
    # Evaluate
    metrics = evaluate_results(sr_output, hr_samples, config)
    print(f"\nResults:")
    print(f"  SSIM: {[f'{s:.3f}' for s in metrics['ssim']]} | Mean: {metrics['ssim_mean']:.3f}")
    print(f"  Physics: {[f'{p:.3f}' for p in metrics['physics_residual']]} | Mean: {metrics['physics_mean']:.3f}")
    
    # Visualize
    plot_results(lr_val_up[idx], sr_output, hr_samples, metrics)
    
    print("\nDone!")
    
    return model_q0, model_q1, history


if __name__ == "__main__":
    model_q0, model_q1, history = main()
