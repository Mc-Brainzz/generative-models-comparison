"""
CycleGAN-style Super-Resolution for Truly Unpaired Data

This implementation uses a "degradation-aware" approach:
- Generator G: LR → HR (learned)
- Degradation D_op: HR → LR (KNOWN - blur + downsample, not learned)
- Discriminator D: Real HR vs Fake HR

Losses:
1. Adversarial Loss: Make generated HR look like real HR distribution
2. Cycle Consistency Loss: G(LR) → degrade → should ≈ LR
3. (Optional) Identity Loss: G(HR_upsampled) ≈ HR_upsampled

This is compared against OT-based Flow Matching for truly unpaired SR.

Author: Research Comparison Study
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
from dataclasses import dataclass
import time
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for CycleGAN-style SR."""
    # Reproducibility
    seed: int = 123
    
    # Data generation
    n_images: int = 600
    points_per_image: int = 140
    hr_resolution: int = 128
    downsample_factor: int = 4
    
    # Degradation (same as your Flow Matching setup)
    blur_sigma: float = 1.0
    blur_radius: int = 3
    
    # Dataset split
    train_ratio: float = 0.85
    
    # Training
    epochs: int = 50  # GANs often need more epochs
    batch_size: int = 16
    lr_g: float = 2e-4  # Generator learning rate
    lr_d: float = 2e-4  # Discriminator learning rate
    beta1: float = 0.5  # Adam beta1 (standard for GANs)
    beta2: float = 0.999
    
    # Loss weights
    lambda_cycle: float = 10.0  # Cycle consistency weight
    lambda_identity: float = 0.0  # Identity loss weight (0 = disabled)
    lambda_edge: float = 4.0  # Edge consistency on degraded LR
    lambda_tv: float = 1e-5  # TV regularization on generated HR
    gan_start_epoch: int = 3  # Warmup with cycle/edge before adversarial loss
    
    # Architecture
    base_channels: int = 64
    n_residual_blocks: int = 6
    
    # Discriminator
    n_discriminator_layers: int = 3
    
    # For truly unpaired: different mu ranges for HR and LR sources
    hr_mu_range: tuple = (0.5, 1.5)
    lr_mu_range: tuple = (1.0, 2.0)


config = Config()


# =============================================================================
# Data Generation (Same as your Flow Matching)
# =============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def van_der_pol_trajectory(mu=1.0, x0=0.1, v0=0.0, t_span=(0, 50), n_points=1000):
    """Generate Van der Pol oscillator trajectory."""
    def dynamics(t, y):
        x, v = y
        dxdt = v
        dvdt = mu * (1 - x**2) * v - x
        return [dxdt, dvdt]
    
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(dynamics, t_span, [x0, v0], t_eval=t_eval, method='RK45')
    return sol.y[0], sol.y[1]


def trajectory_to_image(x, v, resolution=128, sigma=1.5):
    """Convert trajectory to grayscale image."""
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8) * (resolution - 1)
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8) * (resolution - 1)
    
    image = np.zeros((resolution, resolution), dtype=np.float32)
    
    for xi, vi in zip(x_norm, v_norm):
        px, py = int(np.clip(xi, 0, resolution-1)), int(np.clip(vi, 0, resolution-1))
        image[py, px] = 1.0
    
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=sigma)
        if image.max() > 0:
            image = image / image.max()
    
    return image


def create_gaussian_kernel(radius, sigma):
    """Create a Gaussian blur kernel."""
    size = 2 * radius + 1
    x = torch.arange(size, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def degrade_image(hr_image, blur_kernel, downsample_factor):
    """
    Apply degradation: blur + downsample.
    This is the KNOWN forward operator.
    """
    if hr_image.dim() == 2:
        hr_image = hr_image.unsqueeze(0).unsqueeze(0)
    elif hr_image.dim() == 3:
        hr_image = hr_image.unsqueeze(1)
    
    kernel = blur_kernel.unsqueeze(0).unsqueeze(0)
    padding = blur_kernel.shape[0] // 2
    
    blurred = F.conv2d(hr_image, kernel, padding=padding)
    lr_image = F.avg_pool2d(blurred, kernel_size=downsample_factor)
    
    return lr_image.squeeze()


def generate_dataset(n_images, mu_range, seed_offset=0, config=config):
    """Generate dataset of trajectory images."""
    np.random.seed(config.seed + seed_offset)
    
    images = []
    for i in range(n_images):
        mu = np.random.uniform(*mu_range)
        x0 = np.random.uniform(-0.5, 0.5)
        v0 = np.random.uniform(-0.5, 0.5)
        
        x, v = van_der_pol_trajectory(mu=mu, x0=x0, v0=v0, 
                                       n_points=config.points_per_image)
        img = trajectory_to_image(x, v, resolution=config.hr_resolution)
        images.append(img)
    
    return np.stack(images)


# =============================================================================
# Generator Architecture (ResNet-style)
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator for LR → HR super-resolution.
    
    Architecture:
    1. Initial feature extraction (at LR resolution)
    2. Upsampling to HR resolution
    3. Residual blocks for refinement
    4. Output projection
    """
    def __init__(self, config):
        super().__init__()
        
        c = config.base_channels
        scale = config.downsample_factor
        
        # Initial convolution (LR resolution)
        self.initial = nn.Sequential(
            nn.Conv2d(1, c, 7, padding=3),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),  # 2x
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),  # 4x total
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks (HR resolution)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(c) for _ in range(config.n_residual_blocks)]
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 7, padding=3),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, lr):
        """
        Args:
            lr: Low-resolution input [B, 1, H/4, W/4]
        Returns:
            hr: High-resolution output [B, 1, H, W]
        """
        x = self.initial(lr)
        x = self.upsample(x)
        x = self.residual_blocks(x)
        return self.output(x)


# =============================================================================
# Discriminator Architecture (PatchGAN)
# =============================================================================

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    
    Outputs a map of real/fake predictions for each patch,
    rather than a single scalar. This helps with training stability.
    """
    def __init__(self, config):
        super().__init__()
        
        c = config.base_channels
        n_layers = config.n_discriminator_layers
        
        layers = [
            nn.Conv2d(1, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(c * mult_prev, c * mult, 4, stride=2, padding=1),
                nn.InstanceNorm2d(c * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        
        # Final layer
        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(c * mult_prev, c * mult, 4, stride=1, padding=1),
            nn.InstanceNorm2d(c * mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * mult, 1, 4, stride=1, padding=1),
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Returns patch-wise real/fake predictions."""
        return self.model(x)


# =============================================================================
# Loss Functions
# =============================================================================

class GANLoss(nn.Module):
    """GAN loss with label smoothing option."""
    def __init__(self, use_lsgan=True):
        super().__init__()
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target_is_real):
        if target_is_real:
            target = torch.ones_like(pred) * 0.9  # Label smoothing
        else:
            target = torch.zeros_like(pred)
        return self.loss(pred, target)


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude for edge-aware consistency."""
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation regularization to suppress checkerboard artifacts."""
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


# =============================================================================
# Known Degradation Operator (for cycle consistency)
# =============================================================================

class DegradationOperator(nn.Module):
    """
    The KNOWN degradation: HR → LR via blur + downsample.
    This is NOT learned - it's fixed and differentiable.
    """
    def __init__(self, config):
        super().__init__()
        self.downsample_factor = config.downsample_factor
        
        # Create fixed Gaussian blur kernel
        kernel = create_gaussian_kernel(config.blur_radius, config.blur_sigma)
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.padding = config.blur_radius
    
    def forward(self, hr):
        """
        Args:
            hr: High-resolution image [B, 1, H, W]
        Returns:
            lr: Low-resolution image [B, 1, H/4, W/4]
        """
        # Apply Gaussian blur
        blurred = F.conv2d(hr, self.kernel, padding=self.padding)
        # Downsample
        lr = F.avg_pool2d(blurred, kernel_size=self.downsample_factor)
        return lr


# =============================================================================
# Training Loop
# =============================================================================

def train_cyclegan_sr(config):
    """
    Train CycleGAN-style super-resolution on truly unpaired data.
    
    Key difference from standard CycleGAN:
    - We KNOW the degradation (HR → LR), so we don't learn it
    - Only learn G: LR → HR
    - Cycle consistency: degrade(G(LR)) ≈ LR
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(config.seed)
    
    # =========================================================================
    # Generate TRULY UNPAIRED data
    # =========================================================================
    print("\n" + "="*60)
    print("GENERATING TRULY UNPAIRED DATA")
    print("="*60)
    
    # HR images from one distribution
    print(f"Generating HR images with μ ∈ {config.hr_mu_range}...")
    hr_images = generate_dataset(config.n_images, config.hr_mu_range, seed_offset=0)
    
    # LR images from DIFFERENT distribution (truly unpaired!)
    print(f"Generating source images for LR with μ ∈ {config.lr_mu_range}...")
    lr_source_images = generate_dataset(config.n_images, config.lr_mu_range, seed_offset=1000)
    
    # Create blur kernel and degrade LR sources
    blur_kernel = create_gaussian_kernel(config.blur_radius, config.blur_sigma)
    
    lr_images = []
    for img in lr_source_images:
        img_t = torch.tensor(img, dtype=torch.float32)
        lr_img = degrade_image(img_t, blur_kernel, config.downsample_factor)
        lr_images.append(lr_img.numpy())
    lr_images = np.stack(lr_images)
    
    print(f"HR shape: {hr_images.shape}, LR shape: {lr_images.shape}")
    print(f"NOTE: HR and LR are from DIFFERENT trajectories - truly unpaired!")
    
    # Split into train/test
    n_train = int(config.n_images * config.train_ratio)
    
    hr_train = torch.tensor(hr_images[:n_train], dtype=torch.float32).unsqueeze(1)
    lr_train = torch.tensor(lr_images[:n_train], dtype=torch.float32).unsqueeze(1)
    hr_test = torch.tensor(hr_images[n_train:], dtype=torch.float32).unsqueeze(1)
    lr_test = torch.tensor(lr_images[n_train:], dtype=torch.float32).unsqueeze(1)
    
    # For evaluation, we need paired test data (same trajectory)
    print("\nGenerating PAIRED test data for evaluation...")
    test_hr = generate_dataset(50, (0.8, 1.2), seed_offset=9999)
    test_lr = []
    for img in test_hr:
        img_t = torch.tensor(img, dtype=torch.float32)
        lr_img = degrade_image(img_t, blur_kernel, config.downsample_factor)
        test_lr.append(lr_img.numpy())
    test_lr = np.stack(test_lr)
    
    paired_test_hr = torch.tensor(test_hr, dtype=torch.float32).unsqueeze(1).to(device)
    paired_test_lr = torch.tensor(test_lr, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Create data loaders (shuffle independently for truly unpaired)
    hr_loader = DataLoader(TensorDataset(hr_train), batch_size=config.batch_size, shuffle=True)
    lr_loader = DataLoader(TensorDataset(lr_train), batch_size=config.batch_size, shuffle=True)
    
    # =========================================================================
    # Initialize models
    # =========================================================================
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60)
    
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    degradation = DegradationOperator(config).to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr_g, 
                                    betas=(config.beta1, config.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d,
                                    betas=(config.beta1, config.beta2))
    
    # Loss functions
    criterion_GAN = GANLoss(use_lsgan=True)
    criterion_cycle = nn.L1Loss()
    
    # =========================================================================
    # Training
    # =========================================================================
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = {
        'g_loss': [], 'd_loss': [], 'cycle_loss': [], 'edge_loss': [],
        'ssim': [], 'psnr': []
    }
    
    for epoch in range(config.epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_cycle_loss = 0
        epoch_edge_loss = 0
        n_batches = 0
        
        # Iterate through both loaders (truly unpaired - different shuffling)
        hr_iter = iter(hr_loader)
        lr_iter = iter(lr_loader)
        
        pbar = tqdm(range(min(len(hr_loader), len(lr_loader))), 
                    desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for _ in pbar:
            try:
                real_hr = next(hr_iter)[0].to(device)
                real_lr = next(lr_iter)[0].to(device)
            except StopIteration:
                break
            
            batch_size = min(real_hr.size(0), real_lr.size(0))
            real_hr = real_hr[:batch_size]
            real_lr = real_lr[:batch_size]
            
            # =================================================================
            # Train Generator
            # =================================================================
            optimizer_G.zero_grad()
            
            # Generate fake HR from LR
            fake_hr = generator(real_lr)
            
            # Adversarial loss: fool discriminator
            if (epoch + 1) >= config.gan_start_epoch:
                pred_fake = discriminator(fake_hr)
                loss_GAN = criterion_GAN(pred_fake, True)
            else:
                loss_GAN = torch.tensor(0.0, device=device)
            
            # Cycle consistency loss: degrade(G(LR)) ≈ LR
            reconstructed_lr = degradation(fake_hr)
            loss_cycle = criterion_cycle(reconstructed_lr, real_lr) * config.lambda_cycle

            # Edge consistency in LR space: preserve trajectory structure
            edge_real = gradient_magnitude(real_lr)
            edge_recon = gradient_magnitude(reconstructed_lr)
            loss_edge = criterion_cycle(edge_recon, edge_real) * config.lambda_edge
            
            # Identity loss (optional): G(upsample(HR)) ≈ upsample(HR)
            loss_identity = 0
            if config.lambda_identity > 0:
                # Downsample HR to LR size, then try to reconstruct
                hr_as_lr = degradation(real_hr)
                identity_hr = generator(hr_as_lr)
                loss_identity = criterion_cycle(identity_hr, real_hr) * config.lambda_identity

            # TV regularization: discourage high-frequency artifacts
            loss_tv = total_variation_loss(fake_hr) * config.lambda_tv
            
            # Total generator loss
            loss_G = loss_GAN + loss_cycle + loss_edge + loss_identity + loss_tv
            loss_G.backward()
            optimizer_G.step()
            
            # =================================================================
            # Train Discriminator
            # =================================================================
            if (epoch + 1) >= config.gan_start_epoch:
                optimizer_D.zero_grad()
                
                # Real HR
                pred_real = discriminator(real_hr)
                loss_D_real = criterion_GAN(pred_real, True)
                
                # Fake HR (detached)
                pred_fake = discriminator(fake_hr.detach())
                loss_D_fake = criterion_GAN(pred_fake, False)
                
                # Total discriminator loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
            else:
                loss_D = torch.tensor(0.0, device=device)
            
            # Track losses
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            epoch_cycle_loss += loss_cycle.item()
            epoch_edge_loss += loss_edge.item()
            n_batches += 1
            
            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D': f'{loss_D.item():.3f}',
                'Cyc': f'{loss_cycle.item():.3f}',
                'Edge': f'{loss_edge.item():.3f}'
            })
        
        # Average losses
        history['g_loss'].append(epoch_g_loss / n_batches)
        history['d_loss'].append(epoch_d_loss / n_batches)
        history['cycle_loss'].append(epoch_cycle_loss / n_batches)
        history['edge_loss'].append(epoch_edge_loss / n_batches)
        
        # Evaluate on paired test data
        generator.eval()
        with torch.no_grad():
            test_fake_hr = generator(paired_test_lr)
            
            # Compute metrics
            ssim_vals = []
            psnr_vals = []
            for i in range(len(paired_test_hr)):
                hr_np = paired_test_hr[i, 0].cpu().numpy()
                fake_np = test_fake_hr[i, 0].cpu().numpy()
                
                ssim_val = ssim(hr_np, fake_np, data_range=1.0)
                mse = np.mean((hr_np - fake_np) ** 2)
                psnr_val = 10 * np.log10(1.0 / (mse + 1e-10))
                
                ssim_vals.append(ssim_val)
                psnr_vals.append(psnr_val)
            
            avg_ssim = np.mean(ssim_vals)
            avg_psnr = np.mean(psnr_vals)
            
            history['ssim'].append(avg_ssim)
            history['psnr'].append(avg_psnr)
        
        print(f"  Eval - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB")
    
    return generator, discriminator, history, (paired_test_lr, paired_test_hr)


# =============================================================================
# Evaluation and Visualization
# =============================================================================

def evaluate_and_compare(generator, test_data, config):
    """Evaluate CycleGAN results and compare with baselines."""
    device = next(generator.parameters()).device
    test_lr, test_hr = test_data
    
    generator.eval()
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    with torch.no_grad():
        # CycleGAN super-resolution
        cyclegan_hr = generator(test_lr)
        
        # Bicubic baseline
        bicubic_hr = F.interpolate(test_lr, scale_factor=config.downsample_factor, 
                                   mode='bicubic', align_corners=False)
    
    # Compute metrics
    results = {'method': [], 'ssim': [], 'psnr': []}
    
    for name, pred in [('CycleGAN', cyclegan_hr), ('Bicubic', bicubic_hr)]:
        ssim_vals = []
        psnr_vals = []
        
        for i in range(len(test_hr)):
            hr_np = test_hr[i, 0].cpu().numpy()
            pred_np = pred[i, 0].cpu().numpy()
            
            ssim_val = ssim(hr_np, pred_np, data_range=1.0)
            mse = np.mean((hr_np - pred_np) ** 2)
            psnr_val = 10 * np.log10(1.0 / (mse + 1e-10))
            
            ssim_vals.append(ssim_val)
            psnr_vals.append(psnr_val)
        
        results['method'].append(name)
        results['ssim'].append(np.mean(ssim_vals))
        results['psnr'].append(np.mean(psnr_vals))
        
        print(f"{name:12s} - SSIM: {np.mean(ssim_vals):.4f}, PSNR: {np.mean(psnr_vals):.2f} dB")
    
    return results, cyclegan_hr, bicubic_hr


def visualize_results(test_lr, test_hr, cyclegan_hr, bicubic_hr, history, n_samples=4):
    """Visualize training history and sample results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Training curves
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history['g_loss'], label='Generator', color='blue')
    ax1.plot(history['d_loss'], label='Discriminator', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('GAN Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(history['cycle_loss'], label='Cycle Loss', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Cycle Consistency Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(history['ssim'], label='SSIM', color='purple')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.set_title('Validation SSIM')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Sample reconstructions
    for i in range(min(n_samples, 3)):
        ax = fig.add_subplot(2, 3, 4 + i)
        
        lr_up = F.interpolate(test_lr[i:i+1], scale_factor=4, mode='nearest')[0, 0].cpu().numpy()
        hr = test_hr[i, 0].cpu().numpy()
        cyc = cyclegan_hr[i, 0].cpu().numpy()
        bic = bicubic_hr[i, 0].cpu().numpy()
        
        # Create comparison grid
        combined = np.zeros((128, 128*4 + 30))
        combined[:, 0:128] = lr_up
        combined[:, 138:266] = bic
        combined[:, 276:404] = cyc
        combined[:, 414:542] = hr
        
        ax.imshow(combined, cmap='viridis')
        ax.set_title(f'Sample {i+1}: LR | Bicubic | CycleGAN | GT')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cyclegan_sr_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved to 'cyclegan_sr_results.png'")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CycleGAN-Style Super-Resolution for Truly Unpaired Data")
    print("="*60)
    print("\nThis experiment tests whether CycleGAN can overcome OT limitations")
    print("by learning distribution-to-distribution mapping instead of")
    print("relying on mini-batch optimal transport matching.")
    print()
    
    # Train
    start_time = time.time()
    generator, discriminator, history, test_data = train_cyclegan_sr(config)
    train_time = time.time() - start_time
    print(f"\nTotal training time: {train_time/60:.1f} minutes")
    
    # Evaluate
    results, cyclegan_hr, bicubic_hr = evaluate_and_compare(generator, test_data, config)
    
    # Visualize
    test_lr, test_hr = test_data
    visualize_results(test_lr, test_hr, cyclegan_hr.cpu(), bicubic_hr.cpu(), history)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nCompare these results with your Flow Matching + OT results!")
    print("Key questions to answer:")
    print("1. Does CycleGAN achieve higher SSIM than OT-based Flow Matching?")
    print("2. Is training more stable? Check the loss curves.")
    print("3. Do the generated images look more realistic?")
