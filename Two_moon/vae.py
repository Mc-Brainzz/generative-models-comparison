"""
Variational Autoencoder (VAE) for 2D Data Generation

A clean, modular implementation of VAE for learning 2D distributions (e.g., two moons).
Designed for research reproducibility and easy comparison with other generative models.

Key components:
    - VAE with reparameterization trick
    - ELBO loss = Reconstruction (MSE) + KL Divergence
    - Optional beta-annealing for improved latent space structure

References:
    - Kingma & Welling (2014): Auto-Encoding Variational Bayes
    - Higgins et al. (2017): beta-VAE for disentangled representations

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Hyperparameters and settings for VAE training."""
    
    # Data
    n_samples: int = 2000
    noise_level: float = 0.05
    batch_size: int = 128
    
    # Model architecture
    input_dim: int = 2
    latent_dim: int = 2  # 2D latent space for visualization
    hidden_dim: int = 256
    depth: int = 3  # Number of hidden layers
    
    # Training
    learning_rate: float = 1e-3
    n_epochs: int = 2000
    
    # Beta annealing: gradually increase KL weight to improve training stability
    # Starting low encourages reconstruction first, then regularizes latent space
    beta_start: float = 0.001
    beta_end: float = 1.0
    beta_anneal_fraction: float = 0.75  # Fraction of training for annealing
    
    # Reproducibility
    seed: int = 42
    
    # Visualization
    plot_limit: float = 3.0
    n_samples_viz: int = 2000


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch and numpy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data Generation
# =============================================================================

def generate_two_moons_data(
    n_samples: int,
    noise: float,
    seed: int,
    device: torch.device
) -> tuple[torch.Tensor, StandardScaler]:
    """
    Generate standardized two-moons dataset.
    
    The two moons distribution is a common benchmark for generative models
    because it has a non-trivial, non-Gaussian structure that tests whether
    models can capture multi-modal distributions.
    
    Args:
        n_samples: Number of data points to generate
        noise: Standard deviation of Gaussian noise added to points
        seed: Random seed for reproducibility
        device: torch device for tensor placement
    
    Returns:
        X_tensor: Standardized data as torch tensor on device
        scaler: Fitted StandardScaler for inverse transforms
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    
    # Standardize to zero mean, unit variance for stable training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    return X_tensor, scaler


def create_dataloader(
    X: torch.Tensor,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Wrap tensor data in a DataLoader for batched iteration."""
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# Model Definition
# =============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder with configurable depth.
    
    Architecture:
        Encoder: x -> hidden layers -> (mu, log_var)
        Latent: z ~ N(mu, exp(log_var)) via reparameterization
        Decoder: z -> hidden layers -> x_reconstructed
    
    The reparameterization trick (z = mu + eps * sigma where eps ~ N(0,1))
    allows gradients to flow through the sampling operation.
    
    Args:
        input_dim: Dimensionality of input data
        latent_dim: Dimensionality of latent space
        hidden_dim: Width of hidden layers
        depth: Number of hidden layers in encoder/decoder
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 2,
        hidden_dim: int = 256,
        depth: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder: maps input to hidden representation
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
        for _ in range(depth - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Separate heads for mean and log-variance of approximate posterior
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Build decoder: maps latent code back to input space
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
        decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
        for _ in range(depth - 1):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
        # No activation on output: decoder predicts mean of Gaussian likelihood
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample from q(z|x) using the reparameterization trick.
        
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        This makes the sampling differentiable w.r.t. mu and logvar.
        """
        std = torch.exp(0.5 * logvar)  # sigma = sqrt(var) = exp(0.5 * log(var))
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstructed input."""
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> reparameterize -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples by decoding from the prior p(z) = N(0, I)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)


# =============================================================================
# Loss Function
# =============================================================================

def compute_vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss: negative ELBO = Reconstruction + beta * KL.
    
    The Evidence Lower Bound (ELBO) decomposes into:
        - Reconstruction term: E_q[log p(x|z)] (how well decoder reconstructs)
        - KL term: KL(q(z|x) || p(z)) (how close posterior is to prior)
    
    We minimize -ELBO, which equals:
        Reconstruction Loss + KL Divergence
    
    Args:
        x_recon: Reconstructed data from decoder
        x: Original input data
        mu: Mean of approximate posterior q(z|x)
        logvar: Log-variance of approximate posterior
        beta: Weight for KL term (beta=1 is standard VAE, beta<1 is beta-VAE)
    
    Returns:
        total_loss: Combined loss (summed over batch)
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component (unweighted)
    """
    # Reconstruction loss: MSE assumes Gaussian likelihood with fixed variance
    # Using sum reduction for proper scaling with beta
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence from N(mu, sigma^2) to N(0, 1)
    # Analytical form: KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# =============================================================================
# Training
# =============================================================================

def compute_beta(epoch: int, config: Config) -> float:
    """
    Compute beta value with linear annealing schedule.
    
    Beta annealing helps VAE training by:
    1. Initially low beta: focus on reconstruction, learn good encodings
    2. Gradually increase: enforce latent space regularization
    
    This often leads to better latent space structure than fixed beta=1.
    """
    anneal_epochs = int(config.n_epochs * config.beta_anneal_fraction)
    if epoch < anneal_epochs:
        return config.beta_start + (config.beta_end - config.beta_start) * (epoch / anneal_epochs)
    return config.beta_end


def train_vae(
    model: VAE,
    dataloader: DataLoader,
    config: Config,
    device: torch.device
) -> dict:
    """
    Train the VAE model.
    
    Args:
        model: VAE model to train
        dataloader: DataLoader with training data
        config: Training configuration
        device: torch device
    
    Returns:
        Dictionary containing training history (losses per epoch)
    """
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    model.train()
    progress_bar = tqdm(range(config.n_epochs), desc="Training VAE")
    
    for epoch in progress_bar:
        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0
        
        beta = compute_beta(epoch, config)
        
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)
            batch_size = x_batch.shape[0]
            
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(x_batch)
            total_loss, recon_loss, kl_loss = compute_vae_loss(
                x_recon, x_batch, mu, logvar, beta=beta
            )
            
            # Average loss over batch for gradient computation
            (total_loss / batch_size).backward()
            optimizer.step()
            
            # Track losses (per-sample averages)
            epoch_total += total_loss.item() / batch_size
            epoch_recon += recon_loss.item() / batch_size
            epoch_kl += kl_loss.item() / batch_size
            n_batches += 1
        
        # Record epoch averages
        history['total_loss'].append(epoch_total / n_batches)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['kl_loss'].append(epoch_kl / n_batches)
        
        if (epoch + 1) % 100 == 0:
            progress_bar.set_postfix({
                'loss': f"{history['total_loss'][-1]:.3f}",
                'recon': f"{history['recon_loss'][-1]:.3f}",
                'kl': f"{history['kl_loss'][-1]:.3f}",
                'beta': f"{beta:.3f}"
            })
    
    return history


# =============================================================================
# Visualization
# =============================================================================

def plot_training_curves(history: dict, save_path: str = None) -> None:
    """Plot training loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['total_loss'], label='Total Loss', alpha=0.8)
    plt.plot(history['recon_loss'], label='Reconstruction Loss', linestyle='--', alpha=0.8)
    plt.plot(history['kl_loss'], label='KL Divergence', linestyle=':', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_samples_comparison(
    original: np.ndarray,
    generated: np.ndarray,
    limit: float = 3.0,
    save_path: str = None
) -> None:
    """Compare original data distribution with generated samples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].scatter(original[:, 0], original[:, 1], alpha=0.5, s=10, c='tab:orange')
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(generated[:, 0], generated[:, 1], alpha=0.5, s=10, c='tab:purple')
    axes[1].set_title('VAE Generated Samples')
    axes[1].set_xlabel('$x_1$')
    axes[1].set_ylabel('$x_2$')
    axes[1].set_xlim(-limit, limit)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_reconstructions(
    model: VAE,
    X: torch.Tensor,
    n_samples: int = 1000,
    limit: float = 3.0,
    save_path: str = None
) -> None:
    """Visualize original vs reconstructed samples."""
    model.eval()
    
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    x_sample = X[indices]
    
    with torch.no_grad():
        x_recon, _, _ = model(x_sample)
    
    x_np = x_sample.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].scatter(x_np[:, 0], x_np[:, 1], alpha=0.5, s=10, c='tab:orange')
    axes[0].set_title('Original Samples')
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(x_recon_np[:, 0], x_recon_np[:, 1], alpha=0.5, s=10, c='tab:red')
    axes[1].set_title('Reconstructed Samples')
    axes[1].set_xlim(-limit, limit)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    model.train()


def plot_latent_space(
    model: VAE,
    X: torch.Tensor,
    labels: np.ndarray = None,
    save_path: str = None
) -> None:
    """Visualize the learned latent space by encoding data points."""
    model.eval()
    
    with torch.no_grad():
        mu, _ = model.encode(X)
    
    mu_np = mu.cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        mu_np[:, 0], mu_np[:, 1],
        c=labels if labels is not None else 'tab:blue',
        cmap='viridis' if labels is not None else None,
        alpha=0.6, s=10
    )
    
    if labels is not None:
        plt.colorbar(scatter, label='Class Label')
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space (Encoder Means $\\mu$)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    model.train()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training and evaluation pipeline."""
    
    # Setup
    config = Config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("\n[1/4] Generating two moons dataset...")
    X_data, scaler = generate_two_moons_data(
        n_samples=config.n_samples,
        noise=config.noise_level,
        seed=config.seed,
        device=device
    )
    dataloader = create_dataloader(X_data, config.batch_size)
    print(f"  Data shape: {X_data.shape}")
    
    # Also get labels for latent space visualization
    _, labels = make_moons(n_samples=config.n_samples, noise=config.noise_level, random_state=config.seed)
    
    # Initialize model
    print("\n[2/4] Initializing VAE model...")
    model = VAE(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {config.depth} hidden layers, {config.hidden_dim} units each")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Total parameters: {n_params:,}")
    
    # Train
    print("\n[3/4] Training VAE...")
    history = train_vae(model, dataloader, config, device)
    
    # Evaluate and visualize
    print("\n[4/4] Generating visualizations...")
    
    # Plot training curves
    plot_training_curves(history)
    
    # Generate samples and compare
    model.eval()
    with torch.no_grad():
        generated = model.sample(config.n_samples_viz, device).cpu().numpy()
    
    plot_samples_comparison(
        original=X_data.cpu().numpy(),
        generated=generated,
        limit=config.plot_limit
    )
    
    # Visualize latent space
    plot_latent_space(model, X_data, labels=labels)
    
    # Visualize reconstructions
    plot_reconstructions(model, X_data, limit=config.plot_limit)
    
    print("\nDone!")
    
    return model, history


if __name__ == "__main__":
    model, history = main()
