"""
Generative Adversarial Network (GAN) for 2D Data Generation

A clean implementation of vanilla GAN for learning 2D distributions (e.g., two moons).
Includes experiments demonstrating stable training, mode collapse, and the effects
of hyperparameters and architectural choices.

Key concepts:
    - Generator: Maps random noise z ~ N(0,I) to data space
    - Discriminator: Classifies samples as real or fake
    - Adversarial training: G tries to fool D, D tries to distinguish real from fake
    - Common failure modes: mode collapse, training instability, vanishing gradients

References:
    - Goodfellow et al. (2014): Generative Adversarial Networks
    - Radford et al. (2016): DCGANs (architectural guidelines)

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
    """Hyperparameters and settings for GAN training."""
    
    # Data
    n_samples: int = 2000
    noise_level: float = 0.05
    batch_size: int = 128
    data_dim: int = 2
    
    # Model architecture
    latent_dim: int = 5  # Dimension of noise vector z
    hidden_dim: int = 128
    use_batch_norm: bool = True  # BatchNorm in generator helps stability
    activation: str = 'leaky_relu'  # 'leaky_relu' or 'relu'
    
    # Training
    learning_rate: float = 0.0002
    betas: tuple = (0.5, 0.999)  # Adam betas recommended for GANs
    n_epochs: int = 300
    
    # Label smoothing: helps prevent discriminator from becoming too confident
    real_label_smoothing: float = 0.9  # Real labels: 0.9 instead of 1.0
    fake_label_smoothing: float = 0.1  # Fake labels: 0.1 instead of 0.0
    
    # Reproducibility
    seed: int = 42
    
    # Visualization
    n_viz_samples: int = 200
    plot_every: int = 50
    plot_limit: float = 2.5


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
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
) -> tuple[torch.Tensor, np.ndarray, StandardScaler]:
    """
    Generate standardized two-moons dataset.
    
    Standardization to ~zero mean and unit variance helps GAN training
    by keeping values in a reasonable range for the networks.
    
    Returns:
        X_tensor: Data tensor on device
        X_np: Numpy array for plotting
        scaler: Fitted scaler for reference
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    
    return X_tensor, X_scaled, scaler


def create_dataloader(X: torch.Tensor, batch_size: int) -> DataLoader:
    """Create DataLoader for training."""
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# =============================================================================
# Model Definitions
# =============================================================================

class Generator(nn.Module):
    """
    Generator network: Maps latent noise z to data space.
    
    Architecture design choices:
        - LeakyReLU: Prevents dying neurons, more stable than ReLU
        - BatchNorm: Stabilizes training, helps with mode collapse
        - No output activation: Allows network to learn appropriate scale
          (Could use Tanh if data is strictly in [-1, 1])
    
    Args:
        latent_dim: Dimension of input noise vector
        output_dim: Dimension of output (data space)
        hidden_dim: Width of hidden layers
        use_batch_norm: Whether to use batch normalization
        activation: 'leaky_relu' or 'relu'
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        use_batch_norm: bool = True,
        activation: str = 'leaky_relu'
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        
        # Choose activation function
        if activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)
        
        # Build network
        layers = []
        
        # Layer 1: latent_dim -> hidden_dim
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(act_fn)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Layer 2: hidden_dim -> hidden_dim * 2
        layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
        layers.append(act_fn if activation == 'relu' else nn.LeakyReLU(0.2, inplace=True))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim * 2))
        
        # Output layer: hidden_dim * 2 -> output_dim
        layers.append(nn.Linear(hidden_dim * 2, output_dim))
        # No activation: let the network learn the output scale
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from noise z."""
        return self.model(z)
    
    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate n_samples by sampling z ~ N(0, I)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.forward(z)


class Discriminator(nn.Module):
    """
    Discriminator network: Classifies input as real or fake.
    
    Architecture design choices:
        - LeakyReLU: Standard for discriminators, prevents dead neurons
        - No BatchNorm: Often omitted in discriminator (can cause instability)
        - Sigmoid output: Outputs probability p(real)
    
    Args:
        input_dim: Dimension of input (data space)
        hidden_dim: Width of hidden layers
        activation: 'leaky_relu' or 'relu'
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        activation: str = 'leaky_relu'
    ):
        super().__init__()
        
        # Choose activation function
        if activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)
        
        self.model = nn.Sequential(
            # Layer 1: input_dim -> hidden_dim * 2
            nn.Linear(input_dim, hidden_dim * 2),
            act_fn,
            
            # Layer 2: hidden_dim * 2 -> hidden_dim
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            
            # Output: hidden_dim -> 1 (probability)
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output probability that x is real."""
        return self.model(x)


# =============================================================================
# Training
# =============================================================================

def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    dataloader: DataLoader,
    config: Config,
    device: torch.device,
    fixed_noise: torch.Tensor = None
) -> dict:
    """
    Train GAN using the standard minimax objective.
    
    GAN objective:
        min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
    
    In practice, we train:
        - D to maximize: log D(x_real) + log(1 - D(G(z)))
        - G to minimize: log(1 - D(G(z)))  [or maximize log D(G(z)) for stability]
    
    Label smoothing is applied to prevent the discriminator from becoming
    overconfident, which can cause vanishing gradients for the generator.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: DataLoader with real data
        config: Training configuration
        device: torch device
        fixed_noise: Fixed noise for visualization (optional)
    
    Returns:
        Dictionary with training history
    """
    # Loss function: Binary Cross Entropy
    criterion = nn.BCELoss()
    
    # Optimizers (separate for G and D)
    g_optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=config.betas)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=config.betas)
    
    # History
    history = {
        'g_loss': [],
        'd_loss': [],
        'd_real_acc': [],  # Discriminator accuracy on real samples
        'd_fake_acc': []   # Discriminator accuracy on fake samples
    }
    
    # Fixed noise for consistent visualization
    if fixed_noise is None:
        fixed_noise = torch.randn(config.n_viz_samples, config.latent_dim, device=device)
    
    generator.train()
    discriminator.train()
    
    progress_bar = tqdm(range(config.n_epochs), desc="Training GAN")
    
    for epoch in progress_bar:
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real_acc = 0.0
        epoch_d_fake_acc = 0.0
        n_batches = 0
        
        for (real_batch,) in dataloader:
            batch_size = real_batch.size(0)
            real_batch = real_batch.to(device)
            
            # Labels with smoothing
            real_labels = torch.full((batch_size, 1), config.real_label_smoothing, device=device)
            fake_labels = torch.full((batch_size, 1), config.fake_label_smoothing, device=device)
            
            # =====================
            # Train Discriminator
            # =====================
            d_optimizer.zero_grad()
            
            # Loss on real samples: D should output ~1
            d_output_real = discriminator(real_batch)
            d_loss_real = criterion(d_output_real, real_labels)
            
            # Loss on fake samples: D should output ~0
            z = torch.randn(batch_size, config.latent_dim, device=device)
            fake_batch = generator(z)
            d_output_fake = discriminator(fake_batch.detach())  # Detach to not backprop to G
            d_loss_fake = criterion(d_output_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # =====================
            # Train Generator
            # =====================
            g_optimizer.zero_grad()
            
            # Generate new fake samples
            z = torch.randn(batch_size, config.latent_dim, device=device)
            fake_batch = generator(z)
            
            # G wants D to think fake samples are real
            # So we use real_labels for the loss (G tries to maximize D(G(z)))
            d_output_fake_for_g = discriminator(fake_batch)
            g_loss = criterion(d_output_fake_for_g, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            # Track metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_real_acc += (d_output_real > 0.5).float().mean().item()
            epoch_d_fake_acc += (d_output_fake < 0.5).float().mean().item()
            n_batches += 1
        
        # Record epoch averages
        history['g_loss'].append(epoch_g_loss / n_batches)
        history['d_loss'].append(epoch_d_loss / n_batches)
        history['d_real_acc'].append(epoch_d_real_acc / n_batches)
        history['d_fake_acc'].append(epoch_d_fake_acc / n_batches)
        
        # Update progress bar
        progress_bar.set_postfix({
            'G': f"{history['g_loss'][-1]:.3f}",
            'D': f"{history['d_loss'][-1]:.3f}",
            'D_acc': f"{(history['d_real_acc'][-1] + history['d_fake_acc'][-1])/2:.2f}"
        })
    
    return history


# =============================================================================
# Visualization
# =============================================================================

def plot_training_curves(history: dict, title: str = "GAN Training", save_path: str = None) -> None:
    """Plot generator and discriminator loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['g_loss'], label='Generator Loss', alpha=0.8)
    axes[0].plot(history['d_loss'], label='Discriminator Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title(f'{title} - Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Discriminator accuracy
    axes[1].plot(history['d_real_acc'], label='D accuracy on Real', alpha=0.8)
    axes[1].plot(history['d_fake_acc'], label='D accuracy on Fake', alpha=0.8)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Random guess')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Discriminator Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_samples_comparison(
    generator: Generator,
    real_data: np.ndarray,
    device: torch.device,
    n_samples: int = 500,
    limit: float = 2.5,
    title: str = "GAN",
    save_path: str = None
) -> None:
    """Compare real data distribution with generated samples."""
    generator.eval()
    
    with torch.no_grad():
        generated = generator.sample(n_samples, device).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Real data
    axes[0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=10, c='tab:blue')
    axes[0].set_title('Real Data')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Generated data
    axes[1].scatter(generated[:, 0], generated[:, 1], alpha=0.5, s=10, c='tab:red')
    axes[1].set_title(f'{title} Generated Samples')
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
    
    generator.train()


def plot_overlay(
    generator: Generator,
    real_data: np.ndarray,
    device: torch.device,
    n_samples: int = 500,
    limit: float = 2.5,
    title: str = "GAN",
    save_path: str = None
) -> None:
    """Overlay generated samples on real data."""
    generator.eval()
    
    with torch.no_grad():
        generated = generator.sample(n_samples, device).cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.4, s=20, c='tab:blue', label='Real')
    plt.scatter(generated[:, 0], generated[:, 1], alpha=0.4, s=20, c='tab:red', label='Generated')
    plt.title(f'{title} - Real vs Generated')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    generator.train()


# =============================================================================
# Experiments: Demonstrating Training Dynamics
# =============================================================================

def run_experiment(
    config: Config,
    device: torch.device,
    X_data: torch.Tensor,
    real_data_np: np.ndarray,
    experiment_name: str
) -> tuple:
    """
    Run a single GAN training experiment.
    
    Returns:
        generator, discriminator, history
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"  LR: {config.learning_rate}, BatchNorm: {config.use_batch_norm}, "
          f"Activation: {config.activation}")
    
    # Create models
    generator = Generator(
        latent_dim=config.latent_dim,
        output_dim=config.data_dim,
        hidden_dim=config.hidden_dim,
        use_batch_norm=config.use_batch_norm,
        activation=config.activation
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=config.data_dim,
        hidden_dim=config.hidden_dim,
        activation=config.activation
    ).to(device)
    
    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator params: {n_params_g:,}, Discriminator params: {n_params_d:,}")
    
    # Create dataloader
    dataloader = create_dataloader(X_data, config.batch_size)
    
    # Train
    history = train_gan(generator, discriminator, dataloader, config, device)
    
    # Visualize results
    plot_training_curves(history, title=experiment_name)
    plot_samples_comparison(generator, real_data_np, device, title=experiment_name)
    plot_overlay(generator, real_data_np, device, title=experiment_name)
    
    return generator, discriminator, history


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Main function: Run GAN experiments demonstrating different training behaviors.
    
    Experiments:
    1. Stable: Good hyperparameters (baseline)
    2. High LR: Learning rate too high (instability)
    3. No BatchNorm: Without batch normalization (potential mode collapse)
    4. ReLU: Using ReLU instead of LeakyReLU (dying neurons risk)
    """
    
    # Setup
    base_config = Config()
    set_seed(base_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data once (shared across experiments)
    print("\nGenerating two moons dataset...")
    X_data, real_data_np, _ = generate_two_moons_data(
        n_samples=base_config.n_samples,
        noise=base_config.noise_level,
        seed=base_config.seed,
        device=device
    )
    print(f"Data shape: {X_data.shape}")
    
    # Plot real data
    plt.figure(figsize=(8, 8))
    plt.scatter(real_data_np[:, 0], real_data_np[:, 1], alpha=0.5, s=10, c='tab:blue')
    plt.title('Real Two-Moons Dataset (Standardized)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim(-base_config.plot_limit, base_config.plot_limit)
    plt.ylim(-base_config.plot_limit, base_config.plot_limit)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.show()
    
    results = {}
    
    # =========================================================================
    # Experiment 1: Stable Training (Baseline)
    # =========================================================================
    config_stable = Config()
    # Use default settings (already good)
    
    g, d, h = run_experiment(config_stable, device, X_data, real_data_np, "Stable (Baseline)")
    results['stable'] = {'generator': g, 'discriminator': d, 'history': h}
    
    # =========================================================================
    # Experiment 2: High Learning Rate (Instability)
    # =========================================================================
    config_high_lr = Config()
    config_high_lr.learning_rate = 0.01  # 50x higher than stable
    
    g, d, h = run_experiment(config_high_lr, device, X_data, real_data_np, "High Learning Rate")
    results['high_lr'] = {'generator': g, 'discriminator': d, 'history': h}
    
    # =========================================================================
    # Experiment 3: No Batch Normalization (Mode Collapse Risk)
    # =========================================================================
    config_no_bn = Config()
    config_no_bn.use_batch_norm = False
    
    g, d, h = run_experiment(config_no_bn, device, X_data, real_data_np, "No BatchNorm")
    results['no_batchnorm'] = {'generator': g, 'discriminator': d, 'history': h}
    
    # =========================================================================
    # Experiment 4: ReLU Activation (Dying Neurons Risk)
    # =========================================================================
    config_relu = Config()
    config_relu.activation = 'relu'
    
    g, d, h = run_experiment(config_relu, device, X_data, real_data_np, "ReLU Activation")
    results['relu'] = {'generator': g, 'discriminator': d, 'history': h}
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("All Experiments Complete")
    print("="*60)
    print("\nFinal Generator Losses:")
    for name, res in results.items():
        final_g_loss = res['history']['g_loss'][-1]
        final_d_loss = res['history']['d_loss'][-1]
        print(f"  {name:15s}: G={final_g_loss:.4f}, D={final_d_loss:.4f}")
    
    print("\nKey Observations:")
    print("  - Stable: Balanced G/D losses, good coverage of both moons")
    print("  - High LR: Oscillating losses, unstable training")
    print("  - No BatchNorm: May show mode collapse (covering only one moon)")
    print("  - ReLU: Similar to stable, but risk of dying neurons in deeper networks")
    
    return results


if __name__ == "__main__":
    results = main()
