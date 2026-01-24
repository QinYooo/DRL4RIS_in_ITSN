"""
Pre-train Channel Autoencoder
Collects channel samples from random trajectories and trains the autoencoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.scenario import ITSNScenario
from models.channel_autoencoder import (
    ChannelAutoencoder,
    preprocess_channels,
    compute_channel_input_dim,
    normalize_channel_vector
)


class ChannelDataset(Dataset):
    """Dataset of channel samples"""

    def __init__(self, channel_vectors):
        self.data = torch.FloatTensor(channel_vectors)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collect_channel_samples(n_samples=10000, seed=42):
    """
    Collect channel samples from random satellite trajectories

    Args:
        n_samples: Number of channel samples to collect
        seed: Random seed

    Returns:
        Array of channel vectors (n_samples, input_dim)
    """
    print(f"Collecting {n_samples} channel samples...")

    scenario = ITSNScenario(rng_seed=seed)
    rng = np.random.RandomState(seed)

    # Compute input dimension
    input_dim, dims_breakdown = compute_channel_input_dim(scenario)
    print(f"Channel vector dimension: {input_dim}")
    print("Breakdown:", dims_breakdown)

    channel_samples = []

    # Generate diverse trajectories
    n_trajectories = max(100, n_samples // 100)
    samples_per_traj = n_samples // n_trajectories

    for traj_idx in range(n_trajectories):
        # Reset user positions for diversity
        scenario.reset_user_positions()

        # Generate random trajectory
        n_steps = rng.randint(50, 150)
        max_ele = rng.uniform(60, 88)
        start_ele = rng.uniform(15, 40)
        end_ele = rng.uniform(15, 40)

        peak_ratio = rng.uniform(0.3, 0.7)
        peak_step = max(1, int(n_steps * peak_ratio))

        ele_up = np.linspace(start_ele, max_ele, peak_step)
        ele_down = np.linspace(max_ele, end_ele, n_steps - peak_step)
        elevations = np.concatenate([ele_up, ele_down])

        azimuths = np.linspace(
            rng.uniform(45, 135),
            rng.uniform(45, 315),
            n_steps
        )

        # Sample channels along trajectory
        sample_indices = rng.choice(n_steps, size=min(samples_per_traj, n_steps), replace=False)

        for step_idx in sample_indices:
            # Update satellite position
            scenario.update_satellite_position(
                elevations[step_idx],
                azimuths[step_idx],
                orbit_height=500e3
            )

            # Generate channels
            channels = scenario.generate_channels()

            # Preprocess to vector
            channel_vec = preprocess_channels(channels, use_inferred_G_SAT=False)
            channel_samples.append(channel_vec)

        if (traj_idx + 1) % 10 == 0:
            print(f"  Trajectory {traj_idx + 1}/{n_trajectories}, Samples: {len(channel_samples)}")

    channel_samples = np.array(channel_samples[:n_samples])
    print(f"Collected {len(channel_samples)} samples")

    return channel_samples, input_dim


def train_autoencoder(channel_samples, input_dim, latent_dim=32,
                     epochs=100, batch_size=128, lr=1e-3, device='cuda'):
    """
    Train the channel autoencoder

    Args:
        channel_samples: Array of channel vectors
        input_dim: Input dimension
        latent_dim: Latent dimension
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        Trained autoencoder model, normalization stats, training history
    """
    # Normalize data
    print("\nNormalizing data...")
    mean = np.mean(channel_samples)
    std = np.std(channel_samples)
    normalized_samples = (channel_samples - mean) / (std + 1e-8)

    print(f"Data stats: mean={mean:.4f}, std={std:.4f}")

    # Create dataset and dataloader
    dataset = ChannelDataset(normalized_samples)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = ChannelAutoencoder(input_dim, latent_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    print("\nTraining...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * len(batch)

        val_loss /= len(val_dataset)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.6f}")

    normalization_stats = {'mean': mean, 'std': std}

    return model, normalization_stats, history


def visualize_reconstruction(model, channel_samples, normalization_stats, n_samples=5, device='cuda'):
    """Visualize reconstruction quality"""
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    mean = normalization_stats['mean']
    std = normalization_stats['std']

    # Normalize samples
    normalized = (channel_samples[:n_samples] - mean) / (std + 1e-8)

    with torch.no_grad():
        x = torch.FloatTensor(normalized).to(device)
        recon, latent = model(x)
        recon = recon.cpu().numpy()
        latent = latent.cpu().numpy()

    # Denormalize
    recon = recon * std + mean
    original = channel_samples[:n_samples]

    # Plot
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2*n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(original[i], label='Original', alpha=0.7)
        ax.plot(recon[i], label='Reconstructed', alpha=0.7, linestyle='--')
        ax.set_title(f'Sample {i+1} (MSE: {np.mean((original[i] - recon[i])**2):.6f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # Configuration
    N_SAMPLES = 10000
    LATENT_DIM = 32
    EPOCHS = 100
    BATCH_SIZE = 128
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'checkpoints' / 'channel_ae'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    channel_samples, input_dim = collect_channel_samples(n_samples=N_SAMPLES)

    # Train autoencoder
    model, norm_stats, history = train_autoencoder(
        channel_samples, input_dim, latent_dim=LATENT_DIM,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE
    )

    # Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'latent_dim': LATENT_DIM,
        'normalization_stats': norm_stats,
        'history': history
    }
    checkpoint_path = output_dir / 'channel_ae_best.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Channel Autoencoder Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
    print(f"Training curve saved to {output_dir / 'training_curve.png'}")

    # Visualize reconstruction
    fig = visualize_reconstruction(model, channel_samples, norm_stats, n_samples=5, device=DEVICE)
    plt.savefig(output_dir / 'reconstruction_samples.png', dpi=150, bbox_inches='tight')
    print(f"Reconstruction samples saved to {output_dir / 'reconstruction_samples.png'}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
