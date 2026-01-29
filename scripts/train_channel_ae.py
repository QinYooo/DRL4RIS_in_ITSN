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
    Collect channel samples from random satellite trajectories using ITSNEnv

    Args:
        n_samples: Number of channel samples to collect
        seed: Random seed

    Returns:
        Array of channel vectors (n_samples, input_dim)
    """
    print(f"Collecting {n_samples} channel samples using ITSNEnv...")

    # Import ITSNEnv
    from envs.itsn_env import ITSNEnv

    # Create environment with diverse trajectory settings
    env = ITSNEnv(
        rng_seed=seed,
        max_steps_per_episode=50,  # Long episodes for more samples
        n_substeps=1,  # No substeps needed for data collection
        ephemeris_noise_std=0.5  # Include ephemeris noise for robustness
    )

    # Compute input dimension
    input_dim, dims_breakdown = compute_channel_input_dim(env.scenario)
    print(f"Channel vector dimension: {input_dim}")
    print("Breakdown:", dims_breakdown)

    channel_samples = []
    rng = np.random.RandomState(seed)

    # Calculate number of episodes needed
    samples_per_episode = env.max_steps * env.n_substeps
    n_episodes = max(100, (n_samples + samples_per_episode - 1) // samples_per_episode)

    print(f"Collecting from {n_episodes} episodes (~{samples_per_episode} samples/episode)")

    for episode_idx in range(n_episodes):
        # Reset environment with random seed for diversity
        episode_seed = seed + episode_idx
        obs, info = env.reset(seed=episode_seed)

        # Collect initial channel sample
        channels = env.current_channels
        channel_vec = preprocess_channels(channels, use_inferred_G_SAT=False)
        channel_samples.append(channel_vec)

        # Also collect inferred G_SAT version for robustness
        if env.inferred_G_SAT is not None:
            channel_vec_inferred = preprocess_channels(
                channels, use_inferred_G_SAT=True, inferred_G_SAT=env.inferred_G_SAT
            )
            channel_samples.append(channel_vec_inferred)

        # Rollout episode and collect samples
        for step in range(env.max_steps):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Collect channel sample
            channels = env.current_channels
            channel_vec = preprocess_channels(channels, use_inferred_G_SAT=False)
            channel_samples.append(channel_vec)

            # Also collect inferred version
            if env.inferred_G_SAT is not None:
                channel_vec_inferred = preprocess_channels(
                    channels, use_inferred_G_SAT=True, inferred_G_SAT=env.inferred_G_SAT
                )
                channel_samples.append(channel_vec_inferred)

            if terminated or truncated:
                break

        if (episode_idx + 1) % 10 == 0:
            print(f"  Episode {episode_idx + 1}/{n_episodes}, Samples: {len(channel_samples)}")

        # Early stop if we have enough samples
        if len(channel_samples) >= n_samples * 1.2:  # Collect 20% extra for better coverage
            break

    # Shuffle and trim to desired size
    channel_samples = np.array(channel_samples)
    rng.shuffle(channel_samples)
    channel_samples = channel_samples[:n_samples]

    print(f"Collected {len(channel_samples)} samples")

    return channel_samples, input_dim


def train_autoencoder(channel_samples, input_dim, latent_dim=32,
                     epochs=100, batch_size=128, lr=1e-3, device='cuda',
                     checkpoint_dir=None):
    """
    Train the channel autoencoder with checkpointing

    Args:
        channel_samples: Array of channel vectors
        input_dim: Input dimension
        latent_dim: Latent dimension
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        checkpoint_dir: Directory to save checkpoints

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
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 20

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
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Save checkpoint
            if checkpoint_dir is not None:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'input_dim': input_dim,
                    'latent_dim': latent_dim,
                    'normalization_stats': {'mean': mean, 'std': std},
                    'val_loss': best_val_loss,
                    'history': history
                }
                checkpoint_path = checkpoint_dir / 'channel_ae_best.pth'
                torch.save(checkpoint, checkpoint_path)
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"Best={best_val_loss:.6f}, LR={current_lr:.2e}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
            break

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
    N_SAMPLES = 20000  # Increased for better coverage
    LATENT_DIM = 128   # Increased from 32 for better reconstruction
    EPOCHS = 200  # More epochs with early stopping
    BATCH_SIZE = 128
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'checkpoints' / 'channel_ae'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Channel Autoencoder Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  Latent dim: {LATENT_DIM}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Collect data
    channel_samples, input_dim = collect_channel_samples(n_samples=N_SAMPLES)

    # Save raw data for future use
    data_path = output_dir / 'channel_samples.npz'
    np.savez_compressed(data_path, samples=channel_samples, input_dim=input_dim)
    print(f"\nRaw data saved to {data_path}")

    # Train autoencoder
    model, norm_stats, history = train_autoencoder(
        channel_samples, input_dim, latent_dim=LATENT_DIM,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE,
        checkpoint_dir=output_dir
    )

    # Save final model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'latent_dim': LATENT_DIM,
        'normalization_stats': norm_stats,
        'history': history,
        'config': {
            'n_samples': N_SAMPLES,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR
        }
    }
    checkpoint_path = output_dir / 'channel_ae_best.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"\nFinal model saved to {checkpoint_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    ax.plot(history['val_loss'], label='Val Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Learning rate
    ax = axes[1]
    ax.plot(history['learning_rate'], color='green', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
    print(f"Training curve saved to {output_dir / 'training_curve.png'}")

    # Visualize reconstruction
    fig = visualize_reconstruction(model, channel_samples, norm_stats, n_samples=5, device=DEVICE)
    plt.savefig(output_dir / 'reconstruction_samples.png', dpi=150, bbox_inches='tight')
    print(f"Reconstruction samples saved to {output_dir / 'reconstruction_samples.png'}")

    # Compute final statistics
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Compute reconstruction error statistics
    model.eval()
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    mean = norm_stats['mean']
    std = norm_stats['std']

    # Sample 1000 random samples for evaluation
    eval_indices = np.random.choice(len(channel_samples), size=min(1000, len(channel_samples)), replace=False)
    eval_samples = channel_samples[eval_indices]
    normalized = (eval_samples - mean) / (std + 1e-8)

    with torch.no_grad():
        x = torch.FloatTensor(normalized).to(device)
        recon, latent = model(x)
        recon = recon.cpu().numpy()

    # Denormalize
    recon = recon * std + mean

    # Compute errors
    mse = np.mean((eval_samples - recon) ** 2, axis=1)
    mae = np.mean(np.abs(eval_samples - recon), axis=1)

    print(f"\nReconstruction Error (on {len(eval_samples)} samples):")
    print(f"  MSE: mean={np.mean(mse):.6f}, std={np.std(mse):.6f}")
    print(f"  MAE: mean={np.mean(mae):.6f}, std={np.std(mae):.6f}")
    print(f"  Relative error: {np.mean(mae) / (np.abs(eval_samples).mean() + 1e-8) * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Use this checkpoint with ITSNEnvAE:")
    print(f"  env = ITSNEnvAE(ae_checkpoint_path='{checkpoint_path}')")


if __name__ == '__main__':
    main()
