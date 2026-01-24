"""
Quick test script to verify autoencoder pipeline
Tests data collection, training, and environment integration
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from envs.itsn_env import ITSNEnv
from models.channel_autoencoder import (
    ChannelAutoencoder,
    preprocess_channels,
    compute_channel_input_dim
)


def test_data_collection():
    """Test channel data collection from ITSNEnv"""
    print("=" * 60)
    print("Test 1: Data Collection from ITSNEnv")
    print("=" * 60)

    env = ITSNEnv(max_steps_per_episode=5, n_substeps=1, rng_seed=42)

    # Compute expected dimension
    input_dim, dims = compute_channel_input_dim(env.scenario)
    print(f"Expected channel vector dimension: {input_dim}")
    print(f"Breakdown: {dims}")

    # Collect samples from one episode
    samples = []
    obs, _ = env.reset(seed=42)

    # Initial sample
    channel_vec = preprocess_channels(env.current_channels, use_inferred_G_SAT=False)
    samples.append(channel_vec)
    print(f"\nInitial channel vector shape: {channel_vec.shape}")

    # Rollout
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        channel_vec = preprocess_channels(env.current_channels, use_inferred_G_SAT=False)
        samples.append(channel_vec)

        if terminated:
            break

    samples = np.array(samples)
    print(f"Collected {len(samples)} samples, shape: {samples.shape}")
    print(f"Data range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Data mean: {samples.mean():.4f}, std: {samples.std():.4f}")

    print("✓ Data collection test passed\n")
    return samples, input_dim


def test_autoencoder_forward():
    """Test autoencoder forward pass"""
    print("=" * 60)
    print("Test 2: Autoencoder Forward Pass")
    print("=" * 60)

    # Create dummy data
    input_dim = 1000
    latent_dim = 32
    batch_size = 16

    model = ChannelAutoencoder(input_dim, latent_dim)
    print(f"Model created: input_dim={input_dim}, latent_dim={latent_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    recon, latent = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {recon.shape}")

    # Test encode only
    latent_only = model.encode(x)
    print(f"Encode-only latent shape: {latent_only.shape}")

    assert recon.shape == x.shape, "Reconstruction shape mismatch"
    assert latent.shape == (batch_size, latent_dim), "Latent shape mismatch"

    print("✓ Autoencoder forward pass test passed\n")


def test_mini_training():
    """Test mini training loop"""
    print("=" * 60)
    print("Test 3: Mini Training Loop")
    print("=" * 60)

    # Collect small dataset
    print("Collecting mini dataset...")
    env = ITSNEnv(max_steps_per_episode=10, n_substeps=1, rng_seed=42)
    input_dim, _ = compute_channel_input_dim(env.scenario)

    samples = []
    for episode in range(5):
        obs, _ = env.reset(seed=42 + episode)
        for step in range(10):
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)
            channel_vec = preprocess_channels(env.current_channels, use_inferred_G_SAT=False)
            samples.append(channel_vec)
            if terminated:
                break

    samples = np.array(samples)
    print(f"Collected {len(samples)} samples")

    # Normalize
    mean = samples.mean()
    std = samples.std()
    normalized = (samples - mean) / (std + 1e-8)

    # Create model
    latent_dim = 16
    model = ChannelAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Mini training
    print("\nTraining for 10 epochs...")
    x = torch.FloatTensor(normalized)

    for epoch in range(10):
        optimizer.zero_grad()
        recon, _ = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss.item():.6f}")

    print("✓ Mini training test passed\n")


def test_env_ae_integration():
    """Test ITSNEnvAE without pre-trained checkpoint"""
    print("=" * 60)
    print("Test 4: ITSNEnvAE Integration (Random AE)")
    print("=" * 60)

    from envs.itsn_env_ae import ITSNEnvAE

    # Create environment with random AE
    env = ITSNEnvAE(
        ae_checkpoint_path=None,
        latent_dim=32,
        max_steps_per_episode=5,
        n_substeps=1,
        rng_seed=42
    )

    print(f"Environment created")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Test reset
    obs, _ = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Test step
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: reward={reward:.2f}, success={info['success']}")

        if terminated:
            break

    print("✓ ITSNEnvAE integration test passed\n")


def main():
    print("\n" + "=" * 60)
    print("Quick Test: Channel Autoencoder Pipeline")
    print("=" * 60 + "\n")

    try:
        # Test 1: Data collection
        samples, input_dim = test_data_collection()

        # Test 2: Autoencoder forward
        test_autoencoder_forward()

        # Test 3: Mini training
        test_mini_training()

        # Test 4: Environment integration
        test_env_ae_integration()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nReady to run full training:")
        print("  python scripts/train_channel_ae.py")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
