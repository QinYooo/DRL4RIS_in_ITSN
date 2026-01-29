"""
Evaluate Channel Autoencoder Quality
Analyzes reconstruction quality and its impact on RL state representation
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from envs.itsn_env_ae import ITSNEnvAE
from models.channel_autoencoder import preprocess_channels, ChannelAutoencoder


def evaluate_ae_reconstruction(ae_checkpoint_path, n_episodes=10, device='cuda'):
    """
    Evaluate AE reconstruction quality across multiple episodes
    """
    print("=" * 60)
    print("Autoencoder Reconstruction Quality Evaluation")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(ae_checkpoint_path, map_location=device, weights_only=False)
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    norm_stats = checkpoint['normalization_stats']

    print(f"\nAE Configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Compression ratio: {input_dim / latent_dim:.1f}x")

    # Check model architecture
    model_state = checkpoint['model_state_dict']
    print(f"\nModel Architecture:")
    for name, param in model_state.items():
        if 'weight' in name and 'norm' not in name:
            print(f"  {name}: {param.shape}")

    # Initialize model
    model = ChannelAutoencoder(input_dim, latent_dim)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # Create environment
    env = ITSNEnvAE(
        ae_checkpoint_path=ae_checkpoint_path,
        rng_seed=42,
        device=device
    )

    # Collect reconstruction statistics
    mse_list = []
    relative_error_list = []

    print(f"\nEvaluating on {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        step = 0

        while not done and step < 64:
            # Get current channels
            channels = env.current_channels
            channel_vec = preprocess_channels(
                channels,
                use_inferred_G_SAT=True,
                inferred_G_SAT=env.inferred_G_SAT
            )

            # Normalize
            mean = norm_stats['mean']
            std = norm_stats['std']
            normalized = (channel_vec - mean) / (std + 1e-8)

            # Reconstruct
            with torch.no_grad():
                x = torch.FloatTensor(normalized).unsqueeze(0).to(device)
                recon, latent = model(x)
                recon = recon.squeeze(0).cpu().numpy()

            # Compute errors
            mse = np.mean((normalized - recon) ** 2)
            relative_error = np.mean(np.abs(normalized - recon)) / (np.mean(np.abs(normalized)) + 1e-8)

            mse_list.append(mse)
            relative_error_list.append(relative_error)

            # Step environment
            action = np.zeros(env.scenario.N_ris)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

    env.close()

    # Print statistics
    print(f"\n{'=' * 60}")
    print("Reconstruction Statistics")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(mse_list)}")
    print(f"\nMSE (normalized space):")
    print(f"  Mean: {np.mean(mse_list):.6f}")
    print(f"  Std:  {np.std(mse_list):.6f}")
    print(f"  Min:  {np.min(mse_list):.6f}")
    print(f"  Max:  {np.max(mse_list):.6f}")
    print(f"\nRelative Error:")
    print(f"  Mean: {np.mean(relative_error_list)*100:.2f}%")
    print(f"  Std:  {np.std(relative_error_list)*100:.2f}%")

    # Quality assessment
    print(f"\n{'=' * 60}")
    print("Quality Assessment")
    print(f"{'=' * 60}")

    avg_mse = np.mean(mse_list)
    if avg_mse < 0.01:
        print("✅ Excellent: MSE < 0.01 - Minimal information loss")
    elif avg_mse < 0.05:
        print("✅ Good: MSE < 0.05 - Acceptable for RL")
    elif avg_mse < 0.1:
        print("⚠️ Fair: MSE < 0.1 - May affect RL performance")
    else:
        print("❌ Poor: MSE >= 0.1 - Significant information loss, consider:")
        print("   - Increasing latent_dim")
        print("   - Using larger hidden layers")
        print("   - Training for more epochs")

    return {
        'mse_mean': np.mean(mse_list),
        'mse_std': np.std(mse_list),
        'relative_error_mean': np.mean(relative_error_list),
        'input_dim': input_dim,
        'latent_dim': latent_dim
    }


def compare_action_sensitivity(ae_checkpoint_path, n_tests=20, device='cuda'):
    """
    Test if AE-compressed state can distinguish different channel conditions
    """
    print(f"\n{'=' * 60}")
    print("Action Sensitivity Analysis")
    print(f"{'=' * 60}")

    env = ITSNEnvAE(
        ae_checkpoint_path=ae_checkpoint_path,
        rng_seed=42,
        device=device
    )

    # Test: Do different actions lead to distinguishable rewards?
    print(f"\nTesting {n_tests} random actions on same initial state...")

    rewards = []
    for i in range(n_tests):
        obs, _ = env.reset(seed=42)

        if i == 0:
            action = np.zeros(env.scenario.N_ris)  # Zero action
        else:
            action = np.random.uniform(-1, 1, env.scenario.N_ris)

        _, reward, _, _, info = env.step(action)
        rewards.append(reward)

    env.close()

    zero_reward = rewards[0]
    random_rewards = rewards[1:]

    print(f"\nResults:")
    print(f"  Zero action reward: {zero_reward:.2f}")
    print(f"  Random actions: mean={np.mean(random_rewards):.2f}, std={np.std(random_rewards):.2f}")
    print(f"  Reward range: [{min(random_rewards):.2f}, {max(random_rewards):.2f}]")
    print(f"  Actions better than zero: {sum(1 for r in random_rewards if r > zero_reward)}/{len(random_rewards)}")

    # Check if there's enough signal for learning
    reward_range = max(rewards) - min(rewards)
    print(f"\n  Total reward range: {reward_range:.2f}")

    if reward_range > 10:
        print("✅ Good signal: Reward range > 10 - Sufficient for RL learning")
    elif reward_range > 5:
        print("⚠️ Moderate signal: Reward range 5-10 - May need more training")
    else:
        print("❌ Weak signal: Reward range < 5 - Consider reward shaping")

    return {
        'zero_reward': zero_reward,
        'random_mean': np.mean(random_rewards),
        'random_std': np.std(random_rewards),
        'reward_range': reward_range
    }


def analyze_latent_space(ae_checkpoint_path, n_samples=100, device='cuda'):
    """
    Analyze the latent space distribution
    """
    print(f"\n{'=' * 60}")
    print("Latent Space Analysis")
    print(f"{'=' * 60}")

    env = ITSNEnvAE(
        ae_checkpoint_path=ae_checkpoint_path,
        rng_seed=42,
        device=device
    )

    latent_vectors = []

    for seed in range(n_samples):
        obs, _ = env.reset(seed=seed)
        # Extract latent part from observation (indices 6:6+latent_dim)
        latent = obs[6:6+env.latent_dim]
        latent_vectors.append(latent)

    env.close()

    latent_vectors = np.array(latent_vectors)

    print(f"\nLatent space statistics ({n_samples} samples):")
    print(f"  Shape: {latent_vectors.shape}")
    print(f"  Mean: {latent_vectors.mean():.4f}")
    print(f"  Std: {latent_vectors.std():.4f}")
    print(f"  Min: {latent_vectors.min():.4f}")
    print(f"  Max: {latent_vectors.max():.4f}")

    # Check per-dimension variance
    dim_std = latent_vectors.std(axis=0)
    print(f"\nPer-dimension std:")
    print(f"  Mean: {dim_std.mean():.4f}")
    print(f"  Min: {dim_std.min():.4f}")
    print(f"  Max: {dim_std.max():.4f}")

    # Check for dead dimensions (very low variance)
    dead_dims = np.sum(dim_std < 0.01)
    if dead_dims > 0:
        print(f"\n⚠️ Warning: {dead_dims}/{len(dim_std)} dimensions have std < 0.01 (potentially unused)")
    else:
        print(f"\n✅ All {len(dim_std)} latent dimensions are active")

    return {
        'latent_mean': latent_vectors.mean(),
        'latent_std': latent_vectors.std(),
        'dead_dims': dead_dims
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Channel Autoencoder')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to AE checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of episodes for reconstruction evaluation')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return 1

    # Run evaluations
    recon_stats = evaluate_ae_reconstruction(
        str(checkpoint_path),
        n_episodes=args.n_episodes,
        device=args.device
    )

    sensitivity_stats = compare_action_sensitivity(
        str(checkpoint_path),
        device=args.device
    )

    latent_stats = analyze_latent_space(
        str(checkpoint_path),
        device=args.device
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Compression: {recon_stats['input_dim']} -> {recon_stats['latent_dim']} ({recon_stats['input_dim']/recon_stats['latent_dim']:.1f}x)")
    print(f"Reconstruction MSE: {recon_stats['mse_mean']:.6f}")
    print(f"Reward Range: {sensitivity_stats['reward_range']:.2f}")
    print(f"Dead Latent Dims: {latent_stats['dead_dims']}")

    # Overall recommendation
    print(f"\n{'=' * 60}")
    print("RECOMMENDATION")
    print(f"{'=' * 60}")

    issues = []
    if recon_stats['mse_mean'] > 0.1:
        issues.append("High reconstruction error - increase latent_dim or hidden layers")
    if sensitivity_stats['reward_range'] < 5:
        issues.append("Low reward sensitivity - check reward function")
    if latent_stats['dead_dims'] > recon_stats['latent_dim'] * 0.2:
        issues.append("Many dead latent dimensions - reduce latent_dim or improve training")

    if not issues:
        print("✅ AE quality looks good for RL training!")
    else:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"  - {issue}")

    return 0


if __name__ == '__main__':
    exit(main())
