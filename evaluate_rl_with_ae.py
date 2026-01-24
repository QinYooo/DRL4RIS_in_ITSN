"""
Evaluate trained RL agent with AE-compressed state
Compares performance against baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from envs.itsn_env_ae import ITSNEnvAE
from stable_baselines3 import PPO


def evaluate_agent(model, env, n_episodes=100, deterministic=True):
    """
    Evaluate trained agent

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    power_bs_list = []
    power_sat_list = []
    total_power_list = []

    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_successes = []
        episode_power_bs = []
        episode_power_sat = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            episode_successes.append(info['success'])
            episode_power_bs.append(info['P_BS'])
            episode_power_sat.append(info['P_sat'])

            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rates.append(np.mean(episode_successes))
        power_bs_list.append(np.mean(episode_power_bs))
        power_sat_list.append(np.mean(episode_power_sat))
        total_power_list.append(np.mean(episode_power_bs) + np.mean(episode_power_sat))

    results = {
        'episode_rewards': np.array(episode_rewards),
        'episode_lengths': np.array(episode_lengths),
        'success_rates': np.array(success_rates),
        'power_bs': np.array(power_bs_list),
        'power_sat': np.array(power_sat_list),
        'total_power': np.array(total_power_list)
    }

    return results


def print_evaluation_summary(results):
    """Print evaluation summary statistics"""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"\nReward:")
    print(f"  Mean: {results['episode_rewards'].mean():.2f}")
    print(f"  Std:  {results['episode_rewards'].std():.2f}")
    print(f"  Min:  {results['episode_rewards'].min():.2f}")
    print(f"  Max:  {results['episode_rewards'].max():.2f}")

    print(f"\nSuccess Rate:")
    print(f"  Mean: {results['success_rates'].mean() * 100:.2f}%")
    print(f"  Std:  {results['success_rates'].std() * 100:.2f}%")

    print(f"\nPower Consumption:")
    print(f"  BS Power:    {results['power_bs'].mean():.4f} ± {results['power_bs'].std():.4f} W")
    print(f"  Sat Power:   {results['power_sat'].mean():.4f} ± {results['power_sat'].std():.4f} W")
    print(f"  Total Power: {results['total_power'].mean():.4f} ± {results['total_power'].std():.4f} W")

    print(f"\nEpisode Length:")
    print(f"  Mean: {results['episode_lengths'].mean():.1f}")
    print(f"  Std:  {results['episode_lengths'].std():.1f}")


def plot_evaluation_results(results, save_path=None):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rewards
    ax = axes[0, 0]
    ax.hist(results['episode_rewards'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(results['episode_rewards'].mean(), color='red', linestyle='--',
               label=f"Mean: {results['episode_rewards'].mean():.2f}")
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success Rate
    ax = axes[0, 1]
    ax.hist(results['success_rates'] * 100, bins=30, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(results['success_rates'].mean() * 100, color='red', linestyle='--',
               label=f"Mean: {results['success_rates'].mean() * 100:.2f}%")
    ax.set_xlabel('Success Rate (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Success Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Power Consumption
    ax = axes[1, 0]
    ax.scatter(results['power_bs'], results['power_sat'], alpha=0.5, s=20)
    ax.set_xlabel('BS Power (W)')
    ax.set_ylabel('Satellite Power (W)')
    ax.set_title('Power Consumption (BS vs Satellite)')
    ax.grid(True, alpha=0.3)

    # Total Power
    ax = axes[1, 1]
    ax.hist(results['total_power'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(results['total_power'].mean(), color='red', linestyle='--',
               label=f"Mean: {results['total_power'].mean():.4f} W")
    ax.set_xlabel('Total Power (W)')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Power Consumption Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent with AE')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (e.g., logs/PPO_AE_xxx/best_model/best_model.zip)')
    parser.add_argument('--ae-checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to AE checkpoint')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')

    # Environment parameters
    parser.add_argument('--max-steps', type=int, default=40,
                       help='Max steps per episode')
    parser.add_argument('--n-substeps', type=int, default=5,
                       help='Physics substeps per RL step')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='AE latent dimension')

    args = parser.parse_args()

    # Check paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    ae_checkpoint = Path(args.ae_checkpoint)
    if not ae_checkpoint.exists():
        print(f"Error: AE checkpoint not found at {ae_checkpoint}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Evaluating RL Agent with AE-Compressed State")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"AE checkpoint: {ae_checkpoint}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)

    # Create environment
    print("Creating environment...")
    env = ITSNEnvAE(
        ae_checkpoint_path=str(ae_checkpoint),
        max_steps_per_episode=args.max_steps,
        n_substeps=args.n_substeps,
        latent_dim=args.latent_dim,
        rng_seed=args.seed
    )

    # Evaluate
    print(f"\nEvaluating for {args.n_episodes} episodes...")
    results = evaluate_agent(model, env, n_episodes=args.n_episodes,
                            deterministic=args.deterministic)

    # Print summary
    print_evaluation_summary(results)

    # Save results
    results_path = output_dir / 'evaluation_results.npz'
    np.savez(results_path, **results)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_path = output_dir / 'evaluation_plots.png'
    plot_evaluation_results(results, save_path=plot_path)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
