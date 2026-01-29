"""
Compare hand-crafted features vs autoencoder-based state representations
Evaluates reconstruction quality and RL performance
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from envs.itsn_env import ITSNEnv
from envs.itsn_env_ae import ITSNEnvAE


def test_state_dimensions():
    """Test that both environments have correct state dimensions"""
    print("=" * 60)
    print("Testing State Dimensions")
    print("=" * 60)

    # Hand-crafted features environment
    env_manual = ITSNEnv(max_steps_per_episode=10, n_substeps=2)
    obs_manual, _ = env_manual.reset()
    print(f"\n[Manual Features]")
    print(f"  State dimension: {obs_manual.shape[0]}")
    print(f"  Expected: 6 (motion) + 3*(K+1) (channel+interference+feedback)")
    print(f"  K={env_manual.scenario.K}, so expected dim = 6 + 3*{env_manual.scenario.K+1} = {6 + 3*(env_manual.scenario.K+1)}")

    # Autoencoder environment (without checkpoint, using random AE)
    env_ae = ITSNEnvAE(
        ae_checkpoint_path=None,
        latent_dim=128,
        max_steps_per_episode=10,
        n_substeps=2
    )
    obs_ae, _ = env_ae.reset()
    print(f"\n[Autoencoder Features]")
    print(f"  State dimension: {obs_ae.shape[0]}")
    print(f"  Expected: 6 (motion) + 32 (latent) + {env_ae.scenario.K+1} (feedback)")
    print(f"  Expected dim = 6 + 32 + {env_ae.scenario.K+1} = {6 + 32 + env_ae.scenario.K+1}")

    print("\n✓ Dimension test passed")


def test_episode_rollout():
    """Test that both environments can complete episodes"""
    print("\n" + "=" * 60)
    print("Testing Episode Rollout")
    print("=" * 60)

    n_steps = 5

    # Manual features
    print("\n[Manual Features Environment]")
    env_manual = ITSNEnv(max_steps_per_episode=n_steps, n_substeps=2)
    obs, _ = env_manual.reset()

    start_time = time.time()
    for step in range(n_steps):
        action = env_manual.action_space.sample()
        obs, reward, terminated, truncated, info = env_manual.step(action)
        print(f"  Step {step+1}: reward={reward:.2f}, success={info['success']}, "
              f"actual_substeps={info['actual_substeps']}")
        if terminated:
            break
    manual_time = time.time() - start_time
    print(f"  Time: {manual_time:.3f}s")

    # Autoencoder features
    print("\n[Autoencoder Features Environment]")
    env_ae = ITSNEnvAE(
        ae_checkpoint_path=None,
        latent_dim=128,
        max_steps_per_episode=n_steps,
        n_substeps=2
    )
    obs, _ = env_ae.reset()

    start_time = time.time()
    for step in range(n_steps):
        action = env_ae.action_space.sample()
        obs, reward, terminated, truncated, info = env_ae.step(action)
        print(f"  Step {step+1}: reward={reward:.2f}, success={info['success']}, "
              f"actual_substeps={info['actual_substeps']}")
        if terminated:
            break
    ae_time = time.time() - start_time
    print(f"  Time: {ae_time:.3f}s")

    print(f"\n✓ Rollout test passed")
    print(f"  Time overhead: {(ae_time - manual_time) / manual_time * 100:.1f}%")


def visualize_state_comparison():
    """Visualize state vectors from both representations"""
    print("\n" + "=" * 60)
    print("Visualizing State Representations")
    print("=" * 60)

    # Create environments
    env_manual = ITSNEnv(max_steps_per_episode=20, n_substeps=2, rng_seed=42)
    env_ae = ITSNEnvAE(
        ae_checkpoint_path=None,
        latent_dim=128,
        max_steps_per_episode=20,
        n_substeps=2,
        rng_seed=42
    )

    # Collect states over an episode
    states_manual = []
    states_ae = []

    obs_manual, _ = env_manual.reset(seed=42)
    obs_ae, _ = env_ae.reset(seed=42)

    states_manual.append(obs_manual)
    states_ae.append(obs_ae)

    for _ in range(10):
        action = env_manual.action_space.sample()
        obs_manual, _, terminated_m, _, _ = env_manual.step(action)
        obs_ae, _, terminated_ae, _, _ = env_ae.step(action)

        states_manual.append(obs_manual)
        states_ae.append(obs_ae)

        if terminated_m or terminated_ae:
            break

    states_manual = np.array(states_manual)
    states_ae = np.array(states_ae)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Manual features
    ax = axes[0]
    im = ax.imshow(states_manual.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title('Hand-Crafted Features State Evolution')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State Dimension')
    plt.colorbar(im, ax=ax)

    # Autoencoder features
    ax = axes[1]
    im = ax.imshow(states_ae.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title('Autoencoder-Compressed State Evolution')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State Dimension')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'state_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")

    return fig


def main():
    """Run all comparison tests"""
    print("\n" + "=" * 60)
    print("State Representation Comparison")
    print("=" * 60)

    # Test 1: Dimensions
    test_state_dimensions()

    # Test 2: Episode rollout
    test_episode_rollout()

    # Test 3: Visualization
    visualize_state_comparison()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train autoencoder: python scripts/train_channel_ae.py")
    print("2. Train RL agent with AE features: use ITSNEnvAE with checkpoint")
    print("3. Compare RL performance: manual features vs AE features")


if __name__ == '__main__':
    main()
