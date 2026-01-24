"""
Train RL Agent with Autoencoder-Compressed State
Uses pre-trained channel autoencoder for state representation
"""

import numpy as np
import torch
from pathlib import Path
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs.itsn_env_ae import ITSNEnvAE
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


def make_env(ae_checkpoint_path, rank, seed=0, **env_kwargs):
    """
    Utility function for multiprocessed env.

    Args:
        ae_checkpoint_path: Path to pre-trained AE checkpoint
        rank: Index of the subprocess
        seed: Random seed
        **env_kwargs: Additional environment arguments
    """
    def _init():
        env = ITSNEnvAE(
            ae_checkpoint_path=ae_checkpoint_path,
            rng_seed=seed + rank,
            **env_kwargs
        )
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init


def train_rl_agent(
    ae_checkpoint_path,
    total_timesteps=500000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=42,
    device='cuda',
    log_dir='logs',
    checkpoint_freq=50000,
    eval_freq=10000,
    eval_episodes=10,
    **env_kwargs
):
    """
    Train PPO agent with AE-compressed state

    Args:
        ae_checkpoint_path: Path to pre-trained autoencoder
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        seed: Random seed
        device: 'cuda' or 'cpu'
        log_dir: Directory for logs and checkpoints
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        eval_episodes: Number of evaluation episodes
        **env_kwargs: Additional environment arguments
    """

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_AE_{timestamp}"
    log_path = Path(log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training RL Agent with Autoencoder State Compression")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  AE checkpoint: {ae_checkpoint_path}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"  Log directory: {log_path}")
    print(f"  Seed: {seed}")
    print("=" * 60)

    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(ae_checkpoint_path, i, seed, **env_kwargs)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(ae_checkpoint_path, 0, seed, **env_kwargs)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(ae_checkpoint_path, 999, seed, **env_kwargs)])

    # Create PPO agent
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=str(log_path / 'tensorboard'),
        device=device,
        seed=seed
    )

    print(f"\nPPO Agent created:")
    print(f"  Policy: MlpPolicy")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(log_path / 'checkpoints'),
        name_prefix='ppo_ae_model',
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_path / 'best_model'),
        log_path=str(log_path / 'eval'),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    final_model_path = log_path / 'final_model'
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # Close environments
    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Results saved to: {log_path}")
    print(f"View tensorboard: tensorboard --logdir {log_path / 'tensorboard'}")

    return model, log_path


def main():
    parser = argparse.ArgumentParser(description='Train RL agent with AE-compressed state')

    # AE checkpoint
    parser.add_argument('--ae-checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to pre-trained autoencoder checkpoint')

    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=500000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    # Environment parameters
    parser.add_argument('--max-steps', type=int, default=40,
                       help='Max steps per episode')
    parser.add_argument('--n-substeps', type=int, default=5,
                       help='Physics substeps per RL step')
    parser.add_argument('--phase-bits', type=int, default=4,
                       help='RIS phase quantization bits')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='AE latent dimension')

    # Logging
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                       help='Checkpoint frequency')

    args = parser.parse_args()

    # Check if AE checkpoint exists
    ae_checkpoint = Path(args.ae_checkpoint)
    if not ae_checkpoint.exists():
        print(f"Error: AE checkpoint not found at {ae_checkpoint}")
        print("Please train the autoencoder first:")
        print("  python scripts/train_channel_ae.py")
        return 1

    # Environment kwargs
    env_kwargs = {
        'max_steps_per_episode': args.max_steps,
        'n_substeps': args.n_substeps,
        'phase_bits': args.phase_bits,
        'latent_dim': args.latent_dim,
        'device': args.device
    }

    # Train
    model, log_path = train_rl_agent(
        ae_checkpoint_path=str(ae_checkpoint),
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_freq=args.checkpoint_freq,
        **env_kwargs
    )

    return 0


if __name__ == '__main__':
    exit(main())
