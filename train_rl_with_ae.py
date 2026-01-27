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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import json


def make_env(ae_checkpoint_path, rank, seed=0, device='cuda', **env_kwargs):
    """
    Utility function for multiprocessed env.

    Args:
        ae_checkpoint_path: Path to pre-trained AE checkpoint
        rank: Index of the subprocess
        seed: Random seed
        device: Device for AE inference
        **env_kwargs: Additional environment arguments
    """
    def _init():
        env = ITSNEnvAE(
            ae_checkpoint_path=ae_checkpoint_path,
            rng_seed=seed + rank,
            device=device,
            **env_kwargs
        )
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init


def train_rl_agent(
    ae_checkpoint_path,
    total_timesteps=500000,
    n_envs=16,
    learning_rate=1e-4,
    n_steps=64,
    batch_size=64,
    n_epochs=3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
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
            make_env(ae_checkpoint_path, i, seed, device, **env_kwargs)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(ae_checkpoint_path, 0, seed, device, **env_kwargs)])

    # Normalize rewards for better learning
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(ae_checkpoint_path, 999, seed, device, **env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

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

    # 详细轨道评估回调
    trajectory_eval_callback = TrajectoryEvalCallback(
        ae_checkpoint_path=ae_checkpoint_path,
        eval_freq=eval_freq,  # 每eval_freq步进行一次详细评估
        log_path=log_path / 'trajectory_evals',
        device=device,
        verbose=1,
        **env_kwargs
    )

    # ========== Baseline 评估 (零动作) ==========
    print("\n" + "=" * 60)
    print("Baseline evaluation (zero action)...")
    print("=" * 60)
    baseline_results = evaluate_baseline(
        ae_checkpoint_path=ae_checkpoint_path,
        seed=seed,
        device=device,
        save_path=log_path / 'baseline_eval.json',
        **env_kwargs
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, trajectory_eval_callback],
        progress_bar=True
    )

    # Save final model
    final_model_path = log_path / 'final_model'
    model.save(final_model_path)
    env.save(str(log_path / 'final_model' / 'vec_normalize.pkl'))
    print(f"\nFinal model saved to {final_model_path}")

    # 最终详细评估
    print("\n" + "=" * 60)
    print("Final trajectory evaluation...")
    print("=" * 60)
    evaluate_on_trajectory(
        model=model,
        ae_checkpoint_path=ae_checkpoint_path,
        seed=None,
        device=device,
        save_path=log_path / 'final_trajectory_eval.json',
        **env_kwargs
    )

    # Close environments
    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Results saved to: {log_path}")
    print(f"View tensorboard: tensorboard --logdir {log_path / 'tensorboard'}")

    return model, log_path


def evaluate_on_trajectory(
    model,
    ae_checkpoint_path,
    seed=None,
    device='cuda',
    save_path=None,
    **env_kwargs
):
    """
    在随机轨道上评估模型，记录每个step的详细信息

    Args:
        model: 训练好的PPO模型
        ae_checkpoint_path: AE checkpoint路径
        seed: 随机种子（None则随机生成轨道）
        device: 设备
        save_path: 保存结果的路径（None则不保存）
        **env_kwargs: 环境参数

    Returns:
        dict: 包含每个step的P_BS, P_sat, SINR_UE, SINR_SUE
    """
    # 创建评估环境
    if seed is None:
        seed = np.random.randint(0, 100000)

    env = ITSNEnvAE(
        ae_checkpoint_path=ae_checkpoint_path,
        rng_seed=seed,
        device=device,
        **env_kwargs
    )

    # 记录数据
    results = {
        'seed': seed,
        'steps': [],
        'P_BS': [],
        'P_sat': [],
        'P_total': [],
        'SINR_UE': [],  # 每个step的K个UE的SINR
        'SINR_SUE': [],  # 每个step的SUE的SINR
        'SINR_min_dB': [],
        'success': [],
        'true_elevation': [],
        'true_azimuth': [],
        'obs_elevation': [],
        'obs_azimuth': [],
    }

    # 运行一个完整episode
    obs, info = env.reset()
    done = False
    step = 0

    print(f"\n{'='*60}")
    print(f"Evaluating on trajectory (seed={seed})")
    print(f"{'='*60}")
    print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Success':>7}")
    print(f"{'-'*60}")

    while not done:
        # 获取动作
        action, _ = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 记录数据
        P_BS = info.get('P_BS', 0)
        P_sat = info.get('P_sat', 0)
        P_total = P_BS + P_sat
        sinr_ue = info.get('sinr_UE', np.zeros(4))
        sinr_sue = info.get('sinr_SUE', 0)
        success = info.get('success', False)

        # 转换为dB
        sinr_all = np.append(sinr_ue, sinr_sue)
        sinr_min_db = 10 * np.log10(np.min(sinr_all) + 1e-12)
        sinr_ue_db = 10 * np.log10(sinr_ue + 1e-12)
        sinr_sue_db = 10 * np.log10(sinr_sue + 1e-12)

        results['steps'].append(step)
        results['P_BS'].append(float(P_BS))
        results['P_sat'].append(float(P_sat))
        results['P_total'].append(float(P_total))
        results['SINR_UE'].append(sinr_ue_db.tolist())
        results['SINR_SUE'].append(float(sinr_sue_db))
        results['SINR_min_dB'].append(float(sinr_min_db))
        results['success'].append(bool(success))
        results['true_elevation'].append(float(info.get('true_elevation', 0)))
        results['true_azimuth'].append(float(info.get('true_azimuth', 0)))
        results['obs_elevation'].append(float(info.get('obs_elevation', 0)))
        results['obs_azimuth'].append(float(info.get('obs_azimuth', 0)))

        # 打印当前step信息
        success_str = "✓" if success else "✗"
        print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {success_str:>7}")

        step += 1

    env.close()

    # 统计信息
    results['summary'] = {
        'total_steps': step,
        'avg_P_BS': float(np.mean(results['P_BS'])),
        'avg_P_sat': float(np.mean(results['P_sat'])),
        'avg_P_total': float(np.mean(results['P_total'])),
        'avg_SINR_min_dB': float(np.mean(results['SINR_min_dB'])),
        'success_rate': float(np.mean(results['success'])),
    }

    print(f"{'-'*60}")
    print(f"Summary:")
    print(f"  Total steps: {step}")
    print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
    print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
    print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
    print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
    print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")

    # 保存结果
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {save_path}")

    return results


def evaluate_baseline(
    ae_checkpoint_path,
    seed=42,
    device='cuda',
    save_path=None,
    **env_kwargs
):
    """
    Baseline 评估：使用零动作（不调整RIS相位）

    Args:
        ae_checkpoint_path: AE checkpoint路径
        seed: 随机种子
        device: 设备
        save_path: 保存结果的路径
        **env_kwargs: 环境参数

    Returns:
        dict: 包含每个step的详细信息
    """
    env = ITSNEnvAE(
        ae_checkpoint_path=ae_checkpoint_path,
        rng_seed=seed,
        device=device,
        **env_kwargs
    )

    # 记录数据
    results = {
        'seed': seed,
        'type': 'baseline_zero_action',
        'steps': [],
        'P_BS': [],
        'P_sat': [],
        'P_total': [],
        'SINR_UE': [],
        'SINR_SUE': [],
        'SINR_min_dB': [],
        'success': [],
        'rewards': [],
        'true_elevation': [],
        'true_azimuth': [],
    }

    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0

    print(f"\n{'='*60}")
    print(f"Baseline Evaluation (Zero Action, seed={seed})")
    print(f"{'='*60}")
    print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Reward':>8} | {'Success':>7}")
    print(f"{'-'*70}")

    while not done:
        # 零动作
        action = np.zeros(env.scenario.N_ris)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # 记录数据
        P_BS = info.get('P_BS', 0)
        P_sat = info.get('P_sat', 0)
        P_total = P_BS + P_sat
        sinr_ue = info.get('sinr_UE', np.zeros(4))
        sinr_sue = info.get('sinr_SUE', 0)
        success = info.get('success', False)

        sinr_all = np.append(sinr_ue, sinr_sue)
        sinr_min_db = 10 * np.log10(np.min(sinr_all) + 1e-12)
        sinr_ue_db = 10 * np.log10(sinr_ue + 1e-12)
        sinr_sue_db = 10 * np.log10(sinr_sue + 1e-12)

        results['steps'].append(step)
        results['P_BS'].append(float(P_BS))
        results['P_sat'].append(float(P_sat))
        results['P_total'].append(float(P_total))
        results['SINR_UE'].append(sinr_ue_db.tolist())
        results['SINR_SUE'].append(float(sinr_sue_db))
        results['SINR_min_dB'].append(float(sinr_min_db))
        results['success'].append(bool(success))
        results['rewards'].append(float(reward))
        results['true_elevation'].append(float(info.get('true_elevation', 0)))
        results['true_azimuth'].append(float(info.get('true_azimuth', 0)))

        success_str = "Y" if success else "N"
        print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {reward:>8.3f} | {success_str:>7}")

        step += 1

    env.close()

    # 统计信息
    results['summary'] = {
        'total_steps': step,
        'total_reward': float(total_reward),
        'avg_reward': float(total_reward / step),
        'avg_P_BS': float(np.mean(results['P_BS'])),
        'avg_P_sat': float(np.mean(results['P_sat'])),
        'avg_P_total': float(np.mean(results['P_total'])),
        'avg_SINR_min_dB': float(np.mean(results['SINR_min_dB'])),
        'success_rate': float(np.mean(results['success'])),
    }

    print(f"{'-'*70}")
    print(f"Baseline Summary:")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {results['summary']['total_reward']:.3f}")
    print(f"  Avg reward: {results['summary']['avg_reward']:.3f}")
    print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
    print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
    print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
    print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
    print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Baseline results saved to {save_path}")

    return results


class TrajectoryEvalCallback(BaseCallback):
    """
    自定义Callback：每隔一定步数在随机轨道上进行详细评估
    """
    def __init__(
        self,
        ae_checkpoint_path,
        eval_freq=50000,
        log_path=None,
        device='cuda',
        verbose=1,
        **env_kwargs
    ):
        super().__init__(verbose)
        self.ae_checkpoint_path = ae_checkpoint_path
        self.eval_freq = eval_freq
        self.log_path = Path(log_path) if log_path else None
        self.device = device
        self.env_kwargs = env_kwargs
        self.eval_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            save_path = None
            if self.log_path:
                save_path = self.log_path / f'trajectory_eval_{self.eval_count}.json'

            if self.verbose > 0:
                print(f"\n[Callback] Trajectory evaluation #{self.eval_count} at step {self.n_calls}")

            evaluate_on_trajectory(
                model=self.model,
                ae_checkpoint_path=self.ae_checkpoint_path,
                seed=None,  # 随机轨道
                device=self.device,
                save_path=save_path,
                **self.env_kwargs
            )
        return True


def main():
    parser = argparse.ArgumentParser(description='Train RL agent with AE-compressed state')

    # AE checkpoint
    parser.add_argument('--ae-checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to pre-trained autoencoder checkpoint')

    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1024000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=32,
                       help='Number of parallel environments (recommended: 16-32 for episode_length=40)')
    parser.add_argument('--n-steps', type=int, default=64,
                       help='Steps per environment per update (recommended: 64-128 for episode_length=40)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    # Environment parameters
    parser.add_argument('--max-steps', type=int, default=64,
                       help='Max steps per episode (should match n_steps for 1 episode per rollout)')
    parser.add_argument('--n-substeps', type=int, default=10,
                       help='Physics substeps per RL step (use 1 to avoid channel mismatch)')
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

    # Environment kwargs (device will be passed separately to train_rl_agent)
    env_kwargs = {
        'max_steps_per_episode': args.max_steps,
        'n_substeps': args.n_substeps,
        'phase_bits': args.phase_bits,
        'latent_dim': args.latent_dim
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
