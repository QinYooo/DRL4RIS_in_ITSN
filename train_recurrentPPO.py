"""
Train Recurrent PPO Agent with Optional Autoencoder-Compressed State
Uses sb3-contrib's RecurrentPPO with LSTM to handle partial observability

Supports two environment types:
    - ITSNEnv: Full geometry state (6 + 3 + 3*(K+1) + (K+1) dims)
    - ITSNEnvAE: Autoencoder-compressed state (6 + latent_dim + (K+1) dims)
"""

import os
# Fix OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from pathlib import Path
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs.itsn_env import ITSNEnv
from envs.itsn_env_ae import ITSNEnvAE
from sb3_contrib import RecurrentPPO  # Changed from PPO
from baseline.baseline_optimizer import BaselineZFSDROptimizer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import json


def make_env(use_ae, ae_checkpoint_path, rank, seed=0, device='cuda', **env_kwargs):
    """
    Utility function for multiprocessed env.

    Args:
        use_ae: Whether to use autoencoder-compressed state
        ae_checkpoint_path: Path to pre-trained AE checkpoint (required if use_ae=True)
        rank: Index of the subprocess
        seed: Random seed
        device: Device for AE inference (only used if use_ae=True)
        **env_kwargs: Additional environment arguments
    """
    def _init():
        if use_ae:
            env = ITSNEnvAE(
                ae_checkpoint_path=ae_checkpoint_path,
                rng_seed=seed + rank,
                device=device,
                **env_kwargs
            )
        else:
            env = ITSNEnv(
                rng_seed=seed + rank,
                **env_kwargs
            )
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init


def train_rl_agent(
    use_ae=True,
    ae_checkpoint_path=None,
    total_timesteps=500000,
    n_envs=16,
    learning_rate=1e-4,
    n_steps=64,
    batch_size=256,
    n_epochs=2,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.001,
    vf_coef=0.2,
    max_grad_norm=1.0,
    seed=42,
    device='cuda',
    log_dir='logs',
    checkpoint_freq=50000,
    eval_freq=10000,
    eval_episodes=10,
    run_baseline_optimizer=False,
    skip_zero_baseline=False,
    **env_kwargs
):
    """
    Train RecurrentPPO agent (supports both AE-compressed and full geometry state)

    Args:
        use_ae: Whether to use autoencoder-compressed state (True) or full geometry state (False)
        ae_checkpoint_path: Path to pre-trained autoencoder (required if use_ae=True)
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for RecurrentPPO
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
        run_baseline_optimizer: Whether to run baseline optimizer (ZF+SDR) evaluation
        skip_zero_baseline: Whether to skip zero-action baseline evaluation
        **env_kwargs: Additional environment arguments
    """

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_type = "AE" if use_ae else "Geometry"
    run_name = f"RecurrentPPO_{env_type}_{timestamp}"
    log_path = Path(log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training RecurrentPPO Agent with {env_type} State")
    print("=" * 60)
    print(f"Configuration:")
    if use_ae:
        print(f"  Environment: ITSNEnvAE (Autoencoder-compressed state)")
        print(f"  AE checkpoint: {ae_checkpoint_path}")
    else:
        print(f"  Environment: ITSNEnv (Full geometry state)")
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
            make_env(use_ae, ae_checkpoint_path, i, seed, device, **env_kwargs)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(use_ae, ae_checkpoint_path, 0, seed, device, **env_kwargs)])

    # Disable reward normalization to preserve gradient signal
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=10.0)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(use_ae, ae_checkpoint_path, 999, seed, device, **env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_reward=10.0, training=False)

    # Create RecurrentPPO agent with LSTM
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Feature extractor before LSTM
        lstm_hidden_size=128,
        enable_critic_lstm=True,
        n_lstm_layers=1
    )

    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.2,  # Increased from 0.5 to improve value function learning
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=str(log_path / 'tensorboard'),
        device=device,
        seed=seed,
        target_kl=0.01,  # KL divergence target for policy updates
        policy_kwargs=policy_kwargs
    )

    print(f"\nRecurrentPPO Agent created:")
    print(f"  Policy: MlpLstmPolicy")
    print(f"  Feature extractor net_arch: pi=[256, 256], vf=[256, 256]")
    print(f"  LSTM hidden size: 256")
    print(f"  LSTM layers: 1")
    print(f"  Critic LSTM enabled: True")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(log_path / 'checkpoints'),
        name_prefix='recurrent_ppo_ae_model',
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
        use_ae=use_ae,
        ae_checkpoint_path=ae_checkpoint_path,
        eval_freq=eval_freq,  # 每eval_freq步进行一次详细评估
        log_path=log_path / 'trajectory_evals',
        device=device,
        verbose=1,
        **env_kwargs
    )

    # ========== Baseline 评估 (零动作) ==========
    if not skip_zero_baseline:
        print("\n" + "=" * 60)
        print("Baseline evaluation (zero action)...")
        print("=" * 60)
        baseline_results = evaluate_baseline(
            use_ae=use_ae,
            ae_checkpoint_path=ae_checkpoint_path,
            seed=seed,
            device=device,
            save_path=log_path / 'baseline_zero_action_eval.json',
            **env_kwargs
        )
    else:
        print("\n[Info] Skipping zero-action baseline evaluation")

    # ========== Baseline Optimizer 评估 (ZF+SDR) ==========
    if run_baseline_optimizer:
        print("\n" + "=" * 60)
        print("Baseline Optimizer evaluation (ZF+SDR)...")
        print("=" * 60)
        baseline_optimizer_results = evaluate_baseline_optimizer(
            use_ae=use_ae,
            ae_checkpoint_path=ae_checkpoint_path,
            seed=seed,
            device=device,
            save_path=log_path / 'baseline_optimizer_eval.json',
            **env_kwargs
        )
    else:
        print("\n[Info] Skipping baseline optimizer evaluation (use --run-baseline-optimizer to enable)")

    # ========== Baseline 对比摘要 ==========
    if not skip_zero_baseline and run_baseline_optimizer:
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON (Pre-training)")
        print("=" * 60)
        print(f"{'Method':<25} | {'Avg P_total':>12} | {'SINR_min':>10} | {'Success':>10}")
        print(f"-" * 60)
        print(f"{'Zero-Action':<25} | {baseline_results['summary']['avg_P_total']:>12.4f} | {baseline_results['summary']['avg_SINR_min_dB']:>10.2f} | {baseline_results['summary']['success_rate']*100:>9.1f}%")
        print(f"{'Baseline Optimizer':<25} | {baseline_optimizer_results['summary']['avg_P_total']:>12.4f} | {baseline_optimizer_results['summary']['avg_SINR_min_dB']:>10.2f} | {baseline_optimizer_results['summary']['success_rate']*100:>9.1f}%")
        print(f"=" * 60 + "\n")

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
    # 确保目录存在后保存 VecNormalize
    final_model_path.mkdir(parents=True, exist_ok=True)
    env.save(str(final_model_path / 'vec_normalize.pkl'))
    print(f"\nFinal model saved to {final_model_path}")

    # 最终详细评估
    print("\n" + "=" * 60)
    print("Final trajectory evaluation...")
    print("=" * 60)
    evaluate_on_trajectory(
        model=model,
        use_ae=use_ae,
        ae_checkpoint_path=ae_checkpoint_path,
        seed=None,
        device=device,
        save_path=log_path / 'final_trajectory_eval.json',
        **env_kwargs
    )

    # Close environments
    env.close()
    eval_env.close()

    # ========== 最终对比摘要 (DRL vs Baselines) ==========
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY (DRL Model vs Baselines)")
    print("=" * 70)
    print(f"{'Method':<25} | {'Avg P_total':>12} | {'SINR_min':>10} | {'Success':>10}")
    print(f"-" * 70)

    # Load baseline results if they exist
    baseline_zero_path = log_path / 'baseline_zero_action_eval.json'
    baseline_opt_path = log_path / 'baseline_optimizer_eval.json'
    final_eval_path = log_path / 'final_trajectory_eval.json'

    if baseline_zero_path.exists():
        with open(baseline_zero_path, 'r') as f:
            baseline_zero = json.load(f)
        print(f"{'Zero-Action Baseline':<25} | {baseline_zero['summary']['avg_P_total']:>12.4f} | {baseline_zero['summary']['avg_SINR_min_dB']:>10.2f} | {baseline_zero['summary']['success_rate']*100:>9.1f}%")

    if baseline_opt_path.exists():
        with open(baseline_opt_path, 'r') as f:
            baseline_opt = json.load(f)
        print(f"{'Baseline Optimizer':<25} | {baseline_opt['summary']['avg_P_total']:>12.4f} | {baseline_opt['summary']['avg_SINR_min_dB']:>10.2f} | {baseline_opt['summary']['success_rate']*100:>9.1f}%")

    if final_eval_path.exists():
        with open(final_eval_path, 'r') as f:
            final_eval = json.load(f)
        print(f"{'DRL Model (Final)':<25} | {final_eval['summary']['avg_P_total']:>12.4f} | {final_eval['summary']['avg_SINR_min_dB']:>10.2f} | {final_eval['summary']['success_rate']*100:>9.1f}%")

    print(f"=" * 70 + "\n")

    print("Training complete!")
    print("=" * 60)
    print(f"Results saved to: {log_path}")
    print(f"View tensorboard: tensorboard --logdir {log_path / 'tensorboard'}")

    return model, log_path


def evaluate_on_trajectory(
    model,
    use_ae=True,
    ae_checkpoint_path=None,
    seed=None,
    device='cuda',
    save_path=None,
    training_env=None,  # 新增：训练环境（包含归一化统计量）
    **env_kwargs
):
    """
    在随机轨道上评估模型，记录每个step的详细信息

    Args:
        model: 训练好的RecurrentPPO模型
        use_ae: 是否使用 Autoencoder 压缩状态
        ae_checkpoint_path: AE checkpoint路径（如果 use_ae=True 则必需）
        seed: 随机种子（None则随机生成轨道）
        device: 设备
        save_path: 保存结果的路径（None则不保存）
        training_env: 训练环境（用于获取归一化参数，保持评估与训练分布一致）
        **env_kwargs: 环境参数

    Returns:
        dict: 包含每个step的P_BS, P_sat, SINR_UE, SINR_SUE
    """
    # 创建评估环境 (禁用 ephemeris noise 以便公平比较)
    if seed is None:
        seed = np.random.randint(0, 100000)

    if use_ae:
        raw_env = ITSNEnvAE(
            ae_checkpoint_path=ae_checkpoint_path,
            rng_seed=seed,
            device=device,
            ephemeris_noise_std=0.0,
            **env_kwargs
        )
    else:
        raw_env = ITSNEnv(
            rng_seed=seed,
            ephemeris_noise_std=0.0,
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

    # 运行一个完整episode (LSTM需要maintain hidden states)
    obs, info = raw_env.reset()
    done = False
    step = 0

    # Initialize LSTM hidden states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    print(f"\n{'='*60}")
    print(f"Evaluating on trajectory (seed={seed})")
    print(f"{'='*60}")
    print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Success':>7}")
    print(f"{'-'*60}")

    while not done:
        # 关键步骤：使用训练环境的归一化器对观测值进行归一化
        # 这样可以确保评估时的观测值分布与训练时一致
        obs_normalized = training_env.normalize_obs(obs) if training_env is not None else obs

        # 获取动作（使用LSTM states）
        # 显式确保 obs 是二维的 (1, obs_dim)，以匹配 RecurrentPPO 的预期
        action, lstm_states = model.predict(
            obs_normalized[np.newaxis, :],
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        action = action.squeeze()
        episode_starts = np.zeros((1,), dtype=bool)

        # 执行动作
        obs, reward, terminated, truncated, info = raw_env.step(action)
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

    raw_env.close()

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
    use_ae=True,
    ae_checkpoint_path=None,
    seed=42,
    device='cuda',
    save_path=None,
    **env_kwargs
):
    """
    Baseline 评估：使用零动作（不调整RIS相位）

    Args:
        use_ae: 是否使用 Autoencoder 压缩状态
        ae_checkpoint_path: AE checkpoint路径（如果 use_ae=True 则必需）
        seed: 随机种子
        device: 设备
        save_path: 保存结果的路径
        **env_kwargs: 环境参数

    Returns:
        dict: 包含每个step的详细信息
    """
    if use_ae:
        env = ITSNEnvAE(
            ae_checkpoint_path=ae_checkpoint_path,
            rng_seed=seed,
            device=device
            **env_kwargs
        )
    else:
        env = ITSNEnv(
            rng_seed=seed,
            ephemeris_noise_std=0.0,
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

    env.enable_ephemeris_noise = False  # 确保在 reset 前设置
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


def evaluate_baseline_optimizer(
    use_ae=False,
    ae_checkpoint_path=None,
    seed=42,
    device='cuda',
    save_path=None,
    **env_kwargs
):
    """
    Baseline Optimizer 评估：使用 ZF+SDR 方法 (baseline/baseline_optimizer.py)

    Args:
        use_ae: 是否使用 Autoencoder 压缩状态 (与baseline无关，保持接口一致)
        ae_checkpoint_path: AE checkpoint路径
        seed: 随机种子
        device: 设备
        save_path: 保存结果的路径
        **env_kwargs: 环境参数

    Returns:
        dict: 包含每个step的详细信息
    """
    # 创建评估环境 (禁用 ephemeris noise 以便与 baseline 公平比较)
    env = ITSNEnv(
        rng_seed=seed, # 禁用噪声，使用真实卫星位置
        ephemeris_noise_std=0.0,
        **env_kwargs
    )

    # 初始化 baseline optimizer
    sinr_threshold_db = env.sinr_threshold_db
    sinr_threshold_linear = 10 ** (sinr_threshold_db / 10.0)
    gamma_k = np.full(env.scenario.K, sinr_threshold_linear)
    gamma_j = sinr_threshold_linear

    baseline = BaselineZFSDROptimizer(
        K=env.scenario.K,
        J=env.scenario.J,
        N_t=env.scenario.N_t,
        N_s=env.scenario.N_s,
        N=env.scenario.N_ris,
        P_max=env.scenario.P_bs_max,
        sigma2=env.scenario.P_noise,
        gamma_k=gamma_k,
        gamma_j=gamma_j,
        ris_amplitude_gain=env.scenario.ris_amplitude_gain,
        N_iter=10,  # 减少迭代次数以加快评估
        verbose=False
    )

    # 记录数据
    results = {
        'seed': seed,
        'type': 'baseline_optimizer_zf_sdr',
        'steps': [],
        'P_BS': [],
        'P_sat': [],
        'P_total': [],
        'SINR_UE': [],
        'SINR_SUE': [],
        'SINR_min_dB': [],
        'success': [],
        'true_elevation': [],
        'true_azimuth': [],
        'opt_time': [],
    }

    env.enable_ephemeris_noise = False  # 确保在 reset 前设置
    obs, info = env.reset()
    done = False
    step = 0

    print(f"\n{'='*70}")
    print(f"Baseline Optimizer Evaluation (ZF+SDR, seed={seed})")
    print(f"{'='*70}")
    print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Time':>8} | {'Success':>7}")
    print(f"{'-'*75}")

    import time
    total_opt_time = 0

    while not done:
        t_start = time.time()

        # 获取当前信道
        channels = env.current_channels

        # 运行 baseline 优化
        try:
            w_opt, phi_opt, opt_info = baseline.optimize(
                h_k=channels['h_k'],
                h_j=channels['h_j'],
                h_s_k=channels['h_s_k'],
                h_s_j=channels['h_s_j'],
                h_k_r=channels['h_k_r'],
                h_j_r=channels['h_j_r'],
                G_BS=channels['G_BS'],
                G_S=env.true_G_SAT,  # 使用真实 G_SAT (无 ephemeris 不确定性)
                W_sat=channels['W_sat']
            )

            # 计算实际性能
            all_sinr = env.calculate_all_sinrs(phi_opt, w_opt, baseline.P_s, channels)

            P_BS = opt_info['final_P_bs']
            P_sat = opt_info['final_P_sat']
            P_total = P_BS + P_sat

            # 检查是否满足约束
            soft_threshold = 10 ** ((env.sinr_threshold_db - 0.5) / 10.0)
            success = np.all(all_sinr >= soft_threshold)

        except Exception as e:
            print(f"  [Warning] Optimization failed at step {step}: {e}")
            P_BS, P_sat = 10.0, 10.0
            P_total = 20.0
            all_sinr = np.zeros(env.scenario.K + 1)
            success = False

        opt_time = time.time() - t_start
        total_opt_time += opt_time

        # 转换为 dB
        sinr_min_db = 10 * np.log10(np.min(all_sinr) + 1e-12)
        sinr_ue_db = 10 * np.log10(all_sinr[:env.scenario.K] + 1e-12)
        sinr_sue_db = 10 * np.log10(all_sinr[env.scenario.K] + 1e-12)

        results['steps'].append(step)
        results['P_BS'].append(float(P_BS))
        results['P_sat'].append(float(P_sat))
        results['P_total'].append(float(P_total))
        results['SINR_UE'].append(sinr_ue_db.tolist())
        results['SINR_SUE'].append(float(sinr_sue_db))
        results['SINR_min_dB'].append(float(sinr_min_db))
        results['success'].append(bool(success))
        results['true_elevation'].append(float(env.true_elevation))
        results['true_azimuth'].append(float(env.true_azimuth))
        results['opt_time'].append(float(opt_time))

        success_str = "Y" if success else "N"
        print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {opt_time:>8.4f} | {success_str:>7}")

        # Step 环境 (用零动作，因为我们已经在 optimizer 里计算了最优解)
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.scenario.N_ris))
        done = terminated or truncated

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
        'total_opt_time': float(total_opt_time),
        'avg_opt_time': float(total_opt_time / max(step, 1)),
    }

    print(f"{'-'*75}")
    print(f"Baseline Optimizer Summary:")
    print(f"  Total steps: {step}")
    print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
    print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
    print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
    print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
    print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
    print(f"  Avg optimization time: {results['summary']['avg_opt_time']:.4f} s")
    print(f"{'='*70}\n")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Baseline Optimizer results saved to {save_path}")

    return results


class TrajectoryEvalCallback(BaseCallback):
    """
    自定义Callback：每隔一定步数在随机轨道上进行详细评估
    """
    def __init__(
        self,
        use_ae=True,
        ae_checkpoint_path=None,
        eval_freq=50000,
        log_path=None,
        device='cuda',
        verbose=1,
        **env_kwargs
    ):
        super().__init__(verbose)
        self.use_ae = use_ae
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

            # === 修改点开始 ===
            # 获取正在训练的环境 (它包含了当前的均值和方差)
            training_env = self.model.get_env()
            
            evaluate_on_trajectory(
                model=self.model,
                use_ae=self.use_ae,
                ae_checkpoint_path=self.ae_checkpoint_path,
                seed=None,
                device=self.device,
                save_path=save_path,
                training_env=training_env,  # <--- 核心：传入训练环境
                **self.env_kwargs
            )
            # === 修改点结束 ===
        return True


def main():
    parser = argparse.ArgumentParser(description='Train RecurrentPPO agent (supports both AE-compressed and full geometry state)')

    # Environment type
    parser.add_argument('--use-ae', action='store_true',
                       help='Use autoencoder-compressed state (default: False, use full geometry state)')

    # AE checkpoint (required if use-ae=True)
    parser.add_argument('--ae-checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to pre-trained autoencoder checkpoint (required if --use-ae)')

    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=102400,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=32,
                       help='Number of parallel environments (recommended: 16-32 for episode_length=40)')
    parser.add_argument('--n-steps', type=int, default=128,
                       help='Steps per environment per update (recommended: 64-128 for episode_length=40)')
    parser.add_argument('--learning-rate', type=float, default=7e-5,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    # Environment parameters
    parser.add_argument('--max-steps', type=int, default=128,
                       help='Max steps per episode (should match n_steps for 1 episode per rollout)')
    parser.add_argument('--n-substeps', type=int, default=1,
                       help='Physics substeps per RL step (use 1 to avoid channel mismatch)')
    parser.add_argument('--phase-bits', type=int, default=4,
                       help='RIS phase quantization bits')
    parser.add_argument('--latent-dim', type=int, default=128,
                       help='AE latent dimension (will be auto-detected from checkpoint if available)')

    # Logging
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                       help='Checkpoint frequency')

    # Baseline evaluation
    parser.add_argument('--run-baseline-optimizer', action='store_true',
                       help='Run baseline optimizer evaluation (ZF+SDR) before and after training')
    parser.add_argument('--skip-zero-baseline', action='store_true',
                       help='Skip zero-action baseline evaluation')

    args = parser.parse_args()

    # Check AE checkpoint if using AE
    ae_checkpoint_path = None
    if args.use_ae:
        ae_checkpoint = Path(args.ae_checkpoint)
        if not ae_checkpoint.exists():
            print(f"Error: AE checkpoint not found at {ae_checkpoint}")
            print("Please train the autoencoder first:")
            print("  python scripts/train_channel_ae.py")
            return 1
        ae_checkpoint_path = str(ae_checkpoint)
    else:
        print("Using full geometry state (ITSNEnv)")

    # Environment kwargs (device will be passed separately to train_rl_agent)
    env_kwargs = {
        'max_steps_per_episode': args.max_steps,
        'n_substeps': args.n_substeps,
        'phase_bits': args.phase_bits,
    }
    # Only add latent_dim if using AE
    if args.use_ae:
        env_kwargs['latent_dim'] = args.latent_dim

    # Train
    model, log_path = train_rl_agent(
        use_ae=args.use_ae,
        ae_checkpoint_path=ae_checkpoint_path,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_freq=args.checkpoint_freq,
        run_baseline_optimizer=args.run_baseline_optimizer,
        skip_zero_baseline=args.skip_zero_baseline,
        **env_kwargs
    )

    return 0


if __name__ == '__main__':
    exit(main())
