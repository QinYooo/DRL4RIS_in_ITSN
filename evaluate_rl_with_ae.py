"""
Evaluate trained RL agent (supports both AE-compressed and full geometry state)
Load model from logs directory and run detailed trajectory evaluation
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt兼容性问题

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from envs.itsn_env import ITSNEnv
from envs.itsn_env_ae import ITSNEnvAE
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle


def evaluate_on_trajectory(
    model,
    env,
    is_recurrent=False,
    verbose=True
):
    """
    在单条轨道上评估模型，记录每个step的详细信息
    正确处理 RecurrentPPO 的 LSTM state 和 VecNormalize

    Args:
        model: 训练好的PPO或RecurrentPPO模型
        env: 已被VecNormalize包裹的评估环境
        is_recurrent: 是否为RecurrentPPO模型
        verbose: 是否打印详细信息

    Returns:
        dict: 包含每个step的P_BS, P_sat, SINR_UE, SINR_SUE
    """
    # 记录数据
    results = {
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
        'obs_elevation': [],
        'obs_azimuth': [],
        'rewards': [],
    }

    # 初始化 LSTM state (每条轨迹开始时重置)
    lstm_state = None
    episode_start = True  # 标记episode开始，用于LSTM

    # 运行一个完整episode
    # VecEnv.reset() 返回 obs, Gym Env.reset() 返回 (obs, info)
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    done = False
    step = 0

    if verbose:
        print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Success':>7}")
        print(f"{'-'*60}")

    while not done:
        # ====== 关键修改1: 正确处理 RecurrentPPO LSTM state ======
        if is_recurrent:
            # RecurrentPPO predict: 返回 (action, lstm_state)
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=np.array([episode_start]),
                deterministic=True
            )
            # 第一步之后，episode_start 设为 False
            episode_start = False
        else:
            # 普通 PPO predict
            action, _ = model.predict(obs, deterministic=True)

        # 执行动作
        # VecEnv.step() 返回 (obs, reward, done, infos) 4个值, infos 是列表
        # Gym Env.step() 返回 (obs, reward, terminated, truncated, info) 5个值, info 是字典
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            obs, reward, done, infos = step_result
            # VecEnv 的 infos 是列表，取第一个环境的 info
            info = infos[0] if isinstance(infos, list) else infos
        else:
            raise ValueError(f"Unexpected step result: {step_result}")

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
        results['rewards'].append(float(reward))

        # 打印当前step信息
        if verbose:
            success_str = "Y" if success else "N"
            print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {success_str:>7}")

        step += 1

    # 统计信息
    results['summary'] = {
        'total_steps': step,
        'total_reward': float(np.sum(results['rewards'])),
        'avg_P_BS': float(np.mean(results['P_BS'])),
        'avg_P_sat': float(np.mean(results['P_sat'])),
        'avg_P_total': float(np.mean(results['P_total'])),
        'avg_SINR_min_dB': float(np.mean(results['SINR_min_dB'])),
        'min_SINR_min_dB': float(np.min(results['SINR_min_dB'])),
        'success_rate': float(np.mean(results['success'])),
    }

    if verbose:
        print(f"{'-'*60}")
        print(f"Summary:")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {results['summary']['total_reward']:.2f}")
        print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
        print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
        print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
        print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
        print(f"  Min SINR_min: {results['summary']['min_SINR_min_dB']:.2f} dB")
        print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")

    return results


def evaluate_multiple_trajectories(model, use_ae=True, ae_checkpoint_path=None, n_episodes=10,
                                    seed=None, device='cuda', verbose=False, is_recurrent=False,
                                    vecnormalize_path=None, **env_kwargs):
    """
    在多条随机轨道上评估模型（支持 AE 压缩和完整几何状态两种环境）
    正确加载并使用训练时的 VecNormalize 统计量

    Args:
        model: 训练好的模型
        use_ae: 是否使用 Autoencoder 压缩状态
        ae_checkpoint_path: AE checkpoint路径（如果 use_ae=True 则必需）
        n_episodes: 评估的轨道数量
        seed: 基础随机种子
        device: 设备
        verbose: 是否打印每条轨道的详细信息
        is_recurrent: 是否为RecurrentPPO模型（用于正确处理LSTM state）
        vecnormalize_path: VecNormalize .pkl文件路径（用于加载训练统计量）
        **env_kwargs: 环境参数

    Returns:
        list: 每条轨道的评估结果
    """
    all_results = []

    if seed is None:
        seed = np.random.randint(0, 100000)

    for i in tqdm(range(n_episodes), desc="Evaluating trajectories"):
        # ====== 关键修改2: 加载并使用训练时的 VecNormalize 统计量 ======
        # 首先创建环境函数，避免lambda捕获问题
        def make_env():
            if use_ae:
                return ITSNEnvAE(
                    ae_checkpoint_path=ae_checkpoint_path,
                    rng_seed=seed + i,
                    device=device,
                    **env_kwargs
                )
            else:
                return ITSNEnv(
                    rng_seed=seed + i,
                    **env_kwargs
                )

        if vecnormalize_path is not None:
            # 用 DummyVecEnv 包裹，然后用 VecNormalize.load 加载训练统计量
            env = DummyVecEnv([make_env])
            env = VecNormalize.load(vecnormalize_path, env)
            # 评估模式设置
            env.training = False
            env.norm_reward = False  # 评估时不归一化reward，输出原始reward
        else:
            # 没有归一化文件，直接使用原始环境
            env = make_env()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Trajectory {i+1}/{n_episodes} (seed={seed + i})")
            print(f"{'='*60}")

        results = evaluate_on_trajectory(model, env, is_recurrent=is_recurrent, verbose=verbose)
        results['seed'] = seed + i
        all_results.append(results)

        env.close()

    return all_results


def aggregate_results(all_results):
    """汇总多条轨道的评估结果"""
    summaries = [r['summary'] for r in all_results]

    aggregated = {
        'n_trajectories': len(all_results),
        'avg_reward': np.mean([s['total_reward'] for s in summaries]),
        'std_reward': np.std([s['total_reward'] for s in summaries]),
        'avg_P_BS': np.mean([s['avg_P_BS'] for s in summaries]),
        'std_P_BS': np.std([s['avg_P_BS'] for s in summaries]),
        'avg_P_sat': np.mean([s['avg_P_sat'] for s in summaries]),
        'std_P_sat': np.std([s['avg_P_sat'] for s in summaries]),
        'avg_P_total': np.mean([s['avg_P_total'] for s in summaries]),
        'std_P_total': np.std([s['avg_P_total'] for s in summaries]),
        'avg_SINR_min_dB': np.mean([s['avg_SINR_min_dB'] for s in summaries]),
        'std_SINR_min_dB': np.std([s['avg_SINR_min_dB'] for s in summaries]),
        'avg_success_rate': np.mean([s['success_rate'] for s in summaries]),
        'std_success_rate': np.std([s['success_rate'] for s in summaries]),
    }

    return aggregated


def print_aggregated_summary(aggregated):
    """打印汇总结果"""
    print("\n" + "=" * 60)
    print(f"Aggregated Results ({aggregated['n_trajectories']} trajectories)")
    print("=" * 60)
    print(f"  Reward:      {aggregated['avg_reward']:.2f} +/- {aggregated['std_reward']:.2f}")
    print(f"  P_BS:        {aggregated['avg_P_BS']:.4f} +/- {aggregated['std_P_BS']:.4f} W")
    print(f"  P_sat:       {aggregated['avg_P_sat']:.4f} +/- {aggregated['std_P_sat']:.4f} W")
    print(f"  P_total:     {aggregated['avg_P_total']:.4f} +/- {aggregated['std_P_total']:.4f} W")
    print(f"  SINR_min:    {aggregated['avg_SINR_min_dB']:.2f} +/- {aggregated['std_SINR_min_dB']:.2f} dB")
    print(f"  Success:     {aggregated['avg_success_rate']*100:.1f}% +/- {aggregated['std_success_rate']*100:.1f}%")
    print("=" * 60)


def plot_trajectory_results(results, save_path=None):
    """绘制单条轨道的详细结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    steps = results['steps']

    # 1. 功耗随时间变化
    ax = axes[0, 0]
    ax.plot(steps, results['P_BS'], label='P_BS', marker='o', markersize=3)
    ax.plot(steps, results['P_sat'], label='P_sat', marker='s', markersize=3)
    ax.plot(steps, results['P_total'], label='P_total', marker='^', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Consumption over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SINR随时间变化
    ax = axes[0, 1]
    sinr_ue = np.array(results['SINR_UE'])
    for k in range(sinr_ue.shape[1]):
        ax.plot(steps, sinr_ue[:, k], label=f'UE{k+1}', alpha=0.7)
    ax.plot(steps, results['SINR_SUE'], label='SUE', linestyle='--', linewidth=2)
    ax.axhline(y=10.0, color='r', linestyle=':', label='Threshold (10dB)')
    ax.set_xlabel('Step')
    ax.set_ylabel('SINR (dB)')
    ax.set_title('SINR over Time')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 3. 最小SINR随时间变化
    ax = axes[0, 2]
    ax.plot(steps, results['SINR_min_dB'], color='purple', marker='o', markersize=3)
    ax.axhline(y=10.0, color='r', linestyle=':', label='Threshold (10dB)')
    ax.fill_between(steps, results['SINR_min_dB'], 10.0,
                    where=np.array(results['SINR_min_dB']) < 10.0,
                    color='red', alpha=0.3, label='Violation')
    ax.set_xlabel('Step')
    ax.set_ylabel('Min SINR (dB)')
    ax.set_title('Minimum SINR over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 卫星轨迹 (仰角)
    ax = axes[1, 0]
    ax.plot(steps, results['true_elevation'], label='True', marker='o', markersize=3)
    ax.plot(steps, results['obs_elevation'], label='Observed', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Elevation (deg)')
    ax.set_title('Satellite Elevation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 卫星轨迹 (方位角)
    ax = axes[1, 1]
    ax.plot(steps, results['true_azimuth'], label='True', marker='o', markersize=3)
    ax.plot(steps, results['obs_azimuth'], label='Observed', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Azimuth (deg)')
    ax.set_title('Satellite Azimuth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 成功/失败标记
    ax = axes[1, 2]
    success_arr = np.array(results['success']).astype(int)
    colors = ['green' if s else 'red' for s in results['success']]
    ax.bar(steps, success_arr, color=colors, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Success')
    ax.set_title(f"Constraint Satisfaction (Rate: {np.mean(results['success'])*100:.1f}%)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Fail', 'Success'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


def plot_aggregated_results(all_results, save_path=None):
    """绘制多条轨道的汇总结果"""
    summaries = [r['summary'] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 总功耗分布
    ax = axes[0, 0]
    total_powers = [s['avg_P_total'] for s in summaries]
    ax.hist(total_powers, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(total_powers), color='r', linestyle='--',
               label=f'Mean: {np.mean(total_powers):.4f} W')
    ax.set_xlabel('Average Total Power (W)')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Power Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 成功率分布
    ax = axes[0, 1]
    success_rates = [s['success_rate'] * 100 for s in summaries]
    ax.hist(success_rates, bins=20, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(np.mean(success_rates), color='r', linestyle='--',
               label=f'Mean: {np.mean(success_rates):.1f}%')
    ax.set_xlabel('Success Rate (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Success Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. P_BS vs P_sat
    ax = axes[1, 0]
    p_bs = [s['avg_P_BS'] for s in summaries]
    p_sat = [s['avg_P_sat'] for s in summaries]
    ax.scatter(p_bs, p_sat, alpha=0.6, s=50)
    ax.set_xlabel('Average P_BS (W)')
    ax.set_ylabel('Average P_sat (W)')
    ax.set_title('BS Power vs Satellite Power')
    ax.grid(True, alpha=0.3)

    # 4. SINR_min分布
    ax = axes[1, 1]
    sinr_mins = [s['avg_SINR_min_dB'] for s in summaries]
    ax.hist(sinr_mins, bins=20, alpha=0.7, edgecolor='black', color='purple')
    ax.axvline(np.mean(sinr_mins), color='r', linestyle='--',
               label=f'Mean: {np.mean(sinr_mins):.2f} dB')
    ax.axvline(10.0, color='orange', linestyle=':', label='Threshold (10 dB)')
    ax.set_xlabel('Average Min SINR (dB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Minimum SINR Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


def find_latest_model(log_dir='logs'):
    """查找最新的训练模型和对应的VecNormalize文件

    Returns:
        model_path: 模型路径或None
        run_dir: 运行目录或None
        vecnormalize_path: VecNormalize文件路径或None
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None, None, None

    # 查找所有训练目录 (PPO_AE_* 或 RecurrentPPO_*)
    run_dirs = sorted(log_path.glob('PPO_AE_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    run_dirs.extend(sorted(log_path.glob('RecurrentPPO_*'), key=lambda x: x.stat().st_mtime, reverse=True))

    for run_dir in run_dirs:
        # 优先查找best_model
        best_model = run_dir / 'best_model' / 'best_model.zip'
        if best_model.exists():
            # 查找对应的vecnormalize.pkl (搜索 run_dir/best_model/ 和 run_dir/)
            vecnormalize_path = None
            for pkl_file in (run_dir / 'best_model').glob('*.pkl'):
                if 'vec' in pkl_file.name.lower() and 'normalize' in pkl_file.name.lower():
                    vecnormalize_path = pkl_file
                    break
            if vecnormalize_path is None:
                for pkl_file in (run_dir / 'final_model').glob('*.pkl'):
                    if 'vec' in pkl_file.name.lower() and 'normalize' in pkl_file.name.lower():
                        vecnormalize_path = pkl_file
                        break
            return best_model, run_dir, vecnormalize_path

        # 其次查找final_model
        final_model = run_dir / 'final_model.zip'
        if final_model.exists():
            # 查找对应的vecnormalize.pkl (搜索 run_dir/final_model/ 和 run_dir/)
            vecnormalize_path = None
            final_model_dir = run_dir / 'final_model'
            if final_model_dir.is_dir():
                for pkl_file in final_model_dir.glob('*.pkl'):
                    if 'vec' in pkl_file.name.lower() and 'normalize' in pkl_file.name.lower():
                        vecnormalize_path = pkl_file
                        break
            if vecnormalize_path is None:
                for pkl_file in run_dir.glob('*.pkl'):
                    if 'vec' in pkl_file.name.lower() and 'normalize' in pkl_file.name.lower():
                        vecnormalize_path = pkl_file
                        break
            return final_model, run_dir, vecnormalize_path

    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent (supports AE-compressed and full geometry state)')

    # 环境类型
    parser.add_argument('--use-ae', action='store_true',
                       help='Use autoencoder-compressed state (default: False, use full geometry state)')

    # 模型路径
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model. If not specified, uses latest model in logs/')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory to search for models')
    parser.add_argument('--ae-checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to AE checkpoint (required if --use-ae)')

    # 评估参数
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of evaluation trajectories')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (None for random)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed info for each trajectory')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')

    # 环境参数
    parser.add_argument('--max-steps', type=int, default=64,
                       help='Max steps per episode')
    parser.add_argument('--n-substeps', type=int, default=1,
                       help='Physics substeps per RL step')
    parser.add_argument('--latent-dim', type=int, default=128,
                       help='AE latent dimension (will be auto-detected from checkpoint if available)')

    args = parser.parse_args()

    # 查找模型和VecNormalize
    if args.model_path is None:
        model_path, run_dir, vecnormalize_path = find_latest_model(args.log_dir)
        if model_path is None:
            print(f"Error: No model found in {args.log_dir}")
            print("Please specify --model-path or train a model first")
            return 1
        print(f"Using latest model: {model_path}")
        if vecnormalize_path is not None:
            print(f"Using VecNormalize from: {vecnormalize_path}")
    else:
        model_path = Path(args.model_path)
        run_dir = model_path.parent.parent if 'best_model' in str(model_path) else model_path.parent
        # 尝试在运行目录中查找vecnormalize.pkl (搜索 run_dir 和子目录)
        vecnormalize_path = None
        for search_dir in [run_dir, model_path.parent]:
            for pkl_file in search_dir.glob('*.pkl'):
                if 'vec' in pkl_file.name.lower() and 'normalize' in pkl_file.name.lower():
                    vecnormalize_path = pkl_file
                    print(f"Found VecNormalize: {vecnormalize_path}")
                    break
            if vecnormalize_path is not None:
                break

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    # 检查 AE checkpoint（如果使用 AE）
    ae_checkpoint_path = None
    if args.use_ae:
        ae_checkpoint = Path(args.ae_checkpoint)
        if not ae_checkpoint.exists():
            print(f"Error: AE checkpoint not found at {ae_checkpoint}")
            return 1
        ae_checkpoint_path = str(ae_checkpoint)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    if args.use_ae:
        print("Evaluating RL Agent with AE-Compressed State")
    else:
        print("Evaluating RL Agent with Full Geometry State")
    print("=" * 60)
    print(f"Model: {model_path}")
    if args.use_ae:
        print(f"AE checkpoint: {ae_checkpoint_path}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # 加载模型（根据类型选择 PPO 或 RecurrentPPO）
    print("\nLoading model...")
    # 检测模型类型：如果目录名包含 RecurrentPPO 则使用 RecurrentPPO
    is_recurrent = 'RecurrentPPO' in str(run_dir)
    if is_recurrent:
        model = RecurrentPPO.load(model_path)
        print("Using RecurrentPPO (LSTM policy)")
    else:
        model = PPO.load(model_path)
        print("Using PPO")

    # 环境参数
    env_kwargs = {
        'max_steps_per_episode': args.max_steps,
        'n_substeps': args.n_substeps,
    }
    # Only add latent_dim if using AE
    if args.use_ae:
        env_kwargs['latent_dim'] = args.latent_dim

    # 评估多条轨道
    print(f"\nEvaluating on {args.n_episodes} trajectories...")
    all_results = evaluate_multiple_trajectories(
        model=model,
        use_ae=args.use_ae,
        ae_checkpoint_path=ae_checkpoint_path,
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        is_recurrent=is_recurrent,  # ====== 关键修改3: 传递 recurrent 标志 ======
        vecnormalize_path=vecnormalize_path,  # ====== 关键修改4: 传递 VecNormalize 路径 ======
        **env_kwargs
    )

    # 汇总结果
    aggregated = aggregate_results(all_results)
    print_aggregated_summary(aggregated)

    # 保存结果
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'aggregated': aggregated,
            'trajectories': all_results
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # 绘制汇总图
    plot_path = output_dir / 'aggregated_results.png'
    plot_aggregated_results(all_results, save_path=plot_path)

    # 绘制第一条轨道的详细图
    if len(all_results) > 0:
        detail_plot_path = output_dir / 'trajectory_detail.png'
        plot_trajectory_results(all_results[0], save_path=detail_plot_path)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
