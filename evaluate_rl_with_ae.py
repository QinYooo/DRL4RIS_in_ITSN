"""
Evaluate trained RL agent with AE-compressed state
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

from envs.itsn_env_ae import ITSNEnvAE
from stable_baselines3 import PPO


def evaluate_on_trajectory(
    model,
    env,
    verbose=True
):
    """
    在单条轨道上评估模型，记录每个step的详细信息

    Args:
        model: 训练好的PPO模型
        env: 评估环境
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

    # 运行一个完整episode
    obs, info = env.reset()
    done = False
    step = 0

    if verbose:
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


def evaluate_multiple_trajectories(model, ae_checkpoint_path, n_episodes=10,
                                    seed=None, device='cuda', verbose=False, **env_kwargs):
    """
    在多条随机轨道上评估模型

    Args:
        model: 训练好的模型
        ae_checkpoint_path: AE checkpoint路径
        n_episodes: 评估的轨道数量
        seed: 基础随机种子
        device: 设备
        verbose: 是否打印每条轨道的详细信息
        **env_kwargs: 环境参数

    Returns:
        list: 每条轨道的评估结果
    """
    all_results = []

    if seed is None:
        seed = np.random.randint(0, 100000)

    for i in tqdm(range(n_episodes), desc="Evaluating trajectories"):
        env = ITSNEnvAE(
            ae_checkpoint_path=ae_checkpoint_path,
            rng_seed=seed + i,
            device=device,
            **env_kwargs
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Trajectory {i+1}/{n_episodes} (seed={seed + i})")
            print(f"{'='*60}")

        results = evaluate_on_trajectory(model, env, verbose=verbose)
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
    """查找最新的训练模型"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None, None

    # 查找所有PPO_AE开头的目录
    run_dirs = sorted(log_path.glob('PPO_AE_*'), key=lambda x: x.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        # 优先查找best_model
        best_model = run_dir / 'best_model' / 'best_model.zip'
        if best_model.exists():
            return best_model, run_dir

        # 其次查找final_model
        final_model = run_dir / 'final_model.zip'
        if final_model.exists():
            return final_model, run_dir

    return None, None


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent with AE')

    # 模型路径
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model. If not specified, uses latest model in logs/')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory to search for models')
    parser.add_argument('--ae-checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to AE checkpoint')

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
    parser.add_argument('--n-substeps', type=int, default=10,
                       help='Physics substeps per RL step')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='AE latent dimension')

    args = parser.parse_args()

    # 查找模型
    if args.model_path is None:
        model_path, run_dir = find_latest_model(args.log_dir)
        if model_path is None:
            print(f"Error: No model found in {args.log_dir}")
            print("Please specify --model-path or train a model first")
            return 1
        print(f"Using latest model: {model_path}")
    else:
        model_path = Path(args.model_path)
        run_dir = model_path.parent.parent if 'best_model' in str(model_path) else model_path.parent

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    ae_checkpoint = Path(args.ae_checkpoint)
    if not ae_checkpoint.exists():
        print(f"Error: AE checkpoint not found at {ae_checkpoint}")
        return 1

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Evaluating RL Agent with AE-Compressed State")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"AE checkpoint: {ae_checkpoint}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # 加载模型
    print("\nLoading model...")
    model = PPO.load(model_path)

    # 环境参数
    env_kwargs = {
        'max_steps_per_episode': args.max_steps,
        'n_substeps': args.n_substeps,
        'latent_dim': args.latent_dim,
    }

    # 评估多条轨道
    print(f"\nEvaluating on {args.n_episodes} trajectories...")
    all_results = evaluate_multiple_trajectories(
        model=model,
        ae_checkpoint_path=str(ae_checkpoint),
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
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
