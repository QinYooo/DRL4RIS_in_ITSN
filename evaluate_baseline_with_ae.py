"""
Evaluate Baseline (ZF+SDR) with Autoencoder
============================================
Evaluate the ZF+SDR baseline optimizer on the ITSN scenario with AE state.
This script evaluates the baseline method in a single satellite pass,
similar to the zero-action baseline evaluation in train_rl_with_ae.py.

Usage:
    python evaluate_baseline_with_ae.py --ae_checkpoint checkpoints/channel_ae/channel_ae_best.pth
"""

import sys
import os
import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from baseline.baseline_optimizer import BaselineZFSDROptimizer
from envs.itsn_env_ae import ITSNEnvAE


def evaluate_baseline_single_episode(
    ae_checkpoint_path: str,
    seed: int = 42,
    device: str = 'cuda',
    save_path: str = None,
    verbose: bool = True,
    **env_kwargs
):
    """
    Evaluate baseline optimizer for a single satellite pass using AE environment.

    This mimics the evaluate_baseline function in train_rl_with_ae.py,
    but uses the ZF+SDR baseline optimizer instead of zero action.

    Args:
        ae_checkpoint_path: Path to autoencoder checkpoint
        seed: Random seed for trajectory
        device: Device for AE inference
        save_path: Path to save results (None = don't save)
        verbose: Print progress
        **env_kwargs: Additional environment arguments

    Returns:
        dict: Results with per-step metrics
    """
    # Create environment
    env = ITSNEnvAE(
        ae_checkpoint_path=ae_checkpoint_path,
        rng_seed=seed,
        device=device,
        **env_kwargs
    )

    # Record data (same format as train_rl_with_ae.py evaluate_baseline)
    results = {
        'seed': seed,
        'type': 'baseline_ZF_SDR',
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
        'obs_elevation': [],
        'obs_azimuth': [],
    }

    # Initialize optimizer
    K = env.scenario.K
    J = env.scenario.SK
    N_t = env.scenario.N_t
    N_s = env.scenario.N_sat
    N = env.scenario.N_ris
    P_max = env.scenario.P_bs_max
    sigma2 = env.scenario.P_noise

    # SINR thresholds (linear scale)
    target_sinr_dB = 10
    gamma_k = np.ones(K) * (10 ** (target_sinr_dB / 10))
    gamma_j = 10 ** (target_sinr_dB / 10)

    optimizer = BaselineZFSDROptimizer(
        K=K, J=J, N_t=N_t, N_s=N_s, N=N,
        P_max=P_max, sigma2=sigma2,
        gamma_k=gamma_k, gamma_j=gamma_j,
        P_b=env.scenario.P_bs_scale,
        P_s_init=env.scenario.P_sat,
        ris_amplitude_gain=env.scenario.ris_amplitude_gain,
        N_iter=10,  # Reduce iterations for faster evaluation
        verbose=False  # Disable verbose to reduce output
    )

    # Reset environment to get trajectory
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0

    # Store optimization result (update every step for baseline)
    w_opt = None
    Phi_opt = None

    print(f"\n{'='*70}")
    print(f"Baseline Evaluation (ZF+SDR, seed={seed})")
    print(f"{'='*70}")
    print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Reward':>8} | {'Success':>7}")
    print(f"{'-'*70}")

    while not done:
        # Update satellite position and generate channels
        # The environment updates satellite position internally
        # But we need the actual channels for baseline optimization

        # Get current channels from environment
        channels = env.current_channels

        # Extract channels
        h_k = channels['H_BS2UE']
        h_j = channels['H_BS2SUE']
        h_s_k = channels['H_SAT2UE']
        h_s_j = channels['H_SAT2SUE']
        h_k_r = channels['H_RIS2UE']
        h_j_r = channels['H_RIS2SUE']
        G_BS = channels['G_BS']
        G_S = channels['G_SAT']
        W_sat = channels['W_sat']

        # Run baseline optimization at each step
        w_opt, Phi_opt, opt_info = optimizer.optimize(
            h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S, W_sat
        )

        # Convert optimized RIS phase to action for environment step
        # The action is phase angles in radians, quantized
        phase_bits = env.phase_bits
        phase_levels = 2 ** phase_bits
        phase_values = 2 * np.pi * np.arange(phase_levels) / phase_levels

        # Extract phase angles from Phi (diagonal elements)
        phi_angles = np.angle(np.diag(Phi_opt))

        # Quantize to discrete phases
        phi_quantized = phase_values[np.argmin(np.abs(phi_angles[:, None] - phase_values[None, :]), axis=1)]

        # Create action: normalized phase angles in [0, 1] for policy output mapping
        # Env expects action in [0, 1], maps to phase via action * 2pi
        action = phi_quantized / (2 * np.pi)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Record data
        P_BS = info.get('P_BS', 0)
        P_sat = info.get('P_sat', 0)
        P_total = P_BS + P_sat
        sinr_ue = info.get('sinr_UE', np.zeros(K))
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
        results['obs_elevation'].append(float(info.get('obs_elevation', 0)))
        results['obs_azimuth'].append(float(info.get('obs_azimuth', 0)))

        # Print current step info
        success_str = "Y" if success else "N"
        print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {reward:>8.3f} | {success_str:>7}")

        step += 1

    env.close()

    # Compute statistics
    results['summary'] = {
        'total_steps': step,
        'total_reward': float(total_reward),
        'avg_reward': float(total_reward / step) if step > 0 else 0,
        'avg_P_BS': float(np.mean(results['P_BS'])) if results['P_BS'] else 0,
        'avg_P_sat': float(np.mean(results['P_sat'])) if results['P_sat'] else 0,
        'avg_P_total': float(np.mean(results['P_total'])) if results['P_total'] else 0,
        'avg_SINR_min_dB': float(np.mean(results['SINR_min_dB'])) if results['SINR_min_dB'] else 0,
        'success_rate': float(np.mean(results['success'])) if results['success'] else 0,
    }

    print(f"{'-'*70}")
    print(f"Baseline Summary (ZF+SDR):")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {results['summary']['total_reward']:.3f}")
    print(f"  Avg reward: {results['summary']['avg_reward']:.3f}")
    print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
    print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
    print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
    print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
    print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")

    # Save results
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Baseline results saved to {save_path}")

    return results


def evaluate_baseline_with_reoptimize_interval(
    ae_checkpoint_path: str,
    seed: int = 42,
    reoptimize_interval: int = 10,
    device: str = 'cuda',
    save_path: str = None,
    verbose: bool = True,
    **env_kwargs
):
    """
    Evaluate baseline optimizer with periodic re-optimization (robustness test).

    Similar to evaluate_baseline_multiple_episodes in baseline/evaluate_baseline.py,
    but evaluates a single trajectory with periodic re-optimization.

    Args:
        ae_checkpoint_path: Path to autoencoder checkpoint
        seed: Random seed for trajectory
        reoptimize_interval: Re-optimize every N steps
        device: Device for AE inference
        save_path: Path to save results
        verbose: Print progress
        **env_kwargs: Additional environment arguments

    Returns:
        dict: Results with per-step metrics
    """
    # Create environment
    env = ITSNEnvAE(
        ae_checkpoint_path=ae_checkpoint_path,
        rng_seed=seed,
        device=device,
        **env_kwargs
    )

    # Record data
    results = {
        'seed': seed,
        'type': 'baseline_ZF_SDR_interval',
        'reoptimize_interval': reoptimize_interval,
        'steps': [],
        'P_BS': [],
        'P_sat': [],
        'P_total': [],
        'SINR_UE': [],
        'SINR_SUE': [],
        'SINR_min_dB': [],
        'success': [],
        'reoptimized': [],  # Track which steps were re-optimized
        'true_elevation': [],
        'true_azimuth': [],
    }

    # Initialize optimizer
    K = env.scenario.K
    J = env.scenario.SK
    N_t = env.scenario.N_t
    N_s = env.scenario.N_sat
    N = env.scenario.N_ris
    P_max = env.scenario.P_bs_max
    sigma2 = env.scenario.P_noise

    target_sinr_dB = 10
    gamma_k = np.ones(K) * (10 ** (target_sinr_dB / 10))
    gamma_j = 10 ** (target_sinr_dB / 10)

    optimizer = BaselineZFSDROptimizer(
        K=K, J=J, N_t=N_t, N_s=N_s, N=N,
        P_max=P_max, sigma2=sigma2,
        gamma_k=gamma_k, gamma_j=gamma_j,
        P_b=env.scenario.P_bs_scale,
        P_s_init=env.scenario.P_sat,
        ris_amplitude_gain=env.scenario.ris_amplitude_gain,
        N_iter=10,
        verbose=False
    )

    # Reset environment
    obs, info = env.reset()
    done = False
    step = 0

    # Store optimization result (updated every reoptimize_interval steps)
    w_opt_fixed = None
    Phi_opt_fixed = None
    phase_bits = env.phase_bits
    phase_levels = 2 ** phase_bits
    phase_values = 2 * np.pi * np.arange(phase_levels) / phase_levels

    print(f"\n{'='*70}")
    print(f"Baseline Evaluation (ZF+SDR, interval={reoptimize_interval}, seed={seed})")
    print(f"{'='*70}")
    print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Reopt':>6}")
    print(f"{'-'*70}")

    while not done:
        # Get current channels
        channels = env.current_channels

        # Extract channels
        h_k = channels['H_BS2UE']
        h_j = channels['H_BS2SUE']
        h_s_k = channels['H_SAT2UE']
        h_s_j = channels['H_SAT2SUE']
        h_k_r = channels['H_RIS2UE']
        h_j_r = channels['H_RIS2SUE']
        G_BS = channels['G_BS']
        G_S = channels['G_SAT']
        W_sat = channels['W_sat']

        # Re-optimize every reoptimize_interval steps
        reoptimized = False
        if step % reoptimize_interval == 0:
            w_opt, Phi_opt, opt_info = optimizer.optimize(
                h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S, W_sat
            )
            w_opt_fixed = w_opt.copy()
            Phi_opt_fixed = Phi_opt.copy()
            reoptimized = True
        else:
            # Use fixed optimization from most recent step
            w_opt = w_opt_fixed
            Phi_opt = Phi_opt_fixed
            # Evaluate performance (no optimization)
            opt_info = optimizer.evaluate_performance(
                w_opt, Phi_opt, h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S, W_sat
            )

        # Convert to action
        phi_angles = np.angle(np.diag(Phi_opt))
        phi_quantized = phase_values[np.argmin(np.abs(phi_angles[:, None] - phase_values[None, :]), axis=1)]
        action = phi_quantized / (2 * np.pi)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record data
        P_BS = info.get('P_BS', 0)
        P_sat = info.get('P_sat', 0)
        P_total = P_BS + P_sat
        sinr_ue = info.get('sinr_UE', np.zeros(K))
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
        results['reoptimized'].append(bool(reoptimized))
        results['true_elevation'].append(float(info.get('true_elevation', 0)))
        results['true_azimuth'].append(float(info.get('true_azimuth', 0)))

        reopt_str = "Y" if reoptimized else "N"
        print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {reopt_str:>6}")

        step += 1

    env.close()

    # Compute statistics
    results['summary'] = {
        'total_steps': step,
        'avg_P_BS': float(np.mean(results['P_BS'])) if results['P_BS'] else 0,
        'avg_P_sat': float(np.mean(results['P_sat'])) if results['P_sat'] else 0,
        'avg_P_total': float(np.mean(results['P_total'])) if results['P_total'] else 0,
        'avg_SINR_min_dB': float(np.mean(results['SINR_min_dB'])) if results['SINR_min_dB'] else 0,
        'success_rate': float(np.mean(results['success'])) if results['success'] else 0,
    }

    print(f"{'-'*70}")
    print(f"Baseline Summary (ZF+SDR, interval={reoptimize_interval}):")
    print(f"  Total steps: {step}")
    print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
    print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
    print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
    print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
    print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")

    # Save results
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Baseline results saved to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Baseline (ZF+SDR) with Autoencoder')

    # AE checkpoint
    parser.add_argument('--ae_checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to autoencoder checkpoint')

    # Evaluation parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for trajectory')
    parser.add_argument('--reoptimize_interval', type=int, default=1,
                       help='Re-optimize every N steps (default: 1 = re-opt every step)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for AE inference')

    # Environment parameters
    parser.add_argument('--max_steps', type=int, default=128,
                       help='Max steps per episode')
    parser.add_argument('--phase_bits', type=int, default=4,
                       help='RIS phase quantization bits')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='AE latent dimension')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/baseline',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print progress')

    args = parser.parse_args()

    # Check if AE checkpoint exists
    ae_checkpoint = Path(args.ae_checkpoint)
    if not ae_checkpoint.exists():
        print(f"Error: AE checkpoint not found at {ae_checkpoint}")
        return 1

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Environment kwargs
    env_kwargs = {
        'max_steps_per_episode': args.max_steps,
        'phase_bits': args.phase_bits,
        'latent_dim': args.latent_dim
    }

    # Choose evaluation mode based on reoptimize_interval
    if args.reoptimize_interval == 1:
        # Re-optimize every step (standard evaluation)
        save_path = output_dir / f'baseline_eval_step_{timestamp}.json'
        results = evaluate_baseline_single_episode(
            ae_checkpoint_path=str(ae_checkpoint),
            seed=args.seed,
            device=args.device,
            save_path=save_path,
            verbose=args.verbose,
            **env_kwargs
        )
    else:
        # Re-optimize with interval (robustness evaluation)
        save_path = output_dir / f'baseline_eval_interval{args.reoptimize_interval}_{timestamp}.json'
        results = evaluate_baseline_with_reoptimize_interval(
            ae_checkpoint_path=str(ae_checkpoint),
            seed=args.seed,
            reoptimize_interval=args.reoptimize_interval,
            device=args.device,
            save_path=save_path,
            verbose=args.verbose,
            **env_kwargs
        )

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {save_path}")

    return 0


if __name__ == '__main__':
    exit(main())