"""
Baseline Evaluation Script
===========================
Evaluate the ZF+SDR baseline optimizer on the ITSN scenario.
Compare performance with DRL agent.

Usage:
    python baseline/evaluate_baseline.py --num_episodes 10 --save_results
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baseline.baseline_optimizer import BaselineZFSDROptimizer
from envs.scenario import ITSNScenario
from envs.itsn_env import ITSNEnv
import json


def generate_satellite_beamforming(h_s_j, J=1):
    """
    Extract satellite beamforming from channels (already computed by scenario).
    The beamforming is normalized to unit energy (||W_sat||_F^2 = 1).

    Note: Actual transmit power is controlled by P_s, not W_sat.
    """
    W_sat_raw = h_s_j.T  # (N_s, 1) from scenario

    # Normalize to unit Frobenius norm: ||W_sat||_F = 1
    W_sat_norm = W_sat_raw / (np.linalg.norm(W_sat_raw, 'fro') + 1e-12)

    # If J > 1, replicate for multiple satellite users (simplified)
    if J > 1:
        N_s = W_sat_norm.shape[0]
        W_sat = np.tile(W_sat_norm, (1, J))  # (N_s, J)
    else:
        W_sat = W_sat_norm

    return W_sat


def evaluate_baseline_single_episode(
    scenario: ITSNScenario,
    optimizer: BaselineZFSDROptimizer,
    num_steps: int = 50,
    verbose: bool = False,
    reset_users: bool = True,
    elevation_start: float = 50.0,
    elevation_end: float = 70.0,
    azimuth_start: float = 60.0,
    azimuth_end: float = 120.0,
    reoptimize_interval: int = 10
):
    """
    Evaluate baseline optimizer for a single satellite pass.

    Robustness Evaluation Mode:
    - Every reoptimize_interval steps: Re-optimize at current satellite position
    - Other steps: Use most recent optimization, evaluate performance as satellite moves

    Parameters:
    -----------
    reset_users : bool
        If True, reset user positions at the start of this episode
    elevation_start, elevation_end : float
        Satellite elevation angle range (degrees)
    azimuth_start, azimuth_end : float
        Satellite azimuth angle range (degrees)
    reoptimize_interval : int
        Re-optimize every N steps (default: 10)

    Returns:
    --------
    results : dict
        Dictionary containing:
        - power_history: List of total power consumption at each step
        - sum_rate_history: List of sum rates at each step
        - bs_power_history: List of BS power at each step
        - sat_power_history: List of satellite power at each step
        - average_power: Average power consumption
        - average_sum_rate: Average sum rate
    """
    # Reset user positions only if requested (for new episode)
    if reset_users:
        scenario.reset_user_positions()

    # Initialize tracking
    power_history = []
    sum_rate_history = []
    bs_power_history = []
    sat_power_history = []

    # Simulate satellite pass with specified trajectory
    elevation_range = np.linspace(elevation_start, elevation_end, num_steps)
    azimuth_range = np.linspace(azimuth_start, azimuth_end, num_steps)

    # Store optimization result (updated every reoptimize_interval steps)
    w_opt_fixed = None
    Phi_opt_fixed = None

    for step, (ele, azi) in enumerate(zip(elevation_range, azimuth_range)):
        if verbose and step % 10 == 0:
            print(f"\n--- Step {step+1}/{num_steps} ---")
            print(f"Satellite: Elevation={ele:.1f}°, Azimuth={azi:.1f}°")

        # Update satellite position
        scenario.update_satellite_position(ele=ele, azi=azi)

        # Generate channels
        channels = scenario.generate_channels()

        # Extract channels (using scenario's naming convention)
        h_k = channels['H_BS2UE']      # BS -> BS users (K, N_t)
        h_j = channels['H_BS2SUE']     # BS -> SAT users (J, N_t)
        h_s_k = channels['H_SAT2UE']   # SAT -> BS users (K, N_s)
        h_s_j = channels['H_SAT2SUE']  # SAT -> SAT users (J, N_s)
        h_k_r = channels['H_RIS2UE']   # BS users -> RIS (K, N)
        h_j_r = channels['H_RIS2SUE']  # SAT users -> RIS (J, N)
        G_BS = channels['G_BS']        # RIS -> BS (N, N_t)
        G_S = channels['G_SAT']        # RIS -> SAT (N, N_s)

        # Generate satellite beamforming (unit energy, MRT toward satellite user)
        #W_sat = generate_satellite_beamforming(h_s_j, J=optimizer.J)
        W_sat = channels['W_sat']



        # Re-optimize every reoptimize_interval steps
        if step % reoptimize_interval == 0:
            # Perform optimization at current satellite position
            w_opt, Phi_opt, info = optimizer.optimize(
                h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S, W_sat
            )
            # Save optimization result for robustness evaluation
            w_opt_fixed = w_opt.copy()
            Phi_opt_fixed = Phi_opt.copy()

            if verbose:
                print(f"  [Step {step}] Re-optimization completed. Fixing w and Phi until step {step + reoptimize_interval}.")
        else:
            # Use fixed optimization from most recent optimization step
            w_opt = w_opt_fixed
            Phi_opt = Phi_opt_fixed

            # Evaluate performance with fixed optimization
            info = optimizer.evaluate_performance(
                w_opt, Phi_opt, h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S, W_sat
            )

            if verbose and step % 10 == 0:
                print(f"  [Step {step}] Using fixed optimization from most recent optimization.")

        # Record results
        power_history.append(info['final_P_sum'])
        sum_rate_history.append(info['sum_rate_history'][-1] if info['sum_rate_history'] else 0)
        bs_power_history.append(info['final_P_bs'])
        sat_power_history.append(info['final_P_sat'])

        if verbose and step % 10 == 0:
            print(f"  Final Power: BS={info['final_P_bs']:.4f} W, SAT={info['final_P_sat']:.4f} W, Total={info['final_P_sum']:.4f} W")
            print(f"  Sum Rate: {sum_rate_history[-1]:.4f} bps/Hz")

    # Compute statistics
    results = {
        'power_history': power_history,
        'sum_rate_history': sum_rate_history,
        'bs_power_history': bs_power_history,
        'sat_power_history': sat_power_history,
        'average_power': np.mean(power_history),
        'average_sum_rate': np.mean(sum_rate_history),
        'std_power': np.std(power_history),
        'std_sum_rate': np.std(sum_rate_history),
        'max_power': np.max(power_history),
        'min_power': np.min(power_history),
    }

    return results


def evaluate_baseline_multiple_episodes(
    num_episodes: int = 10,
    num_steps_per_episode: int = 50,
    save_results: bool = True,
    output_dir: str = 'baseline/results',
    verbose: bool = True
):
    """
    Evaluate baseline optimizer over multiple episodes.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize scenario
    scenario = ITSNScenario(rng_seed=42)

    # Initialize optimizer
    K = scenario.K
    J = scenario.SK
    N_t = scenario.N_t
    N_s = scenario.N_sat
    N = scenario.N_ris
    P_max = scenario.P_bs_max
    sigma2 = scenario.P_noise

    # SINR thresholds (linear scale)
    # Assuming target SINR of 10 dB for all users
    target_sinr_dB = 10
    gamma_k = np.ones(K) * (10 ** (target_sinr_dB / 10))
    gamma_j = 10 ** (target_sinr_dB / 10)


    # Evaluate multiple episodes
    all_results = []

    print(f"\n{'='*60}")
    print(f"Evaluating Baseline ZF+SDR Optimizer")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {num_steps_per_episode}")
    print(f"Target SINR: {target_sinr_dB} dB")
    print(f"{'='*60}\n")

    # Initialize scenario once (will be reused across episodes)
    scenario = ITSNScenario(rng_seed=42)

    for episode in range(num_episodes):
        print(f"\n[Episode {episode+1}/{num_episodes}]")

        # Reset user positions for new episode with different random seed
        scenario.rng = np.random.RandomState(42 + episode)

        # Generate random satellite trajectory for each episode
        rng_traj = np.random.RandomState(1000 + episode)

        # Random elevation range: start in [30, 45], span [15, 30] degrees
        elevation_start = rng_traj.uniform(40.0, 45.0)
        elevation_span = rng_traj.uniform(25.0, 40.0)
        elevation_end = elevation_start + elevation_span

        # Random azimuth range: start in [60, 120], span [15, 40] degrees
        azimuth_start = rng_traj.uniform(60.0, 120.0)
        azimuth_span = rng_traj.uniform(30.0, 40.0)
        azimuth_end = azimuth_start + azimuth_span

        if verbose:
            print(f"  Satellite trajectory: Elevation [{elevation_start:.1f}° → {elevation_end:.1f}°], "
                  f"Azimuth [{azimuth_start:.1f}° → {azimuth_end:.1f}°]")
        scenario.generate_channels()
        optimizer = BaselineZFSDROptimizer(
            K=K, J=J, N_t=N_t, N_s=N_s, N=N,
            P_max=P_max, sigma2=sigma2,
            gamma_k=gamma_k, gamma_j=gamma_j,
            P_b=scenario.P_bs_scale,
            P_s_init=scenario.P_sat,
            ris_amplitude_gain=scenario.ris_amplitude_gain,
            N_iter=10,  # Reduce iterations for faster evaluation
            verbose=True
        )

        # reset_users=True: Reset user positions at the start of each episode
        results = evaluate_baseline_single_episode(
            scenario, optimizer, num_steps=num_steps_per_episode,
            verbose=verbose, reset_users=True,
            elevation_start=elevation_start, elevation_end=elevation_end,
            azimuth_start=azimuth_start, azimuth_end=azimuth_end,
            reoptimize_interval=10
        )

        all_results.append(results)

        print(f"  Average Power: {results['average_power']:.4f} ± {results['std_power']:.4f} W")
        print(f"  Average Sum Rate: {results['average_sum_rate']:.4f} ± {results['std_sum_rate']:.4f} bps/Hz")
        print(f"  Power Range: [{results['min_power']:.4f}, {results['max_power']:.4f}] W")

    # Aggregate results
    avg_power_all = np.mean([r['average_power'] for r in all_results])
    avg_sum_rate_all = np.mean([r['average_sum_rate'] for r in all_results])
    std_power_all = np.std([r['average_power'] for r in all_results])
    std_sum_rate_all = np.std([r['average_sum_rate'] for r in all_results])

    print(f"\n{'='*60}")
    print(f"Overall Results (over {num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average Power Consumption: {avg_power_all:.4f} ± {std_power_all:.4f} W")
    print(f"Average Sum Rate: {avg_sum_rate_all:.4f} ± {std_sum_rate_all:.4f} bps/Hz")
    print(f"{'='*60}\n")

    # Save results
    if save_results:
        results_summary = {
            'num_episodes': num_episodes,
            'num_steps_per_episode': num_steps_per_episode,
            'target_sinr_dB': target_sinr_dB,
            'avg_power': float(avg_power_all),
            'std_power': float(std_power_all),
            'avg_sum_rate': float(avg_sum_rate_all),
            'std_sum_rate': float(std_sum_rate_all),
            'all_episodes': [
                {
                    'average_power': float(r['average_power']),
                    'average_sum_rate': float(r['average_sum_rate']),
                    'std_power': float(r['std_power']),
                    'std_sum_rate': float(r['std_sum_rate']),
                }
                for r in all_results
            ]
        }

        results_file = os.path.join(output_dir, 'baseline_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"Results saved to {results_file}")

        # Plot results
        plot_baseline_results(all_results, output_dir)

    return all_results


def plot_baseline_results(all_results, output_dir):
    """Plot baseline evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Power consumption over episodes
    ax = axes[0, 0]
    avg_powers = [r['average_power'] for r in all_results]
    std_powers = [r['std_power'] for r in all_results]
    episodes = np.arange(1, len(all_results) + 1)

    ax.errorbar(episodes, avg_powers, yerr=std_powers, fmt='o-', capsize=5, label='Average Power')
    ax.axhline(np.mean(avg_powers), color='r', linestyle='--', label=f'Mean: {np.mean(avg_powers):.2f} W')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Power (W)')
    ax.set_title('Power Consumption per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Sum rate over episodes
    ax = axes[0, 1]
    avg_rates = [r['average_sum_rate'] for r in all_results]
    std_rates = [r['std_sum_rate'] for r in all_results]

    ax.errorbar(episodes, avg_rates, yerr=std_rates, fmt='o-', capsize=5, label='Average Sum Rate', color='green')
    ax.axhline(np.mean(avg_rates), color='r', linestyle='--', label=f'Mean: {np.mean(avg_rates):.2f} bps/Hz')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('Sum Rate per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Power distribution (histogram)
    ax = axes[1, 0]
    all_powers = np.concatenate([r['power_history'] for r in all_results])
    ax.hist(all_powers, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_powers), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_powers):.2f} W')
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Frequency')
    ax.set_title('Power Distribution (All Steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sum rate distribution (histogram)
    ax = axes[1, 1]
    all_rates = np.concatenate([r['sum_rate_history'] for r in all_results])
    ax.hist(all_rates, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(all_rates), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_rates):.2f} bps/Hz')
    ax.set_xlabel('Sum Rate (bps/Hz)')
    ax.set_ylabel('Frequency')
    ax.set_title('Sum Rate Distribution (All Steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'baseline_evaluation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

    plt.close()

    # Plot example trajectory
    plot_example_trajectory(all_results[0], output_dir)


def plot_example_trajectory(results, output_dir):
    """Plot power and sum rate over time for a single episode."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    steps = np.arange(1, len(results['power_history']) + 1)

    # Plot 1: Total power over time
    ax = axes[0]
    ax.plot(steps, results['power_history'], 'o-', label='Total Power', color='blue', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Power (W)')
    ax.set_title('Power Consumption over Satellite Pass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: BS vs SAT power
    ax = axes[1]
    ax.plot(steps, results['bs_power_history'], 's-', label='BS Power', color='orange', markersize=3)
    ax.plot(steps, results['sat_power_history'], '^-', label='SAT Power', color='purple', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Power (W)')
    ax.set_title('BS vs Satellite Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Sum rate over time
    ax = axes[2]
    ax.plot(steps, results['sum_rate_history'], 'o-', label='Sum Rate', color='green', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('Sum Rate over Satellite Pass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'baseline_trajectory_example.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory figure saved to {fig_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate ZF+SDR Baseline Optimizer')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of steps per episode')
    parser.add_argument('--save_results', action='store_true', help='Save results to file')
    parser.add_argument('--output_dir', type=str, default='baseline/results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    evaluate_baseline_multiple_episodes(
        num_episodes=args.num_episodes,
        num_steps_per_episode=args.num_steps,
        save_results=args.save_results,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()