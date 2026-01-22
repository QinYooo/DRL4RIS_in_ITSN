"""
DRL vs Baseline Comparison Tool
================================
Compare the performance of DRL agent against the ZF+SDR baseline.

This script:
1. Loads a trained DRL agent
2. Evaluates both DRL and baseline on the same scenarios
3. Compares power consumption, sum rate, and computation time
4. Generates comparison plots

Usage:
    python baseline/compare_drl_baseline.py --drl_model path/to/model.zip --num_episodes 20
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baseline.baseline_optimizer import BaselineZFSDROptimizer
from baseline.evaluate_baseline import generate_satellite_beamforming
from envs.scenario import ITSNScenario
from envs.itsn_env import ITSNEnv

try:
    from stable_baselines3 import PPO, SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[Warning] stable-baselines3 not available. DRL comparison will be skipped.")


class DRLvsBaselineComparator:
    """
    Compare DRL agent with baseline optimizer.
    """

    def __init__(
        self,
        drl_model_path: str = None,
        num_episodes: int = 20,
        num_steps: int = 50,
        output_dir: str = 'baseline/results/comparison',
        verbose: bool = True
    ):
        self.drl_model_path = drl_model_path
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.output_dir = output_dir
        self.verbose = verbose

        os.makedirs(output_dir, exist_ok=True)

        # Results storage
        self.drl_results = []
        self.baseline_results = []

    def load_drl_model(self):
        """Load trained DRL model."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for DRL comparison")

        if self.drl_model_path is None:
            raise ValueError("DRL model path not provided")

        if not os.path.exists(self.drl_model_path):
            raise FileNotFoundError(f"DRL model not found: {self.drl_model_path}")

        # Try to load as PPO first, then SAC
        try:
            model = PPO.load(self.drl_model_path)
            if self.verbose:
                print(f"[DRL] Loaded PPO model from {self.drl_model_path}")
        except:
            try:
                model = SAC.load(self.drl_model_path)
                if self.verbose:
                    print(f"[DRL] Loaded SAC model from {self.drl_model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load DRL model: {e}")

        return model

    def evaluate_drl_single_episode(self, model, env, scenario, episode_seed):
        """Evaluate DRL agent for a single episode."""
        scenario = ITSNScenario(rng_seed=episode_seed)
        env.scenario = scenario

        obs, _ = env.reset(seed=episode_seed)

        power_history = []
        sum_rate_history = []
        inference_times = []

        for step in range(self.num_steps):
            # Measure inference time
            start_time = time.time()
            action, _ = model.predict(obs, deterministic=True)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

            # Record metrics
            power_history.append(info.get('power_total', 0))
            sum_rate_history.append(info.get('sum_rate', 0))

            if done or truncated:
                break

        return {
            'power_history': power_history,
            'sum_rate_history': sum_rate_history,
            'average_power': np.mean(power_history),
            'average_sum_rate': np.mean(sum_rate_history),
            'avg_inference_time': np.mean(inference_times),
            'total_inference_time': np.sum(inference_times)
        }

    def evaluate_baseline_single_episode(self, optimizer, scenario, episode_seed):
        """Evaluate baseline optimizer for a single episode."""
        scenario = ITSNScenario(rng_seed=episode_seed)
        scenario.reset_user_positions()

        power_history = []
        sum_rate_history = []
        optimization_times = []

        elevation_range = np.linspace(40, 60, self.num_steps)
        azimuth_range = np.linspace(80, 100, self.num_steps)

        for step, (ele, azi) in enumerate(zip(elevation_range, azimuth_range)):
            scenario.update_satellite_position(ele=ele, azi=azi)
            channels = scenario.generate_channels()

            W_sat = generate_satellite_beamforming(scenario, J=optimizer.J)

            # Measure optimization time
            start_time = time.time()
            w_opt, Phi_opt, info = optimizer.optimize(
                channels['h_k'], channels['h_j'],
                channels['h_s_k'], channels['h_s_j'],
                channels['h_k_r'], channels['h_j_r'],
                channels['G_BS'], channels['G_S'], W_sat
            )
            opt_time = time.time() - start_time
            optimization_times.append(opt_time)

            power_history.append(info['final_P_sum'])
            sum_rate_history.append(info['sum_rate_history'][-1] if info['sum_rate_history'] else 0)

        return {
            'power_history': power_history,
            'sum_rate_history': sum_rate_history,
            'average_power': np.mean(power_history),
            'average_sum_rate': np.mean(sum_rate_history),
            'avg_optimization_time': np.mean(optimization_times),
            'total_optimization_time': np.sum(optimization_times)
        }

    def run_comparison(self):
        """Run full comparison between DRL and baseline."""
        print(f"\n{'='*70}")
        print(f"DRL vs Baseline Comparison")
        print(f"{'='*70}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Steps per episode: {self.num_steps}")
        print(f"{'='*70}\n")

        # Initialize scenario and environment
        base_scenario = ITSNScenario(rng_seed=42)

        # Initialize baseline optimizer
        K = base_scenario.K
        J = base_scenario.SK
        N_t = base_scenario.N_t
        N_s = base_scenario.N_sat
        N = base_scenario.N_ris

        target_sinr_dB = 10
        gamma_k = np.ones(K) * (10 ** (target_sinr_dB / 10))
        gamma_j = 10 ** (target_sinr_dB / 10)

        baseline_optimizer = BaselineZFSDROptimizer(
            K=K, J=J, N_t=N_t, N_s=N_s, N=N,
            P_max=base_scenario.P_bs_max,
            sigma2=base_scenario.P_noise,
            gamma_k=gamma_k, gamma_j=gamma_j,
            P_b=base_scenario.P_bs_scale,
            N_iter=10,
            verbose=False
        )

        # Load DRL model and create environment
        if self.drl_model_path and SB3_AVAILABLE:
            drl_model = self.load_drl_model()
            drl_env = ITSNEnv(scenario=base_scenario)
        else:
            drl_model = None
            drl_env = None
            print("[Warning] DRL model not available. Only baseline will be evaluated.")

        # Run comparison for each episode
        for episode in range(self.num_episodes):
            episode_seed = 42 + episode
            print(f"\n[Episode {episode + 1}/{self.num_episodes}] (seed={episode_seed})")

            # Evaluate baseline
            print("  Evaluating Baseline...")
            baseline_result = self.evaluate_baseline_single_episode(
                baseline_optimizer, base_scenario, episode_seed
            )
            self.baseline_results.append(baseline_result)

            print(f"    Baseline - Power: {baseline_result['average_power']:.4f} W, "
                  f"Sum Rate: {baseline_result['average_sum_rate']:.4f} bps/Hz, "
                  f"Time: {baseline_result['total_optimization_time']:.2f} s")

            # Evaluate DRL
            if drl_model is not None:
                print("  Evaluating DRL...")
                drl_result = self.evaluate_drl_single_episode(
                    drl_model, drl_env, base_scenario, episode_seed
                )
                self.drl_results.append(drl_result)

                print(f"    DRL - Power: {drl_result['average_power']:.4f} W, "
                      f"Sum Rate: {drl_result['average_sum_rate']:.4f} bps/Hz, "
                      f"Time: {drl_result['total_inference_time']:.4f} s")

                # Compute relative performance
                power_ratio = (drl_result['average_power'] / baseline_result['average_power']) * 100
                rate_ratio = (drl_result['average_sum_rate'] / baseline_result['average_sum_rate']) * 100
                speedup = baseline_result['total_optimization_time'] / drl_result['total_inference_time']

                print(f"    DRL vs Baseline: Power={power_ratio:.1f}%, Rate={rate_ratio:.1f}%, Speedup={speedup:.1f}x")

        # Aggregate and save results
        self.compute_and_save_summary()

        # Generate plots
        self.plot_comparison()

    def compute_and_save_summary(self):
        """Compute summary statistics and save to file."""
        # Baseline statistics
        baseline_avg_power = np.mean([r['average_power'] for r in self.baseline_results])
        baseline_std_power = np.std([r['average_power'] for r in self.baseline_results])
        baseline_avg_rate = np.mean([r['average_sum_rate'] for r in self.baseline_results])
        baseline_std_rate = np.std([r['average_sum_rate'] for r in self.baseline_results])
        baseline_avg_time = np.mean([r['total_optimization_time'] for r in self.baseline_results])

        summary = {
            'num_episodes': self.num_episodes,
            'num_steps': self.num_steps,
            'baseline': {
                'avg_power': float(baseline_avg_power),
                'std_power': float(baseline_std_power),
                'avg_sum_rate': float(baseline_avg_rate),
                'std_sum_rate': float(baseline_std_rate),
                'avg_computation_time': float(baseline_avg_time)
            }
        }

        # DRL statistics (if available)
        if self.drl_results:
            drl_avg_power = np.mean([r['average_power'] for r in self.drl_results])
            drl_std_power = np.std([r['average_power'] for r in self.drl_results])
            drl_avg_rate = np.mean([r['average_sum_rate'] for r in self.drl_results])
            drl_std_rate = np.std([r['average_sum_rate'] for r in self.drl_results])
            drl_avg_time = np.mean([r['total_inference_time'] for r in self.drl_results])

            summary['drl'] = {
                'avg_power': float(drl_avg_power),
                'std_power': float(drl_std_power),
                'avg_sum_rate': float(drl_avg_rate),
                'std_sum_rate': float(drl_std_rate),
                'avg_inference_time': float(drl_avg_time)
            }

            # Relative performance
            summary['comparison'] = {
                'power_ratio_percent': float((drl_avg_power / baseline_avg_power) * 100),
                'rate_ratio_percent': float((drl_avg_rate / baseline_avg_rate) * 100),
                'speedup_factor': float(baseline_avg_time / drl_avg_time)
            }

        # Print summary
        print(f"\n{'='*70}")
        print(f"Summary Results")
        print(f"{'='*70}")
        print(f"Baseline:")
        print(f"  Power: {baseline_avg_power:.4f} ± {baseline_std_power:.4f} W")
        print(f"  Sum Rate: {baseline_avg_rate:.4f} ± {baseline_std_rate:.4f} bps/Hz")
        print(f"  Avg Computation Time: {baseline_avg_time:.4f} s")

        if self.drl_results:
            print(f"\nDRL:")
            print(f"  Power: {drl_avg_power:.4f} ± {drl_std_power:.4f} W ({summary['comparison']['power_ratio_percent']:.1f}% of baseline)")
            print(f"  Sum Rate: {drl_avg_rate:.4f} ± {drl_std_rate:.4f} bps/Hz ({summary['comparison']['rate_ratio_percent']:.1f}% of baseline)")
            print(f"  Avg Inference Time: {drl_avg_time:.4f} s (Speedup: {summary['comparison']['speedup_factor']:.1f}x)")

        print(f"{'='*70}\n")

        # Save to file
        summary_file = os.path.join(self.output_dir, 'comparison_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to {summary_file}")

    def plot_comparison(self):
        """Generate comparison plots."""
        if not self.drl_results:
            print("[Warning] No DRL results to plot. Skipping comparison plots.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        episodes = np.arange(1, self.num_episodes + 1)

        # Plot 1: Power consumption comparison
        ax = axes[0, 0]
        baseline_powers = [r['average_power'] for r in self.baseline_results]
        drl_powers = [r['average_power'] for r in self.drl_results]

        ax.plot(episodes, baseline_powers, 'o-', label='Baseline (ZF+SDR)', linewidth=2, markersize=6)
        ax.plot(episodes, drl_powers, 's-', label='DRL', linewidth=2, markersize=6)
        ax.axhline(np.mean(baseline_powers), color='blue', linestyle='--', alpha=0.5)
        ax.axhline(np.mean(drl_powers), color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Power (W)')
        ax.set_title('Power Consumption Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Sum rate comparison
        ax = axes[0, 1]
        baseline_rates = [r['average_sum_rate'] for r in self.baseline_results]
        drl_rates = [r['average_sum_rate'] for r in self.drl_results]

        ax.plot(episodes, baseline_rates, 'o-', label='Baseline (ZF+SDR)', linewidth=2, markersize=6)
        ax.plot(episodes, drl_rates, 's-', label='DRL', linewidth=2, markersize=6)
        ax.axhline(np.mean(baseline_rates), color='blue', linestyle='--', alpha=0.5)
        ax.axhline(np.mean(drl_rates), color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sum Rate (bps/Hz)')
        ax.set_title('Sum Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Computation time comparison
        ax = axes[0, 2]
        baseline_times = [r['total_optimization_time'] for r in self.baseline_results]
        drl_times = [r['total_inference_time'] for r in self.drl_results]

        ax.plot(episodes, baseline_times, 'o-', label='Baseline (Optimization)', linewidth=2, markersize=6)
        ax.plot(episodes, drl_times, 's-', label='DRL (Inference)', linewidth=2, markersize=6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Time (s)')
        ax.set_title('Computation Time Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 4: Power efficiency ratio
        ax = axes[1, 0]
        power_ratios = [(d / b) * 100 for d, b in zip(drl_powers, baseline_powers)]
        ax.bar(episodes, power_ratios, alpha=0.7, edgecolor='black')
        ax.axhline(100, color='r', linestyle='--', label='Baseline (100%)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('DRL Power / Baseline Power (%)')
        ax.set_title('Power Efficiency Ratio (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 5: Sum rate ratio
        ax = axes[1, 1]
        rate_ratios = [(d / b) * 100 for d, b in zip(drl_rates, baseline_rates)]
        ax.bar(episodes, rate_ratios, alpha=0.7, color='green', edgecolor='black')
        ax.axhline(100, color='r', linestyle='--', label='Baseline (100%)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('DRL Rate / Baseline Rate (%)')
        ax.set_title('Sum Rate Ratio (Higher is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 6: Speedup factor
        ax = axes[1, 2]
        speedups = [b / d for b, d in zip(baseline_times, drl_times)]
        ax.bar(episodes, speedups, alpha=0.7, color='purple', edgecolor='black')
        ax.axhline(np.mean(speedups), color='r', linestyle='--', label=f'Avg: {np.mean(speedups):.1f}x')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Speedup Factor (Baseline Time / DRL Time)')
        ax.set_title('Computation Speedup (Higher is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.output_dir, 'drl_vs_baseline_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {fig_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare DRL agent with ZF+SDR Baseline')
    parser.add_argument('--drl_model', type=str, default=None, help='Path to trained DRL model (.zip)')
    parser.add_argument('--num_episodes', type=int, default=20, help='Number of episodes to evaluate')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of steps per episode')
    parser.add_argument('--output_dir', type=str, default='baseline/results/comparison', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    comparator = DRLvsBaselineComparator(
        drl_model_path=args.drl_model,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    comparator.run_comparison()


if __name__ == '__main__':
    main()