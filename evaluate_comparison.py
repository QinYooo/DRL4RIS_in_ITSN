"""
Comparison Evaluation Script
==================================
Compare DRL model performance with Baseline Optimizer (ZF+SDR)

Supports:
- Zero-action baseline (no RIS phase control)
- Baseline optimizer (BCD + SDR)
- Trained DRL model (RecurrentPPO)

Results are saved to JSON files with detailed step-by-step metrics.
"""

import os
# Fix OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import json
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from envs.itsn_env import ITSNEnv
from envs.itsn_env_ae import ITSNEnvAE
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import set_random_seed
from baseline.baseline_optimizer import BaselineZFSDROptimizer


def evaluate_baseline_optimizer(seed=42, max_steps=128, verbose=True, save_path=None):
    """
    Evaluate Baseline Optimizer (ZF+SDR) on a random satellite trajectory.

    Args:
        seed: Random seed for reproducibility
        max_steps: Maximum number of evaluation steps
        verbose: Print progress
        save_path: Path to save results (None to skip)

    Returns:
        dict: Evaluation results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BASELINE OPTIMIZER EVALUATION (ZF + SDR)")
        print("=" * 70)

    # Create environment (no ephemeris noise for fair comparison)
    env = ITSNEnv(
        rng_seed=seed,
        max_steps_per_episode=max_steps,
        n_substeps=1,
        phase_bits=4,
        ephemeris_noise_std=0.0,  # No noise for baseline comparison
    )
    env.enable_ephemeris_noise = False
    env.reset()
    # Initialize baseline optimizer
    # Get SINR thresholds from scenario
    sinr_threshold_db = env.sinr_threshold_db
    sinr_threshold_linear = 10 ** (sinr_threshold_db / 10.0)
    gamma_k = np.full(env.scenario.K, sinr_threshold_linear)
    gamma_j = sinr_threshold_linear

    baseline = BaselineZFSDROptimizer(
        K=env.scenario.K,
        J=env.scenario.SK,
        N_t=env.scenario.N_t,
        N_s=env.scenario.N_sat,
        N=env.scenario.N_ris,
        P_max=env.scenario.P_bs_max,
        P_b=env.scenario.P_bs_scale,
        P_s_init=env.P_sat_init,
        sigma2=env.scenario.P_noise,
        gamma_k=gamma_k,
        gamma_j=gamma_j,
        ris_amplitude_gain=env.scenario.ris_amplitude_gain,
        N_iter=5,
        verbose=False  # Suppress verbose output
    )

    # Record results
    results = {
        'method': 'baseline_optimizer_zf_sdr',
        'seed': seed,
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

    obs, info = env.reset()
    done = False
    step = 0

    if verbose:
        print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Success':>7} | {'Time':>8}")
        print(f"-" * 70)

    total_opt_time = 0

    while not done and step < max_steps:
        import time
        t_start = time.time()

        # Get current channels
        channels = env.current_channels

        # Extract channels (using scenario's naming convention)
        h_k = channels['H_BS2UE'].conj()      # BS -> BS users (K, N_t)
        h_j = channels['H_BS2SUE'].conj()     # BS -> SAT users (J, N_t)
        h_s_k = channels['H_SAT2UE'].conj()   # SAT -> BS users (K, N_s)
        h_s_j = channels['H_SAT2SUE'].conj()  # SAT -> SAT users (J, N_s)
        h_k_r = channels['H_RIS2UE'].conj()   # BS users -> RIS (K, N)
        h_j_r = channels['H_RIS2SUE'].conj()  # SAT users -> RIS (J, N)
        G_BS = channels['G_BS'].conj()        # RIS -> BS (N, N_t)
        G_S = channels['G_SAT'].conj()        # RIS -> SAT (N, N_s)

        # Generate satellite beamforming (unit energy, MRT toward satellite user)
        #W_sat = generate_satellite_beamforming(h_s_j, J=optimizer.J)
        W_sat = channels['W_sat']
        # Run baseline optimization
        try:
            w_opt, phi_opt, opt_info = baseline.optimize(
                h_k,
                h_j,
                h_s_k,
                h_s_j,
                h_k_r,
                h_j_r,
                G_BS,
                G_S,  # Use true G_SAT (no ephemeris uncertainty)
                W_sat
            )

            # Calculate actual performance
            all_sinr = opt_info['all_sinr']

            P_BS = opt_info['final_P_bs']
            P_sat = opt_info['final_P_sat']
            P_total = P_BS + P_sat

            # Check success
            soft_threshold = 10 ** ((env.sinr_threshold_db - 0.5) / 10.0)
            success = np.all(all_sinr >= [soft_threshold]*5)

        except Exception as e:
            if verbose:
                print(f"  [Warning] Optimization failed at step {step}: {e}")
            P_BS, P_sat = 10.0, 10.0
            P_total = 20.0
            all_sinr = np.zeros(env.scenario.K + 1)
            success = False

        opt_time = time.time() - t_start
        total_opt_time += opt_time

        # Convert to dB
        sinr_min_db = 10 * np.log10(np.min(all_sinr) + 1e-12)
        sinr_db = 10 * np.log10(all_sinr)
        sinr_ue_db = sinr_db[:-1]
        sinr_sue_db = sinr_db[-1]

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

        if verbose:
            success_str = "Y" if success else "N"
            print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {success_str:>7} | {opt_time:>8.4f}")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.scenario.N_ris))
        done = terminated or truncated
        step += 1

    env.close()

    # Summary statistics
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

    if verbose:
        print(f"-" * 70)
        print(f"Baseline Optimizer Summary:")
        print(f"  Total steps: {step}")
        print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
        print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
        print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
        print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
        print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
        print(f"  Avg optimization time: {results['summary']['avg_opt_time']:.4f} s")
        print(f"=" * 70 + "\n")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to {save_path}")

    return results


def evaluate_drl_model(
    model_path,
    vecnormalize_path,
    use_ae=False,
    ae_checkpoint_path=None,
    seed=42,
    max_steps=128,
    device='cuda',
    verbose=False,
    save_path=None
):
    """
    Evaluate a trained DRL model on a random satellite trajectory.

    Args:
        model_path: Path to the trained model
        vecnormalize_path: Path to VecNormalize stats
        use_ae: Whether to use autoencoder-compressed state
        ae_checkpoint_path: Path to AE checkpoint (if use_ae=True)
        seed: Random seed for reproducibility
        max_steps: Maximum number of evaluation steps
        device: Device for inference
        verbose: Print progress
        save_path: Path to save results (None to skip)

    Returns:
        dict: Evaluation results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("DRL MODEL EVALUATION")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"AE: {use_ae}")

    # Create evaluation environment (no ephemeris noise for fair comparison)
    if use_ae:
        env = ITSNEnvAE(
            ae_checkpoint_path=ae_checkpoint_path,
            rng_seed=seed,
            max_steps_per_episode=max_steps,
            n_substeps=1,
            phase_bits=4,
            ephemeris_noise_std=0.0,  # No noise for fair comparison
            device=device
        )
    else:
        env = ITSNEnv(
            rng_seed=seed,
            max_steps_per_episode=max_steps,
            n_substeps=1,
            phase_bits=4,
            ephemeris_noise_std=0.0,  # No noise for fair comparison
        )
    env.enable_ephemeris_noise = False

    # Load model
    model = RecurrentPPO.load(str(model_path), device=device)

    # Load VecNormalize statistics
    from stable_baselines3.common.vec_env import VecNormalize
    training_env = VecNormalize.load(vecnormalize_path, DummyVecEnv([lambda: env]))

    # Record results
    results = {
        'method': 'drl_recurrent_ppo',
        'seed': seed,
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
    }

    obs, info = env.reset()
    done = False
    step = 0

    # Initialize LSTM hidden states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    # Tracking total inference time
    import time
    total_inference_time = 0

    if verbose:
        print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Success':>7} | {'Time':>8}")
        print(f"-" * 70)

    while not done and step < max_steps:
        t_start = time.time()

        # Normalize observation
        obs_normalized = training_env.normalize_obs(obs)
        obs_2d = obs_normalized[np.newaxis, :] if obs_normalized.ndim == 1 else obs_normalized

        # Get action from model
        action, lstm_states = model.predict(
            obs_2d,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        action = action.flatten()

        inference_time = time.time() - t_start
        total_inference_time += inference_time

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_starts = np.zeros((1,), dtype=bool)

        # Record data
        P_BS = info.get('P_BS', 0)
        P_sat = info.get('P_sat', 0)
        P_total = P_BS + P_sat
        sinr_ue = info.get('sinr_UE', np.zeros(4))
        sinr_sue = info.get('sinr_SUE', 0)
        success = info.get('success', False)

        sinr_min_db = 10 * np.log10(np.min(np.append(sinr_ue, sinr_sue)) + 1e-12)
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

        if verbose:
            success_str = "Y" if success else "N"
            print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {success_str:>7} | {inference_time:>8.6f}")

        step += 1

    env.close()

    # Summary statistics
    results['summary'] = {
        'total_steps': step,
        'avg_P_BS': float(np.mean(results['P_BS'])),
        'avg_P_sat': float(np.mean(results['P_sat'])),
        'avg_P_total': float(np.mean(results['P_total'])),
        'avg_SINR_min_dB': float(np.mean(results['SINR_min_dB'])),
        'success_rate': float(np.mean(results['success'])),
        'total_inference_time': float(total_inference_time),
        'avg_inference_time': float(total_inference_time / max(step, 1)),
    }

    if verbose:
        print(f"-" * 70)
        print(f"DRL Model Summary:")
        print(f"  Total steps: {step}")
        print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
        print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
        print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
        print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
        print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
        print(f"  Avg inference time: {results['summary']['avg_inference_time']:.6f} s")
        print(f"=" * 70 + "\n")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to {save_path}")

    return results


def evaluate_zero_action(seed=42, max_steps=128, verbose=True, save_path=None):
    """
    Evaluate zero-action baseline (no RIS phase control).

    Args:
        seed: Random seed for reproducibility
        max_steps: Maximum number of evaluation steps
        verbose: Print progress
        save_path: Path to save results (None to skip)

    Returns:
        dict: Evaluation results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ZERO-ACTION BASELINE EVALUATION (No RIS Control)")
        print("=" * 70)

    env = ITSNEnv(
        rng_seed=seed,
        max_steps_per_episode=max_steps,
        n_substeps=1,
        phase_bits=4,
        ephemeris_noise_std=0.0,
    )
    env.enable_ephemeris_noise = False

    results = {
        'method': 'zero_action_baseline',
        'seed': seed,
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
    }

    obs, info = env.reset()
    done = False
    step = 0

    if verbose:
        print(f"{'Step':>4} | {'P_BS':>8} | {'P_sat':>8} | {'P_total':>8} | {'SINR_min':>10} | {'Success':>7}")
        print(f"-" * 70)

    while not done and step < max_steps:
        # Zero action (no RIS phase control)
        action = np.zeros(env.scenario.N_ris)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        P_BS = info.get('P_BS', 0)
        P_sat = info.get('P_sat', 0)
        P_total = P_BS + P_sat
        sinr_ue = info.get('sinr_UE', np.zeros(4))
        sinr_sue = info.get('sinr_SUE', 0)
        success = info.get('success', False)

        sinr_min_db = 10 * np.log10(np.min(np.append(sinr_ue, sinr_sue)) + 1e-12)
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

        if verbose:
            success_str = "Y" if success else "N"
            print(f"{step:>4} | {P_BS:>8.4f} | {P_sat:>8.4f} | {P_total:>8.4f} | {sinr_min_db:>10.2f} | {success_str:>7}")

        step += 1

    env.close()

    results['summary'] = {
        'total_steps': step,
        'avg_P_BS': float(np.mean(results['P_BS'])),
        'avg_P_sat': float(np.mean(results['P_sat'])),
        'avg_P_total': float(np.mean(results['P_total'])),
        'avg_SINR_min_dB': float(np.mean(results['SINR_min_dB'])),
        'success_rate': float(np.mean(results['success'])),
    }

    if verbose:
        print(f"-" * 70)
        print(f"Zero-Action Baseline Summary:")
        print(f"  Total steps: {step}")
        print(f"  Avg P_BS: {results['summary']['avg_P_BS']:.4f} W")
        print(f"  Avg P_sat: {results['summary']['avg_P_sat']:.4f} W")
        print(f"  Avg P_total: {results['summary']['avg_P_total']:.4f} W")
        print(f"  Avg SINR_min: {results['summary']['avg_SINR_min_dB']:.2f} dB")
        print(f"  Success rate: {results['summary']['success_rate']*100:.1f}%")
        print(f"=" * 70 + "\n")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to {save_path}")

    return results


def compare_all_methods(
    model_path=None,
    vecnormalize_path=None,
    use_ae=False,
    ae_checkpoint_path=None,
    seed=42,
    max_steps=128,
    device='cuda',
    output_dir='comparison_results'
):
    """
    Run comparison evaluation for all methods.

    Args:
        model_path: Path to trained DRL model (None to skip DRL evaluation)
        vecnormalize_path: Path to VecNormalize stats (required if model_path is provided)
        use_ae: Whether model uses AE-compressed state
        ae_checkpoint_path: Path to AE checkpoint (if use_ae=True)
        seed: Random seed
        max_steps: Maximum evaluation steps
        device: Device for DRL inference
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_dict = {}

    # 1. Zero-action baseline
    results_dict['zero_action'] = evaluate_zero_action(
        seed=seed,
        max_steps=max_steps,
        save_path=output_path / 'zero_action_results.json'
    )

    # 2. Baseline optimizer (ZF+SDR)
    results_dict['baseline_optimizer'] = evaluate_baseline_optimizer(
        seed=seed,
        max_steps=max_steps,
        save_path=output_path / 'baseline_optimizer_results.json'
    )

    # 3. DRL model (if provided)
    if model_path is not None:
        results_dict['drl_model'] = evaluate_drl_model(
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            use_ae=use_ae,
            ae_checkpoint_path=ae_checkpoint_path,
            seed=seed,
            max_steps=max_steps,
            device=device,
            save_path=output_path / 'drl_model_results.json'
        )

    # 4. Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} | {'Avg P_total':>12} | {'SINR_min':>10} | {'Success':>10} | {'Time':>12}")
    print(f"-" * 70)

    for name, results in results_dict.items():
        summary = results['summary']
        avg_p = summary['avg_P_total']
        sinr_min = summary['avg_SINR_min_dB']
        success = summary['success_rate'] * 100

        if 'avg_opt_time' in summary:
            time_str = f"{summary['avg_opt_time']:.4f}s"
        elif 'avg_inference_time' in summary:
            time_str = f"{summary['avg_inference_time']:.6f}s"
        else:
            time_str = "N/A"

        print(f"{name:<25} | {avg_p:>12.4f} | {sinr_min:>10.2f} | {success:>9.1f}% | {time_str:>12}")

    print(f"=" * 70)

    # 5. Save combined results
    with open(output_path / 'comparison_summary.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nCombined results saved to {output_path / 'comparison_summary.json'}")

    return results_dict


def main():
    parser = argparse.ArgumentParser(description='Compare DRL model with Baseline Optimizer')

    # DRL model
    parser.add_argument('--model-path', type=str,
                       default=None,
                       help='Path to trained DRL model (RecurrentPPO)')
    parser.add_argument('--vecnormalize-path', type=str,
                       default=None,
                       help='Path to VecNormalize statistics (required if model-path is provided)')
    parser.add_argument('--use-ae', action='store_true',
                       help='Model uses autoencoder-compressed state')
    parser.add_argument('--ae-checkpoint', type=str,
                       default='checkpoints/channel_ae/channel_ae_best.pth',
                       help='Path to autoencoder checkpoint')

    # Evaluation parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max-steps', type=int, default=128,
                       help='Maximum evaluation steps')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for DRL inference')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for results')

    # Method selection
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'zero', 'baseline', 'drl'],
                       help='Which method(s) to evaluate')

    args = parser.parse_args()

    # Check paths
    if args.method in ['all', 'drl'] and args.model_path is None:
        print("Error: --model-path is required for DRL evaluation")
        return 1

    if args.method in ['all', 'drl'] and args.vecnormalize_path is None:
        print("Error: --vecnormalize-path is required for DRL evaluation")
        return 1

    if args.use_ae and not Path(args.ae_checkpoint).exists():
        print(f"Error: AE checkpoint not found at {args.ae_checkpoint}")
        return 1

    # Run evaluation
    output_dir = args.output_dir

    if args.method == 'all':
        compare_all_methods(
            model_path=args.model_path,
            vecnormalize_path=args.vecnormalize_path,
            use_ae=args.use_ae,
            ae_checkpoint_path=args.ae_checkpoint,
            seed=args.seed,
            max_steps=args.max_steps,
            device=args.device,
            output_dir=output_dir
        )
    elif args.method == 'zero':
        evaluate_zero_action(
            seed=args.seed,
            max_steps=args.max_steps,
            save_path=Path(output_dir) / 'zero_action_results.json'
        )
    elif args.method == 'baseline':
        evaluate_baseline_optimizer(
            seed=args.seed,
            max_steps=args.max_steps,
            save_path=Path(output_dir) / 'baseline_optimizer_results.json'
        )
    elif args.method == 'drl':
        evaluate_drl_model(
            model_path=args.model_path,
            vecnormalize_path=args.vecnormalize_path,
            use_ae=args.use_ae,
            ae_checkpoint_path=args.ae_checkpoint,
            seed=args.seed,
            max_steps=args.max_steps,
            device=args.device,
            save_path=Path(output_dir) / 'drl_model_results.json'
        )

    return 0


if __name__ == '__main__':
    exit(main())
