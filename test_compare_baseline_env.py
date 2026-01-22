"""
Compare baseline optimizer and DRL environment step function
to verify consistency
"""
import numpy as np
from envs.itsn_env import ITSNEnv
from envs.scenario import ITSNScenario
from baseline.baseline_optimizer import BaselineZFSDROptimizer

def test_comparison():
    """Compare baseline and env implementations"""
    print("="*60)
    print("Comparing Baseline Optimizer vs DRL Environment")
    print("="*60)

    # Create scenario
    scenario = ITSNScenario(rng_seed=42)
    scenario.reset_user_positions()

    # Set satellite position
    elevation = 60.0
    azimuth = 90.0
    orbit_height = 500e3
    scenario.update_satellite_position(elevation, azimuth, orbit_height)

    # Generate channels
    channels = scenario.generate_channels()

    print("\n[1] Scenario Setup:")
    print(f"    - K (BS users): {scenario.K}")
    print(f"    - SK (SAT users): {scenario.SK}")
    print(f"    - N_t (BS antennas): {scenario.N_t}")
    print(f"    - N_sat (SAT antennas): {scenario.N_sat}")
    print(f"    - N_ris (RIS elements): {scenario.N_ris}")
    print(f"    - Satellite: ele={elevation}°, azi={azimuth}°")

    # Create baseline optimizer
    gamma_k = 10 ** (10.0 / 10) * np.ones(scenario.K)  # 10 dB SINR threshold
    gamma_j = 10 ** (10.0 / 10)  # 10 dB SINR threshold

    optimizer = BaselineZFSDROptimizer(
        K=scenario.K,
        J=scenario.SK,
        N_t=scenario.N_t,
        N_s=scenario.N_sat,
        N=scenario.N_ris,
        P_max=scenario.P_bs_max,
        sigma2=scenario.P_noise,
        gamma_k=gamma_k,
        gamma_j=gamma_j,
        P_b=scenario.P_bs_scale,
        P_s_init=scenario.P_sat,
        ris_amplitude_gain=scenario.ris_amplitude_gain,
        N_iter=1,  # Only 1 iteration for comparison
        verbose=False
    )

    # Run baseline optimization
    print("\n[2] Running Baseline Optimizer (1 iteration)...")
    w_opt, Phi_opt, info = optimizer.optimize(
        h_k=channels['H_BS2UE'],
        h_j=channels['H_BS2SUE'],
        h_s_k=channels['H_SAT2UE'],
        h_s_j=channels['H_SAT2SUE'],
        h_k_r=channels['H_RIS2UE'],
        h_j_r=channels['H_RIS2SUE'],
        G_BS=channels['G_BS'],
        G_S=channels['G_SAT'],
        W_sat=channels['W_sat']
    )

    P_bs_baseline = info['final_P_bs']
    P_sat_baseline = info['final_P_sat']
    P_total_baseline = info['final_P_sum']

    print(f"    - P_BS: {P_bs_baseline:.6f} W")
    print(f"    - P_SAT: {P_sat_baseline:.6f} W")
    print(f"    - P_TOTAL: {P_total_baseline:.6f} W")

    # Now test DRL environment with same RIS phase
    print("\n[3] Running DRL Environment with same RIS phase...")
    env = ITSNEnv(rng_seed=42, max_steps_per_episode=10, sinr_threshold_db=10.0)
    env.reset(seed=42)

    # Set same satellite position
    env.scenario.update_satellite_position(elevation, azimuth, orbit_height)
    env.current_channels = env.scenario.generate_channels()

    # Extract RIS phases from baseline
    phi_baseline = np.diag(Phi_opt)[:scenario.N_ris] / scenario.ris_amplitude_gain
    phases_baseline = np.angle(phi_baseline)

    # Convert to action space [-1, 1]
    action = 2 * phases_baseline / (2 * np.pi) - 1
    action = np.clip(action, -1, 1)

    # Execute step
    state, reward, terminated, truncated, info_env = env.step(action)

    P_bs_env = info_env['P_BS']
    P_sat_env = info_env['P_sat']
    P_total_env = P_bs_env + P_sat_env

    print(f"    - P_BS: {P_bs_env:.6f} W")
    print(f"    - P_SAT: {P_sat_env:.6f} W")
    print(f"    - P_TOTAL: {P_total_env:.6f} W")
    print(f"    - SINR min: {info_env['sinr_min_db']:.2f} dB")
    print(f"    - SINR mean: {info_env['sinr_mean_db']:.2f} dB")
    print(f"    - ZF success: {info_env['zf_success']}")

    # Compare results
    print("\n[4] Comparison:")
    print(f"    - P_BS diff: {abs(P_bs_baseline - P_bs_env):.6f} W ({abs(P_bs_baseline - P_bs_env)/P_bs_baseline*100:.2f}%)")
    print(f"    - P_SAT diff: {abs(P_sat_baseline - P_sat_env):.6f} W ({abs(P_sat_baseline - P_sat_env)/P_sat_baseline*100:.2f}%)")
    print(f"    - P_TOTAL diff: {abs(P_total_baseline - P_total_env):.6f} W ({abs(P_total_baseline - P_total_env)/P_total_baseline*100:.2f}%)")

    if abs(P_total_baseline - P_total_env) / P_total_baseline < 0.1:
        print("\n    ✓ Results are consistent (< 10% difference)")
    else:
        print("\n    ✗ Results differ significantly (> 10% difference)")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_comparison()