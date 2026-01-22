"""
Test script for new state design in ITSN environment
"""
import numpy as np
from envs.itsn_env import ITSNEnv

def test_state_design():
    """Test the new compressed state design"""
    print("=" * 60)
    print("Testing New State Design")
    print("=" * 60)

    # Create environment
    env = ITSNEnv(
        rng_seed=42,
        max_steps_per_episode=10,
        phase_bits=2,
        sinr_threshold_db=10.0,
        ephemeris_noise_std=0.5
    )

    # Reset environment
    print("\n[1] Resetting environment...")
    state, info = env.reset(seed=42)

    print(f"[OK] Initial state shape: {state.shape}")
    print(f"[OK] Expected shape: ({7 + 4 * env.scenario.K},) = (23,)")

    # Verify state dimensions
    K = env.scenario.K
    expected_dim = 7 + 4 * K
    assert state.shape[0] == expected_dim, f"State dimension mismatch: {state.shape[0]} != {expected_dim}"

    # Parse state components
    idx = 0
    satellite_motion = state[idx:idx+6]
    idx += 6

    channel_features = state[idx:idx+2*K]
    idx += 2*K

    interference_features = state[idx:idx+K+1]
    idx += K+1

    performance_feedback = state[idx:idx+K]

    print(f"\n[2] State components:")
    print(f"  - Satellite motion (6): {satellite_motion}")
    print(f"    * sin(ele)={satellite_motion[0]:.3f}, cos(ele)={satellite_motion[1]:.3f}")
    print(f"    * sin(azi)={satellite_motion[2]:.3f}, cos(azi)={satellite_motion[3]:.3f}")
    print(f"    * delta_ele={satellite_motion[4]:.3f}, delta_azi={satellite_motion[5]:.3f}")

    print(f"  - Channel features (2K={2*K}): {channel_features}")
    print(f"    * g_k (BS signal): {channel_features[:K]}")
    print(f"    * g_k^s (SAT interference): {channel_features[K:]}")

    print(f"  - Interference features (K+1={K+1}): {interference_features}")
    print(f"    * I_k (interference to BS users): {interference_features[:K]}")
    print(f"    * I_s (interference to SAT user): {interference_features[K]:.3f}")

    print(f"  - Performance feedback (K={K}): {performance_feedback}")
    print(f"    * SINR margins: {performance_feedback}")

    # Take a few steps
    print(f"\n[3] Taking environment steps...")
    for step in range(3):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        print(f"\n  Step {step+1}:")
        print(f"    - State shape: {next_state.shape}")
        print(f"    - Reward: {reward:.2f}")
        print(f"    - P_BS: {info['P_BS']:.6f}W, P_sat: {info['P_sat']:.6f}W")
        print(f"    - SINR_min: {info['sinr_min_db']:.2f}dB")
        print(f"    - Satellite motion: ele={next_state[0]:.3f}, azi={next_state[2]:.3f}")
        print(f"    - Delta angles: Δele={next_state[4]:.3f}, Δazi={next_state[5]:.3f}")

        # Verify prev_* variables are updated
        assert env.prev_Phi is not None, "prev_Phi not updated"
        assert env.prev_W_bs is not None, "prev_W_bs not updated"
        assert env.prev_sinr_values is not None, "prev_sinr_values not updated"

        if terminated or truncated:
            break

    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_state_design()
