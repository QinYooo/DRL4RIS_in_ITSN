"""
TD3 Training Script for RIS-Assisted ITSN
==========================================
Train a TD3 agent to control RIS phase shifts under ephemeris uncertainty.

Usage:
    python train.py --total_timesteps 100000 --ephemeris_noise 0.5
"""

import os
# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import gymnasium as gym

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor

from envs.itsn_env import ITSNEnv


class OUEphemerisNoise:
    """
    Ornstein-Uhlenbeck process for correlated ephemeris errors.
    Simulates realistic orbit prediction errors that are temporally correlated.

    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    """

    def __init__(self, mu=0.0, sigma=0.5, theta=0.15, dt=1.0, x0=None):
        """
        Args:
            mu: Long-term mean (typically 0 for error)
            sigma: Volatility (noise magnitude in degrees)
            theta: Mean reversion rate (higher = faster reversion)
            dt: Time step
            x0: Initial value
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        """Reset the noise process."""
        self.x_prev = self.x0 if self.x0 is not None else 0.0

    def __call__(self):
        """Generate next noise sample."""
        x = (self.x_prev
             + self.theta * (self.mu - self.x_prev) * self.dt
             + self.sigma * np.sqrt(self.dt) * np.random.randn())
        self.x_prev = x
        return x


class OUEphemerisWrapper(gym.Wrapper):
    """
    Wrapper to inject OU-correlated ephemeris noise into the environment.
    Replaces Gaussian noise with temporally correlated OU process.

    Works by:
    1. Disabling internal Gaussian noise (enable_ephemeris_noise = False)
    2. Overriding the observed angles with OU-correlated noise before _get_state()
    """

    def __init__(self, env, sigma_ele=0.5, sigma_azi=0.5, theta=0.15):
        """
        Args:
            env: ITSNEnv instance
            sigma_ele: OU noise std for elevation (degrees)
            sigma_azi: OU noise std for azimuth (degrees)
            theta: Mean reversion rate
        """
        super().__init__(env)
        self.ou_ele = OUEphemerisNoise(sigma=sigma_ele, theta=theta)
        self.ou_azi = OUEphemerisNoise(sigma=sigma_azi, theta=theta)

        # Disable internal Gaussian noise - we'll inject OU noise instead
        self.env.enable_ephemeris_noise = False

    def _inject_ou_noise(self):
        """Inject OU noise into observed satellite angles."""
        noise_ele = self.ou_ele()
        noise_azi = self.ou_azi()

        # Override observed angles (used for state construction and inferred CSI)
        self.env.curr_obs_elevation = np.clip(
            self.env.true_elevation + noise_ele, 10.0, 90.0
        )
        self.env.curr_obs_azimuth = np.clip(
            self.env.true_azimuth + noise_azi, 0.0, 360.0
        )

    def reset(self, **kwargs):
        """Reset environment and OU processes."""
        self.ou_ele.reset()
        self.ou_azi.reset()
        obs, info = self.env.reset(**kwargs)
        # Inject OU noise for initial observation
        self._inject_ou_noise()
        # Regenerate state with OU noise
        obs = self.env._get_state()
        return obs, info

    def step(self, action):
        """Step with OU-correlated ephemeris noise injection."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Inject OU noise after physics update
        self._inject_ou_noise()
        # Regenerate inferred G_SAT with new noisy observation
        self.env.inferred_G_SAT = self.env._generate_inferred_G_SAT()
        # Regenerate state with OU noise
        obs = self.env._get_state()
        return obs, reward, terminated, truncated, info


def make_env(ephemeris_noise_std=0.5, use_ou_noise=True, ou_theta=0.15):
    """
    Create and wrap the ITSN environment.

    Args:
        ephemeris_noise_std: Noise magnitude (degrees)
        use_ou_noise: Use OU process instead of Gaussian
        ou_theta: OU mean reversion rate
    """
    env = ITSNEnv(
        ephemeris_noise_std=ephemeris_noise_std if not use_ou_noise else 0.0,
        sinr_threshold_db=10.0,
        sinr_penalty_weight=10.0,
        phase_bits=2,  # 已改为连续相移，此参数仅保留用于兼容性
        max_steps_per_episode=100,
    )

    if use_ou_noise:
        env = OUEphemerisWrapper(
            env,
            sigma_ele=ephemeris_noise_std,
            sigma_azi=ephemeris_noise_std,
            theta=ou_theta
        )

    return env


def train_td3(args):
    """Main TD3 training function."""

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"td3_ris_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Logging to: {log_dir}")

    # Create environments
    train_env = make_env(
        ephemeris_noise_std=args.ephemeris_noise,
        use_ou_noise=args.use_ou_noise,
        ou_theta=args.ou_theta
    )
    train_env = Monitor(train_env, str(log_dir / "train"))

    eval_env = make_env(
        ephemeris_noise_std=args.ephemeris_noise,
        use_ou_noise=args.use_ou_noise,
        ou_theta=args.ou_theta
    )
    eval_env = Monitor(eval_env, str(log_dir / "eval"))

    # Get action dimension for noise
    n_actions = train_env.action_space.shape[0]

    # Action noise for exploration (TD3 uses target policy smoothing)
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.action_noise_std * np.ones(n_actions)
    )

    # Custom policy architecture for physical layer control
    policy_kwargs = dict(
        net_arch=dict(
            pi=[400, 300],  # Actor network
            qf=[400, 300]   # Critic network
        ),
    )

    # Create TD3 agent
    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir / "tensorboard"),
        verbose=1,
        device="auto",
    )

    print(f"[INFO] TD3 Agent created")
    print(f"  - Action dim: {n_actions}")
    print(f"  - Observation dim: {train_env.observation_space.shape[0]}")
    print(f"  - Policy arch: {policy_kwargs['net_arch']}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="td3_ris",
    )

    # Train
    print(f"\n[INFO] Starting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = log_dir / "final_model"
    model.save(str(final_path))
    print(f"[INFO] Final model saved to: {final_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    return model, log_dir


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 Training for RIS-ITSN")

    # Environment
    parser.add_argument("--ephemeris_noise", type=float, default=0.5,
                        help="Ephemeris noise std (degrees)")
    parser.add_argument("--use_ou_noise", action="store_true", default=True,
                        help="Use OU process for ephemeris noise")
    parser.add_argument("--ou_theta", type=float, default=0.15,
                        help="OU mean reversion rate")

    # TD3 hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--action_noise_std", type=float, default=0.1)

    # Training
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--log_dir", type=str, default="logs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, log_dir = train_td3(args)
    print(f"\n[DONE] Training complete. Results in: {log_dir}")
