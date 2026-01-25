"""
Test if environment supports 2048-step rollouts
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from envs.itsn_env_ae import ITSNEnvAE

env = ITSNEnvAE(
    ae_checkpoint_path='checkpoints/channel_ae/channel_ae_best.pth',
    max_steps_per_episode=40,
    n_substeps=5,
    rng_seed=42
)

print('Testing 2048 steps collection...')
print(f'Episode length: {env.max_steps} RL steps')
print(f'Physics steps per episode: {env.total_physics_steps}')
print(f'Expected episodes in 2048 steps: {2048 / env.max_steps:.1f}')
print()

# Simulate collection
total_steps = 0
episodes = 0
obs, _ = env.reset()

for step in range(2048):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_steps += 1

    if terminated or truncated:
        episodes += 1
        obs, _ = env.reset()

print(f'[OK] Successfully collected {total_steps} steps')
print(f'[OK] Completed {episodes} full episodes')
print(f'[OK] Average episode length: {total_steps / max(episodes, 1):.1f}')
print()
print('Environment supports 2048-step rollouts!')
