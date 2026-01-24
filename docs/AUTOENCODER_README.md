# Channel Autoencoder for State Compression

This branch implements an autoencoder-based approach for compressing high-dimensional channel matrices into low-dimensional latent representations for the RL state space.

## Motivation

**Current approach (hand-crafted features):**
- State: satellite motion (6) + channel features (K+1) + interference (K+1) + SINR feedback (K+1)
- Total: 6 + 3*(K+1) = 21 dims for K=4
- Pros: Interpretable, fast, stable
- Cons: May lose spatial structure, requires domain knowledge

**Autoencoder approach:**
- State: satellite motion (6) + compressed channels (latent_dim) + SINR feedback (K+1)
- Total: 6 + 32 + 5 = 43 dims (for latent_dim=32)
- Pros: Preserves more information, end-to-end learning, discovers hidden patterns
- Cons: More complex, higher computational cost, less interpretable

## Implementation

### 1. Channel Autoencoder (`models/channel_autoencoder.py`)
- Compresses all channel matrices (H_BS2UE, G_BS, G_SAT, etc.) into latent vector
- Architecture: Encoder (input → 128 → 64 → latent_dim) + Decoder (symmetric)
- Uses LayerNorm and Dropout for stability

### 2. Pre-training Script (`scripts/train_channel_ae.py`)
Collects channel samples from diverse trajectories and trains the autoencoder:
```bash
python scripts/train_channel_ae.py
```

**Configuration:**
- N_SAMPLES: 10,000 channel samples
- LATENT_DIM: 32 (adjustable)
- EPOCHS: 100
- Output: `checkpoints/channel_ae/channel_ae_best.pth`

### 3. AE-based Environment (`envs/itsn_env_ae.py`)
Extends `ITSNEnv` to use pre-trained autoencoder for state compression:
```python
from envs.itsn_env_ae import ITSNEnvAE

env = ITSNEnvAE(
    ae_checkpoint_path='checkpoints/channel_ae/channel_ae_best.pth',
    latent_dim=32,
    max_steps_per_episode=40,
    n_substeps=5
)
```

### 4. Comparison Script (`scripts/compare_state_representations.py`)
Tests and visualizes both state representations:
```bash
python scripts/compare_state_representations.py
```

## Usage Workflow

### Phase 1: Pre-train Autoencoder
```bash
# Collect channel samples and train AE
python scripts/train_channel_ae.py

# Output:
# - checkpoints/channel_ae/channel_ae_best.pth
# - checkpoints/channel_ae/training_curve.png
# - checkpoints/channel_ae/reconstruction_samples.png
```

### Phase 2: Compare State Representations
```bash
# Test both environments
python scripts/compare_state_representations.py

# Output:
# - results/state_comparison.png
```

### Phase 3: Train RL Agent with AE Features
```python
from envs.itsn_env_ae import ITSNEnvAE
from stable_baselines3 import PPO

# Create environment with pre-trained AE
env = ITSNEnvAE(
    ae_checkpoint_path='checkpoints/channel_ae/channel_ae_best.pth',
    latent_dim=32
)

# Train RL agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

### Phase 4: Performance Comparison
Compare RL performance:
- Baseline: Hand-crafted features (`ITSNEnv`)
- Proposed: AE-compressed features (`ITSNEnvAE`)

Metrics:
- Convergence speed
- Final performance (power consumption, success rate)
- Computational overhead

## File Structure

```
DRL_RIS/
├── models/
│   └── channel_autoencoder.py          # AE model and preprocessing
├── envs/
│   ├── itsn_env.py                     # Original env (hand-crafted features)
│   └── itsn_env_ae.py                  # AE-based env
├── scripts/
│   ├── train_channel_ae.py             # Pre-train autoencoder
│   └── compare_state_representations.py # Compare both approaches
├── checkpoints/
│   └── channel_ae/
│       └── channel_ae_best.pth         # Pre-trained AE checkpoint
└── results/
    └── state_comparison.png            # Visualization
```

## Key Design Decisions

1. **Separate pre-training**: AE is pre-trained on diverse channel samples before RL training
   - Pros: Stable, faster RL training, reusable
   - Cons: Two-stage training

2. **Use inferred G_SAT**: State uses inferred G_SAT (with ephemeris noise) for consistency
   - Agent's belief matches what it observes

3. **Frozen AE during RL**: AE weights are frozen during RL training
   - Future work: End-to-end training with joint loss

## Expected Results

**Hypothesis:**
- AE features should capture more channel structure → better RL performance
- Trade-off: Higher state dimension (43 vs 21) and computational cost

**Validation:**
- Compare final power consumption and success rate
- Analyze latent space structure (t-SNE visualization)
- Measure inference time overhead

## Future Extensions

1. **End-to-end training**: Jointly train AE and RL agent
   - Loss: `L = L_RL + λ * L_reconstruction`

2. **Variational AE (VAE)**: Add stochasticity for robustness

3. **Attention mechanism**: Weight important channel components

4. **Recurrent AE**: Capture temporal dependencies in channel evolution
