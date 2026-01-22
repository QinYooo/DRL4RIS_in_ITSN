# Baseline Implementation: ZF + SDR Method

This directory contains the Python implementation of the MATLAB baseline optimizer for RIS-assisted ITSN interference mitigation.

## Overview

The baseline algorithm uses a **Block Coordinate Descent (BCD)** approach to jointly optimize:
1. **Base Station (BS) Beamforming** using Zero-Forcing (ZF)
2. **RIS Phase Shifts** using Semi-Definite Relaxation (SDR)
3. **Satellite Power** to meet SINR constraints

This implementation is a direct translation of `optimize_ris_zf_SDR.m` from MATLAB to Python.

## Algorithm Flow

```
For each iteration:
  1. Compute effective channels (with current RIS phase Phi)
  2. Zero-Forcing beamforming: W = H^H (H H^H)^{-1}
  3. Power allocation (water-filling if needed)
  4. Optimize RIS phase using SDR:
     - Formulate SDP problem
     - Solve using CVXPY (SCS solver)
     - Gaussian randomization to recover phase
  5. Update satellite power to meet SINR constraint
  6. Check convergence (power change < threshold)
```

## Files

- **`baseline_optimizer.py`**: Main optimizer class (`BaselineZFSDROptimizer`)
- **`evaluate_baseline.py`**: Evaluation script for testing baseline performance
- **`__init__.py`**: Module initialization
- **`README.md`**: This file

## Usage

### 1. Basic Usage (Single Optimization)

```python
from baseline import BaselineZFSDROptimizer
from envs.scenario import ITSNScenario
import numpy as np

# Initialize scenario
scenario = ITSNScenario(rng_seed=42)

# Initialize optimizer
optimizer = BaselineZFSDROptimizer(
    K=scenario.K,          # Number of BS users
    J=scenario.SK,         # Number of SAT users
    N_t=scenario.N_t,      # BS antennas
    N_s=scenario.N_sat,    # SAT antennas
    N=scenario.N_ris,      # RIS elements
    P_max=scenario.P_bs_max,
    sigma2=scenario.P_noise,
    gamma_k=np.ones(scenario.K) * 10,  # SINR threshold (linear)
    gamma_j=10.0,
    P_b=scenario.P_bs_scale,
    N_iter=20,
    verbose=True
)

# Generate channels
scenario.update_satellite_position(ele=45, azi=90)
channels = scenario.generate_channels()

# Run optimization
w_opt, Phi_opt, info = optimizer.optimize(
    h_k=channels['h_k'],
    h_j=channels['h_j'],
    h_s_k=channels['h_s_k'],
    h_s_j=channels['h_s_j'],
    h_k_r=channels['h_k_r'],
    h_j_r=channels['h_j_r'],
    G_BS=channels['G_BS'],
    G_S=channels['G_S'],
    W_sat=np.random.randn(scenario.N_sat, scenario.SK) +
          1j * np.random.randn(scenario.N_sat, scenario.SK)
)

print(f"Final Power: {info['final_P_sum']:.4f} W")
print(f"Sum Rate: {info['sum_rate_history'][-1]:.4f} bps/Hz")
```

### 2. Evaluation Script

Evaluate baseline performance over multiple episodes:

```bash
# Basic evaluation (10 episodes, 50 steps each)
python baseline/evaluate_baseline.py --num_episodes 10 --num_steps 50 --save_results

# Verbose output
python baseline/evaluate_baseline.py --num_episodes 5 --verbose --save_results

# Custom output directory
python baseline/evaluate_baseline.py --num_episodes 20 --output_dir results/baseline_v1
```

**Output:**
- `baseline_results.json`: Numerical results (power, sum rate, etc.)
- `baseline_evaluation.png`: 4-panel figure showing:
  - Power per episode
  - Sum rate per episode
  - Power distribution histogram
  - Sum rate distribution histogram
- `baseline_trajectory_example.png`: Time series for a single episode

## Dependencies

### Required Packages
- `numpy`: Array operations
- `cvxpy`: Convex optimization for SDR
- `scipy`: Linear algebra utilities
- `matplotlib`: Plotting

### Installation
```bash
pip install numpy cvxpy scipy matplotlib
```

**Note:** CVXPY will automatically install a solver (SCS by default). For better performance, you can install MOSEK or Gurobi (requires license).

## Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `N_iter` | Max iterations | 20 | Usually converges in 5-10 iterations |
| `convergence_tol` | Power convergence threshold | 1e-2 | In Watts |
| `P_b` | BS power scaling | 0.001 | Matches MATLAB implementation |
| `P_s_init` | Initial SAT power scaling | 1.0 | Updated during optimization |
| `ris_amplitude_gain` | RIS element gain | 9.0 | Power gain = 81 |
| `gamma_k` | BS user SINR threshold | Array | Linear scale (not dB!) |
| `gamma_j` | SAT user SINR threshold | Scalar | Linear scale |

## Implementation Notes

### 1. Differences from MATLAB
- **Matrix indexing**: Python uses 0-based indexing (MATLAB is 1-based)
- **Complex conjugate transpose**: NumPy uses `.conj().T` (MATLAB uses `'`)
- **SDR solver**: CVXPY (Python) vs CVX (MATLAB)
  - Default solver: SCS (open-source)
  - Optimization precision: `eps=1e-4`, `max_iters=5000`

### 2. Numerical Stability
- Regularization factor `1e-6 * I` added to ZF inversion
- Eigenvalue ratio check in Gaussian randomization
- Numerical scaling factor `1e8` in SDR constraints (matches MATLAB)

### 3. Performance
- **Single optimization**: ~2-5 seconds per iteration (depends on problem size)
- **SDR solve time**: ~0.5-1.5 seconds per iteration
- **Gaussian randomization**: 1000 samples (adjustable)

### 4. Known Limitations
- **Computational cost**: SDR is expensive for large RIS (N > 200)
- **Convergence**: May not converge if SINR constraints are infeasible
- **Phase quantization**: Not implemented (uses continuous phases)

## Comparison with DRL

The baseline serves as a performance upper bound for the DRL agent:

| Metric | Baseline (ZF+SDR) | DRL (PPO/SAC) |
|--------|-------------------|---------------|
| Optimality | Near-optimal (relaxation gap) | Suboptimal (learned policy) |
| Computation | High (~2-5s per step) | Low (~10ms inference) |
| Adaptability | No (re-optimize each time) | Yes (generalize across scenarios) |
| CSI Requirement | Perfect CSI | Robust to ephemeris errors |

**Expected Results:**
- DRL should achieve **80-95%** of baseline performance
- DRL should be **100-1000x faster** in inference
- DRL should be **more robust** to CSI errors

## Troubleshooting

### Issue 1: SDR Solver Fails
**Symptom:** `cvx_status = 'failed'` or `'infeasible'`

**Solutions:**
1. Check SINR thresholds (may be too high)
2. Increase power budget `P_max`
3. Try different solver: `problem.solve(solver=cp.MOSEK)` (requires license)
4. Reduce numerical scaling factor from `1e8` to `1e6`

### Issue 2: Slow Convergence
**Symptom:** Iterations > 20

**Solutions:**
1. Increase `convergence_tol` from `1e-2` to `1e-1`
2. Reduce `N_iter` to force early stopping
3. Check if power oscillates (may indicate infeasibility)

### Issue 3: Poor Performance
**Symptom:** Very high power or low sum rate

**Solutions:**
1. Verify channel generation (check `scenario.generate_channels()`)
2. Check satellite beamforming `W_sat` (should be normalized)
3. Verify SINR thresholds are in linear scale (not dB)
4. Ensure `P_b` and `P_s` scaling are consistent with channels

## Citation

If you use this baseline in your research, please cite:

```bibtex
@article{original_paper,
  title={Energy-Efficient RIS-Assisted Co-channel Interference Mitigation in Integrated Terrestrial-Space Networks},
  author={[Original Authors]},
  journal={[Journal Name]},
  year={[Year]}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.