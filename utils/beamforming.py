"""
Beamforming utilities for ITSN environment
Implements Zero-Forcing (ZF), Water-Filling, and power calculation
"""

import numpy as np
from scipy.linalg import pinv
from scipy.optimize import minimize


def compute_zero_forcing_beamforming(H_eq, rate_requirement, bandwidth, noise_power):
    """
    Compute Zero-Forcing beamforming weights to minimize power
    while satisfying rate constraints for all users

    Parameters:
    -----------
    H_eq : np.ndarray, shape (K, N_t)
        Equivalent channel matrix (including RIS cascade)
    rate_requirement : float
        Required rate per user (bps)
    bandwidth : float
        System bandwidth (Hz)
    noise_power : float
        Noise power (Watts)

    Returns:
    --------
    W : np.ndarray, shape (N_t, K)
        Beamforming matrix (each column is beamforming vector for one user)
    P_total : float
        Total BS transmit power (Watts)
    success : bool
        Whether ZF was successful
    """
    K, N_t = H_eq.shape

    # Zero-Forcing: W = H^H (H H^H)^{-1}
    # This nulls inter-user interference

    try:
        # Compute pseudo-inverse for Zero-Forcing
        # W_zf = H^H (H H^H)^{-1}
        H_hermitian = H_eq.conj().T  # (N_t, K)
        HHH = H_eq @ H_hermitian  # (K, K)

        # Regularization for numerical stability
        reg_factor = 1e-8 * np.eye(K)
        HHH_inv = np.linalg.inv(HHH + reg_factor)

        W_zf = H_hermitian @ HHH_inv  # (N_t, K)

        # Normalize to satisfy rate constraints
        # After ZF, effective channel becomes diagonal: H @ W_zf = I (ideally)
        # SINR_k = |h_k^H w_k|^2 * P_k / noise_power

        # Compute required SINR per user to meet rate requirement
        # R_k = B * log2(1 + SINR_k)
        # SINR_k = 2^(R_k / B) - 1
        required_sinr = 2 ** (rate_requirement / bandwidth) - 1

        # For ZF, SINR_k = |h_k^H w_k|^2 * P_k / noise_power
        # We need: |h_k^H w_k|^2 * P_k >= required_sinr * noise_power

        # Compute channel gains after ZF
        # Since W_zf achieves H @ W_zf ≈ I, the gain is approximately the normalization
        # More precisely: h_k^H w_k for each user k

        gains = np.abs(np.diag(H_eq @ W_zf))  # (K,)

        # Required power per user: P_k = (required_sinr * noise_power) / |gain_k|^2
        required_powers = (required_sinr * noise_power) / (gains ** 2 + 1e-12)

        # Scale each beamforming vector by sqrt(P_k)
        W = W_zf * np.sqrt(required_powers)[np.newaxis, :]  # (N_t, K)

        # Total power
        P_total = np.sum(np.abs(W) ** 2)

        # Sanity check: If power is too large, mark as failure
        if P_total > 1e3:  # 1 kW threshold (adjust as needed)
            return W, P_total, False

        return W, P_total, True

    except np.linalg.LinAlgError:
        # If matrix inversion fails, return failure
        return np.zeros((N_t, K), dtype=complex), 1e6, False


def compute_sinr_and_power(H_eq, W, W_sat, H_SAT2UE, noise_power):
    """
    Compute SINR for each user given beamforming weights

    Parameters:
    -----------
    H_eq : np.ndarray, shape (K, N_t)
        Equivalent BS channel
    W : np.ndarray, shape (N_t, K)
        BS beamforming matrix
    W_sat : np.ndarray, shape (N_sat, 1)
        Satellite beamforming vector
    H_SAT2UE : np.ndarray, shape (K, N_sat)
        Satellite to BS users channel
    noise_power : float
        Noise power

    Returns:
    --------
    sinr_values : np.ndarray, shape (K,)
        SINR for each BS user (linear scale)
    power_bs : float
        Total BS power
    """
    K = H_eq.shape[0]

    # Signal power at each user k: |h_k^H w_k|^2
    signal_power = np.abs(np.diag(H_eq @ W)) ** 2  # (K,)

    # Inter-user interference: sum_{j != k} |h_k^H w_j|^2
    HW = H_eq @ W  # (K, K)
    interference_power = np.sum(np.abs(HW) ** 2, axis=1) - signal_power  # (K,)

    # Satellite interference at each BS user: |h_sat_k^H W_sat|^2 * P_sat
    # Assume unit satellite power here (already included in W_sat)
    sat_interference = np.abs(H_SAT2UE @ W_sat).flatten() ** 2  # (K,)

    # SINR = signal / (interference + sat_interference + noise)
    sinr_values = signal_power / (interference_power + sat_interference + noise_power + 1e-12)

    # Total BS power
    power_bs = np.sum(np.abs(W) ** 2)

    return sinr_values, power_bs


def compute_mrt_beamforming(H_eq, total_power):
    """
    Maximum Ratio Transmission (MRT) beamforming
    Simple alternative to ZF: W_k = h_k^H / ||h_k||

    Parameters:
    -----------
    H_eq : np.ndarray, shape (K, N_t)
        Equivalent channel
    total_power : float
        Total available power to allocate

    Returns:
    --------
    W : np.ndarray, shape (N_t, K)
        MRT beamforming matrix
    """
    K, N_t = H_eq.shape

    # Normalize each channel to unit norm
    W = H_eq.conj().T  # (N_t, K)

    # Normalize columns
    norms = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    W = W / norms

    # Equal power allocation
    power_per_user = total_power / K
    W = W * np.sqrt(power_per_user)

    return W


def compute_zf_waterfilling_baseline(H_eff_k, H_eff_j, H_sat_eff_k, W_sat, P_sat, P_bs_scale,
                                     sinr_threshold_linear, noise_power, max_power=10.0):
    """
    Baseline-style Zero-Forcing beamforming with iterative power allocation.

    This implements the algorithm from baseline_optimizer.py:
    1. Joint ZF for all users (BS + SAT users)
    2. Extract BS beamforming vectors and normalize
    3. Iterative power allocation to meet SINR constraints
    4. Water-filling if power budget exceeded

    Parameters:
    -----------
    H_eff_k : np.ndarray, shape (K, N_t)
        Effective BS-to-BS-user channel (including RIS)
    H_eff_j : np.ndarray, shape (J, N_t)
        Effective BS-to-SAT-user channel (including RIS)
    H_sat_eff_k : np.ndarray, shape (K, N_sat)
        Effective SAT-to-BS-user channel (including RIS)
    W_sat : np.ndarray, shape (N_sat, J)
        Satellite beamforming matrix
    P_sat : float
        Satellite transmit power per antenna
    P_bs_scale : float
        BS power scaling factor
    sinr_threshold_linear : float
        SINR threshold (linear scale)
    noise_power : float
        Noise power
    max_power : float
        Maximum BS power budget

    Returns:
    --------
    W : np.ndarray, shape (N_t, K)
        BS beamforming matrix with power allocation
    P_total : float
        Total BS power (before scaling)
    success : bool
        Whether power allocation succeeded
    info : dict
        Additional information
    """
    K, N_t = H_eff_k.shape
    J = H_eff_j.shape[0]

    try:
        # Step 1: Joint Zero-Forcing for all users (BS + SAT)
        # Following baseline_optimizer.py line 281-297
        H_combine = np.vstack([H_eff_k, H_eff_j]).T  # (N_t, K+J)
        H_H = H_combine.conj().T  # (K+J, N_t)
        HHH = H_H @ H_combine  # (K+J, K+J)

        # Check rank
        rank_H = np.linalg.matrix_rank(H_combine)
        if rank_H < K + J:
            reg_factor = 1e-6 * np.eye(K + J)
        else:
            reg_factor = 1e-6 * np.eye(K + J)

        # Compute ZF weights: W = H (H^H H)^{-1}
        w_all = H_combine @ np.linalg.inv(HHH)  # (N_t, K+J)

        # Extract BS users' beamforming and normalize
        w = w_all[:, :K]  # (N_t, K)
        w = w / (np.linalg.norm(w, 'fro'))

        # Step 2: Iterative power allocation
        p = np.zeros(K)
        for k in range(K):
            # Signal power (before power scaling)
            signal_power = P_bs_scale * np.abs(H_eff_k[k].conj() @ w[:, k]) ** 2

            # Interference from other BS users
            interference = 0
            for m in range(K):
                if m != k:
                    interference += P_bs_scale * np.abs(H_eff_k[k].conj() @ w[:, m]) ** 2

            # Interference from satellite
            for j in range(J):
                interference += P_sat * np.abs(H_sat_eff_k[k].conj() @ W_sat[:, j]) ** 2

            # Required power coefficient: p[k] = gamma * (interference + noise) / signal
            p[k] = max(sinr_threshold_linear * (interference + noise_power) / (signal_power + 1e-20), 0)

        # Step 3: Check power budget
        total_power = np.linalg.norm(np.sqrt(p).reshape(-1, 1) * w.T, 'fro') ** 2

        if total_power > max_power:
            # Use water-filling
            p = _water_filling_power_allocation_baseline(
                H_eff_k, w, H_sat_eff_k, W_sat, P_sat, P_bs_scale,
                sinr_threshold_linear, noise_power, max_power, K, J
            )
            success = False
            note = 'Power budget exceeded, water-filling applied'
        else:
            success = True
            note = 'Power allocation successful'

        # Step 4: Apply power allocation
        W = w * np.sqrt(p)[np.newaxis, :]  # (N_t, K)
        P_total = np.sum(np.abs(W) ** 2)

        # Step 5: Compute actual SINR
        sinr_values = np.zeros(K)
        for k in range(K):
            signal = P_bs_scale * np.abs(H_eff_k[k].conj() @ W[:, k]) ** 2
            interference = noise_power
            for m in range(K):
                if m != k:
                    interference += P_bs_scale * np.abs(H_eff_k[k].conj() @ W[:, m]) ** 2
            for j in range(J):
                interference += P_sat * np.abs(H_sat_eff_k[k].conj() @ W_sat[:, j]) ** 2
            sinr_values[k] = signal / interference

        info = {
            'per_user_powers': p,
            'sinr_values': sinr_values,
            'sinr_values_db': 10 * np.log10(sinr_values + 1e-12),
            'channel_condition_number': np.linalg.cond(HHH),
            'zf_quality': rank_H / (K + J),
            'note': note
        }

        return W, P_total, success, info

    except np.linalg.LinAlgError as e:
        return np.zeros((N_t, K), dtype=complex), 1e6, False, {'error': str(e)}


def _water_filling_power_allocation_baseline(H_eff_k, w, H_sat_eff_k, W_sat, P_sat, P_bs_scale,
                                              gamma_k, noise_power, P_max, K, J):
    """
    Water-filling power allocation (baseline style).

    Uses iterative bisection to find water level mu.
    """
    mu = 1e-3
    step = 1e-2
    max_iter = 1000
    epsilon = 1e-6

    # Compute normalized channel gains
    g = np.zeros(K)
    for k in range(K):
        g_k = np.abs(H_eff_k[k].conj() @ w[:, k]) ** 2

        # Interference
        interference = noise_power
        for j in range(J):
            interference += P_sat * np.abs(H_sat_eff_k[k].conj() @ W_sat[:, j]) ** 2

        g[k] = g_k / (gamma_k * interference)

    # Sort gains
    idx = np.argsort(g)[::-1]
    g_sorted = g[idx]
    idx_map = np.zeros(K, dtype=int)
    idx_map[idx] = np.arange(K)

    # Water-filling iteration
    for iteration in range(max_iter):
        p = np.zeros(K)
        active_users = 0

        for i in range(K):
            if 1 / g_sorted[i] < mu:
                active_users += 1
            else:
                break

        if active_users > 0:
            for i in range(active_users):
                p[idx_map[idx[i]]] = mu - 1 / g_sorted[i]

            total_power = np.sum(p)

            if abs(total_power - P_max) < epsilon:
                break
            elif total_power > P_max:
                mu = mu - step
                step = step / 2
            else:
                mu = mu + step
        else:
            mu = mu + step

    return np.maximum(p, 0)


def compute_zf_waterfilling(H_eq, sinr_threshold_db, noise_power, max_power=10.0,
                            H_sat_eff=None, W_sat=None, P_sat=None):
    """
    Compute Zero-Forcing beamforming with Water-Filling power allocation
    to minimize total power while satisfying SINR constraints

    Considers satellite interference in SINR calculation.

    Algorithm:
    1. Compute ZF beamforming directions: W_zf = H^H(HH^H)^(-1)
    2. Use Water-Filling to allocate optimal power to each user
    3. Subject to: SINR_k >= sinr_threshold for all users

    Parameters:
    -----------
    H_eq : np.ndarray, shape (K, N_t)
        Equivalent channel matrix (including RIS cascade)
    sinr_threshold_db : float
        Minimum required SINR per user (in dB)
    noise_power : float
        Noise power (Watts)
    max_power : float
        Maximum allowable total power (Watts), default 10W
    H_sat_eff : np.ndarray, shape (K, N_sat), optional
        Effective satellite-to-BS-user channel (including RIS path)
    W_sat : np.ndarray, shape (N_sat, 1) or (N_sat,), optional
        Satellite beamforming vector
    P_sat : float, optional
        Satellite transmit power (Watts)

    Returns:
    --------
    W : np.ndarray, shape (N_t, K)
        Beamforming matrix with optimized power allocation
    P_total : float
        Total BS transmit power (Watts)
    success : bool
        Whether optimization was successful
    info : dict
        Additional information (per-user powers, SINR values, etc.)
    """
    K, N_t = H_eq.shape
    sinr_threshold_linear = 10 ** (sinr_threshold_db / 10)

    try:
        # Step 1: Compute Zero-Forcing beamforming directions
        H_hermitian = H_eq.conj().T  # (N_t, K)
        HHH = H_eq @ H_hermitian  # (K, K)

        # Check if channel is well-conditioned
        cond_num = np.linalg.cond(HHH)
        if cond_num > 1e6:
            # Ill-conditioned, use regularized ZF
            reg_factor = 1e-6 * np.trace(HHH) / K
            HHH_reg = HHH + reg_factor * np.eye(K)
            HHH_inv = np.linalg.inv(HHH_reg)
        else:
            # Well-conditioned, standard ZF
            reg_factor = 1e-10 * np.eye(K)
            HHH_inv = np.linalg.inv(HHH + reg_factor)

        W_zf = H_hermitian @ HHH_inv  # (N_t, K), unnormalized

        # Step 2: Compute effective channel gains after ZF
        # After ZF, H @ W_zf should be approximately diagonal
        H_eff = H_eq @ W_zf  # (K, K)

        # Extract diagonal gains (desired signal)
        gains_desired = np.abs(np.diag(H_eff))  # (K,)

        # Extract off-diagonal interference (residual after ZF)
        interference_matrix = H_eff - np.diag(np.diag(H_eff))

        # Normalize ZF vectors to unit norm
        w_zf_norms = np.linalg.norm(W_zf, axis=0)  # (K,)
        W_zf_normalized = W_zf / (w_zf_norms[np.newaxis, :] + 1e-12)

        # Recompute gains for normalized vectors
        gains_normalized = gains_desired / (w_zf_norms + 1e-12)  # (K,)

        # Step 3: Compute satellite interference (if provided)
        sat_interference = np.zeros(K)
        if H_sat_eff is not None and W_sat is not None and P_sat is not None:
            W_sat_vec = W_sat.flatten() if W_sat.ndim > 1 else W_sat
            for k in range(K):
                # Satellite interference to user k: P_sat * |h_sat_k^H * w_sat|^2
                sat_interference[k] = P_sat * np.abs(H_sat_eff[k, :] @ W_sat_vec) ** 2

        # Step 4: Water-Filling power allocation
        # For ZF with perfect interference cancellation:
        # SINR_k = (g_k^2 * P_k) / (sat_interference_k + noise)
        # Constraint: g_k^2 * P_k >= sinr_threshold * (sat_interference_k + noise) for all k
        # Objective: minimize sum(P_k)

        # Total interference + noise per user (including satellite interference)
        total_noise_per_user = sat_interference + noise_power

        # Minimum required power per user (considering satellite interference)
        P_min_per_user = (sinr_threshold_linear * total_noise_per_user) / (gains_normalized ** 2 + 1e-12)

        # Check if minimum required powers are feasible
        if np.any(P_min_per_user > max_power):
            # Infeasible: even allocating all power to one user is insufficient
            return np.zeros((N_t, K), dtype=complex), 1e6, False, {
                'error': 'Infeasible power requirements',
                'min_required_powers': P_min_per_user,
                'max_power': max_power
            }

        if np.sum(P_min_per_user) > max_power:
            # Total minimum power exceeds budget
            # Fall back to proportional scaling
            scale_factor = max_power / np.sum(P_min_per_user)
            P_allocated = P_min_per_user * scale_factor

            success_flag = False  # Cannot meet all SINR constraints
            note = 'Power budget insufficient, scaled proportionally'
        else:
            # Feasible: use minimum required powers (no extra optimization needed)
            # For ZF, minimum power allocation IS optimal (each user gets exactly what it needs)
            P_allocated = P_min_per_user

            success_flag = True
            note = 'Optimal power allocation (minimum required)'

        # Step 5: Construct final beamforming matrix
        W = W_zf_normalized * np.sqrt(P_allocated)[np.newaxis, :]  # (N_t, K)

        # Total power
        P_total = np.sum(np.abs(W) ** 2)

        # Compute actual SINR values for verification
        # SINR_k = |h_k^H w_k|^2 / (inter-user interference + satellite interference + noise)
        H_W = H_eq @ W  # (K, K)
        signal_powers = np.abs(np.diag(H_W)) ** 2  # (K,)

        # Compute residual inter-user interference (off-diagonal terms)
        interference_powers = np.sum(np.abs(H_W) ** 2, axis=1) - signal_powers  # (K,)

        # Total interference + noise (sat_interference already computed in Step 3)
        total_interference = interference_powers + sat_interference + noise_power

        # SINR = signal / (interference + satellite_interference + noise)
        sinr_values = signal_powers / total_interference  # (K,)

        # Package additional info
        info = {
            'per_user_powers': P_allocated,
            'sinr_values': sinr_values,
            'sinr_values_db': 10 * np.log10(sinr_values + 1e-12),
            'channel_condition_number': cond_num,
            'zf_quality': np.sum(np.abs(np.diag(H_eff)) ** 2) / np.sum(np.abs(H_eff) ** 2),
            'sat_interference': sat_interference,
            'note': note
        }

        return W, P_total, success_flag, info

    except np.linalg.LinAlgError as e:
        # Matrix inversion failed
        return np.zeros((N_t, K), dtype=complex), 1e6, False, {
            'error': f'LinAlgError: {str(e)}'
        }

