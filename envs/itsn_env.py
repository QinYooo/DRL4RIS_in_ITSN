"""
ITSN Gym Environment with Ephemeris-Aware State Space
Wraps ITSNScenario for DRL training with hybrid optimization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.scenario import ITSNScenario
from utils.beamforming import compute_zf_waterfilling_baseline


class ITSNEnv(gym.Env):
    """
    Deep Reinforcement Learning Environment for RIS-Assisted ITSN

    State Space (Ephemeris-Aware):
        - Satellite elevation angle (noisy observation)
        - Satellite azimuth angle (noisy observation)
        - Angular velocity (delta_ele, delta_azi)
        - Normalized time step within pass
        - Local CSI features (channel statistics)

    Action Space:
        - Discrete RIS phase shifts (quantized to 1-bit or 2-bit)

    Reward:
        - Negative total power consumption with SINR constraint penalty
    """

    def __init__(self,
                 rng_seed=None,
                 max_steps_per_episode=40,
                 phase_bits=4,  # 1-bit or 2-bit quantization
                 sinr_threshold_db=10.0,  # Minimum SINR requirement
                 sinr_penalty_weight=100.0,
                 ephemeris_noise_std=0.5,  # Degrees of angle noise
                 rate_requirement_bps=1e6,  # Rate constraint per user
                 orbit_height=500e3):

        super().__init__()

        # Initialize scenario
        self.scenario = ITSNScenario(rng_seed=rng_seed if rng_seed else np.random.randint(0, 10000))
        self.rng = np.random.RandomState(rng_seed)

        # Episode parameters
        self.max_steps = max_steps_per_episode
        self.current_step = 0
        self.orbit_height = orbit_height

        # RIS quantization
        self.phase_bits = phase_bits
        self.num_phase_levels = 2 ** phase_bits
        self.phase_codebook = np.linspace(0, 2*np.pi, self.num_phase_levels, endpoint=False)

        # Reward parameters
        self.sinr_threshold_linear = 10 ** (sinr_threshold_db / 10.0)
        self.sinr_penalty_weight = sinr_penalty_weight
        self.rate_requirement = rate_requirement_bps

        # Ephemeris error simulation
        self.ephemeris_noise_std = ephemeris_noise_std

        # Trajectory storage (for ephemeris-aware state)
        self.trajectory_elevation = None
        self.trajectory_azimuth = None
        self.true_elevation = 0.0
        self.true_azimuth = 0.0
        self.prev_elevation = 0.0
        self.prev_azimuth = 0.0

        # Current channels (cached)
        self.current_channels = None

        # Initial RIS phase (identity matrix scaled by amplitude gain)
        self.initial_ris_phase = self.scenario.ris_amplitude_gain * np.eye(self.scenario.N_ris, dtype=complex)

        # Previous step information for compressed state construction
        self.prev_Phi = None  # Previous RIS reflection matrix (N_ris, N_ris)
        self.prev_W_bs = None  # Previous BS beamforming (N_t, K)
        self.prev_sinr_values = None  # Previous SINR values (K,)
        self.prev_obs_elevation = 0.0  # Previous observed elevation (with noise)
        self.prev_obs_azimuth = 0.0  # Previous observed azimuth (with noise)

        # Define action space: Continuous output for each RIS element, then quantize
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.scenario.N_ris,),
            dtype=np.float32
        )

        # State space dimension calculation revious step info)
        # 1. Satellite motion: 6 features
        #    - sin(elevation), cos(elevation), sin(azimuth), cos(azimuth): 4
        #    - delta_elevation, delta_azimuth (from noisy ephemeris): 2
        # 2. Compressed channel features (previous step): 2K features
        #    - g_k,t-1 = |h_k^H + h_r,k^H Phi_t-1 G|^2 (BS user signal strength): K
        #    - g_k,t-1^s = |h_s + h_r,k^H Phi_t-1 G_SAT|^2 |w_sat|^2 (SAT interference to BS users): K
        # 3. Interference features (previous step): K+1 features
        #    - I_k,t-1 = total interference to BS user k (SAT + other BS users): K
        #    - I_s,t-1 = total interference to SAT user: 1
        # 4. Performance feedback: K features
        #    - SINR margin m_k = (SINR_k,t-1 / threshold) - 1: K
        # Totr K=4: state_dim = 7 + 16 = 23
        state_dim = 7 + 4 * self.scenario.K

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset environment and generate a new satellite pass trajectory

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional options:
            - 'trajectory': dict with 'elevation' and 'azimuth' arrays
                If provided, uses external trajectory instead of generating one
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.scenario = ITSNScenario(rng_seed=seed)

        # Reset step counter
        self.current_step = 0

        # Load trajectory from options or generate default
        if options is not None and 'trajectory' in options:
            # Use external trajectory
            traj = options['trajectory']
            self.trajectory_elevation = np.array(traj['elevation'])
            self.trajectory_azimuth = np.array(traj['azimuth'])

            # Validate trajectory length
            if len(self.trajectory_elevation) != self.max_steps or len(self.trajectory_azimuth) != self.max_steps:
                raise ValueError(f"Trajectory length mismatch: expected {self.max_steps}, "
                               f"got elevation={len(self.trajectory_elevation)}, azimuth={len(self.trajectory_azimuth)}")
        else:
            # Generate default trajectory (satellite pass)
            mid_point = self.max_steps // 2
            ele_up = np.linspace(30, 85, mid_point)
            ele_down = np.linspace(85, 30, self.max_steps - mid_point)
            self.trajectory_elevation = np.concatenate([ele_up, ele_down])

            # Azimuth changes gradually
            self.trajectory_azimuth = np.linspace(
                self.rng.uniform(70, 90),
                self.rng.uniform(90, 110),
                self.max_steps
            )

        # Reset user positions for this episode
        self.scenario.reset_user_positions()

        # Reset RIS phase to identity (starting point for each episode)
        self.initial_ris_phase = self.scenario.ris_amplitude_gain * np.eye(self.scenario.N_ris, dtype=complex)

        # Initialize satellite at first trajectory point
        self.true_elevation = self.trajectory_elevation[0]
        self.true_azimuth = self.trajectory_azimuth[0]
        self.prev_elevation = self.true_elevation
        self.prev_azimuth = self.true_azimuth

        # Update scenario satellite position (TRUE position for channel generation)
        self.scenario.update_satellite_position(
            self.true_elevation,
            self.true_azimuth,
            self.orbit_height
        )

        # Generate initial channels
        self.current_channels = self.scenario.generate_channels()

        # Initialize with ZF-waterfilling to verify channel quality and get initial prev_* values
        self._initialize_with_zf_waterfilling()

        # = self.true_elevation + self.rng.normal(0, self   self.prev_obs_azimuth = self.true_azimuth + self.rng.normal(0, self.ephemeris_noise_std)
        
        # The 0 step obsevation:
        # Add ephemeris noise to observed angles
        obs_ele_0 = self.true_elevation + self.rng.normal(0, self.ephemeris_noise_std)
        obs_azi_0 = self.true_azimuth + self.rng.normal(0, self.ephemeris_noise_std)
        self.prev_obs_elevation = obs_ele_0
        self.prev_obs_azimuth = obs_azi_0
        # Current Observation
        self.curr_obs_elevation = obs_ele_0 
        self.curr_obs_azimuth = obs_azi_0

        # Get initial state (with ephemeris noise)
        state = self._get_state()

        return state, {}

    def step(self, action):
        """
        重构后的step函数，采用统一SINR计算，消除数据滞后。
        """
        # --- 1. 动作执行 (t时刻) ---
        action_clipped = np.clip(action, -1.0, 1.0)
        phase_indices = ((action_clipped + 1.0) / 2.0 * (self.num_phase_levels - 1)).astype(int)
        ris_phases = self.phase_codebook[np.clip(phase_indices, 0, self.num_phase_levels - 1)]
        Phi = np.diag(self.scenario.ris_amplitude_gain * np.exp(1j * ris_phases))

        # --- 2. 混合优化 (t时刻) ---
        # 此时 current_channels 是 t 时刻的真实信道
        try:
            # A. 初步计算 BS 波束用于估计干扰
            H_eff_k, H_eff_j, H_sat_eff_k = self._get_all_eff_channels(Phi, self.current_channels)
            
            W_init, _, _, _ = compute_zf_waterfilling_baseline(
                H_eff_k, H_eff_j, H_sat_eff_k, self.current_channels['W_sat'],
                P_sat=self.scenario.P_sat, P_bs_scale=self.scenario.P_bs_scale,
                sinr_threshold_linear=self.sinr_threshold_linear,
                noise_power=self.scenario.P_noise, max_power=self.scenario.P_bs_max
            )

            # B. 确定满足 SU 约束的最小 P_sat
            P_sat_t, _ = self.scenario.compute_sat_power(
                Phi, self.current_channels, W_bs=W_init, sinr_threshold_db=self.sinr_threshold_db
            )

            # C. 确定最终的 BS 波束 W
            W, P_BS_norm, success, _ = compute_zf_waterfilling_baseline(
                H_eff_k, H_eff_j, H_sat_eff_k, self.current_channels['W_sat'],
                P_sat=P_sat_t, P_bs_scale=self.scenario.P_bs_scale,
                sinr_threshold_linear=self.sinr_threshold_linear,
                noise_power=self.scenario.P_noise, max_power=self.scenario.P_bs_max
            )
            P_BS = self.scenario.P_bs_scale * P_BS_norm
            
            # --- 核心：统一计算所有用户在 t 时刻的真实 SINR ---
            # 这确保了 sinr_values 与当前的 W, P_sat, W_sat 完全匹配
            all_sinr_values = self.calculate_all_sinrs(Phi, W, P_sat_t, self.current_channels)

        except Exception:
            success = False
            P_BS, P_sat_t = 10.0, 10.0
            all_sinr_values = np.zeros(self.scenario.K + 1)

        # --- 3. 奖励与快照 ---
        reward = self._calculate_reward(P_BS, P_sat_t, all_sinr_values, success)
        
        # 记录快照 (注意：存入 prev 的是 t 时刻的真实性能反馈)
        self.prev_Phi = Phi
        self.prev_W_bs = W if success else np.zeros_like(W)
        self.prev_sinr_values = all_sinr_values
        self.prev_obs_elevation = self.curr_obs_elevation
        self.prev_obs_azimuth = self.curr_obs_azimuth

        # --- 4. 环境演进 (t -> t+1) ---
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        if not terminated:
            self._advance_physics() # 更新真实位置、CSI、W_sat 
            self._update_observation() # 产生 t+1 时刻噪声观测

        # --- 5. 获取 s_{t+1} ---
        state = self._get_state()
        return state, reward, terminated, False, {"success": success, "P_BS": P_BS, "P_sat": P_sat_t}

    def _calculate_reward(self, P_BS, P_sat, all_sinr_values, success):
        """
        综合考虑系统总能耗和所有用户 (K+1) 的 SINR 满足情况 [cite: 83, 87, 168]
        """
        # 1. 能效奖励：最小化系统总功耗 [cite: 168]
        total_p = P_BS + P_sat
        reward_energy = -10 * np.log10(total_p + 1e-6)

        # 2. SINR 约束检查 (K 个 UE + 1 个 SU)
        sinr_db = 10 * np.log10(all_sinr_values + 1e-12)
        # 计算每个用户相对于阈值的违规量 [cite: 169]
        violations = np.maximum(0, (self.sinr_threshold_db - sinr_db))
        penalty_sinr = -self.sinr_penalty_weight * np.sum(violations)

        # 3. 综合奖励
        # 如果所有约束满足 (success=True 且 violations=0)，给予额外奖励
        bonus_all_satisfied = 5.0 if success and np.all(violations <= 1e-3) else 0.0
        
        return reward_energy + penalty_sinr + bonus_all_satisfied


    def _initialize_with_zf_waterfilling(self):
        """
        Initialize environment with ZF-waterfilling to verify channel quality.
        Uses the initial RIS phase (identity) as starting point.

        Note: Satellite beamforming and power a initialized in generate_channels()
        """
        # Use initial RIS phase (identity matrix)
        Phi_init = self.initial_ris_phase

        # Extract channels (satellite W_sat and P_sat already initialized)
        H_BS2UE = self.current_channels['H_BS2UE']
        H_BS2SUE = self.current_channels['H_BS2SUE']
        H_RIS2UE = self.current_channels['H_RIS2UE']
        H_RIS2SUE = self.current_channels['H_RIS2SUE']
        G_BS = self.current_channels['G_BS']
        G_SAT = self.current_channels['G_SAT']
        H_SAT2UE = self.current_channels['H_SAT2UE']
        W_sat = self.current_channels['W_sat']

        # Compute effective channels with initial RIS phase
        H_eff_k = H_BS2UE + H_RIS2UE @ Phi_init @ G_BS
        H_eff_j = H_BS2SUE + H_RIS2SUE @ Phi_init @ G_BS
        H_sat_eff_k = H_SAT2UE + H_RIS2UE @ Phi_init @ G_SAT

        # Use satellite power from scenario initialization
        P_sat_init = self.scenario.P_sat

        # Run ZF-waterfilling to verify channel quality
        try:
            W_init, P_BS_norm, success, bf_info = compute_zf_waterfilling_baseline(
                H_eff_k, H_eff_j, H_sat_eff_k, W_sat,
                P_sat=P_sat_init,
                P_bs_scale=self.scenario.P_bs_scale,
                sinr_threshold_linear=self.sinr_threshold_linear,
                noise_power=self.scenario.P_noise,
                max_power=self.scenario.P_bs_max
            )

            if success:
                P_BS = self.scenario.P_bs_scale * P_BS_norm
                sinr_min_db = np.min(bf_info['sinr_values_db'])
                print(f"[Init] ZF-waterfilling successful: P_BS={P_BS:.4f}W, P_sat={P_sat_init:.4f}W, SINR_min={sinr_min_db:.2f}dB")
            else:
                print(f"[Init] ZF-waterfilling warning: {bf_info.get('note', 'Unknown issue')}")

        except Exception as e:
            print(f"[Init] ZF-waterfilling failed: {e}")

        # Initialize prev_* variables for state construction (regardless of success)
        self.prev_Phi = Phi_init
        if 'W_init' in locals() and W_init is not None:
            self.prev_W_bs = W_init
            self.prev_sinr_values = bf_info.get('sinr_values', np.ones(self.scenario.K) * self.sinr_threshold_linear)
        else:
            self.prev_W_bs = np.zeros((self.scenario.N_t, self.scenario.K), dtype=complex)
            self.prev_sinr_values = np.ones(self.scenario.K) * self.sinr_threshold_linear


    def calculate_all_sinrs(self, Phi, W, P_sat, channels):
        """
        统一计算当前时刻所有用户(K+1)的瞬时真实SINR，消除滞后性。
        """
        K = self.scenario.K
        noise_pow = self.scenario.P_noise
        W_sat = channels['W_sat']  # 确保使用的是最新的 w_sat
        
        # 1. 提取/构建等效信道
        # BS -> UE 有效信道
        H_eff_k = channels['H_BS2UE'] + channels['H_RIS2UE'] @ Phi @ channels['G_BS']
        # SAT -> UE 干扰有效信道
        H_sat_eff_k = channels['H_SAT2UE'] + channels['H_RIS2UE'] @ Phi @ channels['G_SAT']
        # BS -> SU 干扰有效信道
        H_eff_su = channels['H_BS2SUE'] + channels['H_RIS2SUE'] @ Phi @ channels['G_BS']
        # SAT -> SU 预期信号有效信道
        H_sat_eff_su = channels['H_SAT2SUE'] + channels['H_RIS2SUE'] @ Phi @ channels['G_SAT']

        # 2. 计算 K 个地面用户 (UE) 的 SINR
        sinr_ue = np.zeros(K)
        for k in range(K):
            # 预期信号功率 (使用 P_bs_scale 还原真实功率)
            h_eff_k = H_eff_k[k, :]
            w_k = W[:, k]
            signal_power = self.scenario.P_bs_scale * (np.abs(np.vdot(h_eff_k, w_k))**2)
            
            # 内部干扰 (Intra-system interference from BS)
            intra_interf = 0
            for j in range(K):
                if j != k:
                    intra_interf += self.scenario.P_bs_scale * (np.abs(np.vdot(h_eff_k, W[:, j]))**2)
            
            # 卫星干扰 (Inter-system interference from LEO)
            h_sat_eff_k = H_sat_eff_k[k, :]
            sat_interf = P_sat * (np.abs(np.vdot(h_sat_eff_k, W_sat[:, 0]))**2)
            
            sinr_ue[k] = signal_power / (intra_interf + sat_interf + noise_pow)

        # 3. 计算 1 个卫星用户 (SU) 的 SINR
        # 预期卫星信号功率
        h_sat_eff_su = H_sat_eff_su[0, :]
        sat_signal = P_sat * (np.abs(np.vdot(h_sat_eff_su, W_sat[:, 0]))**2)
        
        # 基站产生的聚合干扰
        bs_to_sat_interf = 0
        for k in range(K):
            h_eff_su_k = H_eff_su[0, :]
            bs_to_sat_interf += self.scenario.P_bs_scale * (np.abs(np.vdot(h_eff_su_k, W[:, k]))**2)
            
        sinr_su = sat_signal / (bs_to_sat_interf + noise_pow)

        return np.append(sinr_ue, sinr_su)  # 返回长度为 K+1 的向量
    def _get_state(self):
        """
        Construct compressed ephemeris-aware state vector with previous step information

        State design (total: 7 + 4K dimensions, K=4 -> 23 dims):
        1. Satellite motion state (6 features):
           - sin(elevation), cos(elevation), sin(azimuth), cos(azimuth)
           - delta_elevation, delta_azimuth (based on noisy observations)
        2. Compressed channel features (2K features, based on prev step):
           - g_k,t-1 = |h_k^H + h_r,k^H Phi_t-1 G_BS|^2 |w_k|^2 (BS user signal strength)
           - g_k,t-1^s = |h_sat,k^H + h_r,k^H Phi_t-1 G_SAT|^2 |w_sat|^2 (SAT interference)
        3. Interference features (K+1 features, based on prev step):
           - I_k,t-1 = SAT interference + other BS user interference to user k
           - I_s,t-1 = BS interference to SAT user
        4. Performance feedback (K features):
           - SINR margin m_k = (SINR_k,t-1 / threshold) - 1
        """
        K = self.scenario.K

        # ========== 1. Satellite Motion State (6 features) ==========
        obs_ele = self.curr_obs_elevation
        obs_azi = self.curr_obs_azimuth
        # Trigonometric encoding (more stable than raw angles)
        ele_rad = np.deg2rad(obs_ele)
        azi_rad = np.deg2rad(obs_azi)

        # Angular velocity based on noisy observations (current obs - previous obs)
        delta_ele = obs_ele - self.prev_obs_elevation
        delta_azi = obs_azi - self.prev_obs_azimuth

        satellite_motion = np.array([
            np.sin(ele_rad), np.cos(ele_rad),
            np.sin(azi_rad), np.cos(azi_rad),
            delta_ele / 10.0,  # Normalize angular velocity
            delta_azi / 10.0
        ])

        # ========== 2. Compressed Channel Features (2K features) ==========
        # Use previous step configuration (Phi_t-1, W_t-1)
        if self.prev_Phi is None or self.prev_W_bs is None:
            # First step: use default values (zeros)
            g_k_bs = np.zeros(K)
            g_k_sat = np.zeros(K)
        else:
            # Extract channels
            H_BS2UE = self.current_channels['H_BS2UE']      # (K, N_t)
            H_RIS2UE = self.current_channels['H_RIS2UE']    # (K, N_ris)
            G_BS = self.current_channels['G_BS']            # (N_ris, N_t)
            H_SAT2UE = self.current_channels['H_SAT2UE']    # (K, N_sat)
            G_SAT = self.current_channels['G_SAT']          # (N_ris, N_sat)
            W_sat = self.current_channels['W_sat']          # (N_sat, 1)

            # Effective channels with previous RIS phase
            H_eff_k = H_BS2UE + H_RIS2UE @ self.prev_Phi @ G_BS  # (K, N_t)
            H_sat_eff_k = H_SAT2UE + H_RIS2UE @ self.prev_Phi @ G_SAT  # (K, N_sat)

            # g_k,t-1 = |h_k^H + h_r,k^H Phi_t-1 G_BS|^2 |w_k|^2
            # This represents the received signal strength for user k from BS
            g_k_bs = np.zeros(K)
            for k in range(K):
                h_eff_k = H_eff_k[k, :]  # (N_t,)
                w_k = self.prev_W_bs[:, k]  # (N_t,)
                g_k_bs[k] = np.abs(np.vdot(h_eff_k, w_k)) ** 2

            # g_k,t-1^s = |h_sat,k^H + h_r,k^H Phi_t-1 G_SAT|^2 |w_sat|^2
            # This represents the interference from SAT to BS user k
            g_k_sat = np.zeros(K)
            for k in range(K):
                h_sat_eff_k = H_sat_eff_k[k, :]  # (N_sat,)
                w_sat = W_sat[:, 0]  # (N_sat,)
                g_k_sat[k] = np.abs(np.vdot(h_sat_eff_k, w_sat)) ** 2

        # Normalize to log scale
        g_k_bs_norm = np.log10(g_k_bs) / 10.0
        g_k_sat_norm = np.log10(g_k_sat) / 10.0

        channel_features = np.concatenate([g_k_bs_norm, g_k_sat_norm])  # (2K,)

        # ========== 3. Interference Features (K+1 features) ==========
        if self.prev_Phi is None or self.prev_W_bs is None:
            # First step: use default values
            I_k = np.zeros(K)
            I_s = 0.0
        else:
            # I_k,t-1 = SAT interference + other BS user interference
            I_k = np.zeros(K)
            for k in range(K):
                # SAT interference to user k
                sat_interference = g_k_sat[k] * self.scenario.P_sat

                # Other BS user interference
                bs_interference = 0.0
                for j in range(K):
                    if j != k:
                        h_eff_k = H_eff_k[k, :]  # (N_t,)
                        w_j = self.prev_W_bs[:, j]  # (N_t,)
                        bs_interference += np.abs(np.vdot(h_eff_k, w_j)) ** 2 * self.scenario.P_bs_scale

                I_k[k] = sat_interference + bs_interference

            # I_s,t-1 = BS interference to SAT user
            H_BS2SUE = self.current_channels['H_BS2SUE']    # (SK, N_t)
            H_RIS2SUE = self.current_channels['H_RIS2SUE']  # (SK, N_ris)
            H_eff_j = H_BS2SUE + H_RIS2SUE @ self.prev_Phi @ G_BS  # (SK, N_t)

            I_s = 0.0
            for k in range(K):
                h_eff_j = H_eff_j[0, :]  # (N_t,) - assume single SAT user
                w_k = self.prev_W_bs[:, k]  # (N_t,)
                I_s += np.abs(np.vdot(h_eff_j, w_k)) ** 2 * self.scenario.P_bs_scale

        # Normalize to log scale
        I_k_norm = np.log10(I_k + 1e-20) / 10.0
        I_s_norm = np.log10(I_s + 1e-20) / 10.0

        interference_features = np.concatenate([I_k_norm, [I_s_norm]])  # (K+1,)

        # ========== 4. Performance Feedback (K features) ==========
        if self.prev_sinr_values is None:
            # First step: assume SINR at threshold
            sinr_margin = np.zeros(K)
        else:
            # m_k = (SINR_k,t-1 / threshold) - 1
            sinr_margin = (self.prev_sinr_values / self.sinr_threshold_linear) - 1.0

        performance_feedback = sinr_margin  # (K,)

        # ========== Concatenate All Features ==========
        state = np.concatenate([
            satellite_motion,       # (6,)
            channel_features,       # (2K,)
            interference_features,  # (K+1,)
            performance_feedback    # (K,)
        ]).astype(np.float32)

        return state


    def render(self):
        """Optional: Visualize current scenario"""
        self.scenario.plot_scenario()
