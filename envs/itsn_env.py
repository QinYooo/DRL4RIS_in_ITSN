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
                 n_substeps=1,  # Number of physics substeps per RL step
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
        self.n_substeps = n_substeps  # Physics substeps per RL decision
        self.total_physics_steps = max_steps_per_episode * n_substeps
        self.current_step = 0  # RL step counter
        self.current_physics_step = 0  # Physics step counter
        self.orbit_height = orbit_height

        # RIS quantization
        self.phase_bits = phase_bits
        self.num_phase_levels = 2 ** phase_bits
        self.phase_codebook = np.linspace(0, 2*np.pi, self.num_phase_levels, endpoint=False)

        # Reward parameters
        self.sinr_threshold_db = sinr_threshold_db
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

        # Dual G_SAT channels for ephemeris uncertainty
        # Only SAT→RIS channel is affected by ephemeris errors
        self.true_G_SAT = None      # True G_SAT based on true satellite position
        self.inferred_G_SAT = None  # Inferred G_SAT based on noisy observations

        # Inference scenario for generating inferred G_SAT (separate from main scenario)
        self.inference_scenario = None

        # Enable/disable ephemeris noise (for ablation studies)
        self.enable_ephemeris_noise = True

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

        # State space dimension calculation
        # 1. Satellite motion: 6 features
        #    - sin(elevation), cos(elevation), sin(azimuth), cos(azimuth): 4
        #    - delta_elevation, delta_azimuth (from noisy ephemeris): 2
        # 2. Channel features (K+1 features):
        #    - Expected signal strength for K UE + 1 SUE
        # 3. Interference features (K+1 features):
        #    - Total interference for K UE + 1 SUE
        # 4. Performance feedback (K+1 features):
        #    - SINR margin for K UE + 1 SUE
        # Total: 6 + 3(K+1) = 6 + 3K + 3, for K=4: state_dim = 21
        state_dim = 6 + 3 * (self.scenario.K + 1)

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

        # Update RNG for trajectory diversity
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.scenario = ITSNScenario(rng_seed=seed)
        else:
            # Generate new random seed for diversity when seed not specified
            new_seed = np.random.randint(0, 1000000)
            self.rng = np.random.RandomState(new_seed)
            self.scenario = ITSNScenario(rng_seed=new_seed)

        # Reset step counters
        self.current_step = 0  # RL step
        self.current_physics_step = 0  # Physics step

        # Load trajectory from options or generate default
        # Trajectory length = total_physics_steps (not max_steps)
        if options is not None and 'trajectory' in options:
            # Use external trajectory
            traj = options['trajectory']
            self.trajectory_elevation = np.array(traj['elevation'])
            self.trajectory_azimuth = np.array(traj['azimuth'])

            # Validate trajectory length
            if len(self.trajectory_elevation) != self.total_physics_steps or len(self.trajectory_azimuth) != self.total_physics_steps:
                raise ValueError(f"Trajectory length mismatch: expected {self.total_physics_steps}, "
                               f"got elevation={len(self.trajectory_elevation)}, azimuth={len(self.trajectory_azimuth)}")
        else:
            # Generate default trajectory (satellite pass) with diversity
            # 1. Randomize peak elevation (60°~88°)
            max_ele = self.rng.uniform(60, 88)

            # 2. Randomize start/end elevation (15°~40°, can differ)
            start_ele = self.rng.uniform(15, 40)
            end_ele = self.rng.uniform(15, 40)

            # 3. Asymmetric trajectory: peak at 30%~70% of pass
            peak_ratio = self.rng.uniform(0.3, 0.7)
            peak_step = max(1, int(self.total_physics_steps * peak_ratio))

            ele_up = np.linspace(start_ele, max_ele, peak_step)
            ele_down = np.linspace(max_ele, end_ele, self.total_physics_steps - peak_step)
            self.trajectory_elevation = np.concatenate([ele_up, ele_down])

            # Azimuth: wider range (45°~135° start, 45°~315° end)
            self.trajectory_azimuth = np.linspace(
                self.rng.uniform(45, 135),
                self.rng.uniform(45, 315),
                self.total_physics_steps
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

        # Initialize true G_SAT
        self.true_G_SAT = self.current_channels['G_SAT']

        # The 0 step obsevation:
        # Add ephemeris noise to observed angles
        obs_ele_0 = self.true_elevation + self.rng.normal(0, self.ephemeris_noise_std)
        obs_azi_0 = self.true_azimuth + self.rng.normal(0, self.ephemeris_noise_std)
        self.prev_obs_elevation = obs_ele_0
        self.prev_obs_azimuth = obs_azi_0
        # Current Observation
        self.curr_obs_elevation = obs_ele_0
        self.curr_obs_azimuth = obs_azi_0

        # Generate inferred G_SAT based on noisy observations
        self.inferred_G_SAT = self._generate_inferred_G_SAT()

        # Initialize with ZF-waterfilling to verify channel quality and get initial prev_* values
        self._initialize_with_zf_waterfilling()

        # Get initial state (with ephemeris noise)
        state = self._get_state()

        return state, {}

    def step(self, action):
        """
        Execute one RL step with n_substeps physics steps:
        1. Update observation for current RL step
        2. Make decision (Phi, W, P_sat) based on current observation
        3. Evaluate performance over n_substeps with fixed decision
        4. Satellite moves during substeps
        5. Return averaged reward and next state
        """
        # --- 1. 动作执行 (固定整个RL step) ---
        action_clipped = np.clip(action, -1.0, 1.0)
        phase_indices = ((action_clipped + 1.0) / 2.0 * (self.num_phase_levels - 1)).astype(int)
        ris_phases = self.phase_codebook[np.clip(phase_indices, 0, self.num_phase_levels - 1)]
        Phi = np.diag(self.scenario.ris_amplitude_gain * np.exp(1j * ris_phases))

        # --- 1.5. 更新当前observation (在决策前) ---
        if self.current_step > 0:  # 第一步在reset中已更新
            self._update_observation()

        # --- 2. 决策阶段：基于更新后的 observation 计算 W 和 P_sat (只做一次) ---
        self.inferred_G_SAT = self._generate_inferred_G_SAT()

        try:
            H_eff_k, H_eff_j, H_sat_eff_k = self._get_all_eff_channels(
                Phi, self.current_channels, G_SAT=self.inferred_G_SAT
            )

            W_init, _, _, _ = compute_zf_waterfilling_baseline(
                H_eff_k, H_eff_j, H_sat_eff_k, self.current_channels['W_sat'],
                P_sat=self.scenario.P_sat, P_bs_scale=self.scenario.P_bs_scale,
                sinr_threshold_linear=self.sinr_threshold_linear,
                noise_power=self.scenario.P_noise, max_power=self.scenario.P_bs_max
            )

            P_sat_fixed, _ = self.scenario.compute_sat_power(
                Phi, self.current_channels, W_bs=W_init, sinr_threshold_db=self.sinr_threshold_db
            )

            W_fixed, P_BS_norm, decision_success, _ = compute_zf_waterfilling_baseline(
                H_eff_k, H_eff_j, H_sat_eff_k, self.current_channels['W_sat'],
                P_sat=P_sat_fixed, P_bs_scale=self.scenario.P_bs_scale,
                sinr_threshold_linear=self.sinr_threshold_linear,
                noise_power=self.scenario.P_noise, max_power=self.scenario.P_bs_max
            )
            P_BS_fixed = self.scenario.P_bs_scale * P_BS_norm

        except Exception:
            decision_success = False
            P_BS_fixed, P_sat_fixed = 10.0, 10.0
            W_fixed = np.zeros((self.scenario.N_t, self.scenario.K), dtype=complex)

        # --- 3. 评估阶段：用固定决策测量 n_substeps 的平均性能 ---
        total_sinr_values = np.zeros(self.scenario.K + 1)
        success_count = 0

        # 实际执行的substeps数量（可能少于n_substeps如果到达轨迹末尾）
        actual_substeps = 0

        for substep in range(self.n_substeps):
            # 检查是否还有剩余physics steps
            if self.current_physics_step >= self.total_physics_steps:
                break

            # 测量真实性能 (用固定的 Phi, W_fixed, P_sat_fixed)
            all_sinr_values = self.calculate_all_sinrs(Phi, W_fixed, P_sat_fixed, self.current_channels)
            total_sinr_values += all_sinr_values
            actual_substeps += 1

            # 检查是否满足约束
            if np.all(all_sinr_values >= self.sinr_threshold_linear):
                success_count += 1

            # 物理演进：卫星移动到下一个physics step
            self.current_physics_step += 1
            if self.current_physics_step < self.total_physics_steps:
                self._advance_physics()

        # --- 4. 计算平均性能 ---
        avg_sinr_values = total_sinr_values / max(actual_substeps, 1)  # 避免除零
        avg_success = (actual_substeps > 0) and (success_count == actual_substeps)

        # --- 5. 计算奖励 (基于平均性能) ---
        # 使用加权和速率作为奖励
        reward = self._calculate_reward_weighted_rate(P_BS_fixed, P_sat_fixed, avg_sinr_values, avg_success)

        # --- 6. 更新快照 ---
        self.prev_Phi = Phi
        self.prev_W_bs = W_fixed if decision_success else np.zeros_like(W_fixed)
        self.prev_sinr_values = avg_sinr_values
        self.prev_obs_elevation = self.curr_obs_elevation
        self.prev_obs_azimuth = self.curr_obs_azimuth

        # --- 7. RL step 计数器更新 ---
        self.current_step += 1
        # 终止条件：达到最大RL步数 或 物理步数耗尽
        terminated = (self.current_step >= self.max_steps) or (self.current_physics_step >= self.total_physics_steps)

        # --- 8. 获取 s_{t+1} ---
        state = self._get_state()

        # --- 9. 构建 info dict ---
        info = {
            "success": avg_success,
            "success_rate": success_count / max(actual_substeps, 1),
            "P_BS": P_BS_fixed,
            "P_sat": P_sat_fixed,
            "actual_substeps": actual_substeps,
            # Ephemeris uncertainty tracking
            "true_elevation": self.true_elevation,
            "true_azimuth": self.true_azimuth,
            "obs_elevation": self.curr_obs_elevation,
            "obs_azimuth": self.curr_obs_azimuth,
            "ephemeris_error_ele": self.curr_obs_elevation - self.true_elevation,
            "ephemeris_error_azi": self.curr_obs_azimuth - self.true_azimuth,
            # G_SAT mismatch tracking
            "G_SAT_mismatch": np.linalg.norm(self.true_G_SAT - self.inferred_G_SAT),
            # SINR values (averaged)
            "sinr_UE": avg_sinr_values[:self.scenario.K],
            "sinr_SUE": avg_sinr_values[self.scenario.K]
        }

        return state, reward, terminated, False, info

    def _calculate_reward(self, P_BS, P_sat, all_sinr_values, success):
        """
        综合考虑系统总能耗和所有用户 (K+1) 的 SINR 满足情况 [cite: 83, 87, 168]
        使用线性尺度的 SINR gap 以提供更平滑的梯度
        """
        # 1. 能效奖励：最小化系统总功耗 [cite: 168]
        total_p = P_BS + P_sat
        reward_energy = -10 * np.log10(total_p + 1e-6)

        # 2. SINR 约束检查 (K 个 UE + 1 个 SU) - 线性尺度
        # 计算每个用户相对于阈值的违规量（线性尺度提供更平滑梯度）
        sinr_gaps = np.maximum(0, self.sinr_threshold_linear - all_sinr_values)
        normalized_gaps = sinr_gaps / (self.sinr_threshold_linear + 1e-12)
        penalty_sinr = -self.sinr_penalty_weight * np.sum(normalized_gaps)

        # 3. 综合奖励
        # 如果所有约束满足 (success=True 且 gaps=0)，给予额外奖励
        bonus_all_satisfied = 5.0 if success and np.all(sinr_gaps <= 1e-6) else 0.0

        return reward_energy + penalty_sinr + bonus_all_satisfied

    def _calculate_reward_weighted_rate(self, P_BS, P_sat, all_sinr_values, success):
        """
        基于加权和速率的奖励函数

        Reward = Σ(weight_i * rate_i) - penalty

        其中:
        - rate_i = log2(1 + SINR_i) (Shannon capacity, bps/Hz)
        - weight: 卫星用户=8, 基站用户=1
        - penalty: SINR低于(10-0.2)dB时施加惩罚

        Args:
            P_BS: BS功率 (W)
            P_sat: 卫星功率 (W)
            all_sinr_values: 所有用户SINR (线性尺度), shape=(K+1,)
            success: 是否所有用户满足SINR约束

        Returns:
            reward: 加权和速率 (bps/Hz)
        """
        # 1. 计算各用户速率 (Shannon capacity)
        # rate = log2(1 + SINR), 单位: bps/Hz
        rates = np.log2(1 + all_sinr_values)

        # 2. 设置权重: 前K个是BS用户(权重=1), 最后1个是卫星用户(权重=8)
        weights = np.ones(self.scenario.K + 1)
        weights[-1] = 8.0  # 卫星用户权重

        # 3. 加权和速率
        weighted_sum_rate = np.sum(weights * rates)

        # 4. SINR惩罚: 低于(10-0.2)=9.8dB时施加惩罚
        sinr_threshold_penalty_db = 9.8
        sinr_threshold_penalty_linear = 10 ** (sinr_threshold_penalty_db / 10.0)

        # 计算违规量 (线性尺度)
        sinr_violations = np.maximum(0, sinr_threshold_penalty_linear - all_sinr_values)

        # 归一化违规量并加权
        normalized_violations = sinr_violations / (sinr_threshold_penalty_linear + 1e-12)
        weighted_violations = weights * normalized_violations

        # 惩罚系数 (可调)
        penalty_weight = 10.0
        penalty = -penalty_weight * np.sum(weighted_violations)

        # 5. 总奖励
        reward = weighted_sum_rate + penalty

        return reward


    def _initialize_with_zf_waterfilling(self):
        """
        Initialize environment with ZF-waterfilling to verify channel quality.
        Uses the initial RIS phase (identity) as starting point.
        Uses INFERRED G_SAT for BS beamforming, evaluates on TRUE channels.

        Note: Satellite beamforming and power are initialized in generate_channels()
        """
        # Use initial RIS phase (identity matrix)
        Phi_init = self.initial_ris_phase

        # Extract channels (satellite W_sat and P_sat already initialized)
        H_BS2UE = self.current_channels['H_BS2UE']
        H_BS2SUE = self.current_channels['H_BS2SUE']
        H_RIS2UE = self.current_channels['H_RIS2UE']
        H_RIS2SUE = self.current_channels['H_RIS2SUE']
        G_BS = self.current_channels['G_BS']
        H_SAT2UE = self.current_channels['H_SAT2UE']
        W_sat = self.current_channels['W_sat']

        # Compute effective channels with initial RIS phase using INFERRED G_SAT
        H_eff_k = H_BS2UE + H_RIS2UE @ Phi_init @ G_BS
        H_eff_j = H_BS2SUE + H_RIS2SUE @ Phi_init @ G_BS
        H_sat_eff_k = H_SAT2UE + H_RIS2UE @ Phi_init @ self.inferred_G_SAT

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

            # Evaluate on TRUE channels
            all_sinr_true = self.calculate_all_sinrs(
                Phi_init, W_init, P_sat_init, self.current_channels
            )

            if success:
                P_BS = self.scenario.P_bs_scale * P_BS_norm
                sinr_min_db = 10 * np.log10(np.min(all_sinr_true) + 1e-12)
                # print(f"[Init] ZF on inferred G_SAT: P_BS={P_BS:.4f}W, P_sat={P_sat_init:.4f}W, "
                #       f"True SINR_min={sinr_min_db:.2f}dB")
            else:
                pass
                # print(f"[Init] ZF-waterfilling warning: {bf_info.get('note', 'Unknown issue')}")

        except Exception as e:
            pass
            # print(f"[Init] ZF-waterfilling failed: {e}")

        # Initialize prev_* variables for state construction (regardless of success)
        self.prev_Phi = Phi_init
        if 'W_init' in locals() and W_init is not None:
            self.prev_W_bs = W_init
            self.prev_sinr_values = all_sinr_true  # All K+1 users (UE + SUE)
        else:
            self.prev_W_bs = np.zeros((self.scenario.N_t, self.scenario.K), dtype=complex)
            self.prev_sinr_values = np.ones(self.scenario.K + 1) * self.sinr_threshold_linear


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

    def _get_all_eff_channels(self, Phi, channels, G_SAT=None):
        """
        Compute effective channels with RIS phase shifts.

        Args:
            Phi: (N_ris, N_ris) diagonal matrix of RIS phase shifts
            channels: Dict containing all channel matrices
            G_SAT: Optional override for G_SAT (for using inferred version)

        Returns:
            Tuple of (H_eff_k, H_eff_j, H_sat_eff_k)
        """
        # Extract channels
        H_BS2UE = channels['H_BS2UE']      # (K, N_t)
        H_BS2SUE = channels['H_BS2SUE']    # (SK, N_t)
        G_BS = channels['G_BS']            # (N_ris, N_t)
        H_RIS2UE = channels['H_RIS2UE']    # (K, N_ris)
        H_RIS2SUE = channels['H_RIS2SUE']  # (SK, N_ris)
        H_SAT2UE = channels['H_SAT2UE']    # (K, N_sat)

        # Use provided G_SAT or default from channels
        if G_SAT is None:
            G_SAT = channels['G_SAT']      # (N_ris, N_sat)

        # Compute effective channels (direct + RIS-assisted)
        H_eff_k = H_BS2UE + H_RIS2UE @ Phi @ G_BS       # BS → UE
        H_eff_j = H_BS2SUE + H_RIS2SUE @ Phi @ G_BS     # BS → SUE
        H_sat_eff_k = H_SAT2UE + H_RIS2UE @ Phi @ G_SAT # SAT → UE (uses G_SAT)

        return H_eff_k, H_eff_j, H_sat_eff_k

    def _advance_physics(self):
        """
        Advance satellite position by one timestep using pre-generated trajectory.
        Updates true_elevation, true_azimuth, and regenerates true channels.
        """
        # Update true position from trajectory
        self.true_elevation = self.trajectory_elevation[self.current_physics_step]
        self.true_azimuth = self.trajectory_azimuth[self.current_physics_step]

        # Update scenario with TRUE position
        self.scenario.update_satellite_position(
            self.true_elevation,
            self.true_azimuth,
            self.orbit_height
        )

        # Generate TRUE channels
        self.current_channels = self.scenario.generate_channels()

        # Store true G_SAT
        self.true_G_SAT = self.current_channels['G_SAT']

    def _update_observation(self):
        """
        Update observed angles by adding Gaussian noise to true angles.
        This simulates ephemeris prediction errors.
        """
        if self.enable_ephemeris_noise:
            # Add Gaussian noise
            noise_ele = self.rng.normal(0, self.ephemeris_noise_std)
            noise_azi = self.rng.normal(0, self.ephemeris_noise_std)

            self.curr_obs_elevation = self.true_elevation + noise_ele
            self.curr_obs_azimuth = self.true_azimuth + noise_azi

            # Clip to valid ranges
            self.curr_obs_elevation = np.clip(self.curr_obs_elevation, 0, 90)
            self.curr_obs_azimuth = self.curr_obs_azimuth % 360
        else:
            # Perfect observation (for ablation studies)
            self.curr_obs_elevation = self.true_elevation
            self.curr_obs_azimuth = self.true_azimuth

    def _generate_inferred_G_SAT(self):
        """
        Generate G_SAT (SAT→RIS channel) based on observed (noisy) satellite position.
        Uses a separate inference scenario to avoid modifying the main scenario.

        Returns:
            G_SAT matrix based on observed position
        """
        # Create inference scenario on first call
        if self.inference_scenario is None:
            self.inference_scenario = ITSNScenario(
                rng_seed=self.rng.randint(0, 10000)
            )
            # Copy user positions from main scenario
            self.inference_scenario.USERS_POS = self.scenario.USERS_POS.copy()
            self.inference_scenario.SATUSERS_POS = self.scenario.SATUSERS_POS.copy()

        # Update inference scenario with OBSERVED position
        self.inference_scenario.update_satellite_position(
            self.curr_obs_elevation,
            self.curr_obs_azimuth,
            self.orbit_height
        )

        # Generate channels at observed position
        inferred_channels = self.inference_scenario.generate_channels()
        G_SAT_inferred = inferred_channels['G_SAT']

        return G_SAT_inferred

    def _get_state(self):
        """
        Construct compressed ephemeris-aware state vector with previous step information

        State design (total: 6 + 3(K+1) dimensions, K=4 -> 21 dims):
        1. Satellite motion state (6 features):
           - sin(elevation), cos(elevation), sin(azimuth), cos(azimuth)
           - delta_elevation, delta_azimuth (based on noisy observations)
        2. Channel features (K+1 features, based on prev step):
           - g_k,t-1: Expected signal strength for K UE (BS → UE)
           - g_s,t-1: Expected signal strength for 1 SUE (SAT → SUE)
        3. Interference features (K+1 features, based on prev step):
           - I_k,t-1: Total interference to K UE (SAT + other BS users)
           - I_s,t-1: Total interference to 1 SUE (BS)
        4. Performance feedback (K+1 features):
           - SINR margin m_k = (SINR_k,t-1 / threshold) - 1 for K UE
           - SINR margin m_s = (SINR_s,t-1 / threshold) - 1 for 1 SUE
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

        # ========== 2. Channel Features (K+1 features) ==========
        # Expected signal strength for each user
        if self.prev_Phi is None or self.prev_W_bs is None:
            # First step: use default values (zeros)
            signal_strength = np.zeros(K + 1)
        else:
            # Extract channels
            H_BS2UE = self.current_channels['H_BS2UE']      # (K, N_t)
            H_RIS2UE = self.current_channels['H_RIS2UE']    # (K, N_ris)
            G_BS = self.current_channels['G_BS']            # (N_ris, N_t)
            H_SAT2SUE = self.current_channels['H_SAT2SUE']  # (SK, N_sat)
            H_RIS2SUE = self.current_channels['H_RIS2SUE']  # (SK, N_ris)
            # Use INFERRED G_SAT for state construction (agent's belief)
            G_SAT = self.inferred_G_SAT                     # (N_ris, N_sat)
            W_sat = self.current_channels['W_sat']          # (N_sat, 1)

            # Effective channels with previous RIS phase
            H_eff_k = H_BS2UE + H_RIS2UE @ self.prev_Phi @ G_BS  # BS → UE (K, N_t)
            H_sat_eff_s = H_SAT2SUE + H_RIS2SUE @ self.prev_Phi @ G_SAT  # SAT → SUE (SK, N_sat)

            # Signal strength for K UE: |h_eff_k^H @ w_k|^2
            signal_strength_ue = np.zeros(K)
            for k in range(K):
                h_eff_k = H_eff_k[k, :]  # (N_t,)
                w_k = self.prev_W_bs[:, k]  # (N_t,)
                signal_strength_ue[k] = np.abs(np.vdot(h_eff_k, w_k)) ** 2

            # Signal strength for 1 SUE: |h_sat_eff_s^H @ w_sat|^2
            h_sat_eff_s = H_sat_eff_s[0, :]  # (N_sat,)
            w_sat = W_sat[:, 0]  # (N_sat,)
            signal_strength_sue = np.abs(np.vdot(h_sat_eff_s, w_sat)) ** 2

            signal_strength = np.append(signal_strength_ue, signal_strength_sue)

        # Normalize to log scale
        signal_strength_norm = np.log10(signal_strength + 1e-20) / 10.0
        channel_features = signal_strength_norm  # (K+1,)

        # ========== 3. Interference Features (K+1 features) ==========
        if self.prev_Phi is None or self.prev_W_bs is None:
            # First step: use default values
            interference = np.zeros(K + 1)
        else:
            # Interference to K UE: SAT interference + other BS user interference
            H_SAT2UE = self.current_channels['H_SAT2UE']    # (K, N_sat)
            H_sat_eff_k = H_SAT2UE + H_RIS2UE @ self.prev_Phi @ G_SAT  # (K, N_sat)

            interference_ue = np.zeros(K)
            for k in range(K):
                # SAT interference to UE k
                h_sat_eff_k = H_sat_eff_k[k, :]  # (N_sat,)
                sat_interference = self.scenario.P_sat * (np.abs(np.vdot(h_sat_eff_k, w_sat)) ** 2)

                # Other BS user interference
                bs_interference = 0.0
                for j in range(K):
                    if j != k:
                        h_eff_k = H_eff_k[k, :]  # (N_t,)
                        w_j = self.prev_W_bs[:, j]  # (N_t,)
                        bs_interference += self.scenario.P_bs_scale * (np.abs(np.vdot(h_eff_k, w_j)) ** 2)

                interference_ue[k] = sat_interference + bs_interference

            # Interference to 1 SUE: BS interference
            H_BS2SUE = self.current_channels['H_BS2SUE']    # (SK, N_t)
            H_eff_j = H_BS2SUE + H_RIS2SUE @ self.prev_Phi @ G_BS  # (SK, N_t)

            interference_sue = 0.0
            for k in range(K):
                h_eff_j = H_eff_j[0, :]  # (N_t,) - assume single SAT user
                w_k = self.prev_W_bs[:, k]  # (N_t,)
                interference_sue += self.scenario.P_bs_scale * (np.abs(np.vdot(h_eff_j, w_k)) ** 2)

            interference = np.append(interference_ue, interference_sue)

        # Normalize to log scale
        interference_norm = np.log10(interference + 1e-20) / 10.0
        interference_features = interference_norm  # (K+1,)

        # ========== 4. Performance Feedback (K+1 features) ==========
        # SINR margin for all K+1 users
        if self.prev_sinr_values is None:
            # First step: assume SINR at threshold
            sinr_margin = np.zeros(K + 1)
        else:
            # m = (SINR_t-1 / threshold) - 1 for all K+1 users
            sinr_margin = (self.prev_sinr_values / self.sinr_threshold_linear) - 1.0

        performance_feedback = sinr_margin  # (K+1,)

        # ========== Concatenate All Features ==========
        state = np.concatenate([
            satellite_motion,       # (6,)
            channel_features,       # (K+1,)
            interference_features,  # (K+1,)
            performance_feedback    # (K+1,)
        ]).astype(np.float32)

        return state


    def render(self):
        """Optional: Visualize current scenario"""
        self.scenario.plot_scenario()
