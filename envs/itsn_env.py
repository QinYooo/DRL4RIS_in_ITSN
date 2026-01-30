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
                 sinr_penalty_weight=10,
                 ephemeris_noise_std=0.5,  # Degrees of angle noise
                 rate_requirement_bps=1e6,  # Rate constraint per user
                 orbit_height=500e3,
                 orbit_duration=600.0,  # 轨道总时间(秒)，默认10分钟
                 coherence_time=0.1):  # 相干时间(秒)，经验值

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

        # 时间参数 (用于NLOS高斯-马尔可夫更新)
        self.orbit_duration = orbit_duration
        self.coherence_time = coherence_time
        self.step_duration = orbit_duration / max_steps_per_episode  # 每RL step时长
        # alpha = J_0(2*pi*f_D*delta_t), 简化为 exp(-delta_t / T_c)
        self.nlos_alpha = np.exp(-self.step_duration / coherence_time)

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

        # State space dimension calculation (Full Geometry State)
        # 1. Sat Incident (6 dims): 卫星相对于 RIS 的几何 (SAT -> RIS)
        #    - sin_ele, cos_ele, sin_azi, cos_azi, d_ele, d_azi
        # 2. BS Incident (3 dims): 基站相对于 RIS 的几何 (BS -> RIS)
        #    - sin_ele, sin_azi, cos_azi
        # 3. User Outgoing (3 * (K+1) dims): 用户相对于 RIS 的几何 (RIS -> Users)
        #    - For each of K UE + 1 SUE: sin_ele, sin_azi, cos_azi
        # 4. Feedback (K+1 dims): SINR Margin 反馈
        #    - For each of K UE + 1 SUE: (SINR / threshold) - 1
        # Total: 6 + 3 + 3*(K+1) + (K+1) = 9 + 4*(K+1)
        # For K=4: state_dim = 9 + 4*5 = 29
        state_dim = 172

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # ===== Adaptive constraint tuning (target success rate) =====
        self.target_success_rate = 0.90     # 你要求的 0.9
        self.lambda_lr = 0.05               # 拉格朗日乘子更新步长(建议 0.01~0.1)
        self.lambda_min = 0.0
        self.lambda_max = 200.0             # 防止过大导致训练崩
        self.lambda_sinr = float(self.sinr_penalty_weight)  # 初值沿用原本的 sinr_penalty_weight
        self.delta_phase_max = 0.3*np.pi  # 每步最大相移变化 (0.1π rad)

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
        # 暂时禁用NLOS缓存以验证
        # self.scenario._generate_nlos_cache()

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

        
        self._initialize_with_zf_waterfilling()

        # Initialize with P_sat
        self._initialize_with_P_sat()
        
        # Get initial state (with ephemeris noise)
        self.SINR_diff = np.zeros(self.scenario.SK+self.scenario.K,)
        state = self._get_state()
        self.prev_Phi = self.initial_ris_phase
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
        # 连续相移：将动作 [-1, 1] 映射到 [-π, π]，使得 action=0 对应 phase=0
        base_phase = np.angle(np.diag(self.prev_Phi)) if self.prev_Phi is not None else np.zeros(self.scenario.N_ris)
        # 增量限制：Δϕ ∈ [-delta_max, delta_max]
        delta_max = self.delta_phase_max  # 比如 0.1π 或更小
        delta_phase = np.clip(action, -1.0, 1.0) * delta_max

        new_phase = (base_phase + delta_phase + np.pi) % (2*np.pi) - np.pi
        Phi = np.diag(self.scenario.ris_amplitude_gain * np.exp(1j * new_phase))
        # action_clipped = np.clip(action, -1.0, 1.0)
        # ris_phases = action_clipped * np.pi  # [-1,1] -> [-π, π]
        # Phi = np.diag(self.scenario.ris_amplitude_gain * np.exp(1j * ris_phases))

        # --- 1.5. 更新当前observation (在决策前) ---
        if self.current_step > 0:  # 第一步在reset中已更新
            self._update_observation()

        # --- 2. 决策阶段：基于更新后的 observation 计算 W 和 P_sat (只做一次) ---
        self.inferred_G_SAT = self._generate_inferred_G_SAT()

        try:
            H_eff_k, H_eff_j, H_sat_eff_k = self._get_all_eff_channels(
                Phi, self.current_channels,self.inferred_G_SAT
            )
            P_sat_fixed = self.P_sat_init

            self.SINR_diff = 10*np.log10(self.calculate_all_sinrs(Phi, self.prev_W_bs, self.prev_P_sat, self.current_channels))-\
                10*np.log10(self.calculate_all_sinrs(self.prev_Phi, self.prev_W_bs, self.prev_P_sat, self.current_channels))
            for _ in range(4):
                W_fixed, P_BS_norm, decision_success, _ = compute_zf_waterfilling_baseline(
                    H_eff_k, H_eff_j, H_sat_eff_k, self.current_channels['W_sat'],
                    P_sat=P_sat_fixed, P_bs_scale=self.scenario.P_bs_scale,
                    sinr_threshold_linear=self.sinr_threshold_linear,
                    noise_power=self.scenario.P_noise, max_power=self.scenario.P_bs_max
                )

                P_sat_fixed, _ = self.scenario.compute_sat_power(
                    Phi, self.current_channels, W_bs=W_fixed, sinr_threshold_db=self.sinr_threshold_db
                )

            P_BS_fixed = self.scenario.P_bs_scale * P_BS_norm

        except Exception as e:
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

            # 检查是否满足约束 (留0.2dB余量)
            soft_threshold = 10 ** ((self.sinr_threshold_db - 0.5) / 10.0)
            if np.all(all_sinr_values >= soft_threshold):
                success_count += 1

            # 物理演进：卫星移动到下一个physics step
            self.current_physics_step += 1
            if self.current_physics_step < self.total_physics_steps:
                self._advance_physics()


        # --- 4. 计算平均性能 ---
        avg_sinr_values = total_sinr_values / max(actual_substeps, 1)  # 避免除零
        avg_success = (actual_substeps > 0) and (success_count == actual_substeps)

        step_success_rate = success_count / max(actual_substeps, 1)

        # 如果成功率低于目标(0.9)，增大惩罚；高于目标则减小惩罚
        self.lambda_sinr = float(np.clip(
            self.lambda_sinr + self.lambda_lr * (self.target_success_rate - step_success_rate),
            self.lambda_min, self.lambda_max
        ))

        # --- 5. 计算奖励 (基于平均性能) ---
        # 使用加权和速率作为奖励
        reward = self._calculate_reward(P_BS_fixed, P_sat_fixed*np.linalg.norm(self.current_channels['W_sat'],'fro'), avg_sinr_values, avg_success)

        # # --- 3.5 更新NLOS缓存 (高斯-马尔可夫模型，为下一step准备) ---
        # self.scenario.update_nlos_cache(self.nlos_alpha)
        # 重新生成channels以应用新的NLOS
        # self.current_channels = self.scenario.generate_channels()
        
        # --- 6. 更新快照 ---
        self.prev_Phi = Phi
        # 无论 decision_success 是否为 True，都用 W_fixed 更新 prev_W_bs
        # 因为 BS 实际上仍在发射，只是可能没满足 SINR 约束
        self.prev_W_bs = W_fixed
        self.prev_sinr_values = avg_sinr_values
        self.prev_obs_elevation = self.curr_obs_elevation
        self.prev_obs_azimuth = self.curr_obs_azimuth
        self.prev_P_sat = P_sat_fixed

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
            "P_BS": P_BS_fixed*np.linalg.norm(self.prev_W_bs)**2,
            "P_sat": P_sat_fixed*np.linalg.norm(self.current_channels['W_sat'])**2,
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
        Adaptive Lagrangian-style reward aiming for target success rate (0.9):
        - Main objective: minimize total power
        - Constraint: SINR >= threshold for all K+1 users (softly enforced via adaptive penalty)
        """

        # 1) Energy term: encourage lower total power
        total_p = float(P_BS + P_sat)
        total_p = max(total_p, 1e-12)
        reward_energy = -10.0 * np.log10(total_p)

        

        # 2) SINR constraint penalty (linear gap, normalized)
        soft_threshold = 10 ** ((self.sinr_threshold_db - 0.5) / 10.0)  # same spirit as your original
        sinr_gaps = np.maximum(0.0, soft_threshold - all_sinr_values)
        normalized_gaps = sinr_gaps / (soft_threshold + 1e-12)

        # 自适应乘子：lambda_sinr 在 step() 里根据成功率动态调整
        penalty_sinr = -self.sinr_penalty_weight * float(np.sum(normalized_gaps))

        # 4) Bonus: encourage strictly satisfying constraints
        # success 在你的实现里是“所有 substeps 都满足约束” :contentReference[oaicite:5]{index=5}
        # 给一个温和的 bonus，避免盖过主目标
        bonus_success = 5.0 if success else 0.0

        # 5) SINR_diff
        diff = np.sum(self.SINR_diff)

        # return reward_energy + penalty_sinr + bonus_success + diff
        return diff + penalty_sinr


    def _calculate_reward_weighted_rate(self, P_BS, P_sat, all_sinr_values, success):
        """
        分层Reward设计 (方案3):

        Layer 1 (必须满足): SINR约束 - 不满足则严重惩罚
        Layer 2 (主目标): 功耗最小化 - 归一化到[0,1]
        Layer 3 (辅助信号): SINR余量bonus - 引导找到更好的RIS相位

        Args:
            P_BS: BS功率 (W)
            P_sat: 卫星功率 (W)
            all_sinr_values: 所有用户SINR (线性尺度), shape=(K+1,)
            success: 是否所有用户满足SINR约束

        Returns:
            reward: 分层奖励值
        """
        # Layer 1: SINR约束检查 (硬约束)
        sinr_threshold_db = 9.8  # 10 - 0.2 dB
        sinr_threshold_linear = 10 ** (sinr_threshold_db / 10.0)

        sinr_gaps = np.maximum(0, sinr_threshold_linear - all_sinr_values)
        constraint_violated = np.any(sinr_gaps > 0)

        if constraint_violated:
            # 严重惩罚，但给出改进方向（gap越大惩罚越重）
            normalized_gaps = sinr_gaps / (sinr_threshold_linear + 1e-12)
            return -10.0 - 5.0 * np.sum(normalized_gaps)

        # --- 以下只有满足SINR约束才执行 ---

        # Layer 2: 功耗最小化 (主目标)
        # 归一化到[0, 1]范围，功耗越低reward越高
        total_power = P_BS + P_sat
        max_power = self.scenario.P_bs_max + self.scenario.P_sat_max  # 10 + 100 = 110W
        min_power = 0.1  # 假设最小功耗0.1W

        # 功耗reward: 功耗越低越好
        # power_ratio ∈ [0, 1], 0表示最小功耗, 1表示最大功耗
        power_ratio = np.clip((total_power - min_power) / (max_power - min_power), 0, 1)
        power_reward = 1.0 - power_ratio  # [0, 1], 越高越好

        # Layer 3: SINR余量bonus (辅助信号)
        # 超过阈值的部分给予小bonus，引导Agent找到更好的RIS相位
        # sinr_margin = (SINR - threshold) / threshold, 限制在[0, 1]
        sinr_margin = np.clip(all_sinr_values / sinr_threshold_linear - 1.0, 0, 1.0)
        margin_bonus = 0.1 * np.mean(sinr_margin)  # 小权重，避免喧宾夺主

        # 综合Reward
        # 主要由功耗决定，SINR余量作为辅助
        reward = 10.0 * power_reward + margin_bonus

        return reward


    def _initialize_with_P_sat(self):
        Phi_init = self.initial_ris_phase
        
        self.prev_P_sat, _ = self.scenario.compute_sat_power(
                Phi_init, self.current_channels, W_bs=self.prev_W_bs, sinr_threshold_db=self.sinr_threshold_db
            )
    
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
        self.P_sat_init = self.scenario.P_sat

        # Run ZF-waterfilling to verify channel quality
        try:
            W_init, P_BS_norm, success, bf_info = compute_zf_waterfilling_baseline(
                H_eff_k, H_eff_j, H_sat_eff_k, W_sat,
                P_sat=self.P_sat_init,
                P_bs_scale=self.scenario.P_bs_scale,
                sinr_threshold_linear=self.sinr_threshold_linear,
                noise_power=self.scenario.P_noise,
                max_power=self.scenario.P_bs_max
            )

            # Evaluate on TRUE channels
            all_sinr_true = self.calculate_all_sinrs(
                Phi_init, W_init, self.P_sat_init, self.current_channels
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
        self.prev_W_bs = W_init
        self.prev_sinr_values = all_sinr_true  # All K+1 users (UE + SUE)
        

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
            signal_power = self.scenario.P_bs_scale * (np.abs(h_eff_k @ w_k)**2)

            # 内部干扰 (Intra-system interference from BS)
            intra_interf = 0
            for j in range(K):
                if j != k:
                    intra_interf += self.scenario.P_bs_scale * (np.abs(h_eff_k @ W[:, j])**2)

            # 卫星干扰 (Inter-system interference from LEO)
            h_sat_eff_k = H_sat_eff_k[k, :]
            sat_interf = P_sat * (np.abs(h_sat_eff_k @ W_sat[:, 0])**2)

            sinr_ue[k] = signal_power / (intra_interf + sat_interf + noise_pow)

        # 3. 计算 1 个卫星用户 (SU) 的 SINR
        # 预期卫星信号功率
        h_sat_eff_su = H_sat_eff_su[0, :]
        sat_signal = P_sat * (np.abs(h_sat_eff_su @ W_sat[:, 0])**2)

        # 基站产生的聚合干扰
        bs_to_sat_interf = 0
        for k in range(K):
            h_eff_su_k = H_eff_su[0, :]
            bs_to_sat_interf += self.scenario.P_bs_scale * (np.abs(h_eff_su_k @ W[:, k])**2)
            
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

    def _get_relative_angles(self, pos_ris, pos_target):
        """
        计算目标相对于 RIS 的角度 (elevation, azimuth)

        Args:
            pos_ris: RIS 位置 (x, y, z)
            pos_target: 目标位置 (x, y, z)

        Returns:
            elevation: 仰角 (度, [0, 90])
            azimuth: 方位角 (度, [0, 360))
        """
        # 计算从 RIS 到目标的向量
        dx = pos_target[0] - pos_ris[0]
        dy = pos_target[1] - pos_ris[1]
        dz = pos_target[2] - pos_ris[2]

        # 计算水平距离
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12  # 避免除零
        # 计算仰角 (elevation)
        # elevation = arctan(dz / horizontal_dist)
        elevation = np.arctan2(dz, horizontal_dist) * 180.0 / np.pi
        elevation = np.clip(elevation, 0.0, 90.0)

        # 计算方位角 (azimuth)
        # azimuth = arctan2(dy, dx)，转换为 [0, 360)
        azimuth = np.arctan2(dy, dx) * 180.0 / np.pi
        azimuth = azimuth % 360.0

        return elevation, azimuth, distance
    def _get_phase_aware_features(self, Phi, channels):
        K = self.scenario.K
        features = []
        noise_std = np.sqrt(self.scenario.P_noise)
        W_bs = self.prev_W_bs  # 利用上一步的预编码
        W_sat = channels['W_sat']

        # 遍历所有用户 (K个UE + 1个SUE)
        for i in range(K + 1):
            if i < K: # 地面用户 UE_i
                h_d = channels['H_BS2UE'][i, :] @ W_bs[:, i]
                h_d_r = (channels['H_RIS2UE'][i, :] @ Phi @ channels['G_BS'] @ W_bs[:, i])
                h_j = 0
                h_j_r = 0
                for j in range(K):
                    if j != i:
                        h_j += channels['H_BS2UE'][i, :] @ W_bs[:, j]
                        h_j_r += channels['H_RIS2UE'][i, :] @ Phi @ channels['G_BS'] @ W_bs[:, j]
                h_j_s = channels['H_SAT2UE'][i, :] @ W_sat[:, 0]
                h_j_s_r = channels['H_RIS2UE'][i, :] @ Phi @ channels['G_SAT'] @ W_sat[:, 0]
                
                p_d = np.abs(h_d + h_d_r)
                p_j = np.abs(h_j + h_j_r)
                p_j_s = np.abs(h_j_s + h_j_s_r)
                delta_phi_d = np.angle(h_d) - np.angle(h_d_r)
                delta_phi_j = np.angle(h_j) - np.angle(h_j_r)
                delta_phi_j_s = np.angle(h_j_s) - np.angle(h_j_s_r)
                features.extend([
                    np.tanh(abs(h_d) / noise_std),
                    np.tanh(abs(h_d_r) / noise_std),
                    np.tanh(abs(h_j) / noise_std),
                    np.tanh(abs(h_j_r) / noise_std),
                    np.tanh(abs(h_j_s) / noise_std),
                    np.tanh(abs(h_j_s_r) / noise_std),
                    delta_phi_d,
                    delta_phi_j,
                    delta_phi_j_s,
                    np.tanh(p_d / noise_std),
                    np.tanh(p_j / noise_std),
                    np.tanh(p_j_s / noise_std),
                ])
            else: # 卫星用户 SUE
                # 直接链路: SAT -> SUE
                h_d = channels['H_SAT2SUE'][0, :] @ W_sat[:, 0]
                h_d_r = (channels['H_RIS2SUE'][0, :] @ Phi @ channels['G_SAT'] @ W_sat[:, 0])
                # 干扰信号（来自 BS 的用户）
                h_j = 0
                h_j_r = 0
                for k in range(K):
                    h_j += channels['H_BS2SUE'][0, :] @ W_bs[:, k]
                    h_j_r += channels['H_RIS2SUE'][0, :] @ Phi @ channels['G_BS'] @ W_bs[:, k]
                # 卫星干扰（来自其他卫星用户，这里假设只有一个卫星用户）
                h_j_s = 0  # 没有其他卫星用户
                h_j_s_r = 0
                p_d = np.abs(h_d + h_d_r)
                p_j = np.abs(h_j + h_j_r)
                p_j_s = np.abs(h_j_s + h_j_s_r)
                delta_phi_d = np.angle(h_d) - np.angle(h_d_r)
                delta_phi_j = np.angle(h_j) - np.angle(h_j_r)
                delta_phi_j_s = np.angle(h_j_s) - np.angle(h_j_s_r)
                features.extend([
                    np.tanh(abs(h_d) / noise_std),
                    np.tanh(abs(h_d_r) / noise_std),
                    np.tanh(abs(h_j) / noise_std),
                    np.tanh(abs(h_j_r) / noise_std),
                    np.tanh(abs(h_j_s) / noise_std),
                    np.tanh(abs(h_j_s_r) / noise_std),
                    delta_phi_d,
                    delta_phi_j,
                    delta_phi_j_s,
                    np.tanh(p_d / noise_std),
                    np.tanh(p_j / noise_std),
                    np.tanh(p_j_s / noise_std),
                ])

        return np.array(features, dtype=np.float32)
    def _get_state(self):
        """
        Construct enhanced state vector with both geometry and channel features

        State design (total: 7 + 12*(K+1) + 3*(K+1) + (K+1) dimensions, K=4 -> 87 dims):
        1. Satellite Geometry (7 dims): 卫星相对于 RIS 的几何和动态信息
           - sin(elevation), cos(elevation), sin(azimuth), cos(azimuth)
           - fixed_orbit_height, delta_elevation, delta_azimuth
        2. Phase-Aware Features (12*(K+1) dims): 信道相关特征 (每个用户12维)
           - 包含幅度、相位、功率等多维特征
        3. RIS Phase States (3*(K+1) dims): RIS 相位状态特征
           - 角度信息用于表示相位配置
        4. SINR Feedback (K+1 dims): 上一步的 SINR 反馈信息
        """
        K = self.scenario.K

        # ========== 1. Sat Incident (6 dims): SAT -> RIS ==========
        obs_ele = self.curr_obs_elevation
        obs_azi = self.curr_obs_azimuth

        # Trigonometric encoding
        ele_rad = np.deg2rad(obs_ele)
        azi_rad = np.deg2rad(obs_azi)

        # Angular velocity based on noisy observations
        delta_ele = obs_ele - self.prev_obs_elevation
        delta_azi = obs_azi - self.prev_obs_azimuth

        sat_geo = np.array([
            np.sin(ele_rad), np.cos(ele_rad),
            np.sin(azi_rad), np.cos(azi_rad), 
            delta_ele,  # Normalize angular velocity
            delta_azi
        ])
        # 信道特征
        features = self._get_phase_aware_features(self.prev_Phi, self.current_channels)
        ris = np.angle(np.diag(self.prev_Phi))/np.pi
        

        # ========== 4. Feedback (K+1 dims): SINR Margin ==========
        if self.prev_sinr_values is None:
            # First step: assume SINR at threshold
            sinr_margin = np.zeros(K + 1)
        else:
            # m = (SINR_t-1 / threshold) - 1 for all K+1 users
            sinr_margin = self.prev_sinr_values

        sinr_feedback = sinr_margin  # (K+1,)
        t = np.atleast_1d(self.current_step / self.max_steps)
        # ========== Concatenate All Features ==========
        state = np.concatenate([
            t,              
            sat_geo,           # (6,)
            features,            # (3,)
            ris,          # (3 * (K+1),)
            sinr_feedback      # (K+1,)
        ]).astype(np.float32)

        return state


    def render(self):
        """Optional: Visualize current scenario"""
        self.scenario.plot_scenario()
