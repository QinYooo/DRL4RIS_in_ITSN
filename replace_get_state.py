"""
Script to replace the _get_state function in itsn_env.py
"""
import re

# Read the file
with open(r"c:\Users\xfy\iCloudDrive\ding.chen\Graduation_Design\DRL_RIS\envs\itsn_env.py", 'r', encoding='utf-8') as f:
    content = f.read()

# New function implementation
new_function = '''    def _get_state(self):
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
        # Add ephemeris noise to observed angles
        obs_elevation = self.true_elevation + self.rng.normal(0, self.ephemeris_noise_std)
        obs_azimuth = self.true_azimuth + self.rng.normal(0, self.ephemeris_noise_std)

        # Trigonometric encoding (more stable than raw angles)
        ele_rad = np.deg2rad(obs_elevation)
        azi_rad = np.deg2rad(obs_azimuth)

        # Angular velocity based on noisy observations (current obs - previous obs)
        delta_ele = obs_elevation - self.prev_obs_elevation
        delta_azi = obs_azimuth - self.prev_obs_azimuth

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
        g_k_bs_norm = np.log10(g_k_bs + 1e-20) / 10.0
        g_k_sat_norm = np.log10(g_k_sat + 1e-20) / 10.0

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
'''

# Find and replace the function using regex
# Pattern: match from "def _get_state(self):" until the next "def " or end of class
pattern = r'(    def _get_state\(self\):.*?)(    def \w+\(|    def render\()'
match = re.search(pattern, content, re.DOTALL)

if match:
    # Replace the old function with the new one
    new_content = content[:match.start(1)] + new_function + '\n\n' + content[match.start(2):]

    # Write back
    with open(r"c:\Users\xfy\iCloudDrive\ding.chen\Graduation_Design\DRL_RIS\envs\itsn_env.py", 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("Successfully replaced _get_state function")
    print(f"Old function length: {len(match.group(1))} chars")
    print(f"New function length: {len(new_function)} chars")
else:
    print("ERROR: Could not find _get_state function")
