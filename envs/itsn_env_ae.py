"""
ITSN Gym Environment with Channel Autoencoder State Compression
Extends ITSNEnv to use pre-trained autoencoder for channel compression
"""

import numpy as np
import torch
from gymnasium import spaces
from pathlib import Path

from envs.itsn_env import ITSNEnv
from models.channel_autoencoder import (
    ChannelAutoencoder,
    preprocess_channels,
    normalize_channel_vector
)


class ITSNEnvAE(ITSNEnv):
    """
    ITSN Environment with Autoencoder-based State Compression

    State Space:
        - Satellite motion features (6 dims)
        - Compressed channel features from AE (latent_dim dims)
        - Performance feedback (K+1 dims)

    Total state dimension: 6 + latent_dim + (K+1)
    """

    def __init__(self,
                 ae_checkpoint_path=None,
                 latent_dim=32,
                 device='cuda',
                 **kwargs):
        """
        Args:
            ae_checkpoint_path: Path to pre-trained autoencoder checkpoint
            latent_dim: Latent dimension of autoencoder
            device: 'cuda' or 'cpu'
            **kwargs: Arguments passed to parent ITSNEnv
        """
        # Initialize parent environment first (to get scenario)
        super().__init__(**kwargs)

        # Setup device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load autoencoder
        self.latent_dim = latent_dim
        self.ae_model = None
        self.normalization_stats = None

        if ae_checkpoint_path is not None:
            self._load_autoencoder(ae_checkpoint_path)
        else:
            print("[Warning] No autoencoder checkpoint provided. Using random initialization.")
            # Create a dummy autoencoder (will need to be trained)
            from models.channel_autoencoder import compute_channel_input_dim
            input_dim, _ = compute_channel_input_dim(self.scenario)
            self.ae_model = ChannelAutoencoder(input_dim, latent_dim).to(self.device)
            self.ae_model.eval()
            self.normalization_stats = {'mean': 0.0, 'std': 1.0}

        # Update observation space to reflect new state dimension
        # State: satellite_motion (6) + channel_latent (latent_dim) + performance_feedback (K+1)
        state_dim = 6 + self.latent_dim + (self.scenario.K + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        print(f"[ITSNEnvAE] State dimension: {state_dim} (6 motion + {self.latent_dim} channel + {self.scenario.K+1} feedback)")

    def _load_autoencoder(self, checkpoint_path):
        """Load pre-trained autoencoder from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")

        print(f"[ITSNEnvAE] Loading autoencoder from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract model parameters
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']

        if latent_dim != self.latent_dim:
            print(f"[Warning] Checkpoint latent_dim ({latent_dim}) != specified latent_dim ({self.latent_dim})")
            self.latent_dim = latent_dim

        # Initialize model
        self.ae_model = ChannelAutoencoder(input_dim, latent_dim).to(self.device)
        self.ae_model.load_state_dict(checkpoint['model_state_dict'])
        self.ae_model.eval()

        # Load normalization stats
        self.normalization_stats = checkpoint['normalization_stats']

        print(f"[ITSNEnvAE] Autoencoder loaded: input_dim={input_dim}, latent_dim={latent_dim}")

    def _encode_channels(self, channels, use_inferred_G_SAT=True):
        """
        Encode channel matrices to latent representation using autoencoder

        Args:
            channels: Channel dictionary
            use_inferred_G_SAT: Whether to use inferred G_SAT

        Returns:
            Latent representation (numpy array)
        """
        # Preprocess channels to vector
        channel_vec = preprocess_channels(
            channels,
            use_inferred_G_SAT=use_inferred_G_SAT,
            inferred_G_SAT=self.inferred_G_SAT if use_inferred_G_SAT else None
        )

        # Normalize
        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']
        normalized_vec = (channel_vec - mean) / (std + 1e-8)

        # Encode
        with torch.no_grad():
            x = torch.FloatTensor(normalized_vec).unsqueeze(0).to(self.device)
            latent = self.ae_model.encode(x)
            latent = latent.squeeze(0).cpu().numpy()

        return latent

    def _get_state(self):
        """
        Construct state vector with autoencoder-compressed channels

        State design (total: 6 + latent_dim + (K+1) dimensions):
        1. Satellite motion state (6 features):
           - sin(elevation), cos(elevation), sin(azimuth), cos(azimuth)
           - delta_elevation, delta_azimuth
        2. Compressed channel features (latent_dim features):
           - Latent representation from autoencoder
        3. Performance feedback (K+1 features):
           - SINR margin for K UE + 1 SUE
        """
        K = self.scenario.K

        # ========== 1. Satellite Motion State (6 features) ==========
        obs_ele = self.curr_obs_elevation
        obs_azi = self.curr_obs_azimuth

        ele_rad = np.deg2rad(obs_ele)
        azi_rad = np.deg2rad(obs_azi)

        delta_ele = obs_ele - self.prev_obs_elevation
        delta_azi = obs_azi - self.prev_obs_azimuth

        satellite_motion = np.array([
            np.sin(ele_rad), np.cos(ele_rad),
            np.sin(azi_rad), np.cos(azi_rad),
            delta_ele / 10.0,
            delta_azi / 10.0
        ])

        # ========== 2. Compressed Channel Features (latent_dim features) ==========
        # Encode current channels using autoencoder
        channel_latent = self._encode_channels(self.current_channels, use_inferred_G_SAT=True)

        # ========== 3. Performance Feedback (K+1 features) ==========
        if self.prev_sinr_values is None:
            # First step: assume SINR at threshold
            sinr_margin = np.zeros(K + 1)
        else:
            # m = (SINR_t-1 / threshold) - 1 for all K+1 users
            sinr_margin = (self.prev_sinr_values / self.sinr_threshold_linear) - 1.0

        performance_feedback = sinr_margin

        # ========== Concatenate All Features ==========
        state = np.concatenate([
            satellite_motion,       # (6,)
            channel_latent,         # (latent_dim,)
            performance_feedback    # (K+1,)
        ]).astype(np.float32)

        return state


def create_env_with_ae(ae_checkpoint_path, **env_kwargs):
    """
    Convenience function to create ITSNEnvAE with autoencoder

    Args:
        ae_checkpoint_path: Path to autoencoder checkpoint
        **env_kwargs: Additional arguments for ITSNEnv

    Returns:
        ITSNEnvAE instance
    """
    return ITSNEnvAE(ae_checkpoint_path=ae_checkpoint_path, **env_kwargs)
