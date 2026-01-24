"""
Channel Autoencoder for State Space Compression
Compresses high-dimensional channel matrices into low-dimensional latent representations
"""

import torch
import torch.nn as nn
import numpy as np


class ChannelAutoencoder(nn.Module):
    """
    Autoencoder for compressing ITSN channel matrices

    Input: Flattened channel matrices (real and imaginary parts)
    Output: Latent representation (compressed features)
    """

    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64]):
        """
        Args:
            input_dim: Total dimension of flattened channel vector
            latent_dim: Dimension of compressed latent space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode channel vector to latent representation"""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation back to channel vector"""
        return self.decoder(z)

    def forward(self, x):
        """Full autoencoder forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def preprocess_channels(channels, use_inferred_G_SAT=False, inferred_G_SAT=None):
    """
    Convert channel dictionary to flattened vector (real and imaginary parts)

    Args:
        channels: Dict containing channel matrices
        use_inferred_G_SAT: Whether to use inferred G_SAT instead of true G_SAT
        inferred_G_SAT: Inferred G_SAT matrix (if use_inferred_G_SAT=True)

    Returns:
        Flattened channel vector (numpy array)
    """
    # Extract all channel matrices
    H_BS2UE = channels['H_BS2UE']      # (K, N_t)
    H_BS2SUE = channels['H_BS2SUE']    # (SK, N_t)
    G_BS = channels['G_BS']            # (N_ris, N_t)
    H_RIS2UE = channels['H_RIS2UE']    # (K, N_ris)
    H_RIS2SUE = channels['H_RIS2SUE']  # (SK, N_ris)
    H_SAT2UE = channels['H_SAT2UE']    # (K, N_sat)
    H_SAT2SUE = channels['H_SAT2SUE']  # (SK, N_sat)

    # Choose G_SAT based on flag
    if use_inferred_G_SAT and inferred_G_SAT is not None:
        G_SAT = inferred_G_SAT
    else:
        G_SAT = channels['G_SAT']      # (N_ris, N_sat)

    # Flatten and concatenate all channels (real and imaginary parts)
    channel_parts = []
    for ch in [H_BS2UE, H_BS2SUE, G_BS, H_RIS2UE, H_RIS2SUE, H_SAT2UE, H_SAT2SUE, G_SAT]:
        channel_parts.append(ch.real.flatten())
        channel_parts.append(ch.imag.flatten())

    channel_vec = np.concatenate(channel_parts)
    return channel_vec.astype(np.float32)


def compute_channel_input_dim(scenario):
    """
    Compute the total dimension of flattened channel vector

    Args:
        scenario: ITSNScenario instance

    Returns:
        Total input dimension for autoencoder
    """
    K = scenario.K
    SK = scenario.SK
    N_t = scenario.N_t
    N_ris = scenario.N_ris
    N_sat = scenario.N_sat

    # Each complex matrix contributes 2 * (rows * cols) to the flattened vector
    dims = {
        'H_BS2UE': 2 * K * N_t,
        'H_BS2SUE': 2 * SK * N_t,
        'G_BS': 2 * N_ris * N_t,
        'H_RIS2UE': 2 * K * N_ris,
        'H_RIS2SUE': 2 * SK * N_ris,
        'H_SAT2UE': 2 * K * N_sat,
        'H_SAT2SUE': 2 * SK * N_sat,
        'G_SAT': 2 * N_ris * N_sat
    }

    total_dim = sum(dims.values())
    return total_dim, dims


def normalize_channel_vector(channel_vec, mean=None, std=None):
    """
    Normalize channel vector using z-score normalization

    Args:
        channel_vec: Raw channel vector
        mean: Pre-computed mean (if None, compute from data)
        std: Pre-computed std (if None, compute from data)

    Returns:
        Normalized channel vector, mean, std
    """
    if mean is None:
        mean = np.mean(channel_vec)
    if std is None:
        std = np.std(channel_vec) + 1e-8

    normalized = (channel_vec - mean) / std
    return normalized, mean, std


def denormalize_channel_vector(normalized_vec, mean, std):
    """Reverse z-score normalization"""
    return normalized_vec * std + mean
