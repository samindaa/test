"""
Finite Scalar Quantization (FSQ)
Reference: "Finite Scalar Quantization: VQ-VAE Made Simple" (Mentzer et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple


class FSQ(nn.Module):
    """
    Finite Scalar Quantization module.

    Each dimension of the latent is independently quantized to L discrete levels.
    Total codebook size = product of all levels.  No learned parameters; gradients
    flow via the Straight-Through Estimator (STE).

    Implementation follows the reference code from Appendix A.1 of the paper:
        z_i  →  bound(z_i)  →  round (STE)  →  normalize to [-1, 1]

    Args:
        levels: Number of quantization levels per dimension.
                Recommended configs from paper (Appendix A.4.1):
                  [8, 5, 5, 5, 5]  →  5_000 codes
                  [8, 8, 8, 5]     →  2_560 codes
                  [8, 8, 8, 8]     →  4_096 codes
                  [8, 8, 8]        →   512 codes
                  [8, 5, 5, 5]     →  1_000 codes
                  [5, 5, 5, 5, 5]  →  3_125 codes
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0)
        self.register_buffer("_basis", _basis)

        self.codebook_size: int = 1
        for l in levels:
            self.codebook_size *= l

    # ------------------------------------------------------------------
    # Straight-through helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_ste(z: Tensor) -> Tensor:
        return z + (z.round() - z).detach()

    # ------------------------------------------------------------------
    # Bound + quantize  (matches Appendix A.1 reference)
    # ------------------------------------------------------------------

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound z and quantize to discrete levels; output is normalized to [-1, 1].

        For odd L:   values in {-(L-1)/2, …, (L-1)/2} / (L//2)
        For even L:  shifted so levels are symmetric, then normalized.
        """
        half_l = (self._levels - 1).float() * (1 + eps) / 2   # eps prevents tanh saturation
        offset = torch.where(self._levels % 2 == 0,
                             torch.full_like(half_l, 0.5),
                             torch.zeros_like(half_l))
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset       # integer-valued range
        half_width = (self._levels // 2).float()
        return self._round_ste(bounded_z) / half_width          # normalized to [-1, 1]

    def quantize(self, z: Tensor) -> Tensor:
        """Bound + round + normalize, with STE gradient."""
        return self.bound(z)

    # ------------------------------------------------------------------
    # Index conversion  (works on normalized [-1, 1] codes)
    # ------------------------------------------------------------------

    def _scale_and_shift(self, codes_normalized: Tensor) -> Tensor:
        """Normalized [-1, 1]  →  integer [0, L-1]."""
        half_width = (self._levels // 2).float()
        return codes_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, integer_codes: Tensor) -> Tensor:
        """Integer [0, L-1]  →  normalized [-1, 1]."""
        half_width = (self._levels // 2).float()
        return (integer_codes - half_width) / half_width

    def codes_to_indices(self, codes: Tensor) -> Tensor:
        """Normalized codes (..., dim) → flat codebook indices (...)."""
        assert codes.shape[-1] == self.dim
        integer_codes = self._scale_and_shift(codes).long()
        return (integer_codes * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Flat codebook indices (...) → normalized codes (..., dim)."""
        integer_codes = torch.stack(
            [(indices // self._basis[i]) % self._levels[i] for i in range(self.dim)],
            dim=-1,
        ).float()
        return self._scale_and_shift_inverse(integer_codes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: (..., dim)  — last dim must equal len(levels).
        Returns:
            quantized: same shape as z, normalized to [-1, 1], STE gradients.
            indices:   (...,) flat codebook indices.
        """
        assert z.shape[-1] == self.dim, (
            f"Input last dim {z.shape[-1]} must match FSQ dim {self.dim}"
        )
        quantized = self.quantize(z)
        indices = self.codes_to_indices(quantized)
        return quantized, indices


# ---------------------------------------------------------------------------
# FSQ-VAE on MNIST — spatial quantization (7×7 grid, D channels per cell)
#
# Old approach: flatten → single vector → 1 code per image  → collapse
# New approach: keep spatial grid → quantize each 7×7 cell  → 49 codes per image
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """28×28 → (B, hidden_dim, 7, 7)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),           # 28 → 14
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # 14 → 7
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),             # 7 → 7, widen
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)   # (B, hidden_dim, 7, 7)


class Decoder(nn.Module):
    """(B, hidden_dim, 7, 7) → 28×28."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14 → 28
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)   # (B, 1, 28, 28)


class FSQVAE(nn.Module):
    """
    FSQ-VAE with spatial quantization.

    Each image is encoded to a 7×7 grid. Every grid cell is independently
    quantized by FSQ, giving 49 code indices per image instead of 1.
    Pre/post-quant projections decouple encoder width from FSQ dim.
    """
    def __init__(self, levels: List[int], hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = hidden_dim
        fsq_dim = len(levels)
        self.encoder = Encoder(hidden_dim)
        self.pre_quant = nn.Conv2d(hidden_dim, fsq_dim, 1)
        self.fsq = FSQ(levels)
        self.post_quant = nn.Conv2d(fsq_dim, hidden_dim, 1)
        self.decoder = Decoder(hidden_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]
        fsq_dim = self.fsq.dim

        z = self.encoder(x)                          # (B, hidden_dim, 7, 7)
        z = self.pre_quant(z)                        # (B, fsq_dim, 7, 7)

        # Rearrange to (B*49, fsq_dim) so FSQ sees a flat batch of vectors
        z = z.permute(0, 2, 3, 1).reshape(-1, fsq_dim)   # (B*49, fsq_dim)
        z_q, indices = self.fsq(z)                        # (B*49, fsq_dim), (B*49,)

        # Restore spatial shape for post-quant projection and decoder
        z_q = z_q.reshape(B, 7, 7, fsq_dim).permute(0, 3, 1, 2)  # (B, fsq_dim, 7, 7)
        indices = indices.reshape(B, 49)                           # (B, 49)

        z_q = self.post_quant(z_q)                                 # (B, hidden_dim, 7, 7)
        x_hat = self.decoder(z_q)
        return x_hat, indices


# ---------------------------------------------------------------------------
# Learnable Conditional Prior
# ---------------------------------------------------------------------------

class ConditionalPrior(nn.Module):
    """
    Learnable conditional prior: P(code_index | condition).

    Models the distribution over FSQ codebook indices conditioned on a
    conditioning signal (e.g. class label). Trained independently after
    the FSQ-VAE is frozen, enabling conditional generation by sampling
    from this prior and decoding through the FSQ decoder.

    Args:
        codebook_size: Total number of FSQ codes (product of all levels).
        num_classes:   Number of conditioning classes.
        hidden_dim:    Width of the MLP layers.
    """

    def __init__(self, codebook_size: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.codebook_size = codebook_size

        # Learned embedding for the conditioning signal
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)

        # MLP that maps condition -> logits over all codebook entries
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, codebook_size),
        )

    def forward(self, labels: Tensor) -> Tensor:
        """
        Args:
            labels: Long tensor of shape (B,) with class indices.
        Returns:
            logits: Tensor of shape (B, codebook_size) — unnormalised log-probs.
        """
        cond = self.class_embedding(labels)   # (B, hidden_dim)
        return self.net(cond)                  # (B, codebook_size)

    def loss(self, labels: Tensor, target_indices: Tensor) -> Tensor:
        """
        Cross-entropy loss: -log P(target_index | label).

        Args:
            labels:         Long tensor (B,) of class labels.
            target_indices: Long tensor (B,) of FSQ code indices from the encoder.
        """
        logits = self.forward(labels)
        return F.cross_entropy(logits, target_indices)

    @torch.no_grad()
    def sample(self, labels: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Sample a code index for each label.

        Args:
            labels:      Long tensor (B,) of class labels.
            temperature: Sampling temperature. 1.0 = standard, <1 = sharper.
        Returns:
            indices: Long tensor (B,) of sampled FSQ code indices.
        """
        logits = self.forward(labels) / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
