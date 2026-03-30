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

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale + shift spatial features by a conditioning vector."""
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, num_features)
        self.shift = nn.Linear(cond_dim, num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, C, H, W),  cond: (B, cond_dim)
        s = self.scale(cond)[:, :, None, None]  # (B, C, 1, 1)
        b = self.shift(cond)[:, :, None, None]
        return x * s + b


class Encoder(nn.Module):
    """28×28 → (B, hidden_dim, 7, 7), label-conditioned via FiLM."""
    def __init__(self, hidden_dim: int, num_classes: int = 10, label_embed_dim: int = 32):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)   # 28 → 14
        self.norm1 = nn.GroupNorm(8, 32)
        self.film1 = FiLM(32, label_embed_dim)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 14 → 7
        self.norm2 = nn.GroupNorm(8, 64)
        self.film2 = FiLM(64, label_embed_dim)
        self.conv3 = nn.Conv2d(64, hidden_dim, 3, padding=1)    # 7 → 7, widen
        self.norm3 = nn.GroupNorm(8, hidden_dim)
        self.film3 = FiLM(hidden_dim, label_embed_dim)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        cond = self.label_embed(labels)                              # (B, label_embed_dim)
        x = F.relu(self.film1(self.norm1(self.conv1(x)), cond))     # (B, 32, 14, 14)
        x = F.relu(self.film2(self.norm2(self.conv2(x)), cond))     # (B, 64, 7, 7)
        x = F.relu(self.film3(self.norm3(self.conv3(x)), cond))     # (B, hidden_dim, 7, 7)
        return x


class Decoder(nn.Module):
    """(B, hidden_dim, 7, 7) + labels → 28×28, conditioned via FiLM."""
    def __init__(self, hidden_dim: int, num_classes: int = 10, label_embed_dim: int = 32):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.conv1 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        self.film1 = FiLM(64, label_embed_dim)
        self.up1   = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 7 → 14
        self.film2 = FiLM(32, label_embed_dim)
        self.up2   = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)   # 14 → 28

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        cond = self.label_embed(labels)                    # (B, label_embed_dim)
        x = F.relu(self.film1(self.conv1(z), cond))       # (B, 64, 7, 7)
        x = F.relu(self.film2(self.up1(x),  cond))        # (B, 32, 14, 14)
        return torch.sigmoid(self.up2(x))                  # (B, 1, 28, 28)


class FSQVAE(nn.Module):
    """
    CVAE-FSQ: label-conditioned encoder + Gaussian bottleneck + FSQ + label-conditioned decoder.

    Both encoder and decoder are conditioned on the class label via FiLM. The encoder
    produces per-spatial-cell μ and log σ² (one Gaussian per 7×7 cell per fsq_dim channel).
    Reparameterization samples z ~ N(μ, σ²); FSQ quantizes z with STE gradients.
    KL loss pushes the posterior toward N(0,I), enabling generation by sampling directly
    from N(0,I) and decoding with any desired label — no encoder needed at inference.
    """
    def __init__(self, levels: List[int], hidden_dim: int = 128,
                 num_classes: int = 10, label_embed_dim: int = 32):
        super().__init__()
        self.latent_dim = hidden_dim
        self.fsq_dim = len(levels)
        self.encoder     = Encoder(hidden_dim, num_classes, label_embed_dim)
        self.mu_head     = nn.Conv2d(hidden_dim, self.fsq_dim, 1)
        self.logvar_head = nn.Conv2d(hidden_dim, self.fsq_dim, 1)
        self.fsq         = FSQ(levels)
        self.post_quant  = nn.Conv2d(self.fsq_dim, hidden_dim, 1)
        self.decoder     = Decoder(hidden_dim, num_classes, label_embed_dim)

    def forward(self, x: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B = x.shape[0]

        h      = self.encoder(x, labels)                           # (B, hidden_dim, 7, 7)
        mu     = self.mu_head(h)                                   # (B, fsq_dim, 7, 7)
        logvar = self.logvar_head(h).clamp(-10.0, 2.0)            # (B, fsq_dim, 7, 7)
        z      = mu + (0.5 * logvar).exp() * torch.randn_like(mu) # reparameterize

        z_flat       = z.permute(0, 2, 3, 1).reshape(-1, self.fsq_dim)  # (B*49, fsq_dim)
        z_q, indices = self.fsq(z_flat)                                  # STE gradients
        z_q          = z_q.reshape(B, 7, 7, self.fsq_dim).permute(0, 3, 1, 2)  # (B, fsq_dim, 7, 7)
        indices      = indices.reshape(B, 49)

        z_q   = self.post_quant(z_q)      # (B, hidden_dim, 7, 7)
        x_hat = self.decoder(z_q, labels) # (B, 1, 28, 28)
        return x_hat, indices, mu, logvar

    @torch.no_grad()
    def sample(self, labels: Tensor, device: torch.device) -> Tensor:
        """Generate images from class labels alone by sampling z ~ N(0, I)."""
        B      = labels.shape[0]
        eps    = torch.randn(B, self.fsq_dim, 7, 7, device=device)
        z_flat = eps.permute(0, 2, 3, 1).reshape(-1, self.fsq_dim)
        z_q, _ = self.fsq(z_flat)
        z_q    = z_q.reshape(B, 7, 7, self.fsq_dim).permute(0, 3, 1, 2)
        z_q    = self.post_quant(z_q)
        return self.decoder(z_q, labels.to(device))  # (B, 1, 28, 28)
