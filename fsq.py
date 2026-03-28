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

    Each dimension of the latent is quantized to a fixed set of levels.
    Total codebook size = product of all levels.

    Args:
        levels: List of integers, number of quantization levels per dimension.
                E.g. [8, 5, 5, 5] gives 8*5*5*5 = 1000 codes.
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)

        # Precompute level tensors as buffers (not learned parameters)
        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels_tensor", levels_tensor)

        # Implicit codebook size
        self.codebook_size = 1
        for l in levels:
            self.codebook_size *= l

        # Basis for converting multi-dim indices to flat code indices
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer("basis", basis)

    def bound(self, z: Tensor) -> Tensor:
        """Bound z to the range expected before rounding to quantization levels."""
        half_l = (self.levels_tensor - 1) / 2  # e.g. for 5 levels: [-2, -1, 0, 1, 2]
        # Shift so levels are symmetric around 0
        offset = torch.where(
            self.levels_tensor % 2 == 0,
            torch.tensor(0.5, device=z.device),
            torch.tensor(0.0, device=z.device),
        )
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantize z to nearest level values (straight-through in backward pass)."""
        bounded = self.bound(z)
        # Round to nearest integer level — straight-through estimator
        quantized = bounded + (bounded.round() - bounded).detach()
        return quantized

    def codes_to_indices(self, codes: Tensor) -> Tensor:
        """
        Convert quantized codes (floating point level values) to flat integer indices.

        Args:
            codes: Tensor of shape (..., dim) with quantized level values.
        Returns:
            indices: Tensor of shape (...,) with flat integer code indices.
        """
        half_l = (self.levels_tensor - 1) // 2
        # Shift from [-half_l, half_l] to [0, levels-1]
        integer_codes = (codes + half_l).long()
        return (integer_codes * self.basis).sum(dim=-1)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """
        Convert flat integer indices back to quantized code values.

        Args:
            indices: Tensor of shape (...,) with flat integer code indices.
        Returns:
            codes: Tensor of shape (..., dim) with quantized level values.
        """
        levels = self.levels_tensor.long()
        half_l = (levels - 1) // 2
        integer_codes = torch.stack(
            [(indices // self.basis[i]) % levels[i] for i in range(self.dim)],
            dim=-1,
        )
        return (integer_codes - half_l).float()

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            z: Input tensor of shape (..., dim). Last dim must match len(levels).
        Returns:
            quantized: Same shape as z, quantized values (straight-through gradients).
            indices: Flat codebook indices of shape (...,).
        """
        assert z.shape[-1] == self.dim, (
            f"Input last dim {z.shape[-1]} must match number of levels {self.dim}"
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
    """28×28 → (B, D, 7, 7)  where D = len(levels)."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),          # 28 → 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),          # 14 → 7
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 1),                        # project to D channels
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)   # (B, D, 7, 7)


class Decoder(nn.Module):
    """(B, D, 7, 7) → 28×28."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 1),
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
    This prevents codebook collapse.
    """
    def __init__(self, levels: List[int]):
        super().__init__()
        self.latent_dim = len(levels)
        self.encoder = Encoder(self.latent_dim)
        self.fsq = FSQ(levels)
        self.decoder = Decoder(self.latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]
        z = self.encoder(x)                          # (B, D, 7, 7)

        # Rearrange to (B*49, D) so FSQ sees a flat batch of vectors
        z = z.permute(0, 2, 3, 1).reshape(-1, self.latent_dim)   # (B*49, D)
        z_q, indices = self.fsq(z)                               # (B*49, D), (B*49,)

        # Restore spatial shape for decoder
        z_q = z_q.reshape(B, 7, 7, self.latent_dim).permute(0, 3, 1, 2)  # (B, D, 7, 7)
        indices = indices.reshape(B, 49)                                   # (B, 49)

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
