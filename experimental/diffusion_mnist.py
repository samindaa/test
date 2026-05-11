"""
Shared machinery for the MNIST diffusion + SDS demo.

Mirrors diffusion_2d.py exactly -- only two things change vs the 2D case:
  - the data is one digit class from MNIST (28x28 = 784-dim images),
  - the model is a wider MLP that takes a 784-dim input.

Everything else (DDPM schedule, add_noise, time embedding, SDS algorithm)
is identical. That is the lesson: SDS does not care what theta is.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Device. Prefer Apple Metal (MPS) when available, then CUDA, then CPU.
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# DDPM noise schedule. See diffusion_2d.py for a longer note; same options:
#   "cosine"  (default, Nichol & Dhariwal 2021) -- best for small/medium T
#   "linear"  (original DDPM)
#   "sigmoid"
# Cosine matters MORE here than for 2D: with images, linear schedules at
# T=200 push x_t too close to pure noise too early, hurting both training
# and SDS quality.
#
# Note: changing the schedule invalidates trained checkpoints. Retrain after.
# ---------------------------------------------------------------------------
def make_schedule(T, kind="cosine"):
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, T)
    if kind == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-4, 0.999).float()
    if kind == "sigmoid":
        s = torch.linspace(-6, 6, T)
        return torch.sigmoid(s) * (0.02 - 1e-4) + 1e-4
    raise ValueError(f"unknown schedule kind: {kind!r}")


SCHEDULE_KIND = "cosine"
T = 200
betas = make_schedule(T, SCHEDULE_KIND).to(DEVICE)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_ab = torch.sqrt(alpha_bars)
sqrt_1mab = torch.sqrt(1 - alpha_bars)

IMG_SIZE = 28
IMG_DIM = IMG_SIZE * IMG_SIZE   # 784


def add_noise(x0, t, eps):
    # x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps,  same form as 2D
    return sqrt_ab[t][:, None] * x0 + sqrt_1mab[t][:, None] * eps


# ---------------------------------------------------------------------------
# Data: load MNIST, filter to one digit class, return as (N, 784) in [-1, 1].
# ---------------------------------------------------------------------------
def get_data(digit_class=8, root="experimental/mnist_data"):
    try:
        from torchvision import datasets
    except ImportError as e:
        raise ImportError(
            "torchvision is required for the MNIST demo.\n"
            "Install it with: pip install torchvision"
        ) from e

    ds = datasets.MNIST(root, train=True, download=True)
    mask = ds.targets == digit_class
    images = ds.data[mask].float() / 255.0          # (N, 28, 28) in [0, 1]
    x = images.view(-1, IMG_DIM) * 2 - 1            # (N, 784) in [-1, 1]
    return x.to(DEVICE)


# ---------------------------------------------------------------------------
# Noise-prediction MLP: (x_t, t) -> predicted noise eps_hat.
# Bigger than the 2D version (more pixels), but architecturally identical.
# ---------------------------------------------------------------------------
class TinyEpsNetMNIST(nn.Module):
    def __init__(self, hidden=512, t_dim=64):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(IMG_DIM + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),          nn.SiLU(),
            nn.Linear(hidden, hidden),          nn.SiLU(),
            nn.Linear(hidden, IMG_DIM),
        )

    def time_embed(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embed(t)], dim=1))
