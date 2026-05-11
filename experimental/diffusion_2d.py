"""
Shared machinery for the 2D-circle diffusion + SDS demo.

Imported by:
    train_diffusion_2d.py  -- trains TinyEpsNet, saves a checkpoint
    sds_2d.py              -- loads the checkpoint, runs SDS

Holds: data sampler, DDPM noise schedule, forward-noising helper, and the
tiny noise-prediction network. Nothing here runs on import.
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
# Data: points uniformly on the unit circle.
# ---------------------------------------------------------------------------
def sample_data(n):
    angles = torch.rand(n, device=DEVICE) * 2 * math.pi
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


# ---------------------------------------------------------------------------
# DDPM noise schedule.
#
# `make_schedule(T, kind)` returns betas. Available kinds:
#   "cosine"  -- Nichol & Dhariwal, Improved DDPM (2021). The current default.
#                Decays the signal-to-noise ratio more gracefully than linear,
#                especially for small T. Almost always a better choice.
#   "linear"  -- Original DDPM (Ho et al. 2020). Destroys signal too fast at
#                the tail when T is small; included for comparison.
#   "sigmoid" -- Smoother variant; a middle ground.
#
# Things further along the SOTA frontier (not implemented here, deliberately):
#   EDM (Karras et al. 2022) -- continuous-time, sigma-parameterized,
#       log-uniform sigma sampling during training. The de-facto modern choice.
#   Flow matching / rectified flow -- different paradigm; SD3 / Flux use it.
#   v-prediction + min-SNR loss weighting -- changes the prediction target
#       and loss weighting rather than the noise schedule itself.
# All of these are bigger refactors than swapping betas. Cosine is the
# best ROI single-line upgrade for this kind of toy demo.
#
# Note: changing the schedule invalidates trained checkpoints. Retrain after.
# ---------------------------------------------------------------------------
def make_schedule(T, kind="cosine"):
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, T)
    if kind == "cosine":
        # Nichol & Dhariwal: define ᾱ_t via a squared cosine, then derive β_t
        # so β_t = 1 - ᾱ_t / ᾱ_{t-1}. The small offset s=0.008 prevents
        # vanishingly small betas at t=0.
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
T = 100
betas = make_schedule(T, SCHEDULE_KIND).to(DEVICE)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_ab = torch.sqrt(alpha_bars)
sqrt_1mab = torch.sqrt(1 - alpha_bars)


def add_noise(x0, t, eps):
    # Forward process: x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps
    return sqrt_ab[t][:, None] * x0 + sqrt_1mab[t][:, None] * eps


# ---------------------------------------------------------------------------
# Tiny noise-prediction network: (x_t, t) -> predicted noise eps_hat.
# ---------------------------------------------------------------------------
class TinyEpsNet(nn.Module):
    def __init__(self, hidden=128, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
            nn.Linear(hidden, 2),
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
