"""
EDM (Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
Generative Models") on the 2D circle, with SDS at the end.

Why EDM. It is now the de-facto modern formulation of diffusion. Compared
to the DDPM in diffusion_2d.py:

  1. Continuous noise level sigma instead of discrete timesteps t. The
     forward process is just  x_t = x + sigma * eps. No betas, no
     alphas, no abars. One number, sigma, controls everything.

  2. Preconditioning. The trainable network F is wrapped:
        D(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma)*x; c_noise(sigma))
     The constants c_*(sigma) are picked so F always sees inputs and
     targets of unit magnitude, no matter how large sigma gets. This is
     why EDM trains stably across the full sigma range while DDPM has
     to be careful about which timesteps it weights.

  3. Log-normal training noise. sigma ~ exp(N(P_mean, P_std^2)). The loss
     is weighted so each sigma contributes equally to learning.

  4. Heun's 2nd-order ODE solver at sample time, with the Karras sigma
     schedule -- a power-rho interpolation between sigma_max and sigma_min.
     Often only ~18 steps are needed to match what DDPM took 1000 for.

This file does THREE things:
    - trains an EDM denoiser on the unit circle
    - generates samples with Heun (proves the denoiser learned the data)
    - runs SDS using the EDM denoiser (mirrors sds_2d.py, but cleaner math)

Run:
    python experimental/edm_2d_demo.py
Outputs:
    experimental/edm_2d.png
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Device.
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# EDM hyperparameters (Karras et al. 2022, table 1, "EDM" column).
# ---------------------------------------------------------------------------
SIGMA_DATA = 0.5       # std of clean data (Karras default; unit circle's
                       # true value is ~0.707, but 0.5 works fine here)
SIGMA_MIN = 0.002      # smallest sigma at sample time
SIGMA_MAX = 80.0       # largest sigma at sample time
P_MEAN = -1.2          # log-normal mean for training-time sigma
P_STD = 1.2            # log-normal std for training-time sigma
RHO = 7.0              # exponent of the Karras sigma schedule for sampling


# ---------------------------------------------------------------------------
# Data: points uniformly on the unit circle.
# ---------------------------------------------------------------------------
def sample_data(n):
    angles = torch.rand(n, device=DEVICE) * 2 * math.pi
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


# ---------------------------------------------------------------------------
# EDM denoiser. The trainable raw network is F; D(x; sigma) wraps F with
# the Karras preconditioning so F always sees nicely-scaled inputs.
# ---------------------------------------------------------------------------
class EDMDenoiser(nn.Module):
    def __init__(self, hidden=128, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        # Same MLP shape as TinyEpsNet, but the time input is a continuous
        # c_noise(sigma) = 0.25 * log(sigma) instead of an integer t.
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def time_embed(self, c_noise):
        half = self.t_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=c_noise.device) / half
        )
        args = c_noise[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=1)

    def F(self, x_in, c_noise):
        return self.net(torch.cat([x_in, self.time_embed(c_noise)], dim=1))

    def forward(self, x, sigma):
        """
        x:     (B, 2)   noisy input
        sigma: (B,)     noise level per sample
        Returns D(x; sigma) -- the denoiser's estimate of clean x, shape (B, 2).
        """
        sigma = sigma.view(-1, 1)
        c_skip  = SIGMA_DATA**2 / (sigma**2 + SIGMA_DATA**2)
        c_out   = sigma * SIGMA_DATA / torch.sqrt(sigma**2 + SIGMA_DATA**2)
        c_in    = 1.0 / torch.sqrt(sigma**2 + SIGMA_DATA**2)
        c_noise = 0.25 * torch.log(sigma).flatten()
        return c_skip * x + c_out * self.F(c_in * x, c_noise)


# ---------------------------------------------------------------------------
# Training. Log-normal sigma, weighted MSE on D vs the clean x.
# ---------------------------------------------------------------------------
def train(steps=4000, batch=512, lr=1e-3):
    print(f"Device: {DEVICE}")
    model = EDMDenoiser().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        x0 = sample_data(batch)                                      # clean
        log_sigma = P_MEAN + P_STD * torch.randn(batch, device=DEVICE)
        sigma = torch.exp(log_sigma)                                 # log-normal
        eps = torch.randn_like(x0)
        x_t = x0 + sigma[:, None] * eps                              # noised

        D = model(x_t, sigma)
        # Karras loss weighting: equalizes effective gradient signal across
        # all sigma so neither tiny nor huge sigma dominates training.
        weight = (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA)**2
        per_sample = ((D - x0) ** 2).sum(dim=-1)
        loss = (weight * per_sample).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}")

    return model


# ---------------------------------------------------------------------------
# Heun sampler with the Karras sigma schedule.
# This is how you'd "sample" from the model the normal way -- nothing to do
# with SDS. Included so we can verify the denoiser learned the data.
# ---------------------------------------------------------------------------
@torch.no_grad()
def heun_sample(model, n=1000, num_steps=18):
    # Karras sigma schedule: power-rho interpolation. Spends more steps near
    # small sigma where details matter, fewer near sigma_max where structure
    # is being decided coarsely.
    i = torch.arange(num_steps, device=DEVICE)
    sigmas = (SIGMA_MAX ** (1/RHO)
              + i / (num_steps - 1)
              * (SIGMA_MIN ** (1/RHO) - SIGMA_MAX ** (1/RHO))) ** RHO
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])     # final sigma = 0

    x = sigmas[0] * torch.randn(n, 2, device=DEVICE)       # start at sigma_max

    for k in range(num_steps):
        s_cur = sigmas[k]
        s_nxt = sigmas[k + 1]

        # Euler step using the EDM ODE: dx/dsigma = (x - D(x; sigma)) / sigma
        D_cur = model(x, s_cur.expand(n))
        d_cur = (x - D_cur) / s_cur
        x_next = x + (s_nxt - s_cur) * d_cur

        # 2nd-order Heun correction (skip on the final step where s_nxt == 0)
        if s_nxt > 0:
            D_nxt = model(x_next, s_nxt.expand(n))
            d_nxt = (x_next - D_nxt) / s_nxt
            x_next = x + (s_nxt - s_cur) * 0.5 * (d_cur + d_nxt)

        x = x_next

    return x


# ---------------------------------------------------------------------------
# SDS with the EDM denoiser.
#
# Derivation (the part that makes EDM-SDS so much cleaner than DDPM-SDS):
#
#   x_t = theta + sigma * eps
#   eps_pred = (x_t - D(x_t; sigma)) / sigma
#   eps_pred - eps
#       = (theta + sigma*eps - D) / sigma  -  eps
#       = (theta - D) / sigma  +  eps  -  eps
#       = (theta - D) / sigma             <- eps cancels EXACTLY, not just
#                                            in expectation. Per-sample.
#
# So the SDS gradient, with weighting w(sigma) = sigma, is just:
#
#       grad = w(sigma) * (eps_pred - eps) * dx/dtheta
#            = sigma * (theta - D) / sigma * I        (identity renderer)
#            = theta - D
#
# Update rule: theta -= lr * (theta - D), i.e. theta moves toward what the
# denoiser thinks the clean version was. Iterating drives theta onto the
# data manifold.
# ---------------------------------------------------------------------------
def run_sds(model, theta_init, steps=500, lr=0.05,
            sigma_lo=0.1, sigma_hi=1.5):
    theta = torch.tensor(theta_init, dtype=torch.float32, device=DEVICE)
    traj = [theta.detach().cpu().numpy().copy()]

    log_lo, log_hi = math.log(sigma_lo), math.log(sigma_hi)
    for _ in range(steps):
        # Sample sigma log-uniformly in a useful middle range. (Avoiding
        # tiny sigma -- D ~ theta, no info -- and huge sigma -- D ~ data
        # mean, pulls toward 0 instead of the manifold.)
        sigma = torch.exp(
            torch.empty(1, device=DEVICE).uniform_(log_lo, log_hi)
        )                                                       # (1,)
        eps = torch.randn(2, device=DEVICE)
        x_t = theta + sigma * eps

        with torch.no_grad():
            D = model(x_t[None], sigma)[0]                      # (2,)

        grad = theta - D                                        # see derivation
        theta = theta - lr * grad

        traj.append(theta.detach().cpu().numpy().copy())

    return np.stack(traj)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot(model, samples, trajectories, save_to):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---- LEFT: Heun samples vs data (sanity-check that training worked) ----
    ax = axes[0]
    data = sample_data(2000).cpu().numpy()
    ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.2, color='gray',
               label='training data')
    samp_np = samples.cpu().numpy()
    ax.scatter(samp_np[:, 0], samp_np[:, 1], s=8, alpha=0.4, color='tab:blue',
               label='Heun samples (18 steps)')
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect('equal')
    ax.set_title("Trained EDM denoiser sampling (proof of life)")
    ax.legend(loc='upper right', fontsize=9)

    # ---- RIGHT: score field + SDS trajectories ----
    ax = axes[1]
    ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.2, color='gray',
               label='training data')

    # score(x; sigma) = (D(x; sigma) - x) / sigma^2 -- points toward data
    grid = torch.linspace(-2.5, 2.5, 22, device=DEVICE)
    X, Y = torch.meshgrid(grid, grid, indexing='xy')
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    sigma_vis = 0.5
    sigma_t = torch.full((pts.shape[0],), sigma_vis, device=DEVICE)
    with torch.no_grad():
        D_grid = model(pts, sigma_t)
    score = (D_grid - pts) / sigma_vis ** 2
    U = score[:, 0].reshape(X.shape).cpu().numpy()
    V = score[:, 1].reshape(Y.shape).cpu().numpy()
    ax.quiver(X.cpu().numpy(), Y.cpu().numpy(), U, V, color='steelblue',
              alpha=0.4, scale=80, width=0.003,
              label=f'score (D−x)/σ² at σ={sigma_vis}')

    colors = ['C1', 'C2', 'C3', 'C4']
    for traj, c in zip(trajectories, colors):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=c, alpha=0.6, linewidth=1)
        ax.plot(traj[0, 0],  traj[0, 1],  'o', color=c, markersize=10)
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=c, markersize=16,
                markeredgecolor='black')

    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')
    ax.set_title("EDM-SDS:  θ ← θ − lr · (θ − D(θ+σε; σ))")
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches='tight')
    print(f"Saved {save_to}")


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)

    print("Training EDM denoiser on 2D unit circle...")
    model = train()
    model.eval()

    print("\nGenerating Heun samples (proof the model learned the data)...")
    samples = heun_sample(model, n=1000)

    print("\nRunning SDS from 4 starting points...")
    starts = [(2.0, 1.6), (-1.8, 1.2), (0.2, -2.2), (-2.2, -0.6)]
    trajectories = [run_sds(model, s) for s in starts]

    print("\nFinal SDS positions (should be near unit circle):")
    for s, traj in zip(starts, trajectories):
        end = traj[-1]
        print(f"  start {s}  ->  end ({end[0]:+.3f}, {end[1]:+.3f})  "
              f"|theta|={np.linalg.norm(end):.3f}")

    plot(model, samples, trajectories, save_to="experimental/edm_2d.png")


if __name__ == "__main__":
    main()
