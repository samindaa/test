"""
SDS on the two-cluster diffusion model -- demonstrates mode-seeking.

Each starting point gets pulled toward the NEAREST cluster, not toward an
average of both. That is the core "mode-seeking" property of SDS that makes
DreamFusion outputs look like one canonical example rather than a blend.

Run:
    python experimental/sds_2d_clusters.py

Reads:
    experimental/ckpt_diffusion_2d_clusters.pt
Writes:
    experimental/sds_2d_clusters.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion_2d import (
    TinyEpsNet, DEVICE, alpha_bars, sqrt_ab, sqrt_1mab,
)
from train_diffusion_2d_clusters import sample_clusters, CENTERS


CKPT_PATH = "experimental/ckpt_diffusion_2d_clusters.pt"
PLOT_PATH = "experimental/sds_2d_clusters.png"


def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Run "
            f"`python experimental/train_diffusion_2d_clusters.py` first."
        )
    model = TinyEpsNet().to(DEVICE)
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


def run_sds(model, theta_init, steps=400, lr=0.05, t_min=20, t_max=80):
    theta = torch.tensor(theta_init, dtype=torch.float32, device=DEVICE)
    traj = [theta.detach().cpu().numpy().copy()]

    for _ in range(steps):
        t = torch.randint(t_min, t_max, (1,), device=DEVICE)
        eps = torch.randn(2, device=DEVICE)
        x_t = sqrt_ab[t] * theta + sqrt_1mab[t] * eps

        with torch.no_grad():
            eps_pred = model(x_t[None], t)[0]

        w = (1 - alpha_bars[t])
        grad = w * (eps_pred - eps)
        theta = theta - lr * grad
        traj.append(theta.detach().cpu().numpy().copy())

    return np.stack(traj)


def plot(model, trajectories, save_to):
    fig, ax = plt.subplots(figsize=(7, 7))

    data = sample_clusters(2000).cpu().numpy()
    ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.25, color='gray',
               label='training data (two clusters)')

    # Score field
    grid = torch.linspace(-3, 3, 24, device=DEVICE)
    X, Y = torch.meshgrid(grid, grid, indexing='xy')
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    t_vis = torch.full((pts.shape[0],), 40, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        score = -model(pts, t_vis) / sqrt_1mab[40]
    U = score[:, 0].reshape(X.shape).cpu().numpy()
    V = score[:, 1].reshape(Y.shape).cpu().numpy()
    ax.quiver(X.cpu().numpy(), Y.cpu().numpy(), U, V, color='steelblue',
              alpha=0.4, scale=80, width=0.003,
              label='learned score (toward nearest mode)')

    # Color trajectories by which cluster they end up nearest to.
    centers_np = CENTERS.cpu().numpy()
    for traj in trajectories:
        end = traj[-1]
        d_left = np.linalg.norm(end - centers_np[0])
        d_right = np.linalg.norm(end - centers_np[1])
        c = 'tab:red' if d_left < d_right else 'tab:blue'
        ax.plot(traj[:, 0], traj[:, 1], '-', color=c, alpha=0.5, linewidth=1)
        ax.plot(traj[0, 0],  traj[0, 1],  'o', color=c, markersize=8)
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=c, markersize=14,
                markeredgecolor='black')

    # Mark the cluster centers
    ax.scatter(centers_np[:, 0], centers_np[:, 1], marker='X', s=200,
               color='black', zorder=5, label='cluster centers')

    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    ax.set_title("Mode-seeking SDS\n"
                 "starts (circles) collapse to the NEAREST mode (stars), "
                 "never to the average")
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_to, dpi=120)
    print(f"Saved {save_to}")


def main():
    torch.manual_seed(0)
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint from {CKPT_PATH}...")
    model = load_model()

    # Spread starts around the plane to show the mode-seeking effect clearly.
    print("Running SDS from 10 starting points...")
    starts = [
        (-2.5,  1.5), (-1.0,  2.0), ( 0.5,  2.0), ( 2.0,  2.0),
        ( 2.5, -1.5), ( 1.0, -2.0), (-0.5, -2.0), (-2.0, -2.0),
        (-0.2,  0.0), ( 0.2,  0.0),  # near the saddle between modes
    ]
    trajectories = [run_sds(model, s) for s in starts]

    print("\nFinal positions and which mode they chose:")
    for s, traj in zip(starts, trajectories):
        end = traj[-1]
        nearest = "LEFT" if end[0] < 0 else "RIGHT"
        print(f"  start {s}  ->  end ({end[0]:+.3f}, {end[1]:+.3f})  [{nearest}]")

    plot(model, trajectories, save_to=PLOT_PATH)


if __name__ == "__main__":
    main()
