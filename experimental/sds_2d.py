"""
Step 2 of the SDS demo: load the trained diffusion model and run SDS.

Run:
    python experimental/sds_2d.py

Reads:
    experimental/ckpt_diffusion_2d.pt   (produced by train_diffusion_2d.py)
Writes:
    experimental/sds_2d.png             (data + score field + trajectories)

The whole point of this script is the run_sds() function below. Each step:
    1) noise theta to get x_t
    2) ask the diffusion model what noise it sees
    3) treat (eps_pred - eps) as the gradient on theta
    4) step theta against it -- no backprop through the model
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion_2d import (
    TinyEpsNet, DEVICE, sample_data, alpha_bars, sqrt_ab, sqrt_1mab,
)


CKPT_PATH = "experimental/ckpt_diffusion_2d.pt"
PLOT_PATH = "experimental/sds_2d.png"


def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. "
            f"Run `python experimental/train_diffusion_2d.py` first."
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
        eps = torch.randn(2, device=DEVICE)             # the smudge
        x_t = sqrt_ab[t] * theta + sqrt_1mab[t] * eps   # smudged theta

        with torch.no_grad():
            eps_pred = model(x_t[None], t)[0]           # critic's guess

        # The SDS "gradient". DreamFusion uses w(t) = (1 - abar_t); using
        # it here keeps small-t (low-noise) steps from dominating.
        w = (1 - alpha_bars[t])
        grad = w * (eps_pred - eps)

        # Identity renderer => no chain rule. With a renderer r(theta) we
        # would multiply by dr/dtheta here (that is the only Jacobian SDS
        # ever computes; the diffusion model's Jacobian is skipped).
        theta = theta - lr * grad
        traj.append(theta.detach().cpu().numpy().copy())

    return np.stack(traj)


def plot(model, trajectories, save_to):
    fig, ax = plt.subplots(figsize=(7, 7))

    data = sample_data(2000).cpu().numpy()
    ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.25, color='gray',
               label='training data (unit circle)')

    # Visualize what the model "knows": the score field at a chosen t.
    # score(x_t) = -eps_pred / sqrt(1 - abar_t)  -- points toward data.
    grid = torch.linspace(-2.5, 2.5, 22, device=DEVICE)
    X, Y = torch.meshgrid(grid, grid, indexing='xy')
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    t_vis = torch.full((pts.shape[0],), 40, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        score = -model(pts, t_vis) / sqrt_1mab[40]
    U = score[:, 0].reshape(X.shape).cpu().numpy()
    V = score[:, 1].reshape(Y.shape).cpu().numpy()
    ax.quiver(X.cpu().numpy(), Y.cpu().numpy(), U, V, color='steelblue',
              alpha=0.4, scale=60, width=0.003,
              label='learned score (toward data)')

    colors = ['C1', 'C2', 'C3', 'C4']
    for traj, c in zip(trajectories, colors):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=c, alpha=0.6, linewidth=1)
        ax.plot(traj[0, 0],  traj[0, 1],  'o', color=c, markersize=10)
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=c, markersize=16,
                markeredgecolor='black')

    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')
    ax.set_title("SDS pulls free parameters onto the data manifold\n"
                 "circles = start, stars = end of optimization")
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_to, dpi=120)
    print(f"Saved {save_to}")


def main():
    torch.manual_seed(0)
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint from {CKPT_PATH}...")
    model = load_model()

    print("Running SDS from 4 different starting points...")
    starts = [(2.0, 1.6), (-1.8, 1.2), (0.2, -2.2), (-2.2, -0.6)]
    trajectories = [run_sds(model, s) for s in starts]

    print("\nFinal positions (should be near the unit circle):")
    for s, traj in zip(starts, trajectories):
        end = traj[-1]
        print(f"  start {s}  ->  end ({end[0]:+.3f}, {end[1]:+.3f})  "
              f"|theta|={np.linalg.norm(end):.3f}")

    plot(model, trajectories, save_to=PLOT_PATH)


if __name__ == "__main__":
    main()
