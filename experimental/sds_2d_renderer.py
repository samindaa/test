"""
SDS through a renderer. Theta is now a SCALAR (1D parameter); the renderer
maps it into 2D image space:

        x(theta) = ( cos(theta), sin(theta) + 0.5 * theta )

This adds the missing piece from the basic demo: the chain rule through
the renderer. SDS produces a "gradient on x" -- (eps_pred - eps) -- but
the parameter we actually want to update is theta, not x. So we multiply
by dx/dtheta. THAT is the only Jacobian SDS ever computes; the diffusion
model's Jacobian is still skipped.

This is the smallest possible illustration of how a NeRF (or any other
differentiable renderer) plugs into SDS.

Note: x(theta) is a curve in 2D that touches the unit circle only at
theta = 0 (giving x = (1, 0)). So SDS should drive theta toward 0 from
any starting value -- that is the only theta whose rendered output lies
on the data manifold.

Run:
    python experimental/sds_2d_renderer.py

Reads:
    experimental/ckpt_diffusion_2d.pt   (the unit-circle checkpoint)
Writes:
    experimental/sds_2d_renderer.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion_2d import (
    TinyEpsNet, DEVICE, sample_data, alpha_bars, sqrt_ab, sqrt_1mab,
)


CKPT_PATH = "experimental/ckpt_diffusion_2d.pt"
PLOT_PATH = "experimental/sds_2d_renderer.png"


# ---------------------------------------------------------------------------
# Renderer + its Jacobian.
#   x(theta)        = (cos(theta), sin(theta) + 0.5*theta)
#   dx/dtheta(theta) = (-sin(theta), cos(theta) + 0.5)
# Both written by hand here so the chain rule is fully explicit.
# ---------------------------------------------------------------------------
def render(theta):
    return torch.stack([torch.cos(theta), torch.sin(theta) + 0.5 * theta])


def render_jacobian(theta):
    return torch.stack([-torch.sin(theta), torch.cos(theta) + 0.5])


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


def run_sds(model, theta_init, steps=600, lr=0.05, t_min=20, t_max=80):
    theta = torch.tensor(float(theta_init), dtype=torch.float32, device=DEVICE)
    theta_traj = [theta.item()]
    x_traj = [render(theta).detach().cpu().numpy().copy()]

    for _ in range(steps):
        x = render(theta)                                 # 1) render: theta -> x

        t = torch.randint(t_min, t_max, (1,), device=DEVICE)
        eps = torch.randn(2, device=DEVICE)
        x_t = sqrt_ab[t] * x + sqrt_1mab[t] * eps         # 2) smudge x

        with torch.no_grad():
            eps_pred = model(x_t[None], t)[0]             # 3) ask the critic

        w = (1 - alpha_bars[t])
        grad_x = w * (eps_pred - eps)                     # 4) "wrongness on x"

        # 5) Chain rule: dL/dtheta = (dL/dx) . (dx/dtheta)
        # For 1D theta, this is just a dot product (scalar result).
        jac = render_jacobian(theta)                       # shape (2,)
        grad_theta = (grad_x * jac).sum()                  # scalar

        theta = theta - lr * grad_theta                    # 6) step in theta-space

        theta_traj.append(theta.item())
        x_traj.append(render(theta).detach().cpu().numpy().copy())

    return np.array(theta_traj), np.stack(x_traj)


def plot(model, runs, starts, save_to):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ---- LEFT panel: image space (where the diffusion model lives) ----
    ax = axes[0]
    data = sample_data(2000).cpu().numpy()
    ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.25, color='gray',
               label='data (unit circle)')

    # The renderer is a 1-parameter curve in 2D. Draw it so you can see
    # what x values are even reachable by varying theta.
    theta_curve = torch.linspace(-4, 4, 300, device=DEVICE)
    x_curve = torch.stack(
        [torch.cos(theta_curve), torch.sin(theta_curve) + 0.5 * theta_curve],
        dim=1,
    ).cpu().numpy()
    ax.plot(x_curve[:, 0], x_curve[:, 1], '--', color='purple', alpha=0.6,
            label='renderer image: { x(θ) : θ ∈ ℝ }')

    # Mark the only intersection of the curve with the data manifold.
    ax.plot(1.0, 0.0, marker='P', color='black', markersize=12, zorder=5,
            label='x(0) = (1, 0)  ← the only θ on the circle')

    colors = ['C1', 'C2', 'C3', 'C4']
    for (theta_t, x_t), c in zip(runs, colors):
        ax.plot(x_t[:, 0], x_t[:, 1], '-', color=c, alpha=0.6, linewidth=1)
        ax.plot(x_t[0, 0],  x_t[0, 1],  'o', color=c, markersize=10)
        ax.plot(x_t[-1, 0], x_t[-1, 1], '*', color=c, markersize=16,
                markeredgecolor='black')

    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    ax.set_title("Image space: rendered point x(θ)")
    ax.legend(loc='upper right', fontsize=8)

    # ---- RIGHT panel: theta space (1D parameter trajectory) ----
    ax = axes[1]
    for (theta_t, _), c, s in zip(runs, colors, starts):
        ax.plot(theta_t, color=c, label=f'start θ={s:+.1f}')
        ax.plot(0, theta_t[0], 'o', color=c, markersize=10)
        ax.plot(len(theta_t) - 1, theta_t[-1], '*', color=c, markersize=14,
                markeredgecolor='black')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6,
               label='θ = 0 (only solution)')
    ax.set_xlabel('SDS step')
    ax.set_ylabel('θ value')
    ax.set_title("Parameter space: θ(t)")
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle("SDS through a renderer:  theta -> x -> diffusion critic", y=1.02)
    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches='tight')
    print(f"Saved {save_to}")


def main():
    torch.manual_seed(0)
    print(f"Device: {DEVICE}")
    print(f"Loading circle checkpoint from {CKPT_PATH}...")
    model = load_model()

    print("Running SDS in θ-space (1D), routed through the renderer...")
    starts = [2.5, 1.0, -1.0, -2.5]
    runs = [run_sds(model, s) for s in starts]

    print("\nFinal θ values (should approach 0):")
    for s, (theta_t, _) in zip(starts, runs):
        print(f"  start θ = {s:+.2f}  ->  end θ = {theta_t[-1]:+.3f}")

    plot(model, runs, starts, save_to=PLOT_PATH)


if __name__ == "__main__":
    main()
