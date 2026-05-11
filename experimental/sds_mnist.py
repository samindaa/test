"""
Step 2 of the MNIST SDS demo: load the trained diffusion model, run SDS to
optimize a 28x28 image starting from random noise, and save snapshots.

Run:
    python experimental/sds_mnist.py

Reads:
    experimental/ckpt_diffusion_mnist.pt   (from train_diffusion_mnist.py)
Writes:
    experimental/sds_mnist.png             (rows = runs, cols = snapshots)

The run_sds() body is the SAME six lines as in sds_2d.py. The only thing
that changed is theta is now a 784-dim image instead of a 2D point. That's
the whole takeaway: SDS is an algorithm on parameters, not on images.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion_mnist import (
    TinyEpsNetMNIST, DEVICE, alpha_bars, sqrt_ab, sqrt_1mab,
    IMG_SIZE, IMG_DIM,
)


CKPT_PATH = "experimental/ckpt_diffusion_mnist.pt"
PLOT_PATH = "experimental/sds_mnist.png"


def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. "
            f"Run `python experimental/train_diffusion_mnist.py` first."
        )
    model = TinyEpsNetMNIST().to(DEVICE)
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


def run_sds(model, theta_init, steps=2000, lr=0.5,
            t_min=20, t_max=180, snapshot_at=None):
    """Returns {step: theta_snapshot} for the requested step indices."""
    if snapshot_at is None:
        snapshot_at = [0, steps // 8, steps // 4, steps // 2, steps]
    snapshot_at = set(snapshot_at)

    theta = theta_init.clone()
    snaps = {0: theta.clone()} if 0 in snapshot_at else {}

    for s in range(1, steps + 1):
        t = torch.randint(t_min, t_max, (1,), device=DEVICE)
        eps = torch.randn(IMG_DIM, device=DEVICE)        # the smudge
        x_t = sqrt_ab[t] * theta + sqrt_1mab[t] * eps    # smudged theta

        with torch.no_grad():
            eps_pred = model(x_t[None], t)[0]            # critic's guess

        w = (1 - alpha_bars[t])
        grad = w * (eps_pred - eps)                      # SDS "gradient"
        theta = theta - lr * grad                        # identity renderer

        if s in snapshot_at:
            snaps[s] = theta.clone()

    return snaps


def to_image(theta):
    img = theta.detach().cpu().view(IMG_SIZE, IMG_SIZE).numpy()
    return np.clip((img + 1) / 2, 0, 1)                  # [-1,1] -> [0,1]


def plot(runs, save_to):
    n_runs = len(runs)
    snap_steps = sorted(runs[0].keys())
    n_cols = len(snap_steps)

    fig, axes = plt.subplots(n_runs, n_cols,
                             figsize=(2 * n_cols, 2 * n_runs))
    if n_runs == 1:
        axes = axes[None]

    for i, snaps in enumerate(runs):
        for j, s in enumerate(snap_steps):
            axes[i, j].imshow(to_image(snaps[s]), cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f"step {s}")
        axes[i, 0].set_ylabel(f"run {i}", rotation=0, labelpad=25)

    fig.suptitle("SDS on MNIST: noise -> digit, one row per run", y=1.0)
    plt.tight_layout()
    plt.savefig(save_to, dpi=120, bbox_inches='tight')
    print(f"Saved {save_to}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint from {CKPT_PATH}...")
    model = load_model()

    print("Running SDS from 4 random initializations...")
    runs = []
    for i in range(4):
        torch.manual_seed(100 + i)
        theta_init = 0.1 * torch.randn(IMG_DIM, device=DEVICE)
        snaps = run_sds(model, theta_init)
        runs.append(snaps)
        print(f"  run {i} done")

    plot(runs, PLOT_PATH)


if __name__ == "__main__":
    main()
