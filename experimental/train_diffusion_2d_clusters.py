"""
Variant of train_diffusion_2d.py: train on a TWO-CLUSTER 2D distribution
instead of the unit circle. Used by sds_2d_clusters.py to demonstrate
SDS's mode-seeking behavior.

Run:
    python experimental/train_diffusion_2d_clusters.py

Outputs:
    experimental/ckpt_diffusion_2d_clusters.pt
"""

import os
import torch

from diffusion_2d import TinyEpsNet, T, DEVICE, add_noise


CKPT_PATH = "experimental/ckpt_diffusion_2d_clusters.pt"

# Two Gaussian clusters at (-1.5, 0) and (+1.5, 0), small spread.
CENTERS = torch.tensor([[-1.5, 0.0], [1.5, 0.0]], device=DEVICE)
STD = 0.2


def sample_clusters(n):
    pick = torch.randint(0, 2, (n,), device=DEVICE)        # which cluster
    centers = CENTERS[pick]                                # (n, 2)
    return centers + STD * torch.randn(n, 2, device=DEVICE)


def train(steps=3000, batch=512, lr=1e-3):
    model = TinyEpsNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        x0 = sample_clusters(batch)
        t = torch.randint(0, T, (batch,), device=DEVICE)
        eps = torch.randn_like(x0)
        x_t = add_noise(x0, t, eps)
        loss = ((model(x_t, t) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}")

    return model


def main():
    torch.manual_seed(0)
    print(f"Device: {DEVICE}")
    print("Training tiny diffusion model on TWO 2D clusters...")
    model = train()

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nSaved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    main()
