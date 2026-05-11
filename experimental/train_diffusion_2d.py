"""
Step 1 of the SDS demo: train the diffusion model and save a checkpoint.

Run:
    python experimental/train_diffusion_2d.py

Outputs:
    experimental/ckpt_diffusion_2d.pt   -- TinyEpsNet state_dict
"""

import os
import torch

from diffusion_2d import TinyEpsNet, T, DEVICE, sample_data, add_noise


CKPT_PATH = "experimental/ckpt_diffusion_2d.pt"


def train(steps=3000, batch=512, lr=1e-3):
    model = TinyEpsNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        x0 = sample_data(batch)
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
    print("Training tiny diffusion model on 2D circle...")
    model = train()

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nSaved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    main()
