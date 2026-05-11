"""
Step 1 of the MNIST SDS demo: train the diffusion model and save a checkpoint.

Run:
    python experimental/train_diffusion_mnist.py

Outputs:
    experimental/mnist_data/             -- MNIST download (one-time)
    experimental/ckpt_diffusion_mnist.pt -- TinyEpsNetMNIST state_dict

The training loop is identical to train_diffusion_2d.py -- only the data
shape and model size differ.
"""

import os
import torch

from diffusion_mnist import TinyEpsNetMNIST, T, DEVICE, get_data, add_noise


CKPT_PATH = "experimental/ckpt_diffusion_mnist.pt"
DIGIT_CLASS = 8                      # train on a single digit; SDS will recreate it


def train(steps=4000, batch=128, lr=1e-3):
    print(f"Device: {DEVICE}")
    print(f"Loading MNIST (digit class = {DIGIT_CLASS})...")
    data = get_data(DIGIT_CLASS)        # already on DEVICE
    print(f"  {data.shape[0]} images of class {DIGIT_CLASS}")

    model = TinyEpsNetMNIST().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch,), device=DEVICE)
        x0 = data[idx]
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
    model = train()
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nSaved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    main()
