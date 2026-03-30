# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "torchvision",
#   "matplotlib",
# ]
# ///
"""
Train an FSQ-VAE on MNIST.
Run: uv run train.py
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fsq import FSQVAE

SPATIAL_POSITIONS = 49  # 7×7 grid cells per image


def train():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4096 codes; pre/post-quant projections keep encoder at 128 channels
    levels = [8, 8, 8]
    model = FSQVAE(levels, hidden_dim=128).to(device)

    print(f"FSQ codebook size: {model.fsq.codebook_size}")
    print(f"Codes per image:   {SPATIAL_POSITIONS} (7×7 spatial grid)")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("./data",
                                   train=True,
                                   download=True,
                                   transform=transform)

    checkpoint = "cvae_fsq_mnist.pt"
    if os.path.exists(checkpoint):
        print(f"\nLoading existing model from {checkpoint}, skipping training.")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=256,
                                  shuffle=True,
                                  num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # -------------------------------------------------------------------------
        # Train the CVAE-FSQ
        # -------------------------------------------------------------------------
        n_epochs    = 30
        beta_max    = 0.5
        warmup_epochs = 10  # β ramps 0 → beta_max over first 10 epochs

        print("\n--- Train CVAE-FSQ ---")
        for epoch in range(1, n_epochs + 1):
            beta = beta_max * min(1.0, epoch / warmup_epochs)
            model.train()
            total_bce = 0.0
            total_kl  = 0.0
            for x, labels in train_loader:
                x, labels = x.to(device), labels.to(device)
                x_hat, _, mu, logvar = model(x, labels)
                bce  = F.binary_cross_entropy(x_hat, x)
                kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = bce + beta * kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_bce += bce.item()
                total_kl  += kl.item()

            model.eval()
            all_indices = []
            with torch.no_grad():
                for x, labels in DataLoader(train_dataset,
                                            batch_size=1024,
                                            num_workers=0):
                    _, idx, _, _ = model(x.to(device), labels.to(device))
                    all_indices.append(idx.cpu())
            usage = torch.cat(all_indices).unique().numel()
            print(
                f"Epoch {epoch:2d} | β={beta:.2f} | BCE: {total_bce/len(train_loader):.4f} | "
                f"KL: {total_kl/len(train_loader):.4f} | "
                f"Codes: {usage}/{model.fsq.codebook_size}"
            )

        torch.save(model.state_dict(), checkpoint)

    # -------------------------------------------------------------------------
    # Visualization: 10 columns (digits 0-9), 3 rows
    #   Row 0: one original image per digit
    #   Row 1: reconstruction
    #   Row 2: generated from label alone (z ~ N(0,I))
    # -------------------------------------------------------------------------
    model.eval()
    num_classes = 10

    # Gather one example per digit from the dataset
    per_class = {}
    for x, y in DataLoader(train_dataset, batch_size=512, num_workers=0):
        for img, label in zip(x, y):
            c = label.item()
            if c not in per_class:
                per_class[c] = img
            if len(per_class) == num_classes:
                break
        if len(per_class) == num_classes:
            break

    class_x      = torch.stack([per_class[c] for c in range(num_classes)])  # (10, 1, 28, 28)
    class_labels = torch.arange(num_classes)                                 # (10,)

    with torch.no_grad():
        recon, _, _, _ = model(class_x.to(device), class_labels.to(device))
        generated      = model.sample(class_labels, device)

    originals     = class_x.squeeze(1).numpy()
    reconstructed = recon.cpu().clamp(0, 1).squeeze(1).numpy()
    generated_np  = generated.cpu().clamp(0, 1).squeeze(1).numpy()

    _, axes = plt.subplots(3, num_classes, figsize=(num_classes * 1.5, 4.5))
    for c in range(num_classes):
        axes[0, c].imshow(originals[c],     cmap="gray", vmin=0, vmax=1)
        axes[0, c].axis("off")
        axes[0, c].set_title(str(c), fontsize=9)
        axes[1, c].imshow(reconstructed[c], cmap="gray", vmin=0, vmax=1)
        axes[1, c].axis("off")
        axes[2, c].imshow(generated_np[c],  cmap="gray", vmin=0, vmax=1)
        axes[2, c].axis("off")
    axes[0, 0].set_ylabel("Original",  fontsize=8)
    axes[1, 0].set_ylabel("Recon",     fontsize=8)
    axes[2, 0].set_ylabel("Generated", fontsize=8)
    plt.tight_layout()
    plt.savefig("reconstructions.png", bbox_inches="tight")
    plt.show()
    print("Saved reconstructions.png")


if __name__ == "__main__":
    train()
