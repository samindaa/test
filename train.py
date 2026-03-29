# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0",
#   "torchvision",
#   "matplotlib",
# ]
# ///
"""
Train an FSQ-VAE + learnable conditional prior on MNIST.
Run: uv run train.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fsq import FSQVAE, ConditionalPrior

SPATIAL_POSITIONS = 49  # 7×7 grid cells per image


def train():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4096 codes; pre/post-quant projections keep encoder at 128 channels
    levels = [8, 8, 8]
    model = FSQVAE(levels, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"FSQ codebook size: {model.fsq.codebook_size}")
    print(f"Codes per image:   {SPATIAL_POSITIONS} (7×7 spatial grid)")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("./data",
                                   train=True,
                                   download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=256,
                              shuffle=True,
                              num_workers=0)

    # -------------------------------------------------------------------------
    # Phase 1: Train the FSQ-VAE
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Train FSQ-VAE ---")
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = F.binary_cross_entropy(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        model.eval()
        all_indices = []
        with torch.no_grad():
            for x, _ in DataLoader(train_dataset,
                                   batch_size=1024,
                                   num_workers=0):
                _, idx = model(x.to(device))  # idx: (B, 49)
                all_indices.append(idx.cpu())
        # Count unique codes across all spatial positions
        usage = torch.cat(all_indices).unique().numel()
        print(
            f"Epoch {epoch:2d} | Recon loss: {avg_loss:.4f} | Codes used: {usage}/{model.fsq.codebook_size}"
        )

    torch.save(model.state_dict(), "fsq_vae_mnist.pt")

    # -------------------------------------------------------------------------
    # Reconstruction visualization
    # -------------------------------------------------------------------------
    n_show = 8
    sample_x, _ = next(
        iter(DataLoader(train_dataset, batch_size=n_show, shuffle=True)))
    model.eval()
    with torch.no_grad():
        recon, _ = model(sample_x.to(device))
    originals = sample_x.squeeze(1).numpy()  # (8, 28, 28)
    reconstructed = recon.cpu().clamp(0, 1).squeeze(1).numpy()  # (8, 28, 28)

    _, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3))
    for i in range(n_show):
        axes[0, i].imshow(originals[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[1, 0].set_ylabel("Recon", fontsize=8)
    plt.tight_layout()
    plt.savefig("reconstructions.png", bbox_inches="tight")
    plt.show()
    print("Saved reconstructions.png")

    # -------------------------------------------------------------------------
    # Phase 2: Train the learnable conditional prior P(code | class)
    # With spatial quantization each image yields 49 (label, code) pairs,
    # giving 49× more training signal for the prior.
    # -------------------------------------------------------------------------
    # print("\n--- Phase 2: Train Conditional Prior ---")
    # model.eval()
    # for p in model.parameters():
    #     p.requires_grad_(False)

    # prior = ConditionalPrior(
    #     codebook_size=model.fsq.codebook_size,
    #     num_classes=10,
    #     hidden_dim=256,
    # ).to(device)
    # prior_optimizer = torch.optim.Adam(prior.parameters(), lr=1e-3)

    # for epoch in range(1, 11):
    #     prior.train()
    #     total_loss = 0.0
    #     for x, labels in train_loader:
    #         x, labels = x.to(device), labels.to(device)

    #         with torch.no_grad():
    #             _, target_indices = model(x)   # (B, 49)

    #         # Expand labels to match every spatial position: (B,) → (B*49,)
    #         labels_expanded = labels.unsqueeze(1).expand(-1, SPATIAL_POSITIONS).reshape(-1)
    #         target_flat = target_indices.reshape(-1)   # (B*49,)

    #         loss = prior.loss(labels_expanded, target_flat)
    #         prior_optimizer.zero_grad()
    #         loss.backward()
    #         prior_optimizer.step()
    #         total_loss += loss.item()

    #     print(f"Epoch {epoch:2d} | Prior CE loss: {total_loss / len(train_loader):.4f}")

    # torch.save(prior.state_dict(), "conditional_prior_mnist.pt")
    # print("\nSaved fsq_vae_mnist.pt and conditional_prior_mnist.pt")

    # -------------------------------------------------------------------------
    # Demo: conditional generation — sample digit "3"
    # Sample one code index per spatial position, then decode.
    # -------------------------------------------------------------------------
    # print("\n--- Conditional generation demo (digit=3) ---")
    # prior.eval()
    # n_samples = 8
    # label_val = 3
    # labels = torch.full((n_samples * SPATIAL_POSITIONS,), fill_value=label_val,
    #                     dtype=torch.long, device=device)

    # # Sample 49 code indices per image
    # sampled_indices = prior.sample(labels)                          # (8*49,)
    # sampled_indices = sampled_indices.reshape(n_samples, SPATIAL_POSITIONS)  # (8, 49)

    # # Decode: indices → FSQ codes → post_quant → decoder
    # fsq_dim = model.fsq.dim
    # codes = model.fsq.indices_to_codes(sampled_indices.reshape(-1))      # (8*49, fsq_dim)
    # codes = codes.reshape(n_samples, 7, 7, fsq_dim).permute(0, 3, 1, 2) # (8, fsq_dim, 7, 7)
    # codes = model.post_quant(codes)                                       # (8, hidden_dim, 7, 7)
    # generated = model.decoder(codes)                                      # (8, 1, 28, 28)
    # print(f"Generated shape: {generated.shape}  (8 images of digit {label_val})")

    # imgs = generated.detach().cpu().clamp(0, 1).squeeze(1)  # (8, 28, 28)
    # fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 1.5, 2))
    # for ax, img in zip(axes, imgs):
    #     ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
    #     ax.axis("off")
    # fig.suptitle(f"Generated digit: {label_val}", y=1.02)
    # plt.tight_layout()
    # plt.savefig("generated_samples.png", bbox_inches="tight")
    # plt.show()
    # print("Saved generated_samples.png")


if __name__ == "__main__":
    train()
