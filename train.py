"""
Train an FSQ-VAE + learnable conditional prior on MNIST.
Run: conda run -n py311_env python train.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fsq import FSQVAE, ConditionalPrior

SPATIAL_POSITIONS = 49  # 7×7 grid cells per image


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 125 codes — small enough to fill quickly with 49 codes/image
    levels = [5, 5, 5]
    model = FSQVAE(levels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"FSQ codebook size: {model.fsq.codebook_size}")
    print(f"Codes per image:   {SPATIAL_POSITIONS} (7×7 spatial grid)")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)

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
            loss = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        model.eval()
        all_indices = []
        with torch.no_grad():
            for x, _ in DataLoader(train_dataset, batch_size=1024, num_workers=0):
                _, idx = model(x.to(device))          # idx: (B, 49)
                all_indices.append(idx.cpu())
        # Count unique codes across all spatial positions
        usage = torch.cat(all_indices).unique().numel()
        print(f"Epoch {epoch:2d} | Recon loss: {avg_loss:.4f} | Codes used: {usage}/{model.fsq.codebook_size}")

    torch.save(model.state_dict(), "fsq_vae_mnist.pt")

    # -------------------------------------------------------------------------
    # Phase 2: Train the learnable conditional prior P(code | class)
    # With spatial quantization each image yields 49 (label, code) pairs,
    # giving 49× more training signal for the prior.
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Train Conditional Prior ---")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    prior = ConditionalPrior(
        codebook_size=model.fsq.codebook_size,
        num_classes=10,
        hidden_dim=256,
    ).to(device)
    prior_optimizer = torch.optim.Adam(prior.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        prior.train()
        total_loss = 0.0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)

            with torch.no_grad():
                _, target_indices = model(x)   # (B, 49)

            # Expand labels to match every spatial position: (B,) → (B*49,)
            labels_expanded = labels.unsqueeze(1).expand(-1, SPATIAL_POSITIONS).reshape(-1)
            target_flat = target_indices.reshape(-1)   # (B*49,)

            loss = prior.loss(labels_expanded, target_flat)
            prior_optimizer.zero_grad()
            loss.backward()
            prior_optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:2d} | Prior CE loss: {total_loss / len(train_loader):.4f}")

    torch.save(prior.state_dict(), "conditional_prior_mnist.pt")
    print("\nSaved fsq_vae_mnist.pt and conditional_prior_mnist.pt")

    # -------------------------------------------------------------------------
    # Demo: conditional generation — sample digit "3"
    # Sample one code index per spatial position, then decode.
    # -------------------------------------------------------------------------
    print("\n--- Conditional generation demo (digit=3) ---")
    prior.eval()
    n_samples = 8
    label_val = 3
    labels = torch.full((n_samples * SPATIAL_POSITIONS,), fill_value=label_val,
                        dtype=torch.long, device=device)

    # Sample 49 code indices per image
    sampled_indices = prior.sample(labels)                          # (8*49,)
    sampled_indices = sampled_indices.reshape(n_samples, SPATIAL_POSITIONS)  # (8, 49)

    # Decode each spatial code to a D-dim float vector, reshape to (B, D, 7, 7)
    D = model.latent_dim
    codes = model.fsq.indices_to_codes(sampled_indices.reshape(-1))  # (8*49, D)
    codes = codes.reshape(n_samples, 7, 7, D).permute(0, 3, 1, 2)    # (8, D, 7, 7)
    generated = model.decoder(codes)                                   # (8, 1, 28, 28)
    print(f"Generated shape: {generated.shape}  (8 images of digit {label_val})")


if __name__ == "__main__":
    train()
