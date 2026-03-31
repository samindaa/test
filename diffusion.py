"""
Diffusion Model with a Small Transformer (DiT-style) on MNIST
=============================================================

Key concepts:
  - Forward process q(x_t | x_{t-1}): add Gaussian noise over T steps
  - Reverse process p_theta(x_{t-1} | x_t): transformer predicts the noise
  - Training: minimize MSE between predicted noise and actual noise added
  - Sampling: start from x_T ~ N(0,I), iteratively denoise to x_0

Architecture: DiT (Diffusion Transformer)
  - Patchify the image into tokens
  - Add sinusoidal timestep embedding to each token (via AdaLN)
  - Stack N transformer blocks
  - Unpatchify back to image
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. DDPM Noise Schedule
# ---------------------------------------------------------------------------

class DDPMSchedule:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021) — much better than linear for MNIST.
    The key fix: linear schedule destroys signal too fast at small t, hurting fine details.
    Cosine keeps more signal at low t and transitions smoothly.
    """
    def __init__(self, T=500, device="cpu"):
        self.T = T
        self.device = device

        # Cosine schedule: alpha_bar_t = cos^2(pi/2 * (t/T + s) / (1 + s))
        s = 0.008
        steps = torch.arange(T + 1, device=device, dtype=torch.float32)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]                    # normalize so alpha_bar_0 = 1
        alpha_bar = alpha_bar[1:]               # shape: (T,), indexed 0..T-1

        betas = 1 - alpha_bar / F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        betas = betas.clamp(0, 0.999)

        # alpha_t = 1 - beta_t
        alphas = 1.0 - betas

        # alpha_bar_t = product of alphas up to t  (used for q(x_t | x_0))
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Store as buffers
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar                          # shape: (T,)
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()

        # For reverse step
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: sample x_t given x_0 and timestep t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # Gather the scalars for each item in the batch
        sqrt_ab   = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_scalar):
        """
        One reverse step: DDPM Algorithm 2 (Ho et al. 2020).

        Direct formula — does NOT reconstruct x0 first.
        mu_t = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_pred)

        Why the old x0_pred approach was broken:
          x0_pred = (x_t - sqrt(1-ab)*eps) / sqrt(ab)
          With the cosine schedule, alpha_bar[T-1] = cos(π/2)² = 0 exactly,
          and the float32 cumprod underflows to 0 for the last ~30 steps.
          Dividing by 0 gives Inf on the very first sample step → NaN propagates
          through the entire chain → pure noise output.
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_scalar, device=self.device, dtype=torch.long)

        eps_pred = model(x_t, t_tensor)

        # coef = beta_t / sqrt(1 - alpha_bar_t)  — always safe, sqrt(1-ab) ≥ 0
        # 1/sqrt(alpha_t) — safe because betas are clamped to 0.999 so alpha ≥ 0.001
        coef = self.betas[t_scalar] / self.sqrt_one_minus_alpha_bar[t_scalar]
        mean = (x_t - coef * eps_pred) / self.alphas[t_scalar].sqrt()

        if t_scalar == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + self.posterior_variance[t_scalar].sqrt() * noise

    @torch.no_grad()
    def sample(self, model, shape):
        """Full reverse chain: x_T -> x_0"""
        model.eval()
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# 2. Sinusoidal Timestep Embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t, dim):
    """
    Standard sinusoidal positional embedding for timesteps.
    t: (B,) integer timesteps
    Returns: (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args  = t[:, None].float() * freqs[None]         # (B, half)
    emb   = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
    return emb


# ---------------------------------------------------------------------------
# 3. Transformer Block with Adaptive Layer Norm (AdaLN)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """
    Adaptive Layer Norm: modulate scale and shift from a conditioning vector.
    Used in DiT to inject the timestep embedding into each transformer layer.
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Project condition to (scale, shift) — initialized near identity
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        # cond: (B, cond_dim) -> scale, shift: (B, 1, dim)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with AdaLN conditioning.
    """
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = AdaLN(dim, cond_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        # Self-attention
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        # MLP
        x = x + self.mlp(self.norm2(x, cond))
        return x


# ---------------------------------------------------------------------------
# 4. DiT: Diffusion Transformer for MNIST
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Small Diffusion Transformer for 28x28 grayscale images.

    Pipeline:
      x (B,1,28,28)
        -> patchify into (B, N, patch_dim)   where N = (28/p)^2
        -> linear embed to (B, N, dim)
        -> add 2D positional embeddings
        -> N transformer blocks conditioned on timestep embedding
        -> linear project back to (B, N, patch_dim)
        -> unpatchify to (B, 1, 28, 28)  <- predicted noise
    """
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_channels=1,
        dim=128,
        depth=4,
        n_heads=4,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.in_channels = in_channels
        n_patches = (img_size // patch_size) ** 2   # 7*7 = 49 for p=4
        patch_dim = in_channels * patch_size ** 2   # 1*16 = 16

        # Timestep embedding
        t_dim = dim * 4
        self.t_embed = nn.Sequential(
            nn.Linear(dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self._t_sinusoid_dim = dim

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim)

        # Learnable 2D positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, cond_dim=t_dim)
            for _ in range(depth)
        ])

        # Final norm + output projection
        self.norm_out = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, patch_dim)

    def patchify(self, x):
        """(B,C,H,W) -> (B, N, C*p*p)"""
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)             # (B, H/p, W/p, C, p, p)
        x = x.reshape(B, -1, C * p * p)             # (B, N, patch_dim)
        return x

    def unpatchify(self, x, H=28, W=28):
        """(B, N, C*p*p) -> (B, C, H, W)"""
        p  = self.patch_size
        C  = self.in_channels
        nh = H // p
        nw = W // p
        x  = x.reshape(x.shape[0], nh, nw, C, p, p)
        x  = x.permute(0, 3, 1, 4, 2, 5)            # (B, C, nh, p, nw, p)
        x  = x.reshape(x.shape[0], C, H, W)
        return x

    def forward(self, x, t):
        """
        x: (B, 1, 28, 28) noisy image
        t: (B,) integer timesteps
        Returns: predicted noise (B, 1, 28, 28)
        """
        # Timestep conditioning
        t_emb = sinusoidal_embedding(t, self._t_sinusoid_dim)   # (B, dim)
        cond  = self.t_embed(t_emb)                              # (B, t_dim)

        # Patchify + embed
        tokens = self.patchify(x)                  # (B, N, patch_dim)
        tokens = self.patch_embed(tokens)          # (B, N, dim)
        tokens = tokens + self.pos_embed           # add position

        # Transformer
        for block in self.blocks:
            tokens = block(tokens, cond)

        # Output
        tokens = self.norm_out(tokens)
        tokens = self.out_proj(tokens)             # (B, N, patch_dim)
        noise  = self.unpatchify(tokens)           # (B, 1, 28, 28)
        return noise


# ---------------------------------------------------------------------------
# 5. EMA (Exponential Moving Average of model weights)
# ---------------------------------------------------------------------------

class EMA:
    """
    Maintains a shadow copy of model weights smoothed with exponential decay.
    Why it matters: the raw model weights oscillate during training; EMA gives
    a stable average that produces much cleaner samples.
    Typical decay = 0.9999 for long runs, 0.999 for shorter ones.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def copy_to(self, model):
        """Load EMA weights into model for sampling."""
        model.load_state_dict({k: v.to(next(model.parameters()).dtype)
                               for k, v in self.shadow.items()})


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------

def train(
    epochs=50,
    batch_size=256,
    lr=1e-3,
    T=500,
    device=None,
    save_every=10,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Data: normalize to [-1, 1]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    schedule = DDPMSchedule(T=T, device=device)
    model    = DiT(img_size=28, patch_size=4, dim=128, depth=4, n_heads=4).to(device)
    ema      = EMA(model, decay=0.999)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr)
    # Cosine LR with eta_min=1e-5 — keeps learning rate from dying to zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            B = x.shape[0]

            # Sample random timesteps
            t = torch.randint(0, T, (B,), device=device)

            # Forward process: add noise
            x_noisy, noise_true = schedule.q_sample(x, t)

            # Predict noise
            noise_pred = model(x_noisy, t)

            # Simple MSE loss (Ho et al. use this exact loss)
            loss = F.mse_loss(noise_pred, noise_true)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)  # update EMA shadow weights every step

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch:03d} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % save_every == 0 or epoch == epochs:
            # Sample using EMA weights (much better quality than raw model)
            ema_model = DiT(img_size=28, patch_size=4, dim=128, depth=4, n_heads=4).to(device)
            ema.copy_to(ema_model)
            samples = schedule.sample(ema_model, shape=(16, 1, 28, 28))
            samples = (samples + 1) / 2  # back to [0,1]
            save_image(samples, f"diffusion_samples_epoch{epoch:03d}.png", nrow=4)
            print(f"  Saved EMA samples -> diffusion_samples_epoch{epoch:03d}.png")

    torch.save(model.state_dict(), "diffusion_dit_mnist.pt")
    torch.save(ema.shadow, "diffusion_dit_mnist_ema.pt")
    print("Saved model -> diffusion_dit_mnist.pt  (+ EMA weights)")

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("DDPM Training Loss (DiT on MNIST)")
    plt.tight_layout()
    plt.savefig("diffusion_loss.png")
    print("Saved loss curve -> diffusion_loss.png")
    return model, schedule


# ---------------------------------------------------------------------------
# 6. Visualize the forward process (educational)
# ---------------------------------------------------------------------------

def visualize_forward_process(T=500, device="cpu"):
    """Show how an image gets progressively noised."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset  = datasets.MNIST("./data", train=True, download=True, transform=tf)
    x0, _   = dataset[0]
    x0       = x0.unsqueeze(0).to(device)

    schedule = DDPMSchedule(T=T, device=device)
    steps    = [0, 25, 50, 100, 200, 350, 499]

    _, axes = plt.subplots(1, len(steps), figsize=(14, 2))
    for ax, t_val in zip(axes, steps):
        t_tensor = torch.tensor([t_val], device=device)
        x_t, _   = schedule.q_sample(x0, t_tensor)
        img      = (x_t.squeeze().cpu() + 1) / 2
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={t_val}")
        ax.axis("off")

    plt.suptitle("Forward Process: Progressive Noising", y=1.05)
    plt.tight_layout()
    plt.savefig("diffusion_forward_process.png", bbox_inches="tight")
    plt.show()
    print("Saved -> diffusion_forward_process.png")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # First: visualize what the forward process looks like
    print("=" * 60)
    print("Visualizing forward (noising) process...")
    visualize_forward_process()

    # Then train
    print("\n" + "=" * 60)
    print("Training DiT diffusion model on MNIST...")
    model, schedule = train(
        epochs=50,
        batch_size=256,
        lr=1e-3,
        T=500,
        save_every=10,
    )
