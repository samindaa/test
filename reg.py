# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "matplotlib",
#   "scikit-learn",
# ]
# ///
"""
Quantile Regression on Moons — using eq. 10 from Dabney et al. 2017
("Distributional Reinforcement Learning with Quantile Regression", QR-DQN).

Loss:
    ρ_τ^κ(u) = |τ − 𝟙{u < 0}| · L_κ(u) / κ

    where u = y − ŷ_τ  (target minus predicted quantile)
    and   L_κ(u) = { 0.5 u²               if |u| ≤ κ
                   { κ(|u| − 0.5κ)        otherwise   (Huber loss)

Quantile midpoints (fixed, not learned):
    τ_i = (2i − 1) / (2N),  i = 1…N

We regress x1 from x0 on the sklearn moons dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ── Step 1: Data ─────────────────────────────────────────────────────────────

np.random.seed(42)
X, _ = make_moons(n_samples=1000, noise=0.15)

x_all = torch.tensor(X[:, 0:1], dtype=torch.float32)   # input
y_all = torch.tensor(X[:, 1:2], dtype=torch.float32)   # target

n_train = 800
x_train, y_train = x_all[:n_train], y_all[:n_train]
x_test,  y_test  = x_all[n_train:], y_all[n_train:]

# ── Step 2: Quantile midpoints τ (eq. just above eq. 10 in the paper) ────────

N = 9  # number of quantiles (odd → one median at 0.50)
# τ_i = (2i - 1) / (2N),  i = 1..N  →  e.g. N=9 gives 0.056, 0.167, ..., 0.944
tau = torch.tensor([(2 * i - 1) / (2 * N) for i in range(1, N + 1)])  # (N,)
print(f"Quantile midpoints τ: {tau.numpy().round(3)}")

# ── Step 3: Quantile Huber loss — eq. 10, Dabney et al. 2017 ─────────────────

def quantile_huber_loss(
    quantiles: torch.Tensor,   # (batch, N)  predicted quantile values
    targets: torch.Tensor,     # (batch, 1)  regression targets
    tau: torch.Tensor,         # (N,)        fixed quantile midpoints
    kappa: float = 1.0,        # Huber threshold (paper uses κ=1)
) -> torch.Tensor:
    u = targets - quantiles                      # (batch, N)  residuals

    # Huber loss  L_κ(u)
    huber = torch.where(
        u.abs() <= kappa,
        0.5 * u.pow(2),
        kappa * (u.abs() - 0.5 * kappa),
    )

    # Asymmetric weight  |τ − 𝟙{u < 0}|
    weight = (tau - (u.detach() < 0).float()).abs()

    rho = weight * huber / kappa                 # (batch, N)
    return rho.sum(dim=-1).mean()               # scalar

# ── Step 4: Model — MLP outputting N quantile values ─────────────────────────

class QuantileNet(nn.Module):
    def __init__(self, in_dim: int, num_quantiles: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (batch, N)

model = QuantileNet(in_dim=1, num_quantiles=N)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ── Step 5: Training loop ─────────────────────────────────────────────────────

EPOCHS = 1000
losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred = model(x_train)                                      # (800, N)
    loss = quantile_huber_loss(pred, y_train, tau, kappa=1.0)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.5f}")

# ── Step 6: Evaluate ──────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    pred_test = model(x_test)   # (200, N)

pred_np = pred_test.numpy()
y_np    = y_test.squeeze().numpy()
x_np    = x_test.squeeze().numpy()

# Indices for q≈0.10, median q≈0.50, q≈0.90  (closest τ values)
idx_lo  = 0          # τ=0.056 ≈ q10
idx_med = N // 2     # τ=0.500 (median)
idx_hi  = N - 1      # τ=0.944 ≈ q90

q_lo  = pred_np[:, idx_lo]
q_med = pred_np[:, idx_med]
q_hi  = pred_np[:, idx_hi]

coverage = np.mean((y_np >= q_lo) & (y_np <= q_hi))
mae      = np.mean(np.abs(y_np - q_med))
print(f"\nτ used  — lo: {tau[idx_lo]:.3f}, mid: {tau[idx_med]:.3f}, hi: {tau[idx_hi]:.3f}")
print(f"Coverage [{tau[idx_lo]:.2f}, {tau[idx_hi]:.2f}]: {coverage:.2%}")
print(f"Median absolute error (τ={tau[idx_med]:.3f}): {mae:.4f}")

# ── Step 7: Plot ──────────────────────────────────────────────────────────────

sort_idx = np.argsort(x_np)
xs = x_np[sort_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.scatter(x_np, y_np, s=10, alpha=0.4, label="True y₁")
ax.plot(xs, q_med[sort_idx], color="red",    lw=2,  label=f"τ={tau[idx_med]:.2f} (median)")
ax.plot(xs, q_lo[sort_idx],  color="orange", lw=1,  linestyle="--", label=f"τ={tau[idx_lo]:.2f}")
ax.plot(xs, q_hi[sort_idx],  color="orange", lw=1,  linestyle="--", label=f"τ={tau[idx_hi]:.2f}")
ax.fill_between(xs, q_lo[sort_idx], q_hi[sort_idx], alpha=0.15, color="orange")

# Draw all N quantile curves in light blue
for i in range(N):
    if i not in (idx_lo, idx_med, idx_hi):
        ax.plot(xs, pred_np[sort_idx, i], color="steelblue", lw=0.5, alpha=0.5)

ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
ax.set_title("Quantile Huber Regression on Moons\n(eq. 10, Dabney et al. 2017)")
ax.legend(fontsize=8)

axes[1].plot(losses)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Quantile Huber Loss")
axes[1].set_title("Training Loss")
axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig("quantile_regression_moons.png", dpi=120)
plt.show()
print("Plot saved to quantile_regression_moons.png")
