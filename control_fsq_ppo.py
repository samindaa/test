"""
Classic Control: PPO with FSQ bottleneck actor
===============================================
Setup:
    uv sync

Run with uv:
    uv run train --env=CartPole-v1
    uv run train --env=Pendulum-v1
    uv run train --env=MountainCarContinuous-v0 --iters=500
    uv run train --env=Acrobot-v1 --num_envs=32
    uv run train --env=InvertedPendulum-v5
    uv run train --env=InvertedDoublePendulum-v5
    uv run train --env=Reacher-v5 --iters=1000 --num_envs=128
    uv run train --env=Pusher-v5 --iters=1000 --num_envs=64
    uv run train --env=Hopper-v5 --iters=3000

Or directly:
    python control_fsq_ppo.py --env=CartPole-v1

Supported environments:
    CartPole-v1                 discrete(2),  obs(4)
    MountainCar-v0              discrete(3),  obs(2)
    MountainCarContinuous-v0    continuous(1),obs(2)
    Pendulum-v1                 continuous(1),obs(3)
    Acrobot-v1                  discrete(3),  obs(6)
    InvertedPendulum-v5         continuous(1),obs(4)  [MuJoCo]
    InvertedDoublePendulum-v5   continuous(1),obs(9)  [MuJoCo]
    Reacher-v5                  continuous(2),obs(10) [MuJoCo]
    Pusher-v5                   continuous(7),obs(23) [MuJoCo]
    Hopper-v5                   continuous(3),obs(11) [MuJoCo]

Architecture:
    observation → EmpiricalNormalizer → Encoder MLP → FSQ → Decoder MLP → action distribution

Action-space conventions:
    discrete-2  (CartPole)     : single continuous output, sign → 0/1
    discrete-N  (MountainCar, Acrobot): N continuous outputs, argmax → 0..N-1
    continuous  (Pendulum, MCC, InvertedPendulum): raw continuous outputs, clipped to action bounds

Recommended FSQ level configs (Appendix A.4.1, Mentzer et al. 2023):
    [8, 5, 5, 5, 5]  →  5_000 codes
    [8, 8, 8, 5]     →  2_560 codes
    [8, 8, 8, 8]     →  4_096 codes
    [8, 8, 8]        →    512 codes
    [8, 5, 5, 5]     →  1_000 codes
    [5, 5, 5, 5, 5]  →  3_125 codes
    Mix L=8 (even, power-of-2) and L=5 (odd, symmetric around 0).
"""

from __future__ import annotations

from absl import app, flags

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable
from rsl_rl.utils.utils import unpad_trajectories

from fsq import FSQ


class MinStdGaussianDistribution(GaussianDistribution):
    """GaussianDistribution with a hard floor on std to preserve exploration."""

    def __init__(self,
                 output_dim: int,
                 min_std: float = 0.1,
                 **kwargs) -> None:
        super().__init__(output_dim, **kwargs)
        self.min_std = min_std

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(mlp_output)
        clamped = self._distribution.stddev.clamp(min=self.min_std)
        self._distribution = torch.distributions.Normal(
            self._distribution.mean, clamped)


# ─────────────────────────────────────────────────────────────────────────────
# Distributional quantile critic (QR-DQN, Dabney et al. 2017)
# ─────────────────────────────────────────────────────────────────────────────


def _quantile_huber_loss(
    quantiles: torch.Tensor,
    returns: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile regression Huber loss (eq. 10, Dabney et al. 2017).

    Args:
        quantiles: Predicted quantile values, shape [B, N].
        returns:   Target returns, shape [B, 1].
        tau:       Quantile midpoints, shape [N].  Fixed; not learned.
        kappa:     Huber threshold (1.0 matches the paper).
    """
    u = returns - quantiles  # [B, N]
    huber = torch.where(
        u.abs() <= kappa,
        0.5 * u.pow(2),
        kappa * (u.abs() - 0.5 * kappa),
    )
    rho = (tau - (u.detach() < 0).float()).abs() * huber / kappa
    return rho.sum(-1).mean()


class QuantileCritic(MLPModel):
    """Distributional critic: outputs N quantile values instead of a single scalar.

    PPO compatibility:
        ``forward()`` returns the mean of quantiles, shape ``[B, 1]`` — identical to
        MLPModel's output so GAE, bootstrapping, and rollout storage all work unchanged.

        ``forward_quantiles()`` returns all N values, shape ``[B, N]`` — used by
        QuantilePPO to compute the quantile regression loss.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,  # always 1 from PPO — accepted but ignored
        num_quantiles: int = 64,
        hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        activation: str = "mish",
        obs_normalization: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.num_quantiles = num_quantiles
        self.obs_groups, self.obs_dim = self._get_obs_dim(
            obs, obs_groups, obs_set)
        self.obs_normalizer = (EmpiricalNormalization(self.obs_dim)
                               if obs_normalization else nn.Identity())
        self.mlp = MLP(self.obs_dim, num_quantiles, list(hidden_dims),
                       activation)

    def _encode(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        if masks is not None:
            x = unpad_trajectories(x, masks)
        return self.obs_normalizer(x)

    def forward_quantiles(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        """Return all N quantile predictions, shape [B, N]."""
        return self.mlp(self._encode(obs, masks))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return mean of quantiles, shape [B, 1] — PPO-compatible."""
        return self.forward_quantiles(obs, masks).mean(-1, keepdim=True)

    def update_normalization(self, obs: TensorDict) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
            self.obs_normalizer.update(x)

    def get_hidden_state(self):
        return None

    def reset(self, dones=None, hidden_state=None) -> None:
        pass


class QuantilePPO(PPO):
    """PPO with quantile regression value loss (QR-DQN critic).

    Replaces the standard MSE value loss with the asymmetric Huber loss over
    ``num_quantiles`` return distribution atoms.  Actor training (surrogate,
    entropy, KL schedule) is unchanged.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: QuantileCritic,
        storage,
        qr_loss_coef: float = 1.0,
        lr_min: float = 5e-5,
        lr_max: float = 1e-2,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.qr_loss_coef = qr_loss_coef
        self.lr_min = lr_min
        self.lr_max = lr_max
        N = critic.num_quantiles
        tau = (torch.arange(N, dtype=torch.float32) + 0.5) / N  # midpoints
        self.tau = tau.to(self.device)

    # ------------------------------------------------------------------
    # Override update() — identical to PPO.update() except:
    #   1. critic call returns quantiles [B, N] via forward_quantiles()
    #   2. value_loss uses _quantile_huber_loss instead of MSE
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run optimization epochs; value loss is quantile regression."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (
                        (batch.advantages - batch.advantages.mean()) /
                        (batch.advantages.std() + 1e-8))  # type: ignore

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry[
                    "data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] /
                              original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(
                    num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(
                batch.actions)  # type: ignore
            distribution_params = tuple(
                p[:original_batch_size]
                for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # ── CHANGED: get full quantile distribution ──────────────
            quantiles = self.critic.forward_quantiles(  # type: ignore[attr-defined]
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )  # [B, N]
            # ─────────────────────────────────────────────────────────

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(  # type: ignore
                        batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.lr_min,
                                                     self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.lr_max,
                                                     self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate,
                                                 device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob -
                torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(
                batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(
                batch.advantages) * torch.clamp(  # type: ignore
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ── CHANGED: quantile regression loss replaces MSE ───────
            value_loss = _quantile_huber_loss(quantiles, batch.returns,
                                              self.tau)
            # ─────────────────────────────────────────────────────────

            loss = (surrogate_loss + self.qr_loss_coef * value_loss -
                    self.entropy_coef * entropy.mean())

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry[
                        "data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations,
                        actions=None,
                        env=self.symmetry["_env"])
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None,
                    actions=action_mean_orig,
                    env=self.symmetry["_env"])
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:])
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(  # type: ignore
                        batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding,
                                              target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(),
                                     self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict: dict[str, float] = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# Per-environment presets
# ─────────────────────────────────────────────────────────────────────────────

ENV_PRESETS: dict[str, dict] = {
    "CartPole-v1": {
        "num_learning_iterations": 300,
        "num_envs": 16,
        "num_steps_per_env": 64,
        "fsq_levels": [8, 8, 8],  # 512 codes, dim=3
        "encoder_hidden_dims": [64, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
    },
    "MountainCar-v0": {
        "num_learning_iterations": 2000,
        "num_envs": 32,
        "num_steps_per_env": 128,
        "fsq_levels": [8, 8, 8],  # 512 codes, dim=3
        "encoder_hidden_dims": [64, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.05,  # higher entropy to aid exploration
    },
    "MountainCarContinuous-v0": {
        "num_learning_iterations": 500,
        "num_envs": 16,
        "num_steps_per_env": 128,
        "fsq_levels": [8, 8, 8],  # 512 codes, dim=3
        "encoder_hidden_dims": [64, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
    },
    "Pendulum-v1": {
        "num_learning_iterations": 300,
        "num_envs": 16,
        "num_steps_per_env": 64,
        "fsq_levels": [8, 8, 8],
        "encoder_hidden_dims": [64, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
    },
    "Acrobot-v1": {
        "num_learning_iterations": 500,
        "num_envs": 32,
        "num_steps_per_env": 64,
        "fsq_levels": [8, 8, 8],  # obs(6) → dim=3 bottleneck
        "encoder_hidden_dims": [64, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
    },
    "InvertedPendulum-v5": {
        "num_learning_iterations": 300,
        "num_envs": 16,
        "num_steps_per_env": 64,
        "fsq_levels": [8, 8, 8],  # obs(4) → dim=3 bottleneck
        "encoder_hidden_dims": [64, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
        "env_kwargs": {
            "reset_noise_scale": 0.1
        },
    },
    "InvertedDoublePendulum-v5": {
        "num_learning_iterations": 500,
        "num_envs": 16,
        "num_steps_per_env": 64,
        "fsq_levels": [8, 8, 8, 8],  # obs(9) → dim=4 bottleneck, 4096 codes
        "encoder_hidden_dims": [128, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.01,
        "env_kwargs": {
            "healthy_reward": 10
        },
    },
    "Reacher-v5": {
        "num_learning_iterations": 3000,
        "num_envs": 128,  # more parallel targets → diverse rollouts
        "num_steps_per_env":
        24,  # short rollout horizon; update ~twice per episode
        # total samples/iter: 64 × 24 = 1,536
        "fsq_levels": [8, 8, 8],  # obs(10) → dim=4 bottleneck, 4096 codes
        "encoder_hidden_dims": [128, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 1e-3,
        "entropy_coef": 0.01,
    },
    "Hopper-v5": {
        "num_learning_iterations": 2000,
        "num_envs": 128,
        "num_steps_per_env": 32,
        "fsq_levels": [8, 8, 8],
        "encoder_hidden_dims": [128, 128],
        "decoder_hidden_dims": [64],
        "learning_rate": 1e-3,
        "entropy_coef": 0.01,
        "distribution_cfg": {
            "class_name": MinStdGaussianDistribution,
            "min_std": 0.1,
        },
        "use_quantile_critic": True,
        "num_quantiles": 32,
    },
    "Pusher-v5": {
        "num_learning_iterations": 2000,
        "num_envs": 1024,
        "num_steps_per_env": 24,
        "fsq_levels": [8, 5, 5, 5],
        "encoder_hidden_dims": [128, 128],
        "decoder_hidden_dims": [64],
        "learning_rate": 1e-3,
        "entropy_coef": 0.01,
        "distribution_cfg": {
            "class_name": MinStdGaussianDistribution,
            "min_std": 0.1,
        },
        "use_quantile_critic": True,
        "num_quantiles": 32,
        "qr_loss_coef": 1.0,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FSQ actor model: obs → Encoder → FSQ → Decoder → action
# ─────────────────────────────────────────────────────────────────────────────


class FSQMLPModel(MLPModel):
    """Actor policy with an FSQ quantization bottleneck.

    Data flow:
        obs  →  obs_normalizer  →  encoder_mlp  →  FSQ  →  decoder_mlp  →  distribution
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        fsq_levels: list[int] | tuple[int, ...] = (8, 8, 8),
        encoder_hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (64, ),
        activation: str = "mish",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        # Bypass MLPModel.__init__; call nn.Module directly so we can define
        # our own sub-modules without the parent creating self.mlp.
        nn.Module.__init__(self)

        self.obs_groups, self.obs_dim = self._get_obs_dim(
            obs, obs_groups, obs_set)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        fsq_dim = len(fsq_levels)

        if distribution_cfg is not None:
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            decoder_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            decoder_output_dim = output_dim

        self.fsq = FSQ(list(fsq_levels))
        self.encoder_mlp = MLP(self.obs_dim, fsq_dim,
                               list(encoder_hidden_dims), activation)
        self.decoder_mlp = MLP(fsq_dim, decoder_output_dim,
                               list(decoder_hidden_dims), activation)

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.decoder_mlp)

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(
            obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        out = self.decoder_mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(out)
                return self.distribution.sample()
            return self.distribution.deterministic_output(out)
        return out

    def get_latent(self,
                   obs: TensorDict,
                   masks: torch.Tensor | None = None,
                   hidden_state=None) -> torch.Tensor:
        """obs → normalise → encode → FSQ-quantise → return quantised codes."""
        obs_list = [obs[g] for g in self.obs_groups]
        x = torch.cat(obs_list, dim=-1)
        x = self.obs_normalizer(x)
        z = self.encoder_mlp(x)
        z_q, _ = self.fsq(z)
        return z_q

    # ------------------------------------------------------------------
    # Normalisation helper (same logic as MLPModel)
    # ------------------------------------------------------------------

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[g] for g in self.obs_groups]
            x = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(x)  # type: ignore

    # ------------------------------------------------------------------
    # Export stubs
    # ------------------------------------------------------------------

    def as_jit(self) -> nn.Module:
        return _FSQTorchModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError(
            "ONNX export is not implemented for FSQMLPModel")


class _FSQTorchModel(nn.Module):
    """Minimal JIT-exportable wrapper for FSQMLPModel."""

    def __init__(self, model: FSQMLPModel) -> None:
        import copy
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_mlp = copy.deepcopy(model.encoder_mlp)
        self.fsq = copy.deepcopy(model.fsq)
        self.decoder_mlp = copy.deepcopy(model.decoder_mlp)
        if model.distribution is not None:
            self.det_out = model.distribution.as_deterministic_output_module()
        else:
            self.det_out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        z = self.encoder_mlp(x)
        z_q, _ = self.fsq(z)
        out = self.decoder_mlp(z_q)
        return self.det_out(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Generic classic-control VecEnv wrapper
# ─────────────────────────────────────────────────────────────────────────────


class ClassicControlVecEnv(VecEnv):
    """Vectorised wrapper for any Gymnasium classic-control environment.

    Action-space conventions
    ------------------------
    discrete-2  (e.g. CartPole):       single continuous output; sign → 0/1
    discrete-N  (e.g. MountainCar-v0): N continuous outputs; argmax → 0..N-1
    continuous  (e.g. Pendulum-v1):    raw continuous; clipped to action bounds

    The observation key is always ``"policy"``.
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int = 8,
        device: str = "cpu",
        env_kwargs: dict | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.cfg = {}
        env_kwargs = env_kwargs or {}

        # Inspect a single env to read spaces and episode limit.
        _probe = gym.make(env_id, **env_kwargs)
        obs_space = _probe.observation_space
        act_space = _probe.action_space
        self.max_episode_length = _probe.spec.max_episode_steps or 1000
        _probe.close()

        self.obs_dim: int = int(np.prod(obs_space.shape))

        if isinstance(act_space, gym.spaces.Discrete):
            self._is_discrete = True
            self._n_discrete: int = int(act_space.n)
            if self._n_discrete == 2:
                self.num_actions = 1  # sign trick
                self._action_mode = "sign"
            else:
                self.num_actions = self._n_discrete  # argmax trick
                self._action_mode = "argmax"
            self._act_low = None
            self._act_high = None
        else:
            self._is_discrete = False
            self.num_actions = int(np.prod(act_space.shape))
            self._action_mode = "continuous"
            self._act_low = torch.tensor(act_space.low,
                                         dtype=torch.float32,
                                         device=device)
            self._act_high = torch.tensor(act_space.high,
                                          dtype=torch.float32,
                                          device=device)

        self._envs = gym.make_vec(env_id, num_envs=num_envs, **env_kwargs)
        raw_obs, _ = self._envs.reset()
        self._obs = torch.tensor(raw_obs, dtype=torch.float32, device=device)
        self.episode_length_buf = torch.zeros(num_envs,
                                              dtype=torch.long,
                                              device=device)

    def _make_obs_td(self) -> TensorDict:
        return TensorDict({"policy": self._obs.clone()},
                          batch_size=[self.num_envs],
                          device=self.device)

    def get_observations(self) -> TensorDict:
        return self._make_obs_td()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self._action_mode == "sign":
            gym_actions = (actions[:, 0] >= 0).long().cpu().numpy()
        elif self._action_mode == "argmax":
            gym_actions = actions.argmax(dim=-1).cpu().numpy()
        else:
            clipped = torch.clamp(actions, self._act_low, self._act_high)
            gym_actions = clipped.cpu().numpy()

        raw_obs, raw_rew, terminated, truncated, _ = self._envs.step(
            gym_actions)

        self._obs = torch.tensor(raw_obs,
                                 dtype=torch.float32,
                                 device=self.device)
        rewards = torch.tensor(raw_rew,
                               dtype=torch.float32,
                               device=self.device)
        dones = torch.tensor((terminated | truncated),
                             dtype=torch.float32,
                             device=self.device)
        time_outs = torch.tensor(truncated,
                                 dtype=torch.float32,
                                 device=self.device)

        self.episode_length_buf += 1
        self.episode_length_buf[dones.bool()] = 0

        return self._make_obs_td(), rewards, dones, {"time_outs": time_outs}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PPO config builder
# ─────────────────────────────────────────────────────────────────────────────


def _base_ppo_cfg(preset: dict) -> dict:
    return {
        "num_steps_per_env": preset["num_steps_per_env"],
        "save_interval": 1000,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"]
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": preset["learning_rate"],
            "entropy_coef": preset["entropy_coef"],
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def make_fsq_ppo_cfg(preset: dict) -> dict:
    """Build a PPO config with FSQ bottleneck actor."""
    cfg = _base_ppo_cfg(preset)
    distribution_cfg = preset.get("distribution_cfg",
                                  {"class_name": "GaussianDistribution"})
    cfg["actor"] = {
        "class_name": FSQMLPModel,
        "fsq_levels": preset["fsq_levels"],
        "encoder_hidden_dims": preset["encoder_hidden_dims"],
        "decoder_hidden_dims": preset["decoder_hidden_dims"],
        "activation": "mish",
        "obs_normalization": True,
        "distribution_cfg": distribution_cfg,
    }
    if preset.get("use_quantile_critic"):
        cfg["critic"] = {
            "class_name": QuantileCritic,
            "num_quantiles": preset.get("num_quantiles", 64),
            "hidden_dims": [128, 128],
            "activation": "mish",
            "obs_normalization": True,
        }
        cfg["algorithm"]["class_name"] = QuantilePPO
        cfg["algorithm"]["qr_loss_coef"] = preset.get("qr_loss_coef", 1.0)
        cfg["algorithm"]["lr_min"] = preset.get("lr_min", 5e-5)
        cfg["algorithm"]["lr_max"] = preset.get("lr_max", 1e-2)
    else:
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "mish",
            "obs_normalization": True,
        }
    return cfg


def make_mlp_ppo_cfg(preset: dict) -> dict:
    """Build a PPO config with a plain MLP actor (no FSQ — baseline)."""
    cfg = _base_ppo_cfg(preset)
    hidden = preset["encoder_hidden_dims"] + preset["decoder_hidden_dims"]
    cfg["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": hidden,
        "activation": "mish",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution"
        },
    }
    cfg["critic"] = {
        "class_name": "MLPModel",
        "hidden_dims": [64, 64],
        "activation": "mish",
        "obs_normalization": True,
    }
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    policy,
    env_id: str,
    num_envs: int = 8,
    num_episodes: int = 10,
    device: str = "cpu",
    env_kwargs: dict | None = None,
) -> float:
    env = ClassicControlVecEnv(env_id,
                               num_envs=num_envs,
                               device=device,
                               env_kwargs=env_kwargs)
    obs = env.get_observations()
    total_rewards = torch.zeros(num_envs, device=device)
    episode_rewards: list[float] = []

    while len(episode_rewards) < num_episodes:
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        total_rewards += rewards
        for i in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
            episode_rewards.append(total_rewards[i].item())
            total_rewards[i] = 0.0
            if len(episode_rewards) >= num_episodes:
                break

    env._envs.close()
    return sum(episode_rewards[:num_episodes]) / num_episodes


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Flags + Main
# ─────────────────────────────────────────────────────────────────────────────

FLAGS = flags.FLAGS

flags.DEFINE_enum("env", "CartPole-v1", list(ENV_PRESETS),
                  "Gymnasium environment ID")
flags.DEFINE_integer("iters", None, "Override num learning iterations")
flags.DEFINE_integer("num_envs", None, "Override number of parallel envs")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_string(
    "fsq_levels", None,
    "Override FSQ levels as comma-separated ints, e.g. 16,16,16,16")
flags.DEFINE_boolean("no_fsq", False,
                     "Use plain MLP actor (no FSQ) as baseline")


def main(_) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(FLAGS.seed)
    print(f"Environment : {FLAGS.env}")
    print(f"Device      : {device}")

    preset = dict(ENV_PRESETS[FLAGS.env])  # copy so we can override safely
    if FLAGS.iters is not None:
        preset["num_learning_iterations"] = FLAGS.iters
    if FLAGS.num_envs is not None:
        preset["num_envs"] = FLAGS.num_envs
    if FLAGS.fsq_levels is not None:
        preset["fsq_levels"] = [int(x) for x in FLAGS.fsq_levels.split(",")]

    use_fsq = not FLAGS.no_fsq
    variant = "mlp" if FLAGS.no_fsq else "fsq_" + "x".join(
        str(l) for l in preset["fsq_levels"])
    env_prefix = FLAGS.env.replace("-", "_").lower()

    env_kwargs = preset.get("env_kwargs", {})
    env = ClassicControlVecEnv(FLAGS.env,
                               num_envs=preset["num_envs"],
                               device=device,
                               env_kwargs=env_kwargs)
    print(
        f"Obs dim     : {env.obs_dim}  |  Num actions: {env.num_actions}  ({env._action_mode})"
    )

    ppo_cfg = make_fsq_ppo_cfg(preset) if use_fsq else make_mlp_ppo_cfg(preset)
    runner = OnPolicyRunner(
        env,
        ppo_cfg,
        log_dir=f"/tmp/rsl_rl_{env_prefix}_{variant}",
        device=device,
    )

    if use_fsq:
        actor = runner.alg.actor
        print(f"\nFSQ codebook size : {actor.fsq.codebook_size}")
        print(f"FSQ latent dim    : {actor.fsq.dim}")
    print(f"\n{'='*60}")
    print(
        f"Training PPO+{'FSQ' if use_fsq else 'MLP'} on {FLAGS.env} for {preset['num_learning_iterations']} iterations …"
    )
    print("=" * 60)

    runner.learn(num_learning_iterations=preset["num_learning_iterations"])

    policy = runner.get_inference_policy(device=device)
    mean_reward = evaluate(policy,
                           FLAGS.env,
                           num_envs=preset["num_envs"],
                           device=device,
                           env_kwargs=env_kwargs)
    variant_label = "MLP" if FLAGS.no_fsq else "FSQ"
    print(f"\nMean episode reward ({variant_label} actor): {mean_reward:.2f}")

    env._envs.close()


def app_run():
    app.run(main)


if __name__ == "__main__":
    app_run()
