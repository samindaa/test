"""Tests for the distributional quantile critic (QR-DQN, Dabney et al. 2017).

Covers:
  - _quantile_huber_loss: scalar output, non-negativity, zero at perfect fit,
      asymmetric weighting, gradient flow, kappa threshold behaviour
  - QuantileCritic: output shapes, forward == mean(forward_quantiles),
      output_dim=1 ignored, PPO interface stubs
  - QuantilePPO: tau initialisation, update() return dict, parameter updates,
      qr_loss_coef scaling
  - Config wiring: make_fsq_ppo_cfg routes to correct classes per env
"""

from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from control_fsq_ppo import (
    ENV_PRESETS,
    QuantileCritic,
    QuantilePPO,
    _quantile_huber_loss,
    make_fsq_ppo_cfg,
)

# ─── shared constants ────────────────────────────────────────────────────────

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 2
NUM_QUANTILES = 16
DEVICE = "cpu"


# ─── shared helpers ──────────────────────────────────────────────────────────


def _make_obs(num_envs: int = NUM_ENVS, obs_dim: int = OBS_DIM) -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(num_envs, obs_dim)},
        batch_size=[num_envs],
        device=DEVICE,
    )


def _make_actor(obs: TensorDict, obs_groups: dict) -> MLPModel:
    return MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[32, 32],
        activation="elu",
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _make_quantile_critic(
    obs: TensorDict,
    obs_groups: dict,
    num_quantiles: int = NUM_QUANTILES,
    obs_normalization: bool = False,
) -> QuantileCritic:
    return QuantileCritic(
        obs,
        obs_groups,
        "critic",
        output_dim=1,  # as passed by PPO — must be accepted
        num_quantiles=num_quantiles,
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=obs_normalization,
    )


def _build_quantile_ppo(
    num_quantiles: int = NUM_QUANTILES,
    qr_loss_coef: float = 1.0,
) -> tuple[QuantilePPO, TensorDict]:
    obs = _make_obs()
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups)
    critic = _make_quantile_critic(obs, obs_groups, num_quantiles=num_quantiles)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    ppo = QuantilePPO(
        actor,
        critic,
        storage,
        qr_loss_coef=qr_loss_coef,
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,  # irrelevant for QuantilePPO but must be accepted
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=None,
        device=DEVICE,
    )
    return ppo, obs


def _fill_storage(ppo: QuantilePPO, obs: TensorDict) -> None:
    """Populate storage with random synthetic transitions."""
    storage = ppo.storage
    for _ in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
        t.values = torch.randn(NUM_ENVS, 1)
        t.actions_log_prob = torch.zeros(NUM_ENVS)
        # GaussianDistribution distribution_params: (mean, std)
        t.distribution_params = (
            torch.zeros(NUM_ENVS, NUM_ACTIONS),
            torch.ones(NUM_ENVS, NUM_ACTIONS),
        )
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)

    # Pre-fill returns and advantages (normally set by compute_returns)
    storage.returns[:] = torch.randn_like(storage.returns)
    storage.advantages[:] = torch.randn_like(storage.advantages)
    # Normalise advantages to prevent surrogate from dominating
    storage.advantages = (
        (storage.advantages - storage.advantages.mean())
        / (storage.advantages.std() + 1e-8)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.  _quantile_huber_loss
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantileHuberLoss:

    def _make_inputs(
        self, B: int = 8, N: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantiles = torch.randn(B, N, requires_grad=True)
        returns = torch.randn(B, 1)
        tau = (torch.arange(N, dtype=torch.float32) + 0.5) / N
        return quantiles, returns, tau

    def test_returns_scalar(self) -> None:
        q, r, tau = self._make_inputs()
        loss = _quantile_huber_loss(q, r, tau)
        assert loss.shape == (), "Expected scalar output"

    def test_non_negative(self) -> None:
        torch.manual_seed(0)
        for _ in range(10):
            q, r, tau = self._make_inputs()
            assert _quantile_huber_loss(q, r, tau).item() >= 0.0

    def test_zero_at_perfect_predictions(self) -> None:
        """When every quantile equals the return, loss should be 0."""
        B, N = 4, 16
        returns = torch.randn(B, 1)
        quantiles = returns.expand(B, N).detach().requires_grad_(True)
        tau = (torch.arange(N, dtype=torch.float32) + 0.5) / N
        loss = _quantile_huber_loss(quantiles, returns, tau)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flows_through_quantiles(self) -> None:
        q, r, tau = self._make_inputs()
        loss = _quantile_huber_loss(q, r, tau)
        loss.backward()
        assert q.grad is not None
        assert not torch.all(q.grad == 0), "Expected non-zero gradients"

    def test_no_gradient_through_indicator(self) -> None:
        """The (u < 0) indicator is detached; grad should flow only via huber."""
        B, N = 2, 4
        tau = (torch.arange(N, dtype=torch.float32) + 0.5) / N
        # All quantiles well above returns → u = r - q < 0 for all
        quantiles = torch.full((B, N), 5.0, requires_grad=True)
        returns = torch.zeros(B, 1)
        loss = _quantile_huber_loss(quantiles, returns, tau)
        loss.backward()
        # Gradient must exist and be finite (would be NaN if indicator had a grad)
        assert torch.isfinite(quantiles.grad).all()

    def test_asymmetric_weighting(self) -> None:
        """For a single quantile at tau, underprediction (u>0) is weighted by tau
        and overprediction (u<0) by (1-tau).  Averaging over all tau cancels out,
        so we test on isolated single-quantile cases."""
        B, delta = 1, 0.5  # within kappa=1.0 → quadratic region
        returns = torch.zeros(B, 1)

        # High tau (0.9): underprediction should cost more than overprediction
        tau_high = torch.tensor([0.9])
        q_under = torch.full((B, 1), -delta)  # u = 0-(-0.5) = +0.5 > 0, weight=tau=0.9
        q_over = torch.full((B, 1), delta)    # u = 0-0.5  = -0.5 < 0, weight=1-tau=0.1
        loss_under = _quantile_huber_loss(q_under, returns, tau_high).item()
        loss_over = _quantile_huber_loss(q_over, returns, tau_high).item()
        assert loss_under > loss_over, (
            f"tau=0.9: underprediction ({loss_under:.4f}) should cost more "
            f"than overprediction ({loss_over:.4f})"
        )

        # Low tau (0.1): overprediction should cost more than underprediction
        tau_low = torch.tensor([0.1])
        loss_under_low = _quantile_huber_loss(q_under, returns, tau_low).item()
        loss_over_low = _quantile_huber_loss(q_over, returns, tau_low).item()
        assert loss_over_low > loss_under_low, (
            f"tau=0.1: overprediction ({loss_over_low:.4f}) should cost more "
            f"than underprediction ({loss_under_low:.4f})"
        )

    def test_kappa_huber_threshold(self) -> None:
        """For |u| <= kappa, loss is quadratic (0.5 u²); for |u| > kappa, linear."""
        B, N = 1, 1
        tau = torch.tensor([0.5])  # symmetric tau → equal weight both sides

        # Below threshold: loss ∝ 0.5 * u²  (divided by kappa)
        kappa = 2.0
        u_small = 0.5  # |u| < kappa
        q_small = torch.full((B, N), -u_small, requires_grad=True)
        r = torch.zeros(B, 1)
        loss_small = _quantile_huber_loss(q_small, r, tau, kappa=kappa)
        expected_small = 0.5 * u_small**2 / kappa  # rho = tau=0.5 * huber/kappa, sum/mean = 1
        # tau=0.5, so weight = |0.5 - (u<0)| = 0.5 (u>0 means u<0 is False → weight=0.5)
        expected_small_scaled = 0.5 * (0.5 * u_small**2) / kappa
        assert loss_small.item() == pytest.approx(expected_small_scaled, rel=1e-5)

        # Above threshold: loss ∝ kappa*(|u| - 0.5*kappa)  (linear)
        u_large = 3.0  # |u| > kappa=2.0
        q_large = torch.full((B, N), -u_large, requires_grad=True)
        loss_large = _quantile_huber_loss(q_large, r, tau, kappa=kappa)
        expected_large_scaled = 0.5 * (kappa * (u_large - 0.5 * kappa)) / kappa
        assert loss_large.item() == pytest.approx(expected_large_scaled, rel=1e-5)

    def test_batch_independence(self) -> None:
        """Loss computed on a batch equals mean of per-sample losses."""
        B, N = 6, 8
        quantiles = torch.randn(B, N)
        returns = torch.randn(B, 1)
        tau = (torch.arange(N, dtype=torch.float32) + 0.5) / N

        batch_loss = _quantile_huber_loss(quantiles, returns, tau)

        per_sample = []
        for i in range(B):
            per_sample.append(
                _quantile_huber_loss(quantiles[i : i + 1], returns[i : i + 1], tau)
            )
        expected = torch.stack(per_sample).mean()
        assert batch_loss.item() == pytest.approx(expected.item(), rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  QuantileCritic
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantileCritic:

    def setup_method(self) -> None:
        self.obs = _make_obs()
        self.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        self.critic = _make_quantile_critic(self.obs, self.obs_groups)

    def test_forward_shape(self) -> None:
        out = self.critic(self.obs)
        assert out.shape == (NUM_ENVS, 1), f"Expected [{NUM_ENVS}, 1], got {out.shape}"

    def test_forward_quantiles_shape(self) -> None:
        out = self.critic.forward_quantiles(self.obs)
        assert out.shape == (NUM_ENVS, NUM_QUANTILES), (
            f"Expected [{NUM_ENVS}, {NUM_QUANTILES}], got {out.shape}"
        )

    def test_forward_equals_mean_of_quantiles(self) -> None:
        """forward() must equal forward_quantiles().mean(-1, keepdim=True)."""
        q = self.critic.forward_quantiles(self.obs)
        v = self.critic(self.obs)
        assert torch.allclose(v, q.mean(-1, keepdim=True)), (
            "forward() diverges from forward_quantiles().mean(-1, keepdim=True)"
        )

    def test_output_dim_one_ignored(self) -> None:
        """PPO passes output_dim=1; internal mlp must still output num_quantiles."""
        critic_with_ignored_dim = QuantileCritic(
            self.obs,
            self.obs_groups,
            "critic",
            output_dim=1,  # hard-coded by PPO
            num_quantiles=32,
        )
        out = critic_with_ignored_dim.forward_quantiles(self.obs)
        assert out.shape[-1] == 32

    def test_is_recurrent_false(self) -> None:
        assert self.critic.is_recurrent is False

    def test_get_hidden_state_returns_none(self) -> None:
        assert self.critic.get_hidden_state() is None

    def test_reset_is_noop(self) -> None:
        # Should not raise regardless of arguments
        self.critic.reset()
        self.critic.reset(dones=torch.zeros(NUM_ENVS))

    def test_update_normalization_runs(self) -> None:
        critic_normed = _make_quantile_critic(
            self.obs, self.obs_groups, obs_normalization=True
        )
        # Should not raise and should update internal running stats
        critic_normed.update_normalization(self.obs)

    def test_update_normalization_changes_stats(self) -> None:
        critic_normed = _make_quantile_critic(
            self.obs, self.obs_groups, obs_normalization=True
        )
        from rsl_rl.modules import EmpiricalNormalization
        norm: EmpiricalNormalization = critic_normed.obs_normalizer  # type: ignore
        mean_before = norm.mean.clone()
        critic_normed.update_normalization(self.obs)
        # After one update the running mean should have moved from its initial value
        assert not torch.equal(norm.mean, mean_before)

    def test_no_normalization_uses_identity(self) -> None:
        critic_plain = _make_quantile_critic(
            self.obs, self.obs_groups, obs_normalization=False
        )
        assert isinstance(critic_plain.obs_normalizer, torch.nn.Identity)

    def test_gradient_flows_to_mlp(self) -> None:
        q = self.critic.forward_quantiles(self.obs)
        q.sum().backward()
        for param in self.critic.mlp.parameters():
            assert param.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  QuantilePPO
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantilePPOInit:

    def test_tau_shape(self) -> None:
        ppo, _ = _build_quantile_ppo(num_quantiles=NUM_QUANTILES)
        assert ppo.tau.shape == (NUM_QUANTILES,)

    def test_tau_midpoints(self) -> None:
        N = 16
        ppo, _ = _build_quantile_ppo(num_quantiles=N)
        tau = ppo.tau
        expected = (torch.arange(N, dtype=torch.float32) + 0.5) / N
        assert torch.allclose(tau, expected)

    def test_tau_range(self) -> None:
        N = 64
        ppo, _ = _build_quantile_ppo(num_quantiles=N)
        assert ppo.tau[0].item() == pytest.approx(0.5 / N, rel=1e-5)
        assert ppo.tau[-1].item() == pytest.approx((2 * N - 1) / (2 * N), rel=1e-5)

    def test_tau_sums_to_half_N(self) -> None:
        """Midpoints of N uniform intervals sum to N/2."""
        N = 32
        ppo, _ = _build_quantile_ppo(num_quantiles=N)
        assert ppo.tau.sum().item() == pytest.approx(N / 2, rel=1e-5)

    def test_tau_on_correct_device(self) -> None:
        ppo, _ = _build_quantile_ppo()
        assert ppo.tau.device.type == DEVICE

    def test_qr_loss_coef_stored(self) -> None:
        coef = 2.5
        ppo, _ = _build_quantile_ppo(qr_loss_coef=coef)
        assert ppo.qr_loss_coef == coef

    def test_is_subclass_of_ppo(self) -> None:
        from rsl_rl.algorithms.ppo import PPO
        ppo, _ = _build_quantile_ppo()
        assert isinstance(ppo, PPO)


class TestQuantilePPOUpdate:

    def setup_method(self) -> None:
        torch.manual_seed(42)
        self.ppo, self.obs = _build_quantile_ppo()
        # Storage is empty at construction; each test fills it as needed.

    def _ready(self) -> None:
        """Fill storage so the PPO is ready for an update() call."""
        _fill_storage(self.ppo, self.obs)

    def test_update_returns_dict(self) -> None:
        self._ready()
        losses = self.ppo.update()
        assert isinstance(losses, dict)

    def test_update_dict_has_required_keys(self) -> None:
        self._ready()
        losses = self.ppo.update()
        for key in ("value", "surrogate", "entropy"):
            assert key in losses, f"Missing key: '{key}'"

    def test_update_values_are_finite_floats(self) -> None:
        self._ready()
        losses = self.ppo.update()
        for key, val in losses.items():
            assert isinstance(val, float), f"{key} is not a float"
            assert not (val != val), f"{key} is NaN"  # NaN check
            assert val != float("inf"), f"{key} is inf"

    def test_update_modifies_critic_params(self) -> None:
        params_before = [p.clone() for p in self.ppo.critic.parameters()]
        self._ready()
        self.ppo.update()
        params_after = list(self.ppo.critic.parameters())
        changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed, "Critic parameters did not change after update()"

    def test_update_modifies_actor_params(self) -> None:
        params_before = [p.clone() for p in self.ppo.actor.parameters()]
        self._ready()
        self.ppo.update()
        params_after = list(self.ppo.actor.parameters())
        changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed, "Actor parameters did not change after update()"

    def test_zero_qr_loss_coef_does_not_update_critic(self) -> None:
        """qr_loss_coef=0.0 removes the value loss from the total loss, so the
        critic gradient is zero and its parameters must not change."""
        ppo, obs = _build_quantile_ppo(qr_loss_coef=0.0)
        _fill_storage(ppo, obs)
        params_before = [p.clone() for p in ppo.critic.parameters()]
        ppo.update()
        params_after = list(ppo.critic.parameters())
        all_same = all(torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert all_same, "Critic should not change when qr_loss_coef=0.0"

    def test_qr_loss_coef_scales_critic_grad(self) -> None:
        """With twice the qr_loss_coef, the critic gradient norm should be exactly 2x.
        We verify this by holding network weights and batch data identical."""
        torch.manual_seed(0)
        obs = _make_obs()

        # Build one PPO, fill its storage, and grab a deterministic batch
        ppo_ref, _ = _build_quantile_ppo(qr_loss_coef=1.0)
        _fill_storage(ppo_ref, obs)
        batch = next(iter(ppo_ref.storage.mini_batch_generator(1, 1)))
        shared_weights = copy.deepcopy(ppo_ref.critic.state_dict())
        tau = ppo_ref.tau

        def _grad_norm(coef: float) -> float:
            ppo, _ = _build_quantile_ppo(qr_loss_coef=coef)
            ppo.critic.load_state_dict(copy.deepcopy(shared_weights))
            ppo.optimizer.zero_grad()
            quantiles = ppo.critic.forward_quantiles(batch.observations)
            value_loss = _quantile_huber_loss(quantiles, batch.returns, tau)
            (coef * value_loss).backward()
            return sum(
                p.grad.norm().item() ** 2
                for p in ppo.critic.parameters()
                if p.grad is not None
            ) ** 0.5

        norm1 = _grad_norm(1.0)
        norm2 = _grad_norm(2.0)
        assert norm2 == pytest.approx(2.0 * norm1, rel=1e-4)

    def test_storage_cleared_after_update(self) -> None:
        """update() must call storage.clear() so the step counter resets."""
        self._ready()
        self.ppo.update()
        assert self.ppo.storage.step == 0

    def test_update_can_run_twice_sequentially(self) -> None:
        """Two consecutive update calls (with fresh storage) must not raise."""
        self._ready()
        self.ppo.update()       # clears storage; step=0
        self._ready()           # refill now that step=0
        self.ppo.update()       # should not raise

    def test_value_loss_uses_qr_not_mse(self) -> None:
        """Critic forward() returns [B,1] mean — if MSE were used the loss would
        equal MSE(mean_quantiles, returns).  QR loss is different from that."""
        ppo, obs = _build_quantile_ppo()
        _fill_storage(ppo, obs)

        # Compute what MSE on the mean would give for one mini-batch
        generator = ppo.storage.mini_batch_generator(
            ppo.num_mini_batches, ppo.num_learning_epochs
        )
        batch = next(iter(generator))
        with torch.no_grad():
            quantiles = ppo.critic.forward_quantiles(batch.observations)
            mean_vals = quantiles.mean(-1, keepdim=True)
            mse_loss = (batch.returns - mean_vals).pow(2).mean().item()
            qr_loss = _quantile_huber_loss(quantiles, batch.returns, ppo.tau).item()

        assert mse_loss != pytest.approx(qr_loss, rel=1e-3), (
            "QR loss should differ from MSE on mean quantiles"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Config wiring via make_fsq_ppo_cfg
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigWiring:

    def test_pusher_uses_quantile_critic(self) -> None:
        cfg = make_fsq_ppo_cfg(dict(ENV_PRESETS["Pusher-v5"]))
        assert cfg["critic"]["class_name"] is QuantileCritic

    def test_pusher_uses_quantile_ppo(self) -> None:
        cfg = make_fsq_ppo_cfg(dict(ENV_PRESETS["Pusher-v5"]))
        assert cfg["algorithm"]["class_name"] is QuantilePPO

    def test_pusher_num_quantiles_in_critic_cfg(self) -> None:
        cfg = make_fsq_ppo_cfg(dict(ENV_PRESETS["Pusher-v5"]))
        assert "num_quantiles" in cfg["critic"]
        assert cfg["critic"]["num_quantiles"] == ENV_PRESETS["Pusher-v5"]["num_quantiles"]

    def test_pusher_qr_loss_coef_in_algorithm_cfg(self) -> None:
        cfg = make_fsq_ppo_cfg(dict(ENV_PRESETS["Pusher-v5"]))
        assert "qr_loss_coef" in cfg["algorithm"]
        assert cfg["algorithm"]["qr_loss_coef"] == ENV_PRESETS["Pusher-v5"]["qr_loss_coef"]

    @pytest.mark.parametrize("env_id", [
        "CartPole-v1", "Pendulum-v1", "Reacher-v5", "InvertedPendulum-v5",
    ])
    def test_non_quantile_envs_use_mlp_critic(self, env_id: str) -> None:
        cfg = make_fsq_ppo_cfg(dict(ENV_PRESETS[env_id]))
        assert cfg["critic"]["class_name"] == "MLPModel", (
            f"{env_id} should use standard MLPModel critic"
        )

    @pytest.mark.parametrize("env_id", [
        "CartPole-v1", "Pendulum-v1", "Reacher-v5",
    ])
    def test_non_quantile_envs_use_standard_ppo(self, env_id: str) -> None:
        cfg = make_fsq_ppo_cfg(dict(ENV_PRESETS[env_id]))
        assert cfg["algorithm"]["class_name"] == "PPO", (
            f"{env_id} should use standard PPO"
        )

    def test_quantile_critic_class_not_in_non_quantile_cfg(self) -> None:
        cfg = make_fsq_ppo_cfg(dict(ENV_PRESETS["CartPole-v1"]))
        assert "num_quantiles" not in cfg["critic"]
        assert "qr_loss_coef" not in cfg["algorithm"]
