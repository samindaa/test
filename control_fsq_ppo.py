"""
Classic Control: PPO with FSQ bottleneck actor
===============================================
Run with:
    python control_fsq_ppo.py
    python control_fsq_ppo.py --env Pendulum-v1
    python control_fsq_ppo.py --env MountainCarContinuous-v0 --iters 500
    python control_fsq_ppo.py --env Acrobot-v1 --num-envs 32
    python control_fsq_ppo.py --env InvertedPendulum-v5
    python control_fsq_ppo.py --env InvertedDoublePendulum-v5

Supported environments:
    CartPole-v1                 discrete(2),  obs(4)
    MountainCar-v0              discrete(3),  obs(2)
    MountainCarContinuous-v0    continuous(1),obs(2)
    Pendulum-v1                 continuous(1),obs(3)
    Acrobot-v1                  discrete(3),  obs(6)
    InvertedPendulum-v5         continuous(1),obs(4)  [MuJoCo]
    InvertedDoublePendulum-v5   continuous(1),obs(9)  [MuJoCo]
    Reacher-v5                  continuous(2),obs(10) [MuJoCo]

Architecture:
    observation → EmpiricalNormalizer → Encoder MLP → FSQ → Decoder MLP → action distribution

Action-space conventions:
    discrete-2  (CartPole)     : single continuous output, sign → 0/1
    discrete-N  (MountainCar, Acrobot): N continuous outputs, argmax → 0..N-1
    continuous  (Pendulum, MCC, InvertedPendulum): raw continuous outputs, clipped to action bounds
"""

from __future__ import annotations

from absl import app, flags

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable
from rsl_rl.utils.utils import unpad_trajectories

from fsq import FSQ

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
        "fsq_levels": [8, 8],  # 64 codes, dim=2 (tiny obs)
        "encoder_hidden_dims": [64, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 3e-4,
        "entropy_coef": 0.05,  # higher entropy to aid exploration
    },
    "MountainCarContinuous-v0": {
        "num_learning_iterations": 500,
        "num_envs": 16,
        "num_steps_per_env": 128,
        "fsq_levels": [8, 8],
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
        "num_learning_iterations": 1000,
        "num_envs": 128,  # more parallel targets → diverse rollouts
        "num_steps_per_env":
        24,  # short rollout horizon; update ~twice per episode
        # total samples/iter: 64 × 24 = 1,536
        "fsq_levels": [8, 8, 8, 8],  # obs(10) → dim=4 bottleneck, 4096 codes
        "encoder_hidden_dims": [128, 64],
        "decoder_hidden_dims": [64],
        "learning_rate": 1e-3,
        "entropy_coef": 0.01,
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
        activation: str = "elu",
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


def make_fsq_ppo_cfg(preset: dict) -> dict:
    """Build a PPO config from a preset dict."""
    return {
        "num_steps_per_env": preset["num_steps_per_env"],
        "save_interval": 1000,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
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
        "actor": {
            "class_name": FSQMLPModel,
            "fsq_levels": preset["fsq_levels"],
            "encoder_hidden_dims": preset["encoder_hidden_dims"],
            "decoder_hidden_dims": preset["decoder_hidden_dims"],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


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

    env_kwargs = preset.get("env_kwargs", {})
    env = ClassicControlVecEnv(FLAGS.env,
                               num_envs=preset["num_envs"],
                               device=device,
                               env_kwargs=env_kwargs)
    print(
        f"Obs dim     : {env.obs_dim}  |  Num actions: {env.num_actions}  ({env._action_mode})"
    )

    runner = OnPolicyRunner(
        env,
        make_fsq_ppo_cfg(preset),
        log_dir=f"/tmp/rsl_rl_fsq_{FLAGS.env.replace('-', '_').lower()}",
        device=device,
    )

    actor = runner.alg.actor
    print(f"\nFSQ codebook size : {actor.fsq.codebook_size}")
    print(f"FSQ latent dim    : {actor.fsq.dim}")
    print(f"\n{'='*60}")
    print(
        f"Training PPO+FSQ on {FLAGS.env} for {preset['num_learning_iterations']} iterations …"
    )
    print("=" * 60)

    runner.learn(num_learning_iterations=preset["num_learning_iterations"])

    policy = runner.get_inference_policy(device=device)
    mean_reward = evaluate(policy,
                           FLAGS.env,
                           num_envs=preset["num_envs"],
                           device=device,
                           env_kwargs=env_kwargs)
    print(f"\nMean episode reward (FSQ actor): {mean_reward:.2f}")

    env._envs.close()


if __name__ == "__main__":
    app.run(main)
