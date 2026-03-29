"""
Play a trained FSQ policy in a MuJoCo viewer.

Setup:
    uv sync

Run with uv:
    uv run play                                                             # latest Reacher-v5 checkpoint
    uv run play --env=InvertedPendulum-v5
    uv run play --env=Reacher-v5 --episodes=5
    uv run play --env=Reacher-v5 --checkpoint=/tmp/rsl_rl_fsq_reacher_v5/model_999.pt
    uv run play --env=Pusher-v5
    uv run play --env=Hopper-v5

Or directly:
    python play.py --env=Reacher-v5

Note: MuJoCo envs render natively on Mac. Classic control envs (CartPole etc.) require pygame.
"""

from __future__ import annotations

import glob
import os

from absl import app, flags

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from control_fsq_ppo import ENV_PRESETS, ClassicControlVecEnv, make_fsq_ppo_cfg

FLAGS = flags.FLAGS

# --env is already registered by control_fsq_ppo (imported above)
flags.DEFINE_string("checkpoint", None, "Path to .pt file (default: latest)")
flags.DEFINE_integer("episodes", 10, "Number of episodes to play")


def latest_checkpoint(env_id: str) -> str:
    env_prefix = env_id.replace("-", "_").lower()
    pts = glob.glob(f"/tmp/rsl_rl_{env_prefix}_*/model_*.pt")
    if not pts:
        raise FileNotFoundError(
            f"No checkpoints found under /tmp/rsl_rl_{env_prefix}_*/")
    # Sort by iteration number embedded in filename.
    pts.sort(
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return pts[-1]


def load_policy(env_id: str, checkpoint: str, device: str):
    preset = ENV_PRESETS[env_id]
    env_kwargs = preset.get("env_kwargs", {})

    # Dummy vec-env just to satisfy OnPolicyRunner's constructor.
    vec_env = ClassicControlVecEnv(env_id,
                                   num_envs=1,
                                   device=device,
                                   env_kwargs=env_kwargs)
    runner = OnPolicyRunner(vec_env,
                            make_fsq_ppo_cfg(dict(preset)),
                            log_dir=None,
                            device=device)
    runner.load(checkpoint)
    vec_env._envs.close()

    return runner.get_inference_policy(device=device)


def play(env_id: str, policy, episodes: int, device: str,
         env_kwargs: dict) -> None:
    env = gym.make(env_id, render_mode="human", **env_kwargs)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            obs_td = TensorDict(
                {
                    "policy":
                    torch.tensor(obs, dtype=torch.float32,
                                 device=device).unsqueeze(0)
                },
                batch_size=[1],
                device=device,
            )
            with torch.no_grad():
                action = policy(obs_td).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1:3d}  reward: {total_reward:.2f}")

    env.close()


def main(_) -> None:
    device = "cpu"  # MuJoCo rendering requires CPU tensors for numpy conversion
    torch.manual_seed(0)

    checkpoint = FLAGS.checkpoint or latest_checkpoint(FLAGS.env)
    print(f"Environment : {FLAGS.env}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Episodes    : {FLAGS.episodes}")

    policy = load_policy(FLAGS.env, checkpoint, device)

    env_kwargs = ENV_PRESETS[FLAGS.env].get("env_kwargs", {})
    play(FLAGS.env, policy, FLAGS.episodes, device, env_kwargs)


def app_run():
    app.run(main)


if __name__ == "__main__":
    app_run()
