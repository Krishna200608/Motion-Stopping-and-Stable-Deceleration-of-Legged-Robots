"""Train PPO policies on the turning environment.

    python train.py --condition base --seed 0 --steps 400000
    python run_all.py            # 2 conditions x 3 seeds, in parallel

Conditions:
  base            rays + speed + goal + previous action
  deadline_aware  the same, plus the analytic decision-deadline slack
                  (the "lookahead" input of RQ3)
"""

import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from turn_env import TurnEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["base", "deadline_aware"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--scenario", choices=["standard", "hard"], default="standard")
    parser.add_argument("--mu", type=float, default=0.9)
    args = parser.parse_args()

    torch.set_num_threads(1)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    suffix = "" if args.scenario == "standard" else "_hard"
    name = f"{args.condition}{suffix}_s{args.seed}"
    ranges = dict(distance_range=(1.0, 5.0), speed_range=(2.0, 3.0)) if args.scenario == "hard" else {}

    env = make_vec_env(
        lambda: Monitor(TurnEnv(mu=args.mu, deadline_aware=args.condition == "deadline_aware", **ranges)),
        n_envs=8, seed=args.seed,
    )
    model = PPO(
        "MlpPolicy", env, seed=args.seed, verbose=0,
        n_steps=256, batch_size=512, n_epochs=10, gamma=0.99, ent_coef=0.01,
        learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=args.steps)
    model.save(os.path.join(MODEL_DIR, name))

    # Training curve: mean episode reward over the last 100 episodes at the end.
    rewards = [ep["r"] for ep in model.ep_info_buffer]
    with open(os.path.join(LOG_DIR, f"{name}.txt"), "w") as f:
        f.write(f"{name}: steps={args.steps} final_mean_episode_reward="
                f"{sum(rewards) / max(1, len(rewards)):.2f}\n")
    print(f"saved {name}")


if __name__ == "__main__":
    main()
