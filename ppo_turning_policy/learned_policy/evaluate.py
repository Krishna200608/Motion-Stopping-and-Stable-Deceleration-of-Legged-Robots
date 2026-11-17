"""Evaluate trained policies and measure RQ2 / RQ3.

    python evaluate.py [n_episodes] [standard|hard]

`hard` evaluates the models trained on the hard scenarios (obstacle 1-5 m
ahead at 2-3 m/s, many cases near or below the analytic deadline) with
Gaussian noise (0.3 m s.d.) added to the ray readings, which stresses the
left/right tie-break on symmetric obstacles.

Every model sees the same scenarios (fixed seeds). Metrics per episode:

  outcome          success / collision / fell / timeout
  feasible         analytic LIP decision deadline at the start speed
                   is <= the obstacle distance (so the obstacle CAN be
                   cleared); success is also reported among feasible cases
  commit_distance  distance to the obstacle at the moment the robot's
                   lateral deviation reaches 0.6 m on the side it finally
                   passes on, never to return
  deadline         the analytic decision deadline at that moment's speed
                   for the true obstacle width
  late             commit_distance < deadline (committed too late)
  flips            changes of side (beyond a 0.3 m dead band) while
                   approaching the obstacle (dithering)
  speed_near       mean speed while within 5 m of the obstacle

Side and commitment are path-based (see commit_metrics): the policy
may chatter its actions step to step, so action signs are not used.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from turn_env import TurnEnv, CLEARANCE, OBSTACLE_DEPTH

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "turning_analysis"))
import lip_step_model as lsm  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
TURN_EPS = 0.15
MU = 0.9


SIDE_BAND = 0.3      # lateral deviation (m) from the obstacle's centre line that counts as "on a side"
COMMIT_DEV = 0.6     # lateral deviation (m) that counts as committed to a side


def commit_metrics(env):
    """Path-based commitment. The side chosen is the side of the obstacle's
    centre line the robot passes on. Commit = the first moment its
    deviation reaches COMMIT_DEV on that side without ever returning to the
    other side afterwards (before the obstacle). Flips = changes of side
    (outside a +-SIDE_BAND dead band) while approaching."""
    pos, vel = env.history["pos"], env.history["vel"]
    k_end = next((k for k, p in enumerate(pos) if p[0] > env.d), len(pos) - 1)
    dev = [p[1] - env.offset for p in pos[:k_end + 1]]
    sides = [int(np.sign(x)) for x in dev if abs(x) > SIDE_BAND]
    flips = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
    final = int(np.sign(dev[-1])) if abs(dev[-1]) > 1e-9 else 0
    if final == 0:
        return None, None, None, flips, 0
    k_commit = None
    for k in range(len(dev) - 1, -1, -1):
        if final * dev[k] >= COMMIT_DEV and all(final * x > 0 for x in dev[k:]):
            k_commit = k
        elif k_commit is not None:
            break
    if k_commit is None:
        return None, None, None, flips, final
    speed = float(np.hypot(*vel[k_commit]))
    deadline = lsm.lip_decision_deadline(speed, env.w, CLEARANCE, MU)[0]
    return env.d - pos[k_commit][0], deadline, speed, flips, final


def run_model(path, deadline_aware, n_episodes, scenario):
    model = PPO.load(path, device="cpu")
    if scenario == "hard":
        env = TurnEnv(mu=MU, deadline_aware=deadline_aware, distance_range=(1.0, 5.0),
                      speed_range=(2.0, 3.0), ray_noise=0.03)
    else:
        env = TurnEnv(mu=MU, deadline_aware=deadline_aware)
    rows = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + i)
        start_speed = float(np.hypot(*env.vel))
        feasible = lsm.lip_decision_deadline(start_speed, env.w, CLEARANCE, MU)[0] <= env.d
        outcome = "running"
        while outcome == "running":
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            outcome = info["outcome"]
        cd, dl, sp, flips, side = commit_metrics(env)
        near = [float(np.hypot(*v)) for p, v in zip(env.history["pos"], env.history["vel"])
                if 0 <= env.d - p[0] <= 5.0]
        rows.append({
            "episode": i, "symmetric": env.symmetric, "distance": env.d, "width": env.w,
            "start_speed": start_speed, "feasible": feasible, "outcome": outcome,
            "side": side, "commit_distance": cd, "deadline": dl, "commit_speed": sp, "flips": flips,
            "late": (cd < dl) if cd is not None else np.nan,
            "speed_near": float(np.mean(near)) if near else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    scenario = sys.argv[2] if len(sys.argv) > 2 else "standard"
    tag = "" if scenario == "standard" else "_hard"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    frames = []
    for fname in sorted(os.listdir(MODEL_DIR)):
        if not fname.endswith(".zip"):
            continue
        name = fname[:-4]
        if ("_hard_" in name) != (scenario == "hard"):
            continue
        condition, seed = name.replace("_hard", "").rsplit("_s", 1)
        df = run_model(os.path.join(MODEL_DIR, fname), condition == "deadline_aware", n_episodes, scenario)
        df.insert(0, "seed", int(seed))
        df.insert(0, "condition", condition)
        frames.append(df)
        print(f"evaluated {name}")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(SCRIPT_DIR, f"eval_episodes{tag}.csv"), index=False)

    def summarize(g):
        feas = g[g.feasible]
        sym = g[g.symmetric]
        turned = g[g.commit_distance.notna()]
        return pd.Series({
            "success_rate": (g.outcome == "success").mean(),
            "feasible_fraction": g.feasible.mean(),
            "success_rate_feasible": (feas.outcome == "success").mean() if len(feas) else np.nan,
            "success_rate_infeasible": (g[~g.feasible].outcome == "success").mean() if (~g.feasible).any() else np.nan,
            "collision_rate": (g.outcome == "collision").mean(),
            "timeout_rate": (g.outcome == "timeout").mean(),
            "speed_near_obstacle": g.speed_near.mean(),
            "mean_commit_distance": turned.commit_distance.mean(),
            "late_commit_rate": turned.late.astype(float).mean(),
            "dither_flips_symmetric": sym.flips.mean(),
            "dither_rate_symmetric": (sym.flips > 0).mean(),
        })

    per_seed = all_df.groupby(["condition", "seed"]).apply(summarize, include_groups=False).reset_index()
    per_seed.to_csv(os.path.join(SCRIPT_DIR, f"eval_per_seed{tag}.csv"), index=False)
    summary = per_seed.drop(columns="seed").groupby("condition").agg(["mean", "std"])
    summary.to_csv(os.path.join(SCRIPT_DIR, f"eval_summary{tag}.csv"))
    pd.set_option("display.width", 220)
    print("\nPer seed:")
    print(per_seed.round(3).to_string(index=False))
    print("\nMean over seeds:")
    print(per_seed.drop(columns="seed").groupby("condition").mean().round(3).T.to_string())

    # ---- plots ----
    metrics = [("success_rate_feasible", "success rate (feasible scenarios)"),
               ("dither_rate_symmetric", "episodes with dithering (symmetric obstacle)"),
               ("late_commit_rate", "committed after the analytic deadline")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    conds = sorted(per_seed.condition.unique())
    for ax, (col, title) in zip(axes, metrics):
        for i, cond in enumerate(conds):
            vals = per_seed[per_seed.condition == cond][col]
            ax.bar(i, vals.mean(), color=["tab:blue", "tab:orange"][i % 2], alpha=0.6)
            ax.scatter([i] * len(vals), vals, color="k", zorder=3, s=18)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Learned policies (dots = individual seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"policy_metrics{tag}.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 5), sharey=True)
    for ax, cond in zip(np.atleast_1d(axes), conds):
        d = all_df[(all_df.condition == cond) & all_df.commit_distance.notna()]
        ax.scatter(d.deadline, d.commit_distance, s=6, alpha=0.35)
        lim = [0, 8]
        ax.plot(lim, lim, "k--", label="commit = analytic deadline")
        ax.set_xlim(lim)
        ax.set_ylim(-1, 9)
        ax.set_xlabel("analytic decision deadline (m)")
        ax.set_title(cond)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("distance at which the policy commits (m)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Commit distance vs. analytic deadline (above the line = in time)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"commit_vs_deadline{tag}.png"), dpi=150)
    plt.close(fig)
    print("plots saved")


if __name__ == "__main__":
    main()
