"""Phase 1 experiments: run the LIP + Capture Point stopping controller (and
the naive baseline) across a sweep of walking speeds, and generate the plots
and results table for the midsem submission.

Run this file directly:

    python run_experiments.py

Outputs land in ./plots/ (created next to this script) and results.csv.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lip_model import G, simulate_lip_segment
from capture_point import capture_point, foot_placement, time_constant
from naive_controller import naive_foot_placement

# ---------------------------------------------------------------------------
# Fixed physical / simulation parameters
# ---------------------------------------------------------------------------
Z0 = 0.8                  # center-of-mass height, m
DT = 0.001                # integration timestep, s
STEP_DURATION = 0.4       # duration of one single-support stance phase, s (post-trigger)
MAX_LEG_REACH = 0.55      # farthest the foot can be planted from the CoM in one step, m
VEL_EPS = 0.02            # velocity below this counts as "stopped", m/s
POS_EPS = 0.03            # |x - p| below this (while stopped) confirms standing over the foot, m
WALK_DURATION = 1.5       # seconds of steady walking before the "stop" command is issued
MAX_STEPS_TO_STOP = 6     # give up (declare "not_converged") after this many post-trigger steps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def run_single_trial(v_des, controller_name, z0=Z0, dt=DT, step_duration=STEP_DURATION,
                      max_leg_reach=MAX_LEG_REACH, walk_duration=WALK_DURATION,
                      max_steps_to_stop=MAX_STEPS_TO_STOP, vel_eps=VEL_EPS, pos_eps=POS_EPS):
    """Simulate walking at v_des, then triggering a stop, with one controller.

    The walking phase is modeled kinematically at a constant v_des (Phase 1's
    focus is the stopping math, not gait generation -- a real walking gait is
    added in Phase 2's PyBullet simulation). It exists purely to hand the
    stopping controller a realistic starting position and velocity. The LIP
    dynamics -- and the instability that makes stopping non-trivial -- only
    kick in once the "stop" command is issued.

    Returns a dict with the full time-series trajectory plus the outcome:
      "stopped"       - velocity and position settled over the foot in time
      "fell"          - the CoM outran the max leg reach mid-stance (tripped)
      "not_converged" - neither happened within max_steps_to_stop steps
    """
    g = G
    tc = time_constant(z0, g)

    # --- Phase A: steady walking at a constant speed ---
    n_walk = max(1, int(round(walk_duration / dt)))
    t_walk = np.arange(n_walk + 1) * dt
    x_walk = v_des * t_walk
    xdot_walk = np.full_like(t_walk, v_des)
    p_walk = x_walk.copy()  # idealized: foot re-planted directly under the CoM each stride
    capture_walk = x_walk + xdot_walk * tc

    t_hist = list(t_walk)
    x_hist = list(x_walk)
    xdot_hist = list(xdot_walk)
    p_hist = list(p_walk)
    capture_hist = list(capture_walk)
    stopping_hist = [False] * len(t_walk)

    # --- Phase B: the "stop" command is issued; hand off to LIP dynamics ---
    x, x_dot = x_hist[-1], xdot_hist[-1]
    p = x  # the stance foot at the instant "stop" is triggered
    t_cursor = t_hist[-1]

    fell = False
    stopped = False
    steps_after_trigger = 0

    while True:
        if controller_name == "capture":
            p, _ = foot_placement(x, x_dot, z0, g, v_des=0.0, max_leg_reach=max_leg_reach, stopping=True)
        elif controller_name == "naive":
            p, _ = naive_foot_placement(x, x_dot, p, z0, g, v_des=0.0, max_leg_reach=max_leg_reach, stopping=True)
        else:
            raise ValueError(f"unknown controller_name: {controller_name}")

        t_seg, x_seg, xdot_seg = simulate_lip_segment(x, x_dot, p, step_duration, dt, z0, g)

        for i in range(1, len(t_seg)):
            t_cursor += dt
            t_hist.append(t_cursor)
            x_hist.append(x_seg[i])
            xdot_hist.append(xdot_seg[i])
            p_hist.append(p)
            capture_hist.append(capture_point(x_seg[i], xdot_seg[i], z0, g, v_des=0.0))
            stopping_hist.append(True)
            if abs(x_seg[i] - p) > max_leg_reach + 1e-9:
                fell = True
                break

        x, x_dot = x_hist[-1], xdot_hist[-1]

        if fell:
            break
        steps_after_trigger += 1
        if abs(x_dot) < vel_eps and abs(x - p) < pos_eps:
            stopped = True
            break
        if steps_after_trigger >= max_steps_to_stop:
            break

    outcome = "fell" if fell else ("stopped" if stopped else "not_converged")
    return {
        "v_des": v_des,
        "controller": controller_name,
        "outcome": outcome,
        "steps_after_trigger": steps_after_trigger,
        "time_to_stop": t_cursor if stopped else None,
        "t": np.array(t_hist),
        "x": np.array(x_hist),
        "x_dot": np.array(xdot_hist),
        "p": np.array(p_hist),
        "capture": np.array(capture_hist),
        "stopping": np.array(stopping_hist),
    }


def run_speed_sweep(speeds, controller_name):
    return [run_single_trial(v, controller_name) for v in speeds]


def plot_trajectories(trial_capture, trial_naive, v_des, outpath):
    """Position and velocity vs time, smart vs naive, for one representative speed."""
    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for trial, label, style in ((trial_capture, "Capture Point", "-"), (trial_naive, "Naive (locked leg)", "--")):
        stop_t = trial["t"][trial["stopping"]][0] if trial["stopping"].any() else None
        ax_pos.plot(trial["t"], trial["x"], style, label=f"{label} - CoM (x)")
        ax_pos.plot(trial["t"], trial["p"], style, alpha=0.5, label=f"{label} - foot (p)")
        ax_vel.plot(trial["t"], trial["x_dot"], style, label=label)
        if stop_t is not None:
            ax_pos.axvline(stop_t, color="gray", linestyle=":", linewidth=1)
            ax_vel.axvline(stop_t, color="gray", linestyle=":", linewidth=1)

    ax_pos.set_ylabel("horizontal position (m)")
    ax_pos.set_title(f"Position vs. time -- starting speed = {v_des:.1f} m/s "
                      f"(dotted line = 'stop' command issued)")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(alpha=0.3)

    ax_vel.axhline(0.0, color="black", linewidth=0.8)
    ax_vel.set_ylabel("horizontal velocity (m/s)")
    ax_vel.set_xlabel("time (s)")
    ax_vel.set_title("Velocity vs. time")
    ax_vel.legend(fontsize=8)
    ax_vel.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_capture_vs_foot(trial, v_des, outpath):
    """Capture point vs. actual foot placement over time, for the capture controller."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trial["t"], trial["capture"], label="capture point (x_capture)")
    ax.plot(trial["t"], trial["p"], label="actual foot placement (p)")
    if trial["stopping"].any():
        stop_t = trial["t"][trial["stopping"]][0]
        ax.axvline(stop_t, color="gray", linestyle=":", linewidth=1, label="'stop' command")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("horizontal position (m)")
    ax.set_title(f"Capture point vs. actual foot placement -- starting speed = {v_des:.1f} m/s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_stability_limit(df, outpath):
    """Outcome (stopped / not_converged / fell) vs. starting speed, for both controllers."""
    outcome_level = {"fell": 0, "not_converged": 1, "stopped": 2}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for controller, marker in (("capture", "o-"), ("naive", "s--")):
        sub = df[df["controller"] == controller].sort_values("v_des")
        levels = sub["outcome"].map(outcome_level)
        ax.plot(sub["v_des"], levels, marker, label=controller)

    max_speed = df[(df["controller"] == "capture") & (df["outcome"] == "stopped")]["v_des"].max()
    if pd.notna(max_speed):
        ax.axvline(max_speed, color="green", linestyle=":", linewidth=1.5,
                   label=f"max stoppable speed ≈ {max_speed:.1f} m/s")

    ax.set_ylim(-0.3, 2.3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["fell", "not converged\n(ran out of steps)", "stopped"])
    ax.set_xlabel("starting walking speed (m/s)")
    ax.set_title("Stopping outcome vs. starting speed")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    speeds = np.round(np.arange(0.2, 4.01, 0.2), 2)

    results_capture = run_speed_sweep(speeds, "capture")
    results_naive = run_speed_sweep(speeds, "naive")
    all_results = results_capture + results_naive

    df = pd.DataFrame([
        {
            "v_des": r["v_des"],
            "controller": r["controller"],
            "outcome": r["outcome"],
            "steps_after_trigger": r["steps_after_trigger"],
            "time_to_stop": r["time_to_stop"],
        }
        for r in all_results
    ])
    csv_path = os.path.join(SCRIPT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table -> {csv_path}")
    print(df.to_string(index=False))

    theoretical_tc = time_constant(Z0)
    theoretical_1step_limit = MAX_LEG_REACH / theoretical_tc
    print(f"\nTc = sqrt(z0/g) = {theoretical_tc:.4f} s")
    print(f"Theoretical 1-step capturable speed limit = max_leg_reach / Tc = {theoretical_1step_limit:.3f} m/s")

    max_stoppable = df[(df["controller"] == "capture") & (df["outcome"] == "stopped")]["v_des"].max()
    print(f"Empirical maximum stoppable speed (within {MAX_STEPS_TO_STOP} steps) = {max_stoppable:.2f} m/s")

    # Representative trajectory + capture-point plots at a few speeds.
    representative_speeds = [0.6, 1.8, 2.8, 3.4]
    for v in representative_speeds:
        trial_capture = run_single_trial(v, "capture")
        trial_naive = run_single_trial(v, "naive")
        plot_trajectories(
            trial_capture, trial_naive, v,
            os.path.join(PLOTS_DIR, f"trajectory_v{v:.1f}.png"),
        )
        plot_capture_vs_foot(
            trial_capture, v,
            os.path.join(PLOTS_DIR, f"capture_vs_foot_v{v:.1f}.png"),
        )
        print(f"Saved plots for v_des={v:.1f} m/s")

    plot_stability_limit(df, os.path.join(PLOTS_DIR, "stability_limit.png"))
    print(f"Saved stability-limit plot -> {os.path.join(PLOTS_DIR, 'stability_limit.png')}")


if __name__ == "__main__":
    main()
