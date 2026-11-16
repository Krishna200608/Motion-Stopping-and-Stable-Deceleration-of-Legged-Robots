"""Generates the decision-deadline, U-turn corridor and feasibility
results for Work Package 3 (planning-level analysis).

Run this file directly:

    python run_deadline_analysis.py

Outputs land in ./plots/ and as CSVs next to this script:
    plots/decision_deadline.png, plots/uturn_corridor.png,
    plots/feasibility_map.png
    decision_deadline.csv, uturn_corridor.csv, feasibility_table.csv
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from decision_deadline import (
    decision_deadline, max_deceleration, stop_distance, swerve_distance,
    uturn_corridor_width, uturn_entry_distance,
)

# ---------------------------------------------------------------------------
# Parameters (body height, step duration, leg reach are Phase 1's values).
# ---------------------------------------------------------------------------
Z0 = 0.8
STEP_DURATION = 0.4
MAX_LEG_REACH = 0.55
CLEARANCE = 0.3        # body half-width + safety gap, m
BODY_WIDTH = 0.5       # m, for the U-turn corridor

MU_VALUES = {
    "low-grip floor (mu=0.3)": 0.3,
    "typical floor (mu=0.6)": 0.6,
    "high-grip floor (mu=0.9)": 0.9,
}
OBSTACLE_WIDTHS = [0.5, 1.0, 2.0, 4.0]   # m

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def deadline(v, w, mu):
    return decision_deadline(v, w, CLEARANCE, mu, STEP_DURATION, MAX_LEG_REACH, Z0)


def plot_deadline(speeds):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (label, mu) in zip(axes, MU_VALUES.items()):
        for w in OBSTACLE_WIDTHS:
            d, d_const = [], []
            for v in speeds:
                dist, v_turn = deadline(v, w, mu)
                const = swerve_distance(v, w, CLEARANCE, mu, STEP_DURATION, MAX_LEG_REACH, Z0)
                d.append(dist)
                d_const.append(const)
                rows.append({"v": v, "mu": mu, "condition": label, "obstacle_width": w,
                             "deadline_brake_then_turn": dist, "brake_to_speed": v_turn,
                             "deadline_constant_speed": const})
            line, = ax.plot(speeds, d_const, linewidth=2, label=f"keep speed, obstacle width {w} m")
            if w == OBSTACLE_WIDTHS[0]:
                ax.plot(speeds, d, "--", color="gray", linewidth=2,
                        label="brake to 0.5 m/s then turn (all widths)")
        a_max = max_deceleration(mu, STEP_DURATION, MAX_LEG_REACH, Z0)
        ax.plot(speeds, [stop_distance(v, a_max) for v in speeds], "k:", label="full-stop distance (ref.)")
        ax.set_title(label)
        ax.set_xlabel("forward speed (m/s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("decision deadline: distance to obstacle (m)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Decision deadline: last distance at which a turn can still clear the obstacle")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "decision_deadline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {path}")
    return pd.DataFrame(rows)


def plot_corridor(speeds):
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (label, mu), color in zip(MU_VALUES.items(), ["tab:blue", "tab:orange", "tab:green"]):
        widths = [uturn_corridor_width(v, BODY_WIDTH, mu, STEP_DURATION, MAX_LEG_REACH, Z0) for v in speeds]
        ax.plot(speeds, widths, color=color, linewidth=2, label=label)
        for v, wd in zip(speeds, widths):
            rows.append({"v": v, "mu": mu, "condition": label, "corridor_width": wd})
    ax.set_xlabel("forward speed (m/s)")
    ax.set_ylabel("minimum corridor width for a U-turn (m)")
    ax.set_title("U-turn corridor width vs. speed\n(2 x turning-envelope radius + body width)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "uturn_corridor.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {path}")
    return pd.DataFrame(rows)


def build_feasibility():
    """Grid of (speed, obstacle distance, obstacle width, grip). A swerve
    is feasible iff the obstacle is at or beyond the decision deadline; a
    full stop (reference) iff it is beyond the stopping distance."""
    speeds = np.arange(0.5, 4.01, 0.5)
    distances = np.arange(0.5, 8.01, 0.5)
    rows = []
    for label, mu in MU_VALUES.items():
        a_max = max_deceleration(mu, STEP_DURATION, MAX_LEG_REACH, Z0)
        for w in OBSTACLE_WIDTHS:
            for v in speeds:
                dl, _ = deadline(v, w, mu)
                dl_const = swerve_distance(v, w, CLEARANCE, mu, STEP_DURATION, MAX_LEG_REACH, Z0)
                for d in distances:
                    rows.append({
                        "v": round(float(v), 2), "mu": mu, "condition": label,
                        "obstacle_width": w, "obstacle_distance": round(float(d), 2),
                        "deadline": dl,
                        "deadline_constant_speed": dl_const,
                        "swerve_feasible": bool(d >= dl),
                        "swerve_keep_speed_feasible": bool(d >= dl_const),
                        "full_stop_feasible": bool(d >= stop_distance(v, a_max)),
                    })
    return pd.DataFrame(rows)


def plot_feasibility(df):
    w = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (label, mu) in zip(axes, MU_VALUES.items()):
        sub = df[(df["mu"] == mu) & (df["obstacle_width"] == w)]
        grid = sub.pivot(index="obstacle_distance", columns="v", values="swerve_keep_speed_feasible").astype(float)
        ax.imshow(grid.values, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  extent=[grid.columns.min() - 0.25, grid.columns.max() + 0.25,
                          grid.index.min() - 0.25, grid.index.max() + 0.25])
        ax.set_title(label)
        ax.set_xlabel("forward speed (m/s)")
    axes[0].set_ylabel("distance to obstacle (m)")
    fig.suptitle(f"Feasibility map, obstacle width {w} m if the robot keeps its speed (green = a turn can still clear it, red = too late)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "feasibility_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {path}")


def uturn_entry_table(speeds):
    rows = []
    for label, mu in MU_VALUES.items():
        for cw in [1.0, 2.0, 3.0, 5.0]:
            for v in speeds:
                dist = uturn_entry_distance(v, cw, BODY_WIDTH, mu, STEP_DURATION, MAX_LEG_REACH, Z0)
                rows.append({"v": v, "mu": mu, "condition": label, "corridor_width": cw,
                             "braking_distance_needed": dist})
    return pd.DataFrame(rows)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    speeds = np.round(np.arange(0.2, 4.01, 0.2), 2)

    dl = plot_deadline(speeds)
    dl.to_csv(os.path.join(SCRIPT_DIR, "decision_deadline.csv"), index=False)

    cor = plot_corridor(speeds)
    entry = uturn_entry_table(speeds)
    cor.to_csv(os.path.join(SCRIPT_DIR, "uturn_corridor.csv"), index=False)
    entry.to_csv(os.path.join(SCRIPT_DIR, "uturn_entry_braking.csv"), index=False)

    feas = build_feasibility()
    feas.to_csv(os.path.join(SCRIPT_DIR, "feasibility_table.csv"), index=False)
    plot_feasibility(feas)

    print("\nDecision deadline (m), brake-then-turn, obstacle width 1.0 m:")
    sel = dl[(dl["obstacle_width"] == 1.0) & (dl["v"].isin([1.0, 2.0, 3.0, 4.0]))]
    print(sel.pivot(index="v", columns="condition", values="deadline_brake_then_turn").round(2).to_string())

    print("\nDecision deadline (m), keeping speed (no braking), typical floor:")
    sel2 = dl[(dl["condition"] == "typical floor (mu=0.6)") & (dl["v"].isin([1.0, 2.0, 3.0, 4.0]))]
    print(sel2.pivot(index="v", columns="obstacle_width", values="deadline_constant_speed").round(2).to_string())

    print("\nFraction of the (speed, distance) grid where the obstacle can still be cleared:")
    summary = feas.groupby(["condition", "obstacle_width"])[["swerve_feasible", "swerve_keep_speed_feasible"]].mean()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
