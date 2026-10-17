"""Generates the turning-envelope plot and results table for the
planning-level analysis requested in the project's revised direction
(Work Package 3).

Run this file directly:

    python run_turning_analysis.py

Outputs land in ./plots/ (created next to this script) and
turning_envelope.csv.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from turning_envelope import r_friction, r_balance

# ---------------------------------------------------------------------------
# Fixed physical / simulation parameters (same body height, step duration,
# and leg reach as Phase 1, so results are directly comparable).
# ---------------------------------------------------------------------------
Z0 = 0.8                # center-of-mass height, m (Phase 1's value)
STEP_DURATION = 0.4     # duration of one single-support stance phase, s (Phase 1's value)
MAX_LEG_REACH = 0.55    # farthest the foot can be planted from the CoM in one step, m (Phase 1's value)

# Three representative ground-friction conditions.
MU_VALUES = {
    "low-grip floor (mu=0.3)": 0.3,
    "typical floor (mu=0.6)": 0.6,
    "high-grip floor (mu=0.9)": 0.9,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    speeds = np.round(np.arange(0.2, 4.01, 0.2), 2)
    rb = np.array([r_balance(v, STEP_DURATION, MAX_LEG_REACH, Z0) for v in speeds])

    rows = []
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(speeds, rb, "k-", linewidth=2.5, label="balance limit (one-legged, new)")

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (label, mu), color in zip(MU_VALUES.items(), colors):
        rf = np.array([r_friction(v, mu) for v in speeds])
        envelope = np.maximum(rf, rb)

        ax.plot(speeds, rf, "--", color=color, alpha=0.6, label=f"friction limit, {label}")
        ax.plot(speeds, envelope, color=color, linewidth=2, label=f"turning envelope, {label}")

        for v, rf_v, rb_v, env_v in zip(speeds, rf, rb, envelope):
            rows.append({
                "v": v,
                "mu": mu,
                "condition": label,
                "r_friction": rf_v,
                "r_balance": rb_v,
                "turning_envelope": env_v,
                "binding_constraint": "friction" if rf_v >= rb_v else "balance",
            })

    ax.set_xlabel("forward speed (m/s)")
    ax.set_ylabel("minimum turning radius (m)")
    ax.set_title("Turning envelope: minimum achievable turn radius vs. speed\n"
                  "(the taller of the friction limit and the one-legged balance limit)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    outpath = os.path.join(PLOTS_DIR, "turning_envelope.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {outpath}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(SCRIPT_DIR, "turning_envelope.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table -> {csv_path}")

    print("\nBinding constraint by floor condition (count across all tested speeds):")
    print(df.groupby("condition")["binding_constraint"].value_counts().to_string())


if __name__ == "__main__":
    main()
