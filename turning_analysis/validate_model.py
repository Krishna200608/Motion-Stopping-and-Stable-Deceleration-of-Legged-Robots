"""Checks the analytic turning / braking / decision-deadline models
against a numerically integrated 2D Linear Inverted Pendulum simulation.

Run this file directly:

    python validate_model.py

The plant is the Phase 1 LIP (explicit Euler, dt = 1 ms) in two
horizontal axes with a foot re-planted every `STEP_DURATION` seconds.
The foot is placed by the deadbeat rule from `lip_step_model.py`, then
clamped to the physical limit (leg reach, friction cone). Two models are
compared against it:

  * "closed-form": the Day-1 bounds (`turning_envelope.py`,
    `decision_deadline.py`): r = v^2 T Tc / reach and a constant a_max.
  * "LIP-step": the exact one-step limits in `lip_step_model.py`.

Four checks:
  1. the single-step capturable speed (Phase 1's 1.93 m/s limit, which
     the review said was never tested in simulation);
  2. the turning envelope (tightest sustainable turn radius vs. speed);
  3. braking distance from v down to the minimum walking speed;
  4. the decision deadline: the smallest obstacle distance for which a
     simulated robot, steering as hard as it can, still clears the
     obstacle, found by bisection.

Outputs: validation/*.csv and plots/validation_*.png.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase 1"))
from lip_model import G  # noqa: E402
from capture_point import time_constant  # noqa: E402

import lip_step_model as lsm  # noqa: E402
from turning_envelope import turning_envelope  # noqa: E402
from decision_deadline import swerve_distance, decision_deadline, max_deceleration  # noqa: E402

Z0 = lsm.Z0
STEP = lsm.STEP_DURATION
REACH = lsm.MAX_LEG_REACH
TC = time_constant(Z0)
DT = 0.001
CLEARANCE = 0.3
OBSTACLE_DEPTH = 1.0
MIN_TURN_SPEED = 0.5
MU_VALUES = (0.3, 0.6, 0.9)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "validation")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


# ---------------------------------------------------------------------------
# The plant: numerically integrated 2D LIP
# ---------------------------------------------------------------------------
def integrate_stance(pos, vel, foot, duration=STEP, dt=DT, z0=Z0, g=G, sample_every=50):
    """Explicit-Euler LIP in x and y with a fixed foot (same scheme as
    Phase 1's euler_step). Returns (end_pos, end_vel, path samples)."""
    k = g / z0
    x, y = pos
    vx, vy = vel
    fx, fy = foot
    path = [(x, y)]
    n = int(round(duration / dt))
    for i in range(1, n + 1):
        ax, ay = k * (x - fx), k * (y - fy)
        x += vx * dt
        y += vy * dt
        vx += ax * dt
        vy += ay * dt
        if i % sample_every == 0:
            path.append((x, y))
    return (x, y), (vx, vy), path


def place_foot(pos, vel, vel_target, mu):
    """Deadbeat foot placement toward vel_target, clamped to u_max.
    Returns (foot, saturated)."""
    u0 = lsm.deadbeat_offset(np.array(vel), np.array(vel_target))
    um = lsm.u_max(mu)
    norm = float(np.hypot(*u0))
    saturated = norm > um + 1e-9
    if saturated:
        u0 = u0 * (um / norm)
    return (pos[0] - u0[0], pos[1] - u0[1]), saturated


def take_step(pos, vel, vel_target, mu):
    foot, sat = place_foot(pos, vel, vel_target, mu)
    pos, vel, path = integrate_stance(pos, vel, foot)
    return pos, vel, path, sat


# ---------------------------------------------------------------------------
# 1. Single-step capturable speed
# ---------------------------------------------------------------------------
def one_step_capturable(v, hold=4.0):
    """Hold ONE foot at the (clamped) capture point and check the CoM
    still converges to rest over it instead of outrunning the leg."""
    foot_x = min(v * TC, REACH)
    pos, vel = (0.0, 0.0), (v, 0.0)
    worst = 0.0
    for _ in range(int(hold / STEP)):
        pos, vel, _ = integrate_stance(pos, vel, (foot_x, 0.0))
        worst = max(worst, abs(pos[0] - foot_x))
        if worst > REACH + 1e-6:
            return False
    return abs(vel[0]) < 0.05


def check_single_step_limit():
    lo, hi = 0.5, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if one_step_capturable(mid) else (lo, mid)
    return lo, REACH / TC


# ---------------------------------------------------------------------------
# 2. Turning envelope
# ---------------------------------------------------------------------------
def turn_sustained(v, dtheta, mu, n_steps=12):
    """Rotate the commanded velocity by dtheta every step; sustained if
    the foot never hits its limit. Returns (ok, radius of traced path)."""
    pos, vel = (0.0, 0.0), (v, 0.0)
    heading, pts = 0.0, [pos]
    for _ in range(n_steps):
        heading += dtheta
        target = (v * np.cos(heading), v * np.sin(heading))
        pos, vel, _, sat = take_step(pos, vel, target, mu)
        pts.append(pos)
        if sat:
            return False, np.nan
    # radius of the traced path: chord of the last step over 2 sin(dtheta/2)
    chord = float(np.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]))
    if dtheta < 1e-9:
        return True, np.inf
    return True, chord / (2.0 * np.sin(dtheta / 2.0))


def sim_turn_radius(v, mu):
    """Bisect the largest sustainable turn per step; return the radius of
    the path actually traced at that limit (inf if v cannot be sustained)."""
    ok, r = turn_sustained(v, 0.0, mu)
    if not ok:
        return np.inf
    lo, hi = 0.0, np.pi
    if turn_sustained(v, hi, mu)[0]:
        lo = hi
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if turn_sustained(v, mid, mu)[0] else (lo, mid)
    return turn_sustained(v, lo, mu)[1] if lo > 1e-6 else np.inf


# ---------------------------------------------------------------------------
# 3. Braking distance
# ---------------------------------------------------------------------------
def sim_braking_distance(v, v_floor, mu, max_steps=30):
    """Brake along +x as hard as the foot limit allows, down to v_floor."""
    pos, vel = (0.0, 0.0), (v, 0.0)
    for _ in range(max_steps):
        if vel[0] <= v_floor + 1e-3:
            return pos[0]
        tc, c, s = lsm._cs()
        nxt = max(v_floor, vel[0] * c - lsm.u_max(mu) / tc * s)
        if nxt >= vel[0] - 1e-9:
            return np.inf
        pos, vel, _, _ = take_step(pos, vel, (nxt, 0.0), mu)
    return np.inf


# ---------------------------------------------------------------------------
# 4. Decision deadline
# ---------------------------------------------------------------------------
def swerve_clears(d, width, v, mu, brake_to=None, max_steps=40):
    """Simulate approaching an obstacle whose front face is at x = d.
    Optionally brake to `brake_to` first (that many steps, from x = 0),
    then steer left as hard as the foot allows until the lateral offset
    clears width/2 + clearance, then hold course. True if the path never
    enters the obstacle."""
    lateral = width / 2.0 + CLEARANCE
    pos, vel = (0.0, 0.0), (v, 0.0)
    braking = brake_to is not None and v > brake_to + 1e-3
    tc, c, s = lsm._cs()
    for _ in range(max_steps):
        if braking:
            nxt = max(brake_to, vel[0] * c - lsm.u_max(mu) / tc * s)
            if nxt >= vel[0] - 1e-9:
                return False
            target = (nxt, 0.0)
            braking = nxt > brake_to + 1e-9
        elif pos[1] < lateral:
            speed = float(np.hypot(*vel))
            target = (0.0, speed)          # as hard a left turn as allowed
        else:
            target = vel                   # clear: hold course
        pos, vel, path, _ = take_step(pos, vel, target, mu)
        for px, py in path:
            if d <= px <= d + OBSTACLE_DEPTH and abs(py) < lateral:
                return False
        if pos[0] > d + OBSTACLE_DEPTH:
            return True
        if pos[0] < -1.0:
            return False
    return True


def sim_deadline(width, v, mu, brake_to=None, d_max=30.0):
    """Smallest obstacle distance that is still clearable (bisection)."""
    if not swerve_clears(d_max, width, v, mu, brake_to):
        return np.inf
    lo, hi = 0.0, d_max
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if swerve_clears(mid, width, v, mu, brake_to) else (mid, hi)
    return hi


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- 1 ----
    v_sim, v_theory = check_single_step_limit()
    print(f"[1] single-step capturable speed: simulated {v_sim:.3f} m/s vs "
          f"theory max_leg_reach/Tc = {v_theory:.3f} m/s")
    pd.DataFrame([{"simulated_limit": v_sim, "theory_limit": v_theory}]).to_csv(
        os.path.join(OUT_DIR, "single_step_limit.csv"), index=False)

    # ---- 2 + 3 ----
    speeds = np.round(np.arange(0.5, 3.51, 0.25), 2)
    rows = []
    for mu in MU_VALUES:
        a_cf = max_deceleration(mu, STEP, REACH, Z0)
        for v in speeds:
            r_cf, _ = turning_envelope(v, mu, STEP, REACH, Z0)
            b_cf = (v**2 - MIN_TURN_SPEED**2) / (2 * a_cf) if v > MIN_TURN_SPEED else 0.0
            rows.append({
                "mu": mu, "v": v,
                "turn_radius_closed_form": r_cf,
                "turn_radius_lip": lsm.lip_turn_radius(v, mu),
                "turn_radius_sim": sim_turn_radius(v, mu),
                "brake_dist_closed_form": b_cf,
                "brake_dist_lip": lsm.lip_braking_distance(v, MIN_TURN_SPEED, mu),
                "brake_dist_sim": sim_braking_distance(v, MIN_TURN_SPEED, mu),
            })
    base = pd.DataFrame(rows)
    base.to_csv(os.path.join(OUT_DIR, "turning_and_braking.csv"), index=False)
    print("[2,3] turning radius and braking distance written")

    # ---- 4 ----
    rows = []
    for mu in MU_VALUES:
        for width in (0.5, 1.0, 2.0, 4.0):
            for v in (1.0, 1.5, 2.0, 2.5, 3.0):
                row = {"mu": mu, "width": width, "v": v}
                row["keep_closed_form"] = swerve_distance(v, width, CLEARANCE, mu, STEP, REACH, Z0)
                row["keep_lip"] = lsm.lip_swerve_distance(v, width, CLEARANCE, mu)
                row["keep_sim"] = sim_deadline(width, v, mu)
                bt_cf, _ = decision_deadline(v, width, CLEARANCE, mu, STEP, REACH, Z0,
                                             min_turn_speed=MIN_TURN_SPEED)
                row["brake_closed_form"] = bt_cf
                row["brake_lip"] = (lsm.lip_braking_distance(v, MIN_TURN_SPEED, mu)
                                    + lsm.lip_swerve_distance(MIN_TURN_SPEED, width, CLEARANCE, mu))
                row["brake_sim"] = sim_deadline(width, v, mu, brake_to=MIN_TURN_SPEED)
                rows.append(row)
    dl = pd.DataFrame(rows)
    dl.to_csv(os.path.join(OUT_DIR, "deadline_validation.csv"), index=False)
    print("[4] decision deadline written")

    # ---- summary ----
    fin = base.replace([np.inf], np.nan)
    print("\nSimulated vs. predicted turn radius (m), mu=0.6:")
    print(base[base.mu == 0.6][["v", "turn_radius_closed_form", "turn_radius_lip", "turn_radius_sim"]]
          .round(2).to_string(index=False))
    print("\nSimulated vs. predicted braking distance to 0.5 m/s (m), mu=0.6:")
    print(base[base.mu == 0.6][["v", "brake_dist_closed_form", "brake_dist_lip", "brake_dist_sim"]]
          .round(2).to_string(index=False))

    ok = dl[np.isfinite(dl.keep_sim)]
    for col in ("keep_closed_form", "keep_lip"):
        err = (ok[col].replace(np.inf, np.nan) - ok.keep_sim).abs()
        print(f"\nkeep-speed deadline, mean abs error vs sim, {col}: {err.mean():.2f} m "
              f"({err.notna().sum()}/{len(ok)} comparable)")
    ok = dl[np.isfinite(dl.brake_sim)]
    for col in ("brake_closed_form", "brake_lip"):
        err = (ok[col].replace(np.inf, np.nan) - ok.brake_sim).abs()
        print(f"brake-then-turn deadline, mean abs error vs sim, {col}: {err.mean():.2f} m "
              f"({err.notna().sum()}/{len(ok)} comparable)")

    # ---- plots ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row_i, mu in enumerate((0.6, 0.9)):
        b = base[base.mu == mu]
        sub = dl[(dl.mu == mu) & (dl.width == 1.0)]
        panels = [
            ("tightest turn radius (m)", [("turn_radius_closed_form", b), ("turn_radius_lip", b), ("turn_radius_sim", b)], b.v),
            ("braking distance to 0.5 m/s (m)", [("brake_dist_closed_form", b), ("brake_dist_lip", b), ("brake_dist_sim", b)], b.v),
            ("decision deadline, keep speed, 1 m obstacle (m)", [("keep_closed_form", sub), ("keep_lip", sub), ("keep_sim", sub)], sub.v),
        ]
        for ax, (ylabel, series, xs) in zip(axes[row_i], panels):
            for col, frame in series:
                kind = col.split("_")[-1] if col.endswith("sim") else ("lip" if col.endswith("lip") else "closed-form")
                style = {"sim": ("k", "o-", "simulation"), "lip": ("tab:red", "s--", "LIP-step model"),
                         "closed-form": ("tab:blue", "^:", "closed-form (Day 1)")}[kind if kind != "form" else "closed-form"]
                y = frame[col].replace(np.inf, np.nan)
                ax.plot(frame.v, y, style[1], color=style[0], label=style[2], markersize=4)
            ax.set_xlabel("forward speed (m/s)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"mu = {mu}")
            ax.set_ylim(bottom=0, top=8)
            ax.grid(alpha=0.3)
        axes[row_i][0].legend(fontsize=8)
    fig.suptitle("Analytic models vs. numerically simulated LIP (missing points = no feasible manoeuvre)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "validation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot -> {path}")


if __name__ == "__main__":
    main()
