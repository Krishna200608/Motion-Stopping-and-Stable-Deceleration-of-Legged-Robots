"""The decision deadline and the U-turn corridor width.

Second half of the reviewer's Work Package 3: "the decision deadline:
the last distance at which a turn can still clear an obstacle of a given
width, and the corridor width a U-turn needs."

Both build directly on `turning_envelope.py` (minimum turn radius at each
speed) and on Phase 1's single-step limit (max_leg_reach / Tc).

----------------------------------------------------------------------
Setup (planning level, point mass, flat ground)

The robot moves along +x at speed v. An obstacle of width `w` is centred
on its path with its front face at distance d. To pass it the robot's
centre must end up laterally offset by

    L = w/2 + clearance

by the time it reaches x = d (clearance = body half-width + safety gap).

----------------------------------------------------------------------
Swerve distance (turn at constant speed)

The tightest turn is a circular arc of radius r = r_min(v) from the
turning envelope. On that arc, lateral offset y and forward progress x
after sweeping angle theta are y = r(1 - cos theta), x = r sin theta.
Solving y = L for x:

    L < r :  x = sqrt(L (2r - L))      (clears before reaching 90 deg)
    L >= r:  x = r                     (turns a full 90 deg, then moves
                                        sideways -- no more forward
                                        progress is needed)

So the robot must START the turn at least swerve_distance(v, w) before
the obstacle.

----------------------------------------------------------------------
Braking first (decelerate, then turn)

Because r_min scales with v^2, slowing down shrinks the turn a lot. The
robot can brake to a lower speed v' before turning. Deceleration is
limited the same two ways turning is:

    friction: a <= mu * g
    balance:  one step can shed at most max_leg_reach / Tc of speed
              (Phase 1's single-step limit), so a <= (max_leg_reach/Tc)
              / step_duration

    a_max = min(mu * g, max_leg_reach / (Tc * step_duration))

Braking from v to v' covers (v^2 - v'^2) / (2 a_max). Then the swerve at
v' covers swerve_distance(v', w). The decision deadline is the best
trade-off over v':

    d_deadline(v, w) = min over v' <= v of
                       [ (v^2 - v'^2)/(2 a_max) + swerve_distance(v', w) ]

If the obstacle is closer than this, no manoeuvre of this kind avoids
it.

Without a floor on v' the minimum is always v' -> 0: braking to a crawl
beats any swerve, and the "deadline" collapses to the stopping distance
(independent of obstacle width). Stopping is out of scope -- the robot
must slow down AND turn -- so v' is bounded below by `min_turn_speed`,
the slowest speed at which the robot is still walking through the turn.
`stop_distance` is provided only as a reference line.

----------------------------------------------------------------------
U-turn (turn back) corridor width

A 180 degree turn at radius r sweeps a half-circle, so the corridor must
be at least 2r + body_width wide. At constant speed v:

    W_uturn(v) = 2 r_min(v) + body_width

and inverting it gives the fastest speed a corridor of width W allows;
the robot must brake down to that speed first (distance by the same
a_max).
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase 1"))
from lip_model import G  # noqa: E402
from lateral_capture import time_constant  # noqa: E402
from turning_envelope import turning_envelope  # noqa: E402


def max_deceleration(mu, step_duration, max_leg_reach, z0, g=G):
    """Strongest braking the robot can apply: the lesser of the friction
    limit and the one-step balance limit (Phase 1's max_leg_reach / Tc,
    spread over one step)."""
    tc = time_constant(z0, g)
    return min(mu * g, max_leg_reach / (tc * step_duration))


def stop_distance(v, a_max):
    """Distance to brake to a standstill (reference only; stopping is
    out of scope for the project)."""
    return v**2 / (2.0 * a_max)


def swerve_distance(v, width, clearance, mu, step_duration, max_leg_reach, z0, g=G):
    """Forward distance needed to shift sideways by width/2 + clearance
    at constant speed v, using the tightest turn the envelope allows."""
    lateral = width / 2.0 + clearance
    r, _ = turning_envelope(v, mu, step_duration, max_leg_reach, z0, g)
    if lateral >= r:
        return r
    return float(np.sqrt(lateral * (2.0 * r - lateral)))


def decision_deadline(v, width, clearance, mu, step_duration, max_leg_reach, z0,
                      min_turn_speed=0.5, reaction_time=0.0, g=G, n_candidates=400):
    """Last distance to the obstacle at which the robot must commit to a
    turn. Returns (deadline_distance, best_turn_speed).

    `min_turn_speed` is the slowest speed the robot may turn at (it
    must keep walking, not stop); if v is already below it, the robot
    just turns at v. `reaction_time` adds v * reaction_time of unavoidable travel before
    any action begins (0 = ideal controller).
    """
    a_max = max_deceleration(mu, step_duration, max_leg_reach, z0, g)
    candidates = np.linspace(min(min_turn_speed, v), v, n_candidates)
    totals = np.array([
        (v**2 - vp**2) / (2.0 * a_max)
        + swerve_distance(vp, width, clearance, mu, step_duration, max_leg_reach, z0, g)
        for vp in candidates
    ])
    i = int(np.argmin(totals))
    return float(totals[i] + v * reaction_time), float(candidates[i])


def uturn_corridor_width(v, body_width, mu, step_duration, max_leg_reach, z0, g=G):
    """Narrowest corridor a constant-speed 180 degree turn fits in."""
    r, _ = turning_envelope(v, mu, step_duration, max_leg_reach, z0, g)
    return 2.0 * r + body_width


def max_uturn_speed(corridor_width, body_width, mu, step_duration, max_leg_reach, z0,
                    g=G, v_hi=10.0):
    """Fastest speed at which a U-turn fits in a corridor of the given
    width (bisection; the width grows monotonically with speed). Returns
    0.0 if even a crawl does not fit."""
    if uturn_corridor_width(1e-3, body_width, mu, step_duration, max_leg_reach, z0, g) > corridor_width:
        return 0.0
    lo, hi = 1e-3, v_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if uturn_corridor_width(mid, body_width, mu, step_duration, max_leg_reach, z0, g) <= corridor_width:
            lo = mid
        else:
            hi = mid
    return lo


def uturn_entry_distance(v, corridor_width, body_width, mu, step_duration, max_leg_reach, z0, g=G):
    """Distance needed to brake from v down to the speed the corridor
    allows (0 if v already fits, None if the corridor is too narrow)."""
    v_turn = max_uturn_speed(corridor_width, body_width, mu, step_duration, max_leg_reach, z0, g)
    if v_turn <= 0.0:
        return None
    if v <= v_turn:
        return 0.0
    a_max = max_deceleration(mu, step_duration, max_leg_reach, z0, g)
    return (v**2 - v_turn**2) / (2.0 * a_max)


if __name__ == "__main__":
    z0, step_duration, max_leg_reach, mu = 0.8, 0.4, 0.55, 0.6
    clearance, body_width = 0.3, 0.5
    v, w = 2.0, 1.0

    a = max_deceleration(mu, step_duration, max_leg_reach, z0)
    d, vp = decision_deadline(v, w, clearance, mu, step_duration, max_leg_reach, z0)
    print(f"v={v} m/s, obstacle width={w} m, mu={mu}:")
    print(f"  max deceleration   = {a:.2f} m/s^2")
    print(f"  decision deadline  = {d:.2f} m (brake to {vp:.2f} m/s, then swerve)")
    print(f"  U-turn corridor    = {uturn_corridor_width(v, body_width, mu, step_duration, max_leg_reach, z0):.2f} m at constant speed")
