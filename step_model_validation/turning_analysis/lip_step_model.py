"""Step-level (exact LIP) limits on braking and turning.

`turning_envelope.py` / `decision_deadline.py` use closed-form bounds
(r = v^2 * T * Tc / reach, a_max = reach / (Tc * T)). Those treat a step
as if the foot could keep pushing for the whole stance phase. The
pendulum is unstable, though: after the CoM passes over the foot it
falls away again. `validate_model.py` shows this makes the closed-form
braking bound far too optimistic at high speed, so this module derives
the limits directly from the LIP solution over one step.

----------------------------------------------------------------------
One stance phase

Let u = (CoM - foot) (a 2D vector; the sagittal and lateral LIP axes are
independent, Kajita et al. 2001) and w the CoM velocity. With
c = cosh(T/Tc), s = sinh(T/Tc), the LIP solution over a stance phase of
duration T with the foot fixed is

    u(T) = u0 * c + w * Tc * s
    w(T) = u0 * s / Tc + w * c

Choosing the foot (that is, u0) fixes the velocity at the end of the
step. To reach a desired end-of-step velocity w_T, solve for u0:

    u0 = (w_T - w * c) * Tc / s                      (deadbeat foot placement)

The foot can only be placed so far from the CoM, and can only push as
hard as friction allows (horizontal force = m g u / z0 <= mu m g):

    |u0| <= u_max = min(max_leg_reach, mu * z0)

----------------------------------------------------------------------
Consequences

* Turning: rotating a speed-v velocity by an angle dtheta per step needs
  |u0|^2 = (v Tc / s)^2 (1 + c^2 - 2 c cos dtheta), so the largest turn
  per step is

      cos dtheta_max = (1 + c^2 - (u_max s / (v Tc))^2) / (2 c)

  and the turn radius follows from the chord of one step: the CoM moves
  by D = u0 (c - 1) + w Tc s, so r = |D| / (2 sin(dtheta_max / 2)). Even walking straight needs |u0| = v Tc tanh(T/2Tc),
  so speeds above u_max / (Tc tanh(T/2Tc)) cannot be sustained at all
  (about 3.19 m/s with Phase 1's parameters on a grippy floor).

* Braking: the lowest end-of-step speed along the direction of travel is

      v_next = v c - (u_max / Tc) s

  which can be negative at low speed (the robot can stop in one step)
  but exceeds v at high speed (the robot cannot brake at all).
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase 1"))
from lip_model import G  # noqa: E402
from lateral_capture import time_constant  # noqa: E402

# Phase 1's parameters.
Z0 = 0.8
STEP_DURATION = 0.4
MAX_LEG_REACH = 0.55


def u_max(mu, z0=Z0, max_leg_reach=MAX_LEG_REACH):
    """Farthest the foot may be from the CoM: leg reach, or the friction cone."""
    return min(max_leg_reach, mu * z0)


def _cs(z0=Z0, step=STEP_DURATION, g=G):
    tc = time_constant(z0, g)
    return tc, np.cosh(step / tc), np.sinh(step / tc)


def deadbeat_offset(w, w_target, z0=Z0, step=STEP_DURATION, g=G):
    """Foot offset u0 (CoM minus foot, 2D) that makes the velocity at the
    end of the step equal w_target. Works on arrays of shape (..., 2)."""
    tc, c, s = _cs(z0, step, g)
    return (np.asarray(w_target) - np.asarray(w) * c) * tc / s


def step_state(u0, w, z0=Z0, step=STEP_DURATION, g=G):
    """Closed-form end of a stance phase. Returns (displacement, w_end)."""
    tc, c, s = _cs(z0, step, g)
    u0, w = np.asarray(u0), np.asarray(w)
    u_end = u0 * c + w * tc * s
    return u_end - u0, u0 * s / tc + w * c


def max_sustainable_speed(mu, z0=Z0, step=STEP_DURATION, max_leg_reach=MAX_LEG_REACH, g=G):
    """Fastest speed at which even straight walking keeps |u0| <= u_max."""
    tc = time_constant(z0, g)
    return u_max(mu, z0, max_leg_reach) / (tc * np.tanh(step / (2.0 * tc)))


def max_turn_per_step(v, mu, z0=Z0, step=STEP_DURATION, max_leg_reach=MAX_LEG_REACH, g=G):
    """Largest heading change (rad) one step can achieve at constant speed
    v. Returns 0.0 if v is too fast to even walk straight, and pi if the
    robot can turn right around in one step."""
    tc, c, s = _cs(z0, step, g)
    um = u_max(mu, z0, max_leg_reach)
    if v * tc * np.tanh(step / (2.0 * tc)) > um:
        return 0.0
    cos_dt = (1.0 + c**2 - (um * s / (v * tc)) ** 2) / (2.0 * c)
    if cos_dt <= -1.0:
        return float(np.pi)
    return float(np.arccos(min(1.0, cos_dt)))


def lip_turn_radius(v, mu, z0=Z0, step=STEP_DURATION, max_leg_reach=MAX_LEG_REACH, g=G):
    """Tightest sustainable turn radius at speed v (inf if cannot turn)."""
    dtheta = max_turn_per_step(v, mu, z0, step, max_leg_reach, g)
    if dtheta <= 0.0:
        return np.inf
    tc, c, s = _cs(z0, step, g)
    w = np.array([v, 0.0])
    w_target = v * np.array([np.cos(dtheta), np.sin(dtheta)])
    u0 = deadbeat_offset(w, w_target, z0, step, g)
    disp, _ = step_state(u0, w, z0, step, g)
    return float(np.hypot(*disp) / (2.0 * np.sin(dtheta / 2.0)))


def next_speed_min(v, mu, z0=Z0, step=STEP_DURATION, max_leg_reach=MAX_LEG_REACH, g=G):
    """Lowest speed reachable after one braking step from speed v."""
    tc, c, s = _cs(z0, step, g)
    return v * c - u_max(mu, z0, max_leg_reach) / tc * s


def lip_braking_distance(v, v_floor, mu, z0=Z0, step=STEP_DURATION, max_leg_reach=MAX_LEG_REACH,
                         g=G, max_steps=50):
    """Distance covered braking from v to v_floor with the strongest
    allowed foot placement each step (the last step lands exactly on
    v_floor). Returns inf if v cannot be braked at all."""
    tc, c, s = _cs(z0, step, g)
    um = u_max(mu, z0, max_leg_reach)
    dist, cur = 0.0, v
    for _ in range(max_steps):
        if cur <= v_floor:
            return dist
        nxt = max(v_floor, next_speed_min(cur, mu, z0, step, max_leg_reach, g))
        if nxt >= cur - 1e-9:
            return np.inf
        u0 = (nxt - cur * c) * tc / s
        disp = u0 * (c - 1.0) + cur * tc * s
        dist += disp
        cur = nxt
    return np.inf


def lip_swerve_distance(v, width, clearance, mu, z0=Z0, step=STEP_DURATION,
                        max_leg_reach=MAX_LEG_REACH, g=G):
    """Forward distance to shift sideways by width/2 + clearance at
    constant speed v, on the tightest sustainable arc."""
    lateral = width / 2.0 + clearance
    r = lip_turn_radius(v, mu, z0, step, max_leg_reach, g)
    if not np.isfinite(r):
        return np.inf
    if lateral >= r:
        return r
    return float(np.sqrt(lateral * (2.0 * r - lateral)))


def lip_decision_deadline(v, width, clearance, mu, min_turn_speed=0.5, z0=Z0, step=STEP_DURATION,
                          max_leg_reach=MAX_LEG_REACH, g=G, n_candidates=60):
    """Decision deadline with the exact LIP step limits: best of braking
    to some v' in [min_turn_speed, v] and then swerving at v'. Returns
    (distance, v'), or (inf, v) if no option works."""
    best, best_v = np.inf, v
    for vp in np.linspace(min(min_turn_speed, v), v, n_candidates):
        total = (lip_braking_distance(v, vp, mu, z0, step, max_leg_reach, g)
                 + lip_swerve_distance(vp, width, clearance, mu, z0, step, max_leg_reach, g))
        if total < best:
            best, best_v = total, vp
    return float(best), float(best_v)


if __name__ == "__main__":
    for mu in (0.3, 0.6, 0.9):
        print(f"mu={mu}: u_max={u_max(mu):.3f} m, max sustainable speed "
              f"{max_sustainable_speed(mu):.2f} m/s")
        for v in (1.0, 2.0, 3.0):
            print(f"   v={v}: turn/step={np.degrees(max_turn_per_step(v, mu)):.0f} deg, "
                  f"radius={lip_turn_radius(v, mu):.2f} m, "
                  f"brake to 0.5 m/s in {lip_braking_distance(v, 0.5, mu):.2f} m")
