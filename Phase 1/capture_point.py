"""Capture Point formula and the "smart" foot-placement rule built on it.

The Capture Point (Pratt et al. / Koolen et al., 2012 -- paper #9 in the
literature review) is the ground location where the *next* foot must land so
that the inverted-pendulum body comes to rest with zero velocity:

    Tc = sqrt(z0 / g)              # time constant of the pendulum
    x_capture = x + x_dot * Tc

We generalize this slightly to `capture_point(..., v_des)` so the same
formula can express two things with one piece of math:

  * v_des = 0   -> the classic Capture Point: place the foot here to STOP.
  * v_des != 0  -> the foot placement that makes the pendulum converge to
                    walking at a constant speed v_des, instead of stopping.
                    (Derivation: substitute x' = x - v_des*t into the LIP
                    equation; it has the same form in the moving frame, so
                    "capturing to zero velocity" in that frame is the same
                    formula shifted by v_des.)

Using the same function for both "keep walking" and "stop" is what lets the
controller below switch cleanly from one to the other.
"""

import numpy as np

from lip_model import G


def time_constant(z0, g=G):
    """Tc = sqrt(z0 / g), the natural time constant of the LIP pendulum."""
    return np.sqrt(z0 / g)


def capture_point(x, x_dot, z0, g=G, v_des=0.0):
    """Instantaneous capture point: the foot location that drives x_dot -> v_des."""
    tc = time_constant(z0, g)
    return x + (x_dot - v_des) * tc


def foot_placement(x, x_dot, z0, g, v_des, max_leg_reach, stopping):
    """Shared walking / stopping foot-placement rule used by the smart controller.

    Computes the ideal capture-point foot location (targeting v_des while
    walking, or 0 while stopping), then clamps it to the farthest the leg can
    physically reach in one step. That physical limit is what eventually
    makes very fast starting speeds impossible to stop from in a single step
    (see run_experiments.py's speed sweep).

    Returns (p_actual, p_ideal).
    """
    target_v = 0.0 if stopping else v_des
    p_ideal = capture_point(x, x_dot, z0, g, v_des=target_v)
    offset = np.clip(p_ideal - x, -max_leg_reach, max_leg_reach)
    p_actual = x + offset
    return p_actual, p_ideal


if __name__ == "__main__":
    # Sanity check: for v_des=0, placing the foot exactly at the capture
    # point should make the velocity decay towards zero (exponentially, per
    # the closed-form LIP solution) instead of diverging.
    from lip_model import simulate_lip_segment

    z0 = 0.8
    x, x_dot = 0.0, 1.2
    p = capture_point(x, x_dot, z0, v_des=0.0)
    print(f"x={x}, x_dot={x_dot}, Tc={time_constant(z0):.3f}, capture point p={p:.3f}")

    t, xs, x_dots = simulate_lip_segment(x, x_dot, p, duration=2.0, dt=0.001, z0=z0)
    print(f"after 2.0s: x={xs[-1]:.4f} (-> p={p:.4f}), x_dot={x_dots[-1]:.4f} (-> 0)")
