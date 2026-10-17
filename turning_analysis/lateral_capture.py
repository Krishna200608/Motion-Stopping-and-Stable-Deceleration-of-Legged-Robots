"""Lateral (sideways) Capture Point -- the "turning" counterpart of
Phase 1's forward-direction Capture Point.

Phase 1's `Phase 1/capture_point.py` only ever asks "where must the foot
land, *along the direction of travel*, to stop (or walk at v_des)?"
Turning needs the same question asked *sideways*: how much lateral
(side-to-side) velocity can the robot build up or cancel out in one
stance phase, given the same physical limits (leg reach, step duration,
body height) Phase 1 already uses?

This is not a new physical model -- it is the standard result (Kajita et
al., 2001, "The 3D Linear Inverted Pendulum Mode") that on flat ground
the forward (sagittal) and sideways (lateral) planes of the LIP model are
independent of each other: the same equation

    accel = (g / z0) * (pos - foot)

applies separately on each axis, with the same time constant
Tc = sqrt(z0 / g) in both. Phase 1 only ever used the forward axis; this
module applies the identical equations to the lateral axis, which is
what turning requires.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase 1"))
from lip_model import G  # noqa: E402


def time_constant(z0, g=G):
    """Tc = sqrt(z0 / g) -- identical to Phase 1's capture_point.time_constant,
    since it is the same pendulum, just balancing sideways instead of forwards.
    """
    return np.sqrt(z0 / g)


def lateral_capture_point(y, y_dot, z0, g=G, v_lat_des=0.0):
    """Lateral counterpart of Phase 1's capture_point().

    Same formula, same derivation, applied to the sideways (y) axis
    instead of the forward (x) axis: the foot location that drives the
    lateral velocity y_dot towards v_lat_des (0 by default -- "don't
    drift sideways").
    """
    tc = time_constant(z0, g)
    return y + (y_dot - v_lat_des) * tc


def max_lateral_velocity_per_step(max_leg_reach, z0, g=G):
    """The largest lateral velocity the robot can cancel out (or build up
    from rest) within a single stance step, given how far the foot can
    physically reach to the side.

    Directly mirrors Phase 1's single-step capturable-speed limit
    (`run_experiments.py`: theoretical_1step_limit = max_leg_reach / Tc),
    just pointed sideways: the lateral capture point formula says the
    foot must land `y_dot * Tc` away from the body to fully cancel a
    lateral velocity y_dot in one step. The fastest lateral velocity that
    is physically reachable in one step is whatever makes that required
    distance exactly equal to max_leg_reach.
    """
    tc = time_constant(z0, g)
    return max_leg_reach / tc


if __name__ == "__main__":
    z0, max_leg_reach = 0.8, 0.55
    tc = time_constant(z0)
    v_lat_max = max_lateral_velocity_per_step(max_leg_reach, z0)
    print(f"Tc = {tc:.4f} s (same as Phase 1's forward-direction Tc)")
    print(f"Max lateral velocity that can be cancelled in one step = "
          f"{v_lat_max:.3f} m/s (max_leg_reach={max_leg_reach} m)")
