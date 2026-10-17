"""The turning envelope: the tightest (smallest-radius) turn a legged
robot can make at a given forward speed, derived two independent ways.

This directly answers the question posed in the project's revised
direction (reviewer's Work Package 3, "Planning-level analysis"): "the
minimum turning radius at each speed, limited by friction (r ~= v^2 /
(mu*g)) and by lateral balance." Neither bound by itself is new physics;
applying *both* to a single-support legged robot and comparing them is
the part the reviewer flagged as missing from the literature: "No work
asks when a biped must commit to a direction, given its speed and
turning limits."

----------------------------------------------------------------------
Bound 1 -- friction limit (classical circular motion; the same reason a
car skids if it corners too fast)

A foot in contact with the ground can only push sideways as hard as
friction allows: lateral force <= mu * m * g. Turning in a circle of
radius r at speed v requires centripetal acceleration v^2 / r. Setting
the force needed equal to the force available and solving for r gives
the tightest turn radius physics allows before the foot slips:

    r_friction(v) = v^2 / (mu * g)

A wheeled vehicle (4 contact points, always touching the ground) is
limited by this bound alone.

----------------------------------------------------------------------
Bound 2 -- balance limit (new: derived from the lateral Capture Point)

A legged robot balances on ONE foot at a time, so there is a second
question bound 1 does not capture: even if the ground has enough grip,
can the robot's one-legged balance actually redirect its momentum
sideways fast enough, within a single step, to achieve that turn?

Executing a turn of radius r at speed v means the velocity vector rotates
by v/r radians per second; over one step of duration `step_duration`,
this requires building up a lateral velocity change of approximately

    delta_v_lateral = (v^2 / r) * step_duration

`lateral_capture.py` gives the largest lateral velocity the robot's
one-legged balance can supply in a single step:
`max_leg_reach / Tc` (see max_lateral_velocity_per_step). Setting the
required lateral velocity change equal to that balance budget and
solving for r gives the tightest turn radius one-legged balance allows,
independent of friction:

    r_balance(v) = v^2 * step_duration * Tc / max_leg_reach

----------------------------------------------------------------------
The turning envelope is the tighter (larger, i.e. more restrictive) of
the two bounds at each speed -- whichever constraint binds first:

    r_min(v) = max(r_friction(v), r_balance(v))
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase 1"))
from lip_model import G  # noqa: E402
from lateral_capture import time_constant  # noqa: E402


def r_friction(v, mu, g=G):
    """Tightest turn radius allowed by ground friction alone."""
    return v**2 / (mu * g)


def r_balance(v, step_duration, max_leg_reach, z0, g=G):
    """Tightest turn radius allowed by one-legged lateral balance alone."""
    tc = time_constant(z0, g)
    return (v**2 * step_duration * tc) / max_leg_reach


def turning_envelope(v, mu, step_duration, max_leg_reach, z0, g=G):
    """The actual minimum achievable turn radius: whichever bound binds,
    plus which constraint ("friction" or "balance") is the binding one.
    """
    rf = r_friction(v, mu, g)
    rb = r_balance(v, step_duration, max_leg_reach, z0, g)
    binding = "friction" if rf >= rb else "balance"
    return max(rf, rb), binding


if __name__ == "__main__":
    # Sanity check at a representative walking speed.
    v = 1.5
    z0, step_duration, max_leg_reach, mu = 0.8, 0.4, 0.55, 0.6

    rf = r_friction(v, mu, G)
    rb = r_balance(v, step_duration, max_leg_reach, z0, G)
    r_env, binding = turning_envelope(v, mu, step_duration, max_leg_reach, z0, G)

    print(f"v={v} m/s, mu={mu}:")
    print(f"  r_friction = {rf:.3f} m")
    print(f"  r_balance  = {rb:.3f} m")
    print(f"  envelope   = {r_env:.3f} m  (binding constraint: {binding})")
