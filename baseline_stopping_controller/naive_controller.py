"""Baseline "naive" controller: the failure case we compare the smart
Capture Point controller against.

Before "stop" is triggered it walks exactly like the smart controller (same
`foot_placement` rule), so the comparison isolates a single variable: what
happens at the moment the robot is told to stop.

Once "stop" is triggered, the naive controller does the intuitive-but-wrong
thing -- it simply keeps the foot fixed where it last was, as if the legs
locked in place. Because the LIP equation is unstable away from the foot,
this reliably makes the center of mass run away from the foot and "trip"
(exceed the maximum leg reach), which is exactly the behaviour real animals
and robots avoid by taking a deliberate braking step instead.
"""

from capture_point import foot_placement


def naive_foot_placement(x, x_dot, current_p, z0, g, v_des, max_leg_reach, stopping):
    """Returns (p_actual, p_ideal) using the "lock the legs" strategy once stopping.

    `current_p` is the stance foot position from the step that is ending;
    while stopping, it is returned unchanged forever.
    """
    if stopping:
        return current_p, current_p
    return foot_placement(x, x_dot, z0, g, v_des, max_leg_reach, stopping=False)
