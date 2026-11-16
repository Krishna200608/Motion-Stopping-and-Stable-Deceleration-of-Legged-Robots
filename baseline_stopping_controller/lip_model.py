"""Linear Inverted Pendulum (LIP) dynamics.

The robot's body is simplified to a point mass at a fixed height `z0` above the
ground, balanced over a single foot contact point `p`. The only equation of
motion we need for Phase 1 is the horizontal acceleration of that point mass:

    x_ddot = (g / z0) * (x - p)

This is an *unstable* equation: whenever the center of mass (x) is not
directly above the foot (p), the acceleration pushes it further away, not
back towards balance. That instability is exactly why a legged robot has to
keep placing its foot in the right spot (see capture_point.py) instead of
just standing still on one leg forever.

Reference: Papers #3 (Chen et al., 2023) and #9 (Koolen et al., 2012) in the
project's literature review.
"""

import numpy as np

G = 9.81  # standard gravity, m/s^2


def lip_acceleration(x, p, z0, g=G):
    """Horizontal acceleration of the center of mass for a given stance foot p."""
    return (g / z0) * (x - p)


def euler_step(x, x_dot, p, dt, z0, g=G):
    """Advance the LIP state by one explicit-Euler integration step.

    The foot position `p` is treated as fixed during the step, matching a
    single-support walking phase (the foot doesn't move once it has touched
    down).
    """
    x_ddot = lip_acceleration(x, p, z0, g)
    x_new = x + x_dot * dt
    x_dot_new = x_dot + x_ddot * dt
    return x_new, x_dot_new


def simulate_lip_segment(x0, x_dot0, p, duration, dt, z0, g=G):
    """Integrate the LIP dynamics for `duration` seconds with a fixed foot `p`.

    Returns three NumPy arrays (t, x, x_dot) sampled every `dt`, including the
    starting sample at t=0.
    """
    n_steps = max(1, int(round(duration / dt)))
    t = np.zeros(n_steps + 1)
    x = np.zeros(n_steps + 1)
    x_dot = np.zeros(n_steps + 1)
    x[0], x_dot[0] = x0, x_dot0

    cur_x, cur_x_dot = x0, x_dot0
    for i in range(1, n_steps + 1):
        cur_x, cur_x_dot = euler_step(cur_x, cur_x_dot, p, dt, z0, g)
        t[i] = i * dt
        x[i] = cur_x
        x_dot[i] = cur_x_dot
    return t, x, x_dot


if __name__ == "__main__":
    # Quick sanity demo: walking forward at a constant speed for a few steps,
    # with the foot simply re-planted under the CoM at each step boundary
    # (this is deliberately *not* a stable walking controller -- it's here
    # only to show the raw, uncontrolled LIP dynamics blowing up, which is
    # the whole reason capture_point.py exists).
    z0 = 0.8
    dt = 0.001
    step_duration = 0.4
    x, x_dot = 0.0, 1.0

    print(f"{'t':>6} {'x':>8} {'x_dot':>8}")
    t_total = 0.0
    for step in range(4):
        p = x  # naive: plant the foot exactly under the CoM every step
        t, xs, x_dots = simulate_lip_segment(x, x_dot, p, step_duration, dt, z0)
        x, x_dot = xs[-1], x_dots[-1]
        t_total += step_duration
        print(f"{t_total:6.2f} {x:8.3f} {x_dot:8.3f}")
