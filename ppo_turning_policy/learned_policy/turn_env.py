"""Obstacle-triggered slow-down-and-turn environment (Work Packages 1-2,
scaled down).

The robot is the validated step-level LIP from `turning_analysis/`
(`lip_step_model.py`): one environment step is one stance phase of 0.4 s.
The action chooses the CoM velocity wanted at the end of the step; the
foot offset needed for that is computed by the deadbeat rule and clamped
to what the leg and the floor allow (|u| <= min(reach, mu * z0)), so the
robot can only turn / brake as hard as `validate_model.py` showed the
LIP can. Nothing here is hand-tuned to make turning easier.

Task: the robot starts at the origin heading +x at 1.5-2.8 m/s (the
"standard" scenarios; "hard" uses 2.0-3.0 m/s and d = 1-5 m, where many
cases sit at or below the analytic deadline). A block
obstacle (width 1-4 m) is d metres ahead, either centred on the robot's
path (symmetric: left and right are equally good) or offset. A goal is
placed beyond it. The robot must reach the goal without hitting the
obstacle or the arena walls, choosing left or right on its own.

Observation (all scaled to about [-1, 1]):
  * 16 ray distances over 360 degrees (range 10 m), relative to the
    direction of motion;
  * speed, goal direction (cos, sin) and distance, previous action;
  * optional `deadline_aware` extra: how much distance to the obstacle
    is left beyond the analytic decision deadline at the current speed
    (computed for a nominal 2.5 m obstacle), and the deadline itself.
    This is the "lookahead" input of RQ3: it tells the policy how soon it
    must commit, using the Work Package 3 model.

Action (2 values in [-1, 1]): change in speed (x 1 m/s) and change in
heading (x 90 degrees) wanted over the next step. A small action-rate
penalty stops the policy from alternating +-1 every step (without it the
foot clamp averages the chatter into a net swerve and the policy "works"
but weaves).

Scope: "turn back" (a dead end that needs a U-turn) is not trained;
the U-turn corridor is handled analytically only.
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "turning_analysis"))
import lip_step_model as lsm  # noqa: E402

STEP = lsm.STEP_DURATION
CLEARANCE = 0.3
OBSTACLE_DEPTH = 1.0
ARENA_HALF_WIDTH = 7.0
MAX_RAY = 10.0
N_RAYS = 16
V_MIN, V_MAX, V_FELL = 0.5, 3.0, 4.0
NOMINAL_WIDTH = 2.5
GOAL_RADIUS = 1.0
MAX_STEPS = 40
ACTION_RATE_PENALTY = 0.05   # discourages bang-bang chatter (cost per squared action change)

_RAY_ANGLES = np.arange(N_RAYS) * 2.0 * np.pi / N_RAYS
_TC, _C, _S = lsm._cs()


def deadline_table(mu, width=NOMINAL_WIDTH):
    """Decision deadline vs speed (capped at 10 m) for the nominal obstacle."""
    speeds = np.linspace(0.3, 3.5, 33)
    vals = np.array([min(10.0, lsm.lip_decision_deadline(v, width, CLEARANCE, mu)[0]) for v in speeds])
    return speeds, vals


class TurnEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mu=0.9, deadline_aware=False, symmetric_prob=0.5, seed=None,
                 distance_range=(3.0, 8.0), speed_range=(1.5, 2.8), ray_noise=0.0):
        super().__init__()
        self.mu = mu
        self.um = lsm.u_max(mu)
        self.deadline_aware = deadline_aware
        self.symmetric_prob = symmetric_prob
        self.distance_range = distance_range
        self.speed_range = speed_range
        self.ray_noise = ray_noise
        self._dl_speeds, self._dl_vals = deadline_table(mu)
        n_obs = N_RAYS + 1 + 3 + 2 + (2 if deadline_aware else 0)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None, symmetric=None, distance=None, width=None, speed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        self.d = float(distance if distance is not None else rng.uniform(*self.distance_range))
        self.w = float(width if width is not None else rng.uniform(1.0, 4.0))
        sym = symmetric if symmetric is not None else (rng.random() < self.symmetric_prob)
        self.offset = 0.0 if sym else float(rng.uniform(-1.0, 1.0))
        self.symmetric = bool(sym)
        lat = self.w / 2.0 + CLEARANCE
        # boxes: x0, x1, y0, y1 (lateral clearance included, as in validate_model.py)
        self.boxes = np.array([
            [self.d, self.d + OBSTACLE_DEPTH, self.offset - lat, self.offset + lat],
            [-5.0, 60.0, ARENA_HALF_WIDTH, ARENA_HALF_WIDTH + 1.0],
            [-5.0, 60.0, -ARENA_HALF_WIDTH - 1.0, -ARENA_HALF_WIDTH],
        ])
        self.goal = np.array([self.d + OBSTACLE_DEPTH + 5.0, 0.0])
        speed = float(speed if speed is not None else rng.uniform(*self.speed_range))
        self.pos = np.zeros(2)
        self.vel = np.array([speed, 0.0])
        self.t = 0
        self.prev_action = np.zeros(2)
        self.history = {"pos": [self.pos.copy()], "vel": [self.vel.copy()], "action": []}
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _rays(self):
        heading = np.arctan2(self.vel[1], self.vel[0])
        ang = heading + _RAY_ANGLES
        dx = np.cos(ang)[None, :]                                   # (1, R)
        dy = np.sin(ang)[None, :]
        b = self.boxes                                              # (B, 4)
        # slab method; parallel rays get +-inf slabs (inside) or an empty interval (outside)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx0 = (b[:, 0:1] - self.pos[0]) / dx
            tx1 = (b[:, 1:2] - self.pos[0]) / dx
            ty0 = (b[:, 2:3] - self.pos[1]) / dy
            ty1 = (b[:, 3:4] - self.pos[1]) / dy
        par_x, par_y = np.abs(dx) < 1e-9, np.abs(dy) < 1e-9
        in_x = (b[:, 0:1] <= self.pos[0]) & (self.pos[0] <= b[:, 1:2])
        in_y = (b[:, 2:3] <= self.pos[1]) & (self.pos[1] <= b[:, 3:4])
        lo_x = np.where(par_x, np.where(in_x, -np.inf, np.inf), np.minimum(tx0, tx1))
        hi_x = np.where(par_x, np.where(in_x, np.inf, -np.inf), np.maximum(tx0, tx1))
        lo_y = np.where(par_y, np.where(in_y, -np.inf, np.inf), np.minimum(ty0, ty1))
        hi_y = np.where(par_y, np.where(in_y, np.inf, -np.inf), np.maximum(ty0, ty1))
        t_in = np.maximum(lo_x, lo_y)
        t_out = np.minimum(hi_x, hi_y)
        hit = (t_out >= t_in) & (t_in > 0.0)
        return np.minimum(MAX_RAY, np.where(hit, t_in, np.inf).min(axis=0))

    def _deadline(self, speed):
        return float(np.interp(speed, self._dl_speeds, self._dl_vals))

    def _obs(self):
        rays = self._rays()
        if self.ray_noise > 0.0:
            rays = np.clip(rays + self._rng.normal(0.0, self.ray_noise * MAX_RAY, N_RAYS), 0.0, MAX_RAY)
        speed = float(np.hypot(*self.vel))
        heading = np.arctan2(self.vel[1], self.vel[0])
        to_goal = self.goal - self.pos
        gdist = float(np.hypot(*to_goal))
        gang = np.arctan2(to_goal[1], to_goal[0]) - heading
        obs = list(rays / MAX_RAY) + [speed / V_MAX, np.cos(gang), np.sin(gang), gdist / 15.0]
        obs += list(self.prev_action)
        if self.deadline_aware:
            dl = self._deadline(speed)
            obs += [float(np.clip((rays[0] - dl) / 5.0, -1.0, 1.0)), dl / 5.0]
        return np.array(obs, dtype=np.float32)

    def _collides(self, p):
        for x0, x1, y0, y1 in self.boxes:
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return True
        return False

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        speed = float(np.hypot(*self.vel))
        heading = np.arctan2(self.vel[1], self.vel[0])
        target_speed = float(np.clip(speed + action[0] * 1.0, V_MIN, V_MAX))
        target_heading = heading + action[1] * np.pi / 2.0
        target = target_speed * np.array([np.cos(target_heading), np.sin(target_heading)])

        u0 = lsm.deadbeat_offset(self.vel, target)
        norm = float(np.hypot(*u0))
        if norm > self.um:
            u0 = u0 * (self.um / norm)
        old_pos = self.pos.copy()
        prev_action = self.prev_action.copy()
        collided = False
        for frac in np.linspace(1.0 / 8.0, 1.0, 8):
            tc, c, s = _TC, np.cosh(frac * STEP / _TC), np.sinh(frac * STEP / _TC)
            p = old_pos + (u0 * c + self.vel * tc * s) - u0
            if self._collides(p):
                collided = True
                break
        disp, self.vel = lsm.step_state(u0, self.vel)
        self.pos = old_pos + disp
        self.t += 1
        self.prev_action = action.copy()
        self.history["pos"].append(self.pos.copy())
        self.history["vel"].append(self.vel.copy())
        self.history["action"].append(action.copy())

        new_speed = float(np.hypot(*self.vel))
        d_prev = float(np.hypot(*(self.goal - old_pos)))
        d_now = float(np.hypot(*(self.goal - self.pos)))
        reward = (d_prev - d_now) - 0.02 - ACTION_RATE_PENALTY * float(np.sum((action - prev_action) ** 2))
        terminated, outcome = False, "running"
        if collided:
            reward, terminated, outcome = reward - 10.0, True, "collision"
        elif new_speed > V_FELL:
            reward, terminated, outcome = reward - 10.0, True, "fell"
        elif d_now < GOAL_RADIUS:
            reward, terminated, outcome = reward + 10.0, True, "success"
        truncated = (not terminated) and self.t >= MAX_STEPS
        if truncated:
            outcome = "timeout"
        return self._obs(), float(reward), terminated, truncated, {"outcome": outcome}
