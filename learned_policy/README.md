# Learned turning policy (Work Packages 1-2, scaled down)

A small PPO policy that sees an obstacle ahead through a ray-cast distance
scan, slows down and chooses left or right, trained on CPU in minutes.
It answers RQ2 and RQ3 on a **reduced-fidelity model**; RQ1 (the G1
humanoid port of SmoothTurn) is **not done**.

## What is here

- `turn_env.py` -- gymnasium environment. The robot is the validated
  step-level LIP from `turning_analysis/lip_step_model.py`: one environment
  step is one 0.4 s stance phase, the action picks the wanted end-of-step
  velocity, and the foot offset needed is clamped to
  `min(leg reach, mu * z0)`. So the policy can only brake and turn as hard
  as the LIP allows (`turning_analysis/validate_model.py` checks this
  against a numerically integrated simulation).
  Observation: 16 rays over 360 degrees, speed, goal direction and distance,
  previous action; the `deadline_aware` condition adds the distance left
  beyond the analytic decision deadline and the deadline itself (nominal
  2.5 m obstacle) -- the "lookahead" input for RQ3.
- `train.py`, `run_all.py` -- PPO (stable-baselines3, 2x128 MLP,
  8 environments). `python run_all.py 300000 3` trains 2 conditions x 3 seeds
  in parallel; `python run_all.py 400000 3 hard` trains on the hard scenarios.
- `evaluate.py` -- runs every model on 300 fixed scenarios and writes
  `eval_*.csv` and `plots/`. `models/` holds the trained policies.

## Scenarios

A block obstacle (width 1-4 m), centred on the path half the time (left and
right equally good) and offset otherwise, with the goal beyond it.
Arena walls at y = +-7 m. mu = 0.9.

- **standard**: obstacle 3-8 m ahead, start speed 1.5-2.8 m/s.
- **hard**: obstacle 1-5 m ahead, start speed 2.0-3.0 m/s, so many cases sit
  at or below the analytic deadline; evaluated with Gaussian noise (0.3 m)
  on the rays to stress the left/right tie-break.

## Results (3 seeds per condition, 300 episodes each, mean over seeds)

| | standard base | standard deadline-aware | hard base | hard deadline-aware |
|---|---|---|---|---|
| success | 100% | 99.9% | 90.4% | 90.4% |
| success on analytically feasible cases | 100% | 99.9% | 97.4% | 97.4% |
| success on infeasible cases | -- | -- | 3.0% | 3.0% |
| scenarios that are feasible | 100% | 100% | 92.7% | 92.7% |
| speed within 5 m of the obstacle (m/s) | 1.85 | 2.03 | 1.93 | 2.09 |
| commit distance (m) | 3.9 | 4.0 | 1.6 | 1.7 |
| committed after the analytic deadline | 0% | 0.1% | 11.6% | 13.8% |
| side changes on symmetric obstacles | 0.00 | 0.00 | 0.00 | 0.00 |

Per-seed numbers: `eval_per_seed.csv`, `eval_per_seed_hard.csv`. Plots:
`plots/policy_metrics*.png`, `plots/commit_vs_deadline*.png`.

**RQ2 (does the policy commit before the analytic deadline?)** Yes in the
standard scenarios (about 4 m out against deadlines under 1.7 m). On the
hard scenarios it commits about 1.7 m out and 12-14% of commitments fall
after the analytic deadline, yet 97% of feasible cases still succeed. The
policy's lateral deviation is measured at 0.6 m, which is after steering
starts, so "late" is biased upward; the deadline also assumes the tightest
possible turn.

**RQ3 (dithering / freezing, and does lookahead help?)** No dithering and no
freezing (no timeouts) was found, even with ray noise, and giving the policy
the analytic deadline made no measurable difference (differences are within
seed-to-seed variation; with 3 seeds we cannot claim an effect). Each trained
model breaks the symmetric tie the same way every time (one fixed side per
model). **This is a weak test of RQ3**: the goal is always ahead, there are no
dead ends, so the classic local-minimum trap cannot occur.

## Caveats (read before citing)

- Point-mass LIP, not the G1 humanoid, and a flat 2D arena.
- Braking is cheap in this model and turns are almost free at low speed, so
  the policy mostly learns "slow down, then go around". Its speed near the
  obstacle is 1.9-2.1 m/s on average.
- A first version without an action-rate penalty learned to alternate +-1
  turn commands every step (bang-bang) and weave; the foot clamp averaged it
  into a net swerve. It "succeeded" but made action-based dithering metrics
  meaningless, so the final runs use an action-rate penalty (0.05 per squared
  action change) and path-based side and commit metrics.
- Turn-back (U-turn) scenarios are not trained; the U-turn corridor is
  analytic only (`turning_analysis/`).
- Seeds: 3 per condition. No significance tests.
