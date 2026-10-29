# Turning Analysis (Work Package 3: Planning-Level Analysis)

This folder extends Phase 1's Linear Inverted Pendulum (LIP) + Capture
Point model from pure stopping to **turning**, as requested in the
project's revised direction (`G2_Progress_Review.docx`), Work Package 3:

> "Extend your pendulum model to turning and derive ... the turning
> envelope: the minimum turning radius at each speed, limited by
> friction (r ≈ v²/μg) and by lateral balance."

## What's here

- `lateral_capture.py` -- the sideways counterpart of Phase 1's
  `capture_point.py`. Same LIP equations, same time constant
  `Tc = sqrt(z0/g)`, applied to the lateral (side-to-side) axis instead
  of the forward axis. This is the standard decoupling result for the
  3D LIP model (Kajita et al., 2001) -- Phase 1 only ever used the
  forward axis.
- `turning_envelope.py` -- derives and computes two independent lower
  bounds on achievable turn radius at a given speed:
  - **Friction limit** `r_friction(v) = v² / (μg)` -- classical circular
    motion, the same reason a car skids if it corners too fast.
  - **Balance limit** `r_balance(v) = v² · step_duration · Tc / max_leg_reach`
    (derived here) -- a legged robot balances on one foot at a time, so
    even with enough ground grip, its one-legged balance may not be able
    to redirect momentum sideways fast enough within a single step. This
    is the part the review report flagged as missing from the
    literature: *"No work asks when a biped must commit to a direction,
    given its speed and turning limits."*

  The actual turning envelope is whichever bound is more restrictive:
  `r_min(v) = max(r_friction(v), r_balance(v))`.
- `run_turning_analysis.py` -- sweeps speed from 0.2 to 4.0 m/s (same
  range as Phase 1) across three floor-friction conditions and plots the
  turning envelope. Generates `plots/turning_envelope.png` and
  `turning_envelope.csv`.

## Result

Using Phase 1's own physical parameters (`z0=0.8 m`, `step_duration=0.4 s`,
`max_leg_reach=0.55 m`):

- On a **low-grip floor** (μ=0.3), the **friction** limit always binds --
  turning is limited by how well the foot grips the ground.
- On a **typical or high-grip floor** (μ≥0.6), the **balance** limit
  always binds instead -- the robot's one-legged balance, not the floor,
  is what limits how tightly it can turn.

This gives the team a concrete, falsifiable baseline: once a trained
turning policy exists (Work Packages 1-2), its measured turning radius
at each speed can be checked against this envelope (RQ1 in the review
report).

## Decision deadline and U-turn corridor

- `decision_deadline.py` -- builds on the turning envelope plus Phase 1's
  single-step limit. Full derivation is in the module docstring.
  - **Swerve distance**: with obstacle width `w` and clearance `c`, the
    robot must shift sideways by `L = w/2 + c`. On the tightest arc
    `r = r_min(v)` this takes `sqrt(L(2r - L))` of forward distance
    (`r` if `L >= r`).
  - **Decision deadline**: the last distance at which the robot must
    commit. The robot may brake first (max deceleration
    `a_max = min(mu*g, (max_leg_reach/Tc)/step_duration)`) to a speed
    `v' >= min_turn_speed` (0.5 m/s, since stopping is out of scope),
    then swerve: `min over v' of (v^2 - v'^2)/(2 a_max) + swerve(v')`.
  - **U-turn corridor**: `W = 2 r_min(v) + body_width`, plus the inverse
    (fastest U-turn speed for a given corridor) and the braking distance
    needed to get down to it.
- `run_deadline_analysis.py` -- generates `plots/decision_deadline.png`,
  `plots/uturn_corridor.png`, `plots/feasibility_map.png`, and
  `decision_deadline.csv`, `uturn_corridor.csv`, `uturn_entry_braking.csv`,
  `feasibility_table.csv` (every (speed, distance, width, grip)
  combination, flagged feasible or not).

### Findings (z0=0.8, step 0.4 s, reach 0.55 m, clearance 0.3 m)

- **If the robot keeps its speed**, the deadline grows with speed and
  obstacle width. On a typical floor at 4 m/s: 1.8 m (0.5 m wide) up to
  3.2 m (4 m wide). At 2 m/s it is only 0.8 m.
- **Braking first beats turning at speed.** The turn radius and the
  braking distance both scale with `v^2`, but braking is cheaper, so the
  brake-then-turn deadline is much shorter (1.7 m at 4 m/s on a typical
  floor) and almost independent of obstacle width, because at 0.5 m/s the
  turn radius is under 0.1 m. Real robots will not turn that sharply at
  low speed; this is a point-mass limit, and the learned policy
  (RQ2) is where it gets tested.
- **Grip matters only on a low-grip floor.** For mu >= 0.6 the balance
  limit binds both turning and braking, so mu = 0.6 and 0.9 give
  identical results.
- **U-turn corridor**: about 2.2 m at 2 m/s and 7.1 m at 4 m/s
  (typical or high grip); 11.4 m at 4 m/s on the low-grip floor.

### Assumptions

Point mass, flat ground, obstacle centred on the path, brake then turn
(no braking during the turn), instant reaction (`reaction_time` is
available), turn radius at any speed taken from the envelope.

## Quick start

```bash
cd turning_analysis
python run_turning_analysis.py
python run_deadline_analysis.py
```
