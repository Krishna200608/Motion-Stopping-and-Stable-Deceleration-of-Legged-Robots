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

## Quick start

```bash
cd turning_analysis
python run_turning_analysis.py
```
