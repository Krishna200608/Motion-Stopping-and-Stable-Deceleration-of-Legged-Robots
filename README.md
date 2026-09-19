# Motion Stopping and Stable Deceleration of Legged Robots

Course project for Foundations of Robotics (Mathematical Foundation of Robotics), BTech IT.

**Team:** Krishna Sikheriya (IIT2023139),  Grish Gautam (IIT2023131), Nitya Bhavsar (IIT2023140), Priyam Jyoti Chakrabarty (IIT2023147), Tavish Chawla (IIT2023150),
Biswajit Dash (MRM2025003)

## What this is

A legged robot walking forward has momentum. Simply locking its legs to stop is like tripping --
the momentum carries the body forward and it falls. This project implements and tests the
**Capture Point** method: computing exactly where the next foot must land to bring the robot to a
stable stop, instead of falling.


See [`docs/literature_review.md`](docs/literature_review.md) for the ten papers this project is
based on.

## Repository structure

```
phase1_math_model/
    lip_model.py          # LIP dynamics: x_ddot = (g/z0) * (x - p)
    capture_point.py       # Capture point formula + smart stopping/walking foot placement
    naive_controller.py    # Baseline "lock the legs" controller (the failure case)
    run_experiments.py     # Speed sweep, results table, and all plots
    plots/                 # Generated graphs (created by run_experiments.py)
    results.csv             # Generated results table (created by run_experiments.py)
docs/
    literature_review.md   # Full write-up of the 10 reviewed papers
requirements.txt
```

## Quick start

```bash
python -m venv venv
venv\Scripts\activate        # on Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cd phase1_math_model
python run_experiments.py
```

This generates `results.csv` and all plots under `phase1_math_model/plots/`. See `MANUAL.md` for
more detail, including what each plot shows and how to interpret it.

## The core math

```
LIP acceleration:   x_ddot = (g / z0) * (x - p)
Time constant:      Tc = sqrt(z0 / g)
Capture point:      x_capture = x + x_dot * Tc
```

where `x` is the center-of-mass horizontal position, `x_dot` its velocity, `p` the current stance
foot position, `z0` the (constant) center-of-mass height, and `g = 9.81 m/s^2`.
