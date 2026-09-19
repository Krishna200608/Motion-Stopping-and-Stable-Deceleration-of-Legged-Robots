# Literature Review: Motion Stopping and Stable Deceleration of Legged Robots

**Course:** Foundations of Robotics (Mathematical Foundation of Robotics), BTech IT
**Team:** Krishna Sikheriya, Priyam Jyoti Chakrabarty, Grish Gautam, Nitya Bhavsar, Tavish Chawla,
Biswajit Dash

## Scope

Ten papers were reviewed, prioritizing 2022-2025 work, with two foundational papers from 2012 and
2021 included because they underlie nearly all of the more recent work. Together they motivate
this project's design: model the robot's balance dynamics with the **Linear Inverted Pendulum
(LIP)**, compute the **Capture Point** at every instant, and treat "stopping" as a distinct
control mode with a clear trigger to switch into it.

---

## 1. Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion

- **Authors:** Tairan He, Chong Zhang, Wenli Xiao, Guanqi He, Changliu Liu, Guanya Shi
- **Venue / Year:** RSS 2024 (Best Paper Award)
- **Institutions:** Carnegie Mellon University; ETH Zurich
- **Link:** https://arxiv.org/abs/2401.17583

**Summary.** Trains two neural-network policies for a quadruped: an "agile" policy for fast
running, and a "recovery" policy that slows and stops the robot safely. A value function based on
Hamilton-Jacobi reachability theory decides, in real time, when to switch from agile to recovery
mode as collision risk approaches.

**What we borrow.** The *two-mode* design -- normal locomotion vs. an explicit stopping/recovery
controller -- with a clear, computable trigger for switching between them. We do not use
reachability theory or reinforcement learning; instead we replicate the concept with a simple
"stop command" flag that switches our controller from its walking foot-placement rule to its
Capture Point foot-placement rule (`phase1_math_model/capture_point.py`).

**Relevance.** Frames "stopping" as a distinct control mode, which is exactly how our controller
is structured.

---

## 2. Fall Prediction, Control, and Recovery of Quadruped Robots

- **Authors:** Hao Sun, Junjie Yang, Yinghao Jia, Chong Zhang, Xudong Yu, Changhong Wang
- **Venue / Year:** ISA Transactions, 2024
- **Institutions:** Harbin Institute of Technology; China Academy of Launch Vehicle Technology;
  ETH Zurich
- **Link:** https://doi.org/10.1016/j.isatra.2024.05.039

**Summary.** Builds an online algorithm that predicts, via convex optimization, whether a
quadruped is about to fall, based on N-step capturability (see paper #9 below). If a fall is
predicted, the robot switches to a dedicated fall/stopping controller instead of continuing its
normal gait.

**What we borrow.** *Prediction before action*: at every timestep, ask "can I still stop safely
from this state?" Our speed-sweep experiment (`run_experiments.py`) is a static, offline version
of this question -- for a given starting speed, we determine in advance whether the Capture Point
controller can bring the robot to a safe stop.

**Relevance.** Template for the fall-prediction check that would run inside a real-time control
loop (a natural Phase 2 / future-work extension).

---

## 3. Quadruped Capturability and Push Recovery via a Switched-Systems Characterization of
   Dynamic Balance

- **Authors:** Hua Chen, Zejun Hong, Shunpeng Yang, Patrick M. Wensing, Wei Zhang
- **Venue / Year:** IEEE Transactions on Robotics, 2023
- **Institutions:** Southern University of Science and Technology; University of Notre Dame
- **Link:** https://arxiv.org/abs/2201.11928

**Summary.** Extends N-step capturability (originally developed for bipeds) to quadrupeds, using
a Linear Inverted Pendulum (LIP) model and an explicit Model Predictive Control (MPC) scheme,
validated on a Mini-Cheetah-class robot.

**What we borrow.** **This is our primary mathematical reference.** The LIP model
(`phase1_math_model/lip_model.py`) and the Capture Point formula
(`phase1_math_model/capture_point.py`) implemented in this project come directly from this line
of work.

**Relevance.** Core mathematical foundation for the entire project.

---

## 4. Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking

- **Authors:** Bart van Marum, Aayam Shrestha, Helei Duan, Pranay Dugar, Jeremy Dao, Alan Fern
- **Venue / Year:** IROS 2024
- **Institutions:** Oregon State University
- **Link:** https://arxiv.org/abs/2404.19173

**Summary.** Designs a controller supporting two explicit commands, "Walk" and "Stand" (stop),
and proposes a repeatable benchmarking method for stopping/standing controllers: command
following, disturbance recovery, and energy use.

**What we borrow.** Their benchmarking metrics are the template for how we evaluate our own
controller: does it stop within a bounded number of steps, how far does it drift before stopping,
and (in Phase 2) does it survive a given push.

**Relevance.** Evaluation criteria, not a controller design -- shapes our `run_experiments.py`
success/failure definitions (`stopped` / `not_converged` / `fell`).

---

## 5. Towards Standardized Disturbance Rejection Testing of Legged Robot Locomotion with Linear
   Impactor

- **Authors:** Bowen Weng, Guillermo A. Castillo, Yun-Seok Kang, Ayonga Hereid
- **Venue / Year:** ICRA 2024
- **Institutions:** The Ohio State University
- **Link:** https://arxiv.org/abs/2308.14636

**Summary.** Proposes a standardized mechanical "linear impactor" to reliably and repeatably test
how well a legged robot rejects disturbances, replacing inconsistent hand-pushing.

**What we borrow.** We cannot build a physical impactor, but we replicate it in simulation: a
precisely controlled, repeatable impulse (an instantaneous velocity or force kick) applied to the
simulated robot at a chosen time and magnitude.

**Relevance.** Defines the disturbance-testing protocol for Phase 2
(`phase2_pybullet_sim/disturbance_tests.py`).

---

## 6. A Survey on Control of Humanoid Fall Over

- **Authors:** Rajesh Subburaman, Dimitrios Kanoulas, Nikos Tsagarakis, Jinoh Lee
- **Venue / Year:** Robotics and Autonomous Systems, 2023
- **Institutions:** University College London; Istituto Italiano di Tecnologia; German Aerospace
  Center (DLR); KAIST
- **Link:** https://doi.org/10.1016/j.robot.2023.104443

**Summary.** A broad survey categorizing humanoid fall-handling approaches into three stages: fall
**prediction**, controlled **falling/stopping**, and post-fall **recovery**.

**What we borrow.** The three-stage taxonomy (predict -> control the stop/fall -> recover)
organizes both our code architecture (state estimator -> stop predictor -> controller, see the
project brief's system diagram) and this report's structure.

**Relevance.** Background/structuring reference rather than a specific algorithm.

---

## 7. Learning to Stop: A Unifying Principle for Legged Locomotion in Varying Environments

- **Authors:** Thomas George Thuruthel, G. Picardi, F. Iida, C. Laschi, M. Calisti
- **Venue / Year:** Royal Society Open Science, 2021
- **Institutions:** University of Cambridge; Scuola Superiore Sant'Anna
- **Link:** https://doi.org/10.1098/rsos.210223

**Summary.** Proposes "learning to stop" as a general reinforcement-learning training strategy:
training a controller to reliably stop first makes it learn other locomotion behaviors faster and
more robustly across changing environments.

**What we borrow.** The conceptual framing that stopping is a foundational skill, not an
afterthought -- stated explicitly in this report's motivation.

**Relevance.** Conceptual/motivational; the paper's title matches our project's premise almost
exactly.

---

## 8. Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion

- **Authors:** Yunho Kim, Hyunsik Oh, Jeonghyun Lee, Jinhyeok Choi, Gwanghyeon Ji, Moonkyu Jung,
  Donghoon Youm, Jemin Hwangbo
- **Venue / Year:** IEEE Transactions on Robotics, 2024
- **Institutions:** KAIST
- **Link:** https://arxiv.org/abs/2308.12517

**Summary.** Trains legged robot controllers using explicit safety constraints alongside normal
reward signals, so the controller naturally avoids unsafe/unstable states without hand-tuned
penalty rewards.

**What we borrow.** Conceptual support for using a *hard constraint* (the capture-point condition
and the maximum leg reach) rather than a learned reward. Our controller enforces
"can the foot physically reach the capture point" as a hard clamp
(`phase1_math_model/capture_point.py::foot_placement`) instead of shaping a reward function.

**Relevance.** Supporting design justification.

---

## 9. Capturability-Based Analysis and Control of Legged Locomotion (Parts 1 & 2)

- **Authors:** Twan Koolen, Tomas de Boer, John Rebula, Ambarish Goswami, Jerry Pratt, et al.
- **Venue / Year:** International Journal of Robotics Research, 2012
- **Institution:** Florida Institute for Human & Machine Cognition (IHMC)
- **Link:** https://doi.org/10.1177/0278364912452673

**Summary.** The original, most-cited paper defining **N-step capturability** -- a precise way to
state whether a legged robot can come to a stop without falling within N steps -- and the
**Capture Point**: the exact ground location a foot must reach to stop in one step.

**What we borrow.** **The second pillar of our core math**, alongside paper #3. The Capture Point
formula:

```
Tc = sqrt(z0 / g)
x_capture = x + x_dot * Tc
```

is implemented directly in `phase1_math_model/capture_point.py`, and our speed-sweep experiment
reproduces the paper's N-step capturability idea empirically: the theoretical *1-step* capturable
speed limit (`max_leg_reach / Tc`) is far lower than the *N-step* limit we observe once the
controller is allowed several corrective steps (see `docs/final_report.md` / Phase 1 results).

**Relevance.** Foundational theory; underlies nearly every other paper on this list.

---

## 10. 3D-SLIP Model Based Dynamic Stability Strategy for Legged Robots with Impact Disturbance
    Rejection

- **Authors:** Bin Han, Haoyuan Yi, Zhenyu Xu, Xin Yang, Xin Luo
- **Venue / Year:** Scientific Reports, 2022
- **Institutions:** Huazhong University of Science and Technology; Guangdong Intelligent Robotics
  Institute
- **Link:** https://doi.org/10.1038/s41598-022-09937-9

**Summary.** Uses the 3D Spring-Loaded Inverted Pendulum (SLIP) model and a three-part controller
(touchdown angle control, body attitude control, energy compensation) to keep a running quadruped
stable under sideways impacts.

**What we borrow.** If time permits after the core LIP-based controller is working, the
energy-compensation idea is a good stretch-goal extension for Phase 2. Otherwise, useful
background for modeling terrain/impact disturbances.

**Relevance.** Stretch-goal / background reference for Phase 2's terrain-and-disturbance testing.

---

## Summary Table

| # | Paper (short) | Year | Primary contribution used |
|---|---|---|---|
| 1 | Agile But Safe | 2024 | Two-mode (walk/stop) control structure |
| 2 | Fall Prediction (Sun et al.) | 2024 | "Predict before you act" stopping trigger |
| 3 | Quadruped Capturability (Chen et al.) | 2023 | **LIP model + Capture Point (core math)** |
| 4 | Reward Design for Standing/Walking | 2024 | Evaluation / benchmarking metrics |
| 5 | Linear Impactor | 2024 | Simulated disturbance-testing protocol |
| 6 | Survey on Humanoid Fall Over | 2023 | Predict-Control-Recover taxonomy |
| 7 | Learning to Stop | 2021 | Motivation: stopping as a foundational skill |
| 8 | Not Only Rewards But Also Constraints | 2024 | Hard-constraint design justification |
| 9 | Capturability (Koolen/Pratt et al.) | 2012 | **Capture Point formula (core math)** |
| 10 | 3D-SLIP Model | 2022 | Stretch goal: energy compensation |
