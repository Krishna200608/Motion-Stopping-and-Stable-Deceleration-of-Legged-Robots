# Draft message to the reviewer (not sent)

**Subject: Group 2 – turning analysis update, and decisions we need from you**

Hello,

Since your review we have redirected the project to obstacle-triggered
slowing and turning for a biped. Status, all in the repo
(`turning_analysis/`, `learned_policy/`):

**Done**
- Turning envelope and decision deadline / U-turn corridor derived (work
  package 3).
- We checked them against a simulated Linear Inverted Pendulum. Phase 1's
  single-step limit (1.926 m/s) is confirmed in simulation. Our first
  closed-form bounds were too optimistic above ~2 m/s (the pendulum is
  unstable, so a long stance phase cannot keep braking), so we replaced them
  with exact one-step LIP limits that match the simulation to about 0.1 m.
- A small PPO policy (point-mass LIP, 16-ray distance scan, left/right
  choice) on CPU, 3 seeds per condition, with and without the analytic
  deadline as an input.

**Not done (scoped out for time)**
- The G1 humanoid port of the SmoothTurn recipe in MuJoCo Playground or
  Isaac Lab (work package 1), so RQ1 is not answered yet.
- Rendered depth, and a "turn back" dead-end scenario in the learned policy.

**Decisions we need from you**
1. Is the reduced-fidelity LIP environment acceptable as the final
   evaluation for RQ2 and RQ3, or must these be answered on the G1?
2. If the G1 is required: MuJoCo Playground or Isaac Lab, and is there GPU
   access we can use (SmoothTurn trains in ~2.5 h on an RTX 3060)?
3. We assumed the robot may not drop below 0.5 m/s while turning (stopping is
   out of scope). Is that threshold reasonable?

Could we have 20 minutes this week with the whole team?

Thank you,
Group 2
