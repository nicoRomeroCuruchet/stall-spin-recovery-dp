# 3DOF Reduced Banked Pullout Model

Research code for aircraft stall and spin upset recovery using VRAM-accelerated Dynamic Programming.
The core approach solves the minimal altitude loss recovery problem as an infinite-horizon optimal control
problem via massively parallel Policy Iteration on continuous-state MDPs. Dynamics are integrated
on-the-fly using 4th-order Runge-Kutta entirely within GPU registers, avoiding the memory-bound
limitations of traditional transition table methods. Reference aircraft: **Grumman AA-1 Yankee**
(Riley 1985, NASA TM-86309).

---

## 3DOF Reduced Banked Pullout Model

*Based on: Bunge, Pavone & Kroo, "Minimal Altitude Loss Pullout Maneuvers," AIAA GNC 2018.*

### Model Description

This branch implements the reduced-order 3-DOF point-mass model from the paper, derived from
the full 6-DOF equations under the following simplifying assumptions:

- **β ≈ 0**: sideslip angle remains near zero throughout the maneuver.
- **CL and μ̇ are directly commanded** by inner-loop controllers (high-bandwidth, dynamics neglected).
- **CD = CD(CL)**: drag is a function of lift coefficient only (no sideslip dependency).
- **CY ≈ 0**: lateral aerodynamic side force is negligible.
- **Idle power** (δt not a control input; this investigation is limited to idle power maneuvers).

Under these assumptions the full equations of motion (Appendix A.2 of the paper) reduce to a
3-state system with state **x = (V, γ, μ)** and control **a = (CL_cmd, μ̇_cmd)**.

### Equations of Motion

```
V̇  = -g sin γ  -  (1/2) ρ (S/m) V² CD(CL_cmd)          (3a)

γ̇  =  (1/2) ρ (S/m) V CL_cmd cos μ  -  (g/V) cos γ      (3b)

μ̇  =  μ̇_cmd                                               (3c)
```

Where CD is a quadratic function of CL (stability-derivative model, Appendix B of the paper):

```
CD = CD₀ + CD_α α  +  CD_α² α²
```

### Control Bounds

To prevent secondary stalls, CL_cmd is constrained within a safety margin of 0.2 from the
stall limits (positive stall CL ≈ 1.2, negative stall CL ≈ −0.7 for the AA-1 Yankee):

```
-0.5  ≤  CL_cmd  ≤  1.0                                   (4)
```

The bank rate command is bounded by the steady-state roll rate achievable with maximum aileron
deflection at the stall speed reference:

```
|μ̇_cmd|  ≤  μ̇_max  ≈  p_max                              (5a)

p_max  ≈  p̂_max (2 V_ref / b)  =  |Cl_δa / Cl_p| δa_max (2 V_ref / b)   (5b)
```

For the AA-1 Yankee: Cl_p ≈ −0.5, Cl_δa ≈ −0.0595 1/deg, δa_max = 25 deg, b = 7.41 m,
Vs ≈ 32 m/s → **p_max ≈ 30 deg/s**.

### State-Space Discretization (Table 1 of the paper)

| Variable | Lower Bound | Increment | Upper Bound | Units |
|---|---|---|---|---|
| V | 0.9 | 0.1 | 4.0 | 1/Vs |
| γ | −180 | 5 | 0 | deg |
| μ | −20 | 5 | 200 | deg |
| CL_cmd | −0.5 | 0.25 | 1.0 | — |
| μ̇_cmd | −30 | 5 | 30 | deg/s |

### Optimal Control Problem

The minimum altitude-loss pullout is cast as an infinite-horizon optimal control problem
(absorbing state at level flight, γ = 0) with stage cost **g(x, a) = −V sin γ Δt + 0.01 μ̇_cmd²**
and solved via value iteration (Bellman equation):

```
J*(x) = min  { g(x, a) + J*(x'(x, a)) }
         a∈A
```

The extra quadratic term on μ̇_cmd smooths the bank-rate policy, which would otherwise be
purely bang-bang.

### Key Results (from the paper)

- The optimal policy is **bang-bang** in both CL and μ̇: controls switch between their
  limits upon crossing switching surfaces in state space.
- For most initial conditions the optimal action is to **roll back to wings-level** (positive
  CL, roll back) and execute an upright pullout.
- For near-inverted initial attitudes (μ ≈ 150–180 deg) it can be optimal to **roll over and
  perform an inverted pullout** (negative CL), which is cheaper in altitude than rolling back.
- Minimum altitude loss is highly sensitive to maximum achievable CL: every reduction of 0.1
  in CL_max costs approximately 20 m of additional altitude loss.