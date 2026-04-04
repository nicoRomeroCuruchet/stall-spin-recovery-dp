# 4DOF Symmetric Stall Recovery

Research code for aircraft stall upset recovery using Deep Reinforcement Learning (PPO) and Dynamic Programming.
The core approach solves the minimal altitude loss recovery problem as an infinite-horizon optimal control problem.
Reference aircraft: **Grumman AA-1 Yankee** (Riley 1985, NASA TM-86309).

> **Reference paper:**
> Grillo, C., Torre, F., & Bunge, R. A. (2023).
> *Optimal Stall Recovery via Deep Reinforcement Learning for a General Aviation Aircraft.*
> AIAA SciTech Forum, National Harbor, MD.
> Universidad de San Andrés, Argentina.

---

### Running

Train the policy (or load from cache if `results/SymmetricStall_policy.npz` exists) and generate all figures:

```bash
python main.py
```

Output is written to `results/`:

| File | Description |
|---|---|
| `SymmetricStall_policy.npz` | Trained value function and policy |
| `symmetric_stall_trajectory.png` | Recovery trajectory |
| `symmetric_stall_heatmaps.png` | Optimal policy heatmaps |

---

## Equations of Motion

Full nonlinear EOM in flight path and flow angle representation with symmetric flight assumptions:

$$\dot{V} = -g\sin\gamma - \frac{D - T\cos\alpha}{m}$$

$$\dot{\gamma} = \frac{L + T\sin\alpha}{mV} - \frac{g}{V}\cos\gamma$$

$$\dot{\alpha} = q - \frac{L + T\sin\alpha}{mV} + \frac{g}{V}\cos\gamma$$

$$\dot{q} = \frac{M_y}{I_{yy}}$$

Aerodynamic forces and moments use the same AA-1 Yankee coefficients as Bunge et al. 2018 (Tables 1 & 2).



**Symmetric flight assumptions:** $\beta = 0$, $\mu \approx 0$, $p = r = 0$.

Under these assumptions the full 8-state nonlinear EOM reduce to a 4-state system:

| State | Symbol | Description |
|---|---|---|
| Flight path angle | $\gamma$ | angle between velocity vector and horizon |
| Airspeed | $V$ | total airspeed |
| Angle of attack | $\alpha$ | angle between velocity and body x-axis |
| Pitch rate | $q$ | body-axis pitch rate |

| Control | Symbol | Description |
|---|---|---|
| Elevator deflection | $\delta_e$ | positive trailing edge down |
| Throttle | $\delta_t$ | engine thrust command |

---

## Discretization

### State Space — 3,388,896 nodes

| State | Symbol | Min | Max | Bins | Resolution |
|---|---|---|---|---|---|
| Flight path angle | $\gamma$ | $-90°$ | $5°$ | 56 | $\approx 1.7°$ |
| Normalized airspeed | $V/V_s$ | $0.9$ | $2.0$ | 41 | $0.028$ |
| Angle of attack | $\alpha$ | $-14°$ | $20°$ | 36 | $\approx 0.97°$ |
| Pitch rate | $q$ | $-50\,°/s$ | $50\,°/s$ | 41 | $\approx 2.5\,°/s$ |

Total: $56 \times 41 \times 36 \times 41 = 3{,}388{,}896$ states.

### Action Space — 147 actions

| Control | Min | Max | Bins | Resolution |
|---|---|---|---|---|
| Elevator $\delta_e$ | $-25°$ | $15°$ | 21 | $2°$ |
| Throttle $\delta_t$ | $0$ | $1$ | 7 | $\approx 0.17$ |

Total: $21 \times 7 = 147$ discrete actions.

### Terminal Conditions

| Condition | Type |
|---|---|
| $\gamma \geq 0°$ | Success (absorbing) |
| $|\alpha| \geq 40°$ | Failure — deep stall / structural limit |
| $\gamma \leq -175°$ | Failure — unrecoverable dive |

### Solver

| Parameter | Value |
|---|---|
| Discount factor | $1.0$ (undiscounted) |
| Convergence threshold $\theta$ | $5 \times 10^{-6}$ |
| Max iterations | $1000$ |
| Integration step $dt$ | $0.01\,\text{s}$ |
| Interpolation | 4D Barycentric (CUDA) |

---

## Results

### Optimal Policy (Elevator, Throttle, Altitude Loss)

![Optimal Policy Heatmaps](results/symmetric_stall_heatmaps.png)

Optimal elevator deflection, throttle command and expected altitude loss as a function of
flight path angle $\gamma$ and angle of attack $\alpha$, for three airspeeds ($V/V_s = 0.9,\,1.0,\,1.1$)
at zero pitch rate. At stalled angles of attack ($\alpha > 15°$) and near-stall speeds the
optimal policy commands full nose-down elevator with full throttle. Once the angle of attack
drops below stall, the policy reverses to pitch-up elevator to complete the pullout.

### Stall Recovery Trajectory

<img src="results/symmetric_stall_trajectory.png" width="800"/>

Sample recovery from $\gamma = 0°$, $V/V_s = 0.95$, $\alpha = 20°$, $q = 0\,\text{deg/s}$.
The DP policy commands immediate full nose-down elevator and full throttle, reducing $\alpha$
below the stall angle, then transitions to a moderate pitch-up input to recover level flight
with minimal altitude loss.
