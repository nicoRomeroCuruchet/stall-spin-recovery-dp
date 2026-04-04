# 4DOF Symmetric Stall Recovery — Riley (1985) Aerodynamic Model

Research code for optimal stall recovery using GPU-accelerated Policy Iteration.
The problem is formulated as an infinite-horizon optimal control problem solved via
massively parallel Policy Iteration on continuous-state MDPs. Dynamics are integrated
on-the-fly using 4th-order Runge-Kutta entirely within GPU registers, avoiding the
memory-bound limitations of traditional transition table methods.

This branch uses the **full nonlinear aerodynamic model** of the Grumman AA-1 Yankee
derived from wind-tunnel measurements reported in Riley (1985), NASA TM-86309. Unlike
the linear approximation used in other branches, all aerodynamic coefficients are
tabulated as a function of angle of attack $\alpha$ and thrust coefficient $C_T$, capturing
the nonlinear stall and post-stall behavior of the aircraft.

> **Aerodynamic model reference:**
> Riley, D. R. (1985).
> *Simulator Study of the Stall Departure Characteristics of a Light General Aviation
> Airplane With and Without a Wing-Leading-Edge Modification.*
> NASA Technical Memorandum 86309, Langley Research Center, Hampton, Virginia.

> **Optimal control reference:**
> Grillo, C., Torre, F., & Bunge, R. A. (2023).
> *Optimal Stall Recovery via Deep Reinforcement Learning for a General Aviation Aircraft.*
> AIAA SciTech Forum, National Harbor, MD.

---

### Running

Train the policy (or load from cache if `SymmetricStall_policy.npz` exists) and generate all figures:

```bash
python main.py
```

Output files:

| File | Description |
|---|---|
| `SymmetricStall_policy.npz` | Trained value function and policy |
| `img/symmetric_stall_trajectory.png` | Recovery trajectory (7-panel) |
| `img/symmetric_stall_heatmaps.png` | Optimal policy heatmaps |

---

## Aerodynamic Model

### Background

Riley (1985) developed a six-degree-of-freedom nonlinear simulation of the Grumman AA-1
Yankee, a two-place, single-engine, low-wing general aviation aircraft, for the stall and
initial departure region of flight. The mathematical model was established from
**full-scale powered wind-tunnel tests** conducted at the NASA Langley 30- by 60-Foot
Tunnel. The tests covered the complete angle-of-attack range from $-10°$ to $+40°$ and
included both power-off ($C_T = 0$) and power-on ($C_T = 0.5$) conditions.

The key feature of this model is that **thrust effects are not added as a separate term
in the equations of motion**. Instead, the propeller slipstream modifies the aerodynamic
environment over the wing, so its effect is absorbed directly into the aerodynamic
coefficient tables indexed by the thrust coefficient $C_T$. This coupling is particularly
significant at high angles of attack and near-stall speeds.

### Aerodynamic Coefficient Tables

Each longitudinal coefficient is tabulated at 14 angle-of-attack breakpoints:

$$\alpha \in \{-10°,\ -5°,\ 0°,\ 5°,\ 10°,\ 12°,\ 14°,\ 16°,\ 18°,\ 20°,\ 25°,\ 30°,\ 35°,\ 40°\}$$

and at two thrust coefficient values, $C_T = 0$ (power-off) and $C_T = 0.5$ (power-on).
During simulation, a **bilinear interpolation** is performed: first linearly in $\alpha$
within each $C_T$ table, then linearly between the two $C_T$ tables:

$$f(\alpha, C_T) = f(\alpha,\, 0) + \frac{C_T}{0.5}\,\bigl[f(\alpha,\, 0.5) - f(\alpha,\, 0)\bigr]$$

The six coefficients used in the symmetric flight model are:

| Coefficient | Symbol | Description |
|---|---|---|
| Lift (baseline) | $C_{L,o}(\alpha, C_T)$ | Nonlinear lift; flat-top plateau 14°–18° at $C_T=0$ ($C_{L,\max}=1.26$) |
| Lift (pitch rate) | $C_{L,q}(\alpha, C_T)$ | Pitch-rate damping contribution to lift |
| Lift (elevator) | $C_{L,\delta_e}(\alpha, C_T)$ | Elevator effectiveness on lift |
| Drag (baseline) | $C_{D,o}(\alpha, C_T)$ | Strong post-stall rise; negative values at low $\alpha$ for $C_T=0.5$ (propulsive) |
| Pitching moment (baseline) | $C_{m,o}(\alpha, C_T)$ | Nose-down moment increasing with $\alpha$ |
| Pitching moment (pitch rate) | $C_{m,q}(\alpha, C_T)$ | Pitch damping; large negative values near stall |
| Pitching moment (elevator) | $C_{m,\delta_e}(\alpha, C_T)$ | Elevator effectiveness on pitching moment |

The total lift, drag and pitching moment coefficients used in the EOM are assembled as:

$$C_L = C_{L,o} + C_{L,\delta_e}\,\delta_e + C_{L,q}\,\hat{q}$$

$$C_D = C_{D,o}$$

$$C_m = C_{m,o} + C_{m,\delta_e}\,\delta_e + C_{m,q}\,\hat{q}$$

where the non-dimensional pitch rate is $\hat{q} = q\bar{c}/(2V)$.

### Thrust Coefficient

The thrust coefficient $C_T$ is computed from the throttle command $\delta_t$ and the
current airspeed $V$:

$$C_T = \min\!\left(0.5,\;\max\!\left(0,\;\frac{K_t\,\delta_t}{\frac{1}{2}\rho V^2 S}\right)\right)$$

where $K_t$ is calibrated so that full throttle ($\delta_t = 1$) produces level flight at
twice the stall speed ($V = 2V_s$).

### Physical Constants (Riley 1985, Table I)

| Parameter | Symbol | Value |
|---|---|---|
| Mass | $m$ | 715.21 kg |
| Wing area | $S$ | 9.1147 m² |
| Mean aerodynamic chord | $\bar{c}$ | 1.22 m |
| Wing span | $b$ | 8.066 m |
| Pitch inertia | $I_{yy}$ | 1000.60 kg·m² |
| Roll inertia | $I_{xx}$ | 808.06 kg·m² |
| Air density (sea level) | $\rho$ | 1.225 kg/m³ |

---

## Equations of Motion

Under **symmetric flight assumptions** ($\beta = 0$, $\mu = 0$, $p = r = 0$), the full
6DOF nonlinear rigid-body equations reduce to a 4-state system in wind-axis
representation:

$$\dot{V} = -g\sin\gamma - \frac{D}{m}$$

$$\dot{\gamma} = \frac{L}{mV} - \frac{g}{V}\cos\gamma$$

$$\dot{\alpha} = q - \dot{\gamma}$$

$$\dot{q} = \frac{M_y}{I_{yy}}$$

where aerodynamic forces and moment are:

$$L = \tfrac{1}{2}\rho V^2 S\, C_L(\alpha, C_T, \delta_e, \hat{q})$$

$$D = \tfrac{1}{2}\rho V^2 S\, C_D(\alpha, C_T)$$

$$M_y = \tfrac{1}{2}\rho V^2 S\bar{c}\, C_m(\alpha, C_T, \delta_e, \hat{q})$$

Note that thrust effects are fully embedded in $C_L$, $C_D$ and $C_m$ through the
$C_T$ parameter, consistent with the Riley (1985) formulation. No explicit thrust term
appears in the equations of motion.

The state and control variables are:

| Variable | Symbol | Description |
|---|---|---|
| Flight path angle | $\gamma$ | angle between velocity vector and horizon |
| Normalized airspeed | $V/V_s$ | airspeed normalized by stall speed |
| Angle of attack | $\alpha$ | angle between velocity vector and body x-axis |
| Pitch rate | $q$ | body-axis pitch rate |
| Elevator deflection | $\delta_e$ | control input, $[-25°,\, +15°]$ |
| Throttle | $\delta_t$ | control input, $[0,\, 1]$ |

Integration is performed with a fixed-step **4th-order Runge-Kutta** scheme at
$dt = 0.01\,\text{s}$.

---

## Discretization

### State Space — 3,388,896 nodes

| State | Min | Max | Bins | Resolution |
|---|---|---|---|---|
| $\gamma$ | $-90°$ | $5°$ | 56 | $\approx 1.7°$ |
| $V/V_s$ | $0.9$ | $2.0$ | 41 | $0.028$ |
| $\alpha$ | $-14°$ | $20°$ | 36 | $\approx 0.97°$ |
| $q$ | $-50°/\text{s}$ | $50°/\text{s}$ | 41 | $\approx 2.5°/\text{s}$ |

Total: $56 \times 41 \times 36 \times 41 = 3{,}388{,}896$ states.

### Action Space — 147 actions

| Control | Min | Max | Bins |
|---|---|---|---|
| $\delta_e$ | $-25°$ | $15°$ | 21 |
| $\delta_t$ | $0$ | $1$ | 7 |

Total: $21 \times 7 = 147$ discrete actions.

### Terminal Conditions

| Condition | Type |
|---|---|
| $\gamma \geq 0°$ | Success (absorbing) |
| $\|\alpha\| \geq 40°$ | Failure — deep stall / structural limit |
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

![Optimal Policy Heatmaps](img/symmetric_stall_heatmaps.png)

Optimal elevator deflection, throttle command and expected altitude loss as a function of
$\gamma$ and $\alpha$, for three airspeeds ($V/V_s = 0.9,\,1.0,\,1.1$) at zero pitch rate.

### Stall Recovery Trajectory

<img src="img/symmetric_stall_trajectory.png" width="850"/>

Sample recovery from $\gamma = 0°$, $V/V_s = 0.95$, $\alpha = 20°$, $q = 0\,\text{deg/s}$.

---

## Installation

**Requirements:** Python 3.10+, NVIDIA GPU with CUDA-capable driver.

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install CuPy (GPU backend)

CuPy must be installed separately because the correct wheel depends on your CUDA driver version.
Check your CUDA version with:

```bash
nvidia-smi
```

Then install the matching wheel:

| CUDA version (nvidia-smi) | Install command |
|---|---|
| 11.x | `pip install cupy-cuda11x` |
| 12.x | `pip install cupy-cuda12x` |
| 13.x | `pip install cupy-cuda12x` *(use 12x, backward-compatible)* |

> CuPy does not require the full CUDA toolkit (`nvcc`) — only the NVIDIA driver.
> If you get a `libnvrtc.so not found` error, install the CUDA runtime libraries:
> ```bash
> pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
> ```
