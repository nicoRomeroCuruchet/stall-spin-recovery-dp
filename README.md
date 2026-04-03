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

## Problem

Stall is one of the leading causes of fatal general aviation accidents. Upon entering a stall, the pilot must
execute a recovery maneuver that minimizes altitude loss. This environment formulates stall recovery as an
infinite-horizon optimal control problem:

$$\min_{u(\cdot)} \int_0^{T} g(x, u)\, dt, \qquad g(x, u) = -V \sin\gamma$$

with absorbing state at $\gamma = 0$ (level flight recovered). The state space is 4-dimensional —
$x = (\gamma, V, \alpha, q)$ — and the problem is solved using Proximal Policy Optimization (PPO), a
model-free deep RL algorithm, which scales where exact Dynamic Programming is intractable due to the
curse of dimensionality. The PPO policy is validated against a DP/Value Iteration baseline on simpler
subproblems (reproducing Bunge 2018 results).

---

## Model

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

## Equations of Motion

Full nonlinear EOM in flight path and flow angle representation (Eq. 7, Grillo et al. 2023):

$$\dot{V} = -g \sin\gamma - \frac{(D - T\cos\alpha)\cos\beta - Y\sin\beta}{m} \tag{7a}$$

$$\dot{\gamma} = \frac{L + T\sin\alpha}{mV}\cos\mu - \frac{g}{V}\cos\gamma - \frac{(D - T\cos\alpha)\sin\beta + Y\cos\beta}{mV}\sin\mu \tag{7b}$$

$$\dot{\mu} = (\cos\beta + \tan\beta\sin\beta)(p\cos\alpha + r\sin\alpha) + \left(\sin\mu\tan\gamma + \tan\beta\right)\frac{L + T\sin\alpha}{mV}$$
$$\quad + \frac{(D - T\cos\alpha)\sin\beta + Y\cos\beta}{mV}\cos\mu\tan\gamma - \frac{g}{V}\cos\gamma\cos\mu\tan\beta \tag{7c}$$

$$\dot{\alpha} = q - \sec\beta\left(\frac{L + T\sin\alpha}{mV} - \frac{g}{V}\cos\gamma\cos\mu\right) - \tan\beta\,(p\cos\alpha + r\sin\alpha) \tag{7d}$$

$$\dot{\beta} = \frac{(D - T\cos\alpha)\sin\beta + Y\cos\beta}{mV} + \frac{g}{V}\cos\gamma\sin\mu - (r\cos\alpha - p\sin\alpha) \tag{7e}$$

$$\dot{p} = \frac{M_x}{I_{xx}} - qr\,\frac{I_{zz} - I_{yy}}{I_{xx}} \tag{7f}$$

$$\dot{q} = \frac{M_y}{I_{yy}} + pr\,\frac{I_{zz} - I_{xx}}{I_{yy}} \tag{7g}$$

$$\dot{r} = \frac{M_z}{I_{zz}} - pq\,\frac{I_{yy} - I_{xx}}{I_{zz}} \tag{7h}$$

Under symmetric flight ($\beta = 0$, $\mu = 0$, $p = r = 0$), these collapse to:

$$\dot{V} = -g\sin\gamma - \frac{D - T\cos\alpha}{m}$$

$$\dot{\gamma} = \frac{L + T\sin\alpha}{mV} - \frac{g}{V}\cos\gamma$$

$$\dot{\alpha} = q - \frac{L + T\sin\alpha}{mV} + \frac{g}{V}\cos\gamma$$

$$\dot{q} = \frac{M_y}{I_{yy}}$$

Aerodynamic forces and moments use the same AA-1 Yankee coefficients as Bunge et al. 2018 (Tables 1 & 2).

---

## Reward Function

$$r = r_h + r_{\delta_e} + r_\alpha$$

- $r_h = -V\sin\gamma \cdot dt$ — altitude loss per step (primary objective)
- $r_{\delta_e}$ — elevator rate penalty (control smoothness)
- $r_\alpha = -20$ if $|\alpha| > 40°$ — deep stall penalty (absorbing)

---

## Results

### Optimal Policy (Elevator, Throttle, Altitude Loss)

![Optimal Policy Heatmaps](img/symmetric_stall_Fig6_Stall_Heatmaps.png)

Optimal elevator deflection, throttle command and expected altitude loss as a function of
flight path angle $\gamma$ and angle of attack $\alpha$, for three airspeeds ($V/V_s = 0.9,\,1.0,\,1.1$)
at zero pitch rate. At stalled angles of attack ($\alpha > 15°$) and near-stall speeds the
optimal policy commands full nose-down elevator with full throttle. Once the angle of attack
drops below stall, the policy reverses to pitch-up elevator to complete the pullout.

### Stall Recovery Trajectory

<img src="img/symmetric_stall_Markovian_DP.png" width="500" height="600"/>

Sample recovery from $\gamma = 0°$, $V/V_s = 0.95$, $\alpha = 20°$, $q = 0\,\text{deg/s}$.
The DP policy commands immediate full nose-down elevator and full throttle, reducing $\alpha$
below the stall angle, then transitions to a moderate pitch-up input to recover level flight
with minimal altitude loss.

---

## Nomenclature

| Symbol | Meaning |
|---|---|
| $\rho$ | air density |
| $b$ | wing span |
| $\bar{c}$ | mean aerodynamic chord |
| $S$ | wing surface area |
| $V$ | airspeed |
| $\alpha$ | angle of attack |
| $\beta$ | sideslip angle |
| $\phi, \theta, \psi$ | roll, pitch and yaw angles |
| $\gamma$ | flight path angle |
| $\mu$ | bank angle |
| $p, q, r$ | roll, pitch and yaw rate |
| $\delta_e$ | elevator deflection, positive trailing edge down |
| $\delta_r$ | rudder deflection, positive trailing edge to the left |
| $\delta_a$ | aileron deflection, positive trailing edge down of right aileron |
| $\delta_t$ | throttle position |
| $L, D, Y$ | aerodynamic lift, drag and side force |
| $M_x, M_y, M_z$ | aerodynamic rolling, pitching and yawing moment about the c.g. |
| $C_L, C_D, C_Y$ | lift, drag and side force coefficients |
| $C_l, C_m, C_n$ | rolling, pitching and yawing moment coefficients |
| $T$ | engine thrust |
| $J$ | value function |
| $g$ | stage cost |
| $a$ | vector of actions |
