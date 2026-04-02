# stall-spin

Research code for aircraft stall and spin upset recovery using VRAM-accelerated Dynamic Programming.
The core approach solves the minimal altitude loss recovery problem as an infinite-horizon optimal control
problem via massively parallel Policy Iteration on continuous-state MDPs. Dynamics are integrated
on-the-fly using 4th-order Runge-Kutta entirely within GPU registers, avoiding the memory-bound
limitations of traditional transition table methods. Reference aircraft: **Grumman AA-1 Yankee**
(Riley 1985, NASA TM-86309).

---

## Environments

Each environment is a branch in this repository. Complexity increases with index; strikethrough entries
are planned but not yet implemented.

| # | Name | Observation Space | Action Space | Constraints | Status |
|---|---|---|---|---|---|
| 0 | Base Plane | γ | δe | V = const | — |
| 1 | Reduced Symmetric Glider Pullout | γ, V | CL (or α) | β = 0 | — |
| 1.5 | Symmetric Glider Pullout | γ, V, α, q | δe | β = 0 | — |
| 2 | Symmetric Stall | γ, V, α, q | δe, δt | β = 0 | — |
| 3 | Reduced Symmetric Pullout | γ, V | CL (or α), δt | β = 0 | — |
| 4 | Reduced Banked Glider Pullout | γ, V, μ | CL, μ̇ | β = 0 | — |
| — | ~~Banked Glider Spin~~ | ~~γ, V, α, μ, p, q~~ | ~~δe, δa, δr~~ | ~~δr = 0, β = 0~~ | Planned |
| — | ~~Banked Pullout~~ | ~~γ, V, α, μ, p, q~~ | ~~δe, δa, δt, δr~~ | ~~β = 0~~ | Planned |
| 5 | Banked Spin | γ, V, α, μ, p, q | δe, δa, δt, ~~δr~~ | β = 0 | Implemented |
| 6 | Full Environment | γ, V, α, β, μ, p, q, r | δe, δa, δt, δr | — | — |

---

## Nomenclature

| Symbol | Meaning |
|---|---|
| ρ | air density |
| b | wing span |
| c | chord length |
| S | wing surface area |
| px, py, pz | northward, eastward and down position |
| h | altitude, from the ground |
| u, v, w | body-x, y and z velocity |
| V | airspeed |
| α | angle of attack |
| β | sideslip angle |
| φ, θ, ψ | roll, pitch and yaw angles |
| γ | flight path angle |
| μ | bank angle |
| p̂ | dimensionless roll rate, p̂ = pb/2V |
| p, q, r | roll, pitch and yaw rate |
| δe | elevator deflection, positive trailing edge down |
| δr | rudder deflection, positive trailing edge to the left |
| δa | aileron deflection, positive trailing edge down of right aileron |
| δt | throttle position |
| L, D, Y | aerodynamic lift, drag and side force |
| Mx, My, Mz | aerodynamic rolling, pitching and yawing moment about the c.g. |
| CL, CD, CY | aerodynamic lift, drag and side force coefficient |
| Cl, Cm, Cn | aerodynamic rolling, pitching and yawing moment coefficient about the c.g. |
| f | system dynamic equation of motion |
| J | value function |
| g | stage cost |
| a | vector of actions |

