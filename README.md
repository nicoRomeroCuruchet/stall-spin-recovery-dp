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

| # | Name | Observation Space | Action Space | Constraints | Status | Branch
|---|---|---|---|---|---|---|
| 0 | Base Plane | γ | δe | V = const | — |
| 1 | Reduced Symmetric Glider Pullout | γ, V | CL (or α) | β = 0 | — |
| 1.5 | Symmetric Glider Pullout | γ, V, α, q | δe | β = 0 | — |
| 2 | Symmetric Stall | γ, V, α, q | δe, δt | β = 0 | — | 4dof-symmetric-stall
| 3 | Reduced Symmetric Pullout | γ, V | CL (or α), δt | β = 0 | — |
| 4 | Reduced Banked Glider Pullout | γ, V, μ | CL, μ̇ | β = 0 | — | 3dof-reduced-banked-pullout
| — | ~~Banked Glider Spin~~ | ~~γ, V, α, μ, p, q~~ | ~~δe, δa, δr~~ | ~~δr = 0, β = 0~~ | Planned |
| — | ~~Banked Pullout~~ | ~~γ, V, α, μ, p, q~~ | ~~δe, δa, δt, δr~~ | ~~β = 0~~ | Planned |
| 5 | Banked Spin | γ, V, α, μ, p, q | δe, δa, δt, ~~δr~~ | β = 0 | — |
| 6 | Full Environment | γ, V, α, β, μ, p, q, r | δe, δa, δt, δr | — | — |

