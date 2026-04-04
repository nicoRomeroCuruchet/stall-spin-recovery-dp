# stall-spin

Research code for aircraft stall and spin upset recovery using VRAM-accelerated Dynamic Programming.
The core approach solves the minimal altitude loss recovery problem as an infinite-horizon optimal control
problem via massively parallel Policy Iteration on continuous-state MDPs. Dynamics are integrated
on-the-fly using 4th-order Runge-Kutta entirely within GPU registers, avoiding the memory-bound
limitations of traditional transition table methods. Reference aircraft: **Grumman AA-1 Yankee**
(Riley 1985, NASA TM-86309).

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

Troubleshooting:

CuPy does not require the full CUDA toolkit (`nvcc`) — only the NVIDIA driver. If you get a `libnvrtc.so not found` error, install the 

CUDA runtime libraries:
```bash
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
```



## Environments

Each environment is a branch in this repository. Complexity increases with index; strikethrough entries
are planned but not yet implemented.

| # | Name | Observation Space | Action Space | Constraints | Status | Branch
|---|---|---|---|---|---|---|
| 0 | Base Plane | γ | δe | V = const | — |
| 1 | Reduced Symmetric Glider Pullout | γ, V | CL (or α) | β = 0 | — |
| 1.5 | Symmetric Glider Pullout | γ, V, α, q | δe | β = 0 | — |
| 2 | Symmetric Stall | γ, V, α, q | δe, δt | β = 0 | implemented | 4dof-symmetric-stall
| 2.5 | Symmetric Stall with Riley | γ, V, α, q | δe, δt | β = 0 | implemented | 4dof-symmetric-stall-riley
| 3 | Reduced Symmetric Pullout | γ, V | CL (or α), δt | β = 0 | — |
| 4 | Reduced Banked Glider Pullout | γ, V, μ | CL, μ̇ | β = 0 | implemented | 3dof-reduced-banked-pullout
| — | ~~Banked Glider Spin~~ | ~~γ, V, α, μ, p, q~~ | ~~δe, δa, δr~~ | ~~δr = 0, β = 0~~ | Planned |
| — | ~~Banked Pullout~~ | ~~γ, V, α, μ, p, q~~ | ~~δe, δa, δt, δr~~ | ~~β = 0~~ | Planned |
| 5 | Banked Spin | γ, V, α, μ, p, q | δe, δa, δt, ~~δr~~ | β = 0 | — |
| 6 | Full Environment | γ, V, α, β, μ, p, q, r | δe, δa, δt, δr | — | — |

---