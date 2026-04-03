"""
experiments.py
--------------
Grid configurations and experiment setup functions for the Banked Glider Policy Iteration.
Also contains the Numba-JIT procedural state/action space generators and the GPU profiling routine.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Any, Optional

# Force Numba to use OpenMP threading layer to avoid TBB version warnings
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import gymnasium as gym
import numba as nb
import numpy as np

# Allow imports from the parent project directory

from envs.reduced_banked_pullout import ReducedBankedGliderPullout
from solver.policy_iteration import PolicyIteration, PolicyIterationConfig

logger = logging.getLogger(__name__)


# =====================================================================
# NUMBA: PROCEDURAL SPACE GENERATORS
# =====================================================================

@nb.njit(parallel=True, fastmath=True, cache=True)
def generate_state_space(
    gamma_bins: np.ndarray,
    v_bins: np.ndarray,
    mu_bins: np.ndarray
) -> np.ndarray:
    """
    Generates the Cartesian product of the state space without memory spikes.
    Pre-allocates the exact memory block required and uses multi-threaded C-level loops.
    """
    n_gamma = len(gamma_bins)
    n_v = len(v_bins)
    n_mu = len(mu_bins)
    total_states = n_gamma * n_v * n_mu

    states = np.empty((total_states, 3), dtype=np.float32)

    for i in nb.prange(n_gamma):
        for j in range(n_v):
            for k in range(n_mu):
                idx = i * (n_v * n_mu) + j * n_mu + k
                states[idx, 0] = gamma_bins[i]
                states[idx, 1] = v_bins[j]
                states[idx, 2] = mu_bins[k]

    return states


@nb.njit(parallel=True, fastmath=True, cache=True)
def generate_action_space(
    cl_bins: np.ndarray,
    br_bins: np.ndarray
) -> np.ndarray:
    """Generates the Cartesian product of the action space efficiently."""
    n_cl = len(cl_bins)
    n_br = len(br_bins)
    total_actions = n_cl * n_br

    actions = np.empty((total_actions, 2), dtype=np.float32)

    for i in nb.prange(n_cl):
        for j in range(n_br):
            idx = i * n_br + j
            actions[idx, 0] = cl_bins[i]
            actions[idx, 1] = br_bins[j]

    return actions


# =====================================================================
# EXPERIMENT CONFIGURATIONS — 4 GRID LEVELS
# =====================================================================
#
#  Level 1 — Paper (Bunge 2018, Table 1): 37 × 32 × 45  ≈   53 k states
#  Level 2 — 2× refinement             : 73 × 63 × 89  ≈  409 k states
#  Level 3 — High fidelity             :181 × 125 × 177 ≈    4 M states
#  Level 4 — Maximum (24 GB VRAM)      :361 × 250 × 354 ≈   32 M states
#
# Actions: all levels use 7 × 13 = 91-action grid (paper values).
# =====================================================================

def _make_config() -> PolicyIterationConfig:
    return PolicyIterationConfig(
        gamma=1.0,
        theta=1e-4,
        n_steps=1000,
        log=False,
        log_interval=10,
        img_path=Path("./img")
    )


def _standard_actions() -> np.ndarray:
    cl_vals = np.linspace(-0.5, 1.0, 7, dtype=np.float32)
    br_vals = np.linspace(np.deg2rad(-30), np.deg2rad(30), 13, dtype=np.float32)
    return generate_action_space(cl_vals, br_vals)


def setup_level_1() -> Tuple[gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig]:
    """Level 1 — Paper grid (Bunge 2018, Table 1): 37 × 32 × 45 ≈ 53k states."""
    logger.info("[*] Level 1 — Paper grid (37 × 32 × 45, Bunge 2018 Table 1)")
    env = ReducedBankedGliderPullout()

    gamma_bins = np.linspace(np.deg2rad(-180), 0.0, 37, dtype=np.float32)   # 5 deg increment
    v_bins     = np.linspace(0.9, 4.0, 32, dtype=np.float32)                # 0.1 Vs increment
    mu_bins    = np.linspace(np.deg2rad(-20), np.deg2rad(200), 45, dtype=np.float32)  # 5 deg increment

    states = generate_state_space(gamma_bins, v_bins, mu_bins)
    logger.info(f"[*] State space: {states.shape}")
    return env, states, _standard_actions(), _make_config()


def setup_level_2() -> Tuple[gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig]:
    """Level 2 — 2× paper refinement: 73 × 63 × 89 ≈ 409k states."""
    logger.info("[*] Level 2 — 2× refinement (73 × 63 × 89)")
    env = ReducedBankedGliderPullout()

    gamma_bins = np.linspace(np.deg2rad(-180), 0.0, 73, dtype=np.float32)   # 2.5 deg increment
    v_bins     = np.linspace(0.9, 4.0, 63, dtype=np.float32)                # 0.05 Vs increment
    mu_bins    = np.linspace(np.deg2rad(-20), np.deg2rad(200), 89, dtype=np.float32)  # 2.5 deg increment

    states = generate_state_space(gamma_bins, v_bins, mu_bins)
    logger.info(f"[*] State space: {states.shape}")
    return env, states, _standard_actions(), _make_config()


def setup_level_3() -> Tuple[gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig]:
    """Level 3 — High fidelity: 181 × 125 × 177 ≈ 4M states."""
    logger.info("[*] Level 3 — High fidelity (181 × 125 × 177)")
    env = ReducedBankedGliderPullout()

    gamma_bins = np.linspace(np.deg2rad(-180), 0.0, 181, dtype=np.float32)   # 1 deg increment
    v_bins     = np.linspace(0.9, 4.0, 125, dtype=np.float32)                # 0.025 Vs increment
    mu_bins    = np.linspace(np.deg2rad(-20), np.deg2rad(200), 177, dtype=np.float32)  # 1.25 deg increment

    states = generate_state_space(gamma_bins, v_bins, mu_bins)
    logger.info(f"[*] State space: {states.shape}")
    return env, states, _standard_actions(), _make_config()


def setup_level_4() -> Tuple[gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig]:
    """Level 4 — Maximum resolution: 361 × 250 × 354 ≈ 32M states. Requires 24 GB VRAM."""
    logger.info("[*] Level 4 — Maximum grid (361 × 250 × 354)")
    env = ReducedBankedGliderPullout()

    gamma_bins = np.linspace(np.deg2rad(-180), 0.0, 361, dtype=np.float32)   # 0.5 deg increment
    v_bins     = np.linspace(0.9, 4.0, 250, dtype=np.float32)                # ~0.0125 Vs increment
    mu_bins    = np.linspace(np.deg2rad(-20), np.deg2rad(200), 354, dtype=np.float32)  # 0.625 deg increment

    states = generate_state_space(gamma_bins, v_bins, mu_bins)
    logger.info(f"[*] State space: {states.shape}")
    return env, states, _standard_actions(), _make_config()


GRID_LEVELS = {
    1: setup_level_1,
    2: setup_level_2,
    3: setup_level_3,
    4: setup_level_4,
}


def get_setup_for_level(level: int):
    """Return the setup function for the given grid level (1–4)."""
    if level not in GRID_LEVELS:
        raise ValueError(f"Invalid level {level}. Choose from 1, 2, 3, 4.")
    return GRID_LEVELS[level]


# =====================================================================
# GPU PROFILING
# =====================================================================

def run_profiling(setup_func: Optional[Any] = None) -> None:
    """
    GPU-accurate profiling of the Policy Iteration pipeline using CUDA events.
    """
    import cupy as cp

    logger.info("=" * 60)
    logger.info("  GPU PROFILING — CUDA Event Timing")
    logger.info("=" * 60)

    t0_setup = time.perf_counter()
    if setup_func is None:
        setup_func = setup_level_1
    env, states, actions, config = setup_func()
    t_setup = time.perf_counter() - t0_setup

    n_states = len(states)
    n_actions = len(actions)

    logger.info(f"  States: {n_states:,}  |  Actions: {n_actions}")
    logger.info(f"  Setup time: {t_setup:.2f} s")

    pi = PolicyIteration(env, states, actions, config)
    eval_times = []
    improve_times = []
    n_converged = 0

    for n in range(config.n_steps):
        cp.cuda.Device(0).synchronize()
        ev_start = cp.cuda.Event()
        ev_end = cp.cuda.Event()
        ev_start.record()
        pi.policy_evaluation()
        ev_end.record()
        ev_end.synchronize()
        eval_times.append(cp.cuda.get_elapsed_time(ev_start, ev_end) / 1000.0)

        cp.cuda.Device(0).synchronize()
        im_start = cp.cuda.Event()
        im_end = cp.cuda.Event()
        im_start.record()
        is_stable = pi.policy_improvement()
        im_end.record()
        im_end.synchronize()
        improve_times.append(cp.cuda.get_elapsed_time(im_start, im_end) / 1000.0)

        n_converged = n + 1
        logger.info(f"  Iter {n+1}: eval={eval_times[-1]:.3f}s  improve={improve_times[-1]:.3f}s")

        if is_stable:
            logger.info(f"  Converged at iteration {n_converged}.")
            break

    cp.cuda.Device(0).synchronize()
    t0_pull = time.perf_counter()
    pi._pull_tensors_from_gpu()
    t_pull = time.perf_counter() - t0_pull

    total_eval = sum(eval_times)
    total_improve = sum(improve_times)
    total_wall = t_setup + total_eval + total_improve + t_pull

    avg_eval = total_eval / n_converged if n_converged else 0
    avg_improve = total_improve / n_converged if n_converged else 0
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    def fmt(t):
        if t < 0.001: return f"{t * 1e6:.1f} µs"
        elif t < 1.0: return f"{t * 1000:.2f} ms"
        elif t < 60.0: return f"{t:.3f} s"
        else: return f"{t:.1f} s ({t/60:.1f} min)"

    report_lines = [
        "", "=" * 65, "  PROFILING REPORT — CUDA Event Timing", "=" * 65,
        f"  GPU:                  {gpu_name}",
        f"  State space:          {n_states:,} states",
        f"  Action space:         {n_actions} actions",
        f"  Convergence:          {n_converged} iterations", "-" * 65,
        f"  Setup (CPU):          {fmt(t_setup):>14s}",
        f"  Policy Evaluation:    {fmt(total_eval):>14s}  (avg {fmt(avg_eval)}/iter)",
        f"  Policy Improvement:   {fmt(total_improve):>14s}  (avg {fmt(avg_improve)}/iter)",
        f"  GPU->CPU Transfer:    {fmt(t_pull):>14s}", "-" * 65,
        f"  TOTAL WALL TIME:      {fmt(total_wall):>14s}", "=" * 65, ""
    ]
    report = "\n".join(report_lines)
    print(report)

    latex = f"""
% Auto-generated by run_profiling(setup_level_4_profiling) — {gpu_name}
\\begin{{table}}[h]
\\centering
\\caption{{Computational cost of Policy Iteration on {n_states:,} states, {n_actions} actions ({gpu_name}).}}
\\label{{tab:profiling}}
\\begin{{tabular}}{{l r r}}
\\toprule
Phase & Total (ms) & Per Iteration (ms) \\\\
\\midrule
State grid construction (CPU) & {t_setup*1000:.2f} & -- \\\\
Policy Evaluation (GPU)       & {total_eval*1000:.2f} & {avg_eval*1000:.2f} \\\\
Policy Improvement (GPU)      & {total_improve*1000:.2f} & {avg_improve*1000:.2f} \\\\
GPU $\\to$ CPU Transfer        & {t_pull*1000:.2f} & -- \\\\
\\midrule
\\textbf{{Total}}              & \\textbf{{{total_wall*1000:.2f}}} & -- \\\\
\\bottomrule
\\end{{tabular}}
\\\\[4pt]
\\footnotesize Converged in {n_converged} iterations. Timing via CUDA events.
\\end{{table}}
"""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "profiling_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(results_dir / "profiling_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)

    logger.info(f"Reports saved to {results_dir / 'profiling_report.txt'} and {results_dir / 'profiling_table.tex'}")
