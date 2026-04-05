"""
main.py
-------
Policy Iteration pipeline for the 2-DOF Symmetric Pullout with thrust.

Usage:
    python main.py --level 1            # coarse grid, load cached policy if present
    python main.py --level 2            # medium grid
    python main.py --level 3            # fine grid
    python main.py --level 4            # very fine grid (needs ≥8 GB VRAM)
    python main.py --level 1 --retrain  # force retraining
    python main.py --level 2 --no-plots # train only, skip plots

All outputs are written to results/.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from aircraft.reduced_symmetric_pullout import ReducedSymmetricPullout
from PolicyIteration import PolicyIteration, PolicyIterationConfig

matplotlib.use("Agg")

RESULTS_DIR = Path("results")


# -----------------------------------------------------------------------
# Grid definitions for each refinement level
# -----------------------------------------------------------------------

def make_grid(
    n_gamma: int, n_vn: int, n_cl: int, n_throttle: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a flat (N, 2) state grid and a (n_cl*n_throttle, 2) action grid."""
    env = ReducedSymmetricPullout()
    gamma_lo, vn_lo = env.observation_space.low
    gamma_hi, vn_hi = env.observation_space.high

    gamma_vals = np.linspace(gamma_lo, gamma_hi, n_gamma, dtype=np.float32)
    vn_vals    = np.linspace(vn_lo,    vn_hi,    n_vn,    dtype=np.float32)

    gg, vv = np.meshgrid(gamma_vals, vn_vals, indexing="ij")
    states = np.column_stack([gg.ravel(), vv.ravel()]).astype(np.float32)

    cl_lo,  cl_hi  = env.action_space.low[0],  env.action_space.high[0]
    th_lo,  th_hi  = env.action_space.low[1],  env.action_space.high[1]

    cl_vals       = np.linspace(cl_lo, cl_hi, n_cl,       dtype=np.float32)
    throttle_vals = np.linspace(th_lo, th_hi, n_throttle, dtype=np.float32)

    # Cartesian product: (n_cl * n_throttle, 2)
    CL, TH = np.meshgrid(cl_vals, throttle_vals, indexing="ij")
    actions = np.column_stack([CL.ravel(), TH.ravel()]).astype(np.float32)

    env.close()
    return states, actions


LEVEL_GRIDS = {
    #  level: (n_gamma, n_vn, n_cl, n_throttle)
    1: (51,    31,   16,  8),    # ~1.6 k states,  128 actions
    2: (101,   61,   21, 11),    # ~6.2 k states,  231 actions
    3: (201,  121,   31, 16),    # ~24 k states,   496 actions
    4: (401,  241,   41, 21),    # ~97 k states,   861 actions
    5: (801,  481,   51, 26),    # ~385 k states, 1326 actions
    6: (1601, 961,   61, 31),    # ~1.5 M states, 1891 actions
}


def get_setup_for_level(level: int):
    n_gamma, n_vn, n_cl, n_throttle = LEVEL_GRIDS[level]
    states, actions = make_grid(n_gamma, n_vn, n_cl, n_throttle)
    env = ReducedSymmetricPullout()
    config = PolicyIterationConfig(
        maximum_iterations=20_000,
        gamma=0.995,
        theta=1e-4,
        n_steps=100,
        log=True,
        log_interval=200,
        results_path=RESULTS_DIR,
    )
    logger.info(
        f"Level {level}: {len(states):,} states × {len(actions)} actions  "
        f"(γ:{n_gamma}, V:{n_vn}, CL:{n_cl}, δT:{n_throttle})"
    )
    return env, states, actions, config


# -----------------------------------------------------------------------
# Train or load
# -----------------------------------------------------------------------

def train_or_load(
    env, states: np.ndarray, actions: np.ndarray,
    config: PolicyIterationConfig, retrain: bool = False,
) -> PolicyIteration:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    policy_path = RESULTS_DIR / f"{env.unwrapped.__class__.__name__}_policy.npz"

    if not retrain and policy_path.exists():
        logger.info(f"[+] Loading policy from {policy_path}  (--retrain to force)")
        try:
            pi = PolicyIteration.load(policy_path, env=env)
            pi.states_space = states
            return pi
        except Exception as exc:
            logger.error(f"[-] Load failed ({exc}), retraining …")

    if retrain and policy_path.exists():
        logger.info(f"[!] --retrain: ignoring cached policy at {policy_path}")

    logger.info("[*] Training new policy …")
    pi = PolicyIteration(env, states, actions, config)
    pi.run()
    pi.save(policy_path)
    cwd_copy = Path.cwd() / policy_path.name
    if cwd_copy.exists() and cwd_copy.resolve() != policy_path.resolve():
        cwd_copy.unlink()
    return pi


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_altitude_loss(pi: PolicyIteration, prefix: str) -> None:
    """
    Colour map of minimum altitude loss vs. initial conditions.

    Altitude loss [m] = -V(s) * V_stall
    X-axis: Initial Relative Airspeed (V/Vs)
    Y-axis: Initial Flight Path Angle [deg]
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cl_ref  = 0.41 + 4.6983 * np.deg2rad(15)
    v_stall = float(np.sqrt(697.18 * 9.81 / (0.5 * 1.225 * 9.1147 * cl_ref)))

    ng, nv   = pi.grid_shape
    vf       = pi.value_function.reshape(ng, nv)
    alt_loss = -vf * v_stall   # [m], positive = altitude lost

    gamma_lo, gamma_hi = pi.bounds_low[0], pi.bounds_high[0]
    vn_lo,    vn_hi    = pi.bounds_low[1], pi.bounds_high[1]

    gamma_vals = np.linspace(np.rad2deg(gamma_lo), np.rad2deg(gamma_hi), ng)
    vn_vals    = np.linspace(vn_lo, vn_hi, nv)

    g_mask = (gamma_vals >= -90.0) & (gamma_vals <= 0.0)
    v_mask = (vn_vals    >= 1.0)   & (vn_vals    <= 4.0)
    VN, GG = np.meshgrid(vn_vals[v_mask], gamma_vals[g_mask])

    fig, ax = plt.subplots(figsize=(8, 6))

    show_edges = max(g_mask.sum(), v_mask.sum()) <= 80
    pcm = ax.pcolormesh(
        VN, GG, alt_loss[np.ix_(g_mask, v_mask)], cmap="viridis", shading="nearest",
        vmin=0, vmax=110,
        edgecolors="k" if show_edges else "none",
        linewidth=0.3 if show_edges else 0,
    )
    fig.colorbar(pcm, ax=ax, fraction=0.025, pad=0.02, shrink=0.85, aspect=40)

    ax.set_xlabel(r"Initial Airspeed  $V/V_s$")
    ax.set_ylabel("Initial Flight Path Angle (deg)")
    ax.set_title("Minimum altitude loss — δ_throttle and C_L jointly optimised")

    fig.tight_layout()
    out = RESULTS_DIR / f"{prefix}_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[+] Heatmaps saved → {out}")


def plot_policy(pi: PolicyIteration, prefix: str) -> None:
    """Heatmaps of optimal C_L and δ_throttle as functions of initial conditions."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ng, nv = pi.grid_shape
    optimal_cl       = pi.action_space[pi.policy, 0].reshape(ng, nv)
    optimal_throttle = pi.action_space[pi.policy, 1].reshape(ng, nv)

    gamma_lo, gamma_hi = pi.bounds_low[0], pi.bounds_high[0]
    vn_lo,    vn_hi    = pi.bounds_low[1], pi.bounds_high[1]

    gamma_vals = np.linspace(np.rad2deg(gamma_lo), np.rad2deg(gamma_hi), ng)
    vn_vals    = np.linspace(vn_lo, vn_hi, nv)

    g_mask = (gamma_vals >= -90.0) & (gamma_vals <= 0.0)
    v_mask = (vn_vals    >= 1.0)   & (vn_vals    <= 4.0)
    VN, GG = np.meshgrid(vn_vals[v_mask], gamma_vals[g_mask])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, cmap in zip(
        axes,
        [optimal_cl[np.ix_(g_mask, v_mask)], optimal_throttle[np.ix_(g_mask, v_mask)]],
        ["Optimal lift coefficient $C_L$", "Optimal throttle $\\delta_{throttle}$"],
        ["viridis", "plasma"],
    ):
        pcm = ax.pcolormesh(VN, GG, data, cmap=cmap, shading="nearest")
        fig.colorbar(pcm, ax=ax, fraction=0.025, pad=0.02, shrink=0.85, aspect=40)
        ax.set_xlabel(r"Initial Airspeed  $V/V_s$")
        ax.set_ylabel("Initial Flight Path Angle (deg)")
        ax.set_title(title)

    fig.suptitle("Optimal policy — 2-DOF Symmetric Pullout with thrust", fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / f"{prefix}_policy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[+] Policy heatmaps saved → {out}")


def plot_trajectory(pi: PolicyIteration, prefix: str) -> None:
    """Simulate a single recovery from a steep dive and plot the trajectory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = ReducedSymmetricPullout()
    cl_ref  = 0.41 + 4.6983 * np.deg2rad(15)
    v_stall = float(np.sqrt(697.18 * 9.81 / (0.5 * 1.225 * 9.1147 * cl_ref)))

    gamma_0 = np.deg2rad(-60.0)
    vn_0    = 1.4
    env.airplane.reset(gamma_0, vn_0, 0.0)
    obs = env._get_obs()

    ng, nv             = pi.grid_shape
    gamma_lo, gamma_hi = pi.bounds_low[0], pi.bounds_high[0]
    vn_lo,    vn_hi    = pi.bounds_low[1], pi.bounds_high[1]

    hist = {"t": [], "gamma": [], "vn": [], "cl": [], "throttle": [], "reward": []}
    t  = 0.0
    dt = env.airplane.TIME_STEP

    for _ in range(2000):
        g_idx = int(np.clip(
            (obs[0] - gamma_lo) / (gamma_hi - gamma_lo) * (ng - 1), 0, ng - 1
        ))
        v_idx = int(np.clip(
            (obs[1] - vn_lo) / (vn_hi - vn_lo) * (nv - 1), 0, nv - 1
        ))
        s_idx    = g_idx * nv + v_idx
        a_idx    = int(pi.policy[s_idx])
        cl       = float(pi.action_space[a_idx, 0])
        throttle = float(pi.action_space[a_idx, 1])

        hist["t"].append(t)
        hist["gamma"].append(np.rad2deg(obs[0]))
        hist["vn"].append(obs[1])
        hist["cl"].append(cl)
        hist["throttle"].append(throttle)

        obs, reward, terminated, _, _ = env.step([cl, throttle])
        hist["reward"].append(reward)
        t += dt

        if terminated:
            if obs[0] >= 0.0:
                logger.info(f"[+] Pull-out at t={t:.1f}s")
            else:
                logger.warning(f"[-] Catastrophic dive at t={t:.1f}s")
            break

    env.close()

    altitude = np.cumsum(
        np.array(hist["vn"]) * v_stall * np.sin(np.deg2rad(hist["gamma"])) * dt
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(hist["t"], hist["gamma"])
    axes[0, 0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("γ [°]")
    axes[0, 0].set_title("Flight-path angle")

    axes[0, 1].plot(hist["t"], hist["vn"])
    axes[0, 1].axhline(1.0, color="k", lw=0.8, ls="--", label="Vs")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("V/Vs [-]")
    axes[0, 1].set_title("Normalised airspeed")
    axes[0, 1].legend()

    axes[0, 2].plot(hist["t"], altitude)
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].set_ylabel("Altitude change [m]")
    axes[0, 2].set_title("Altitude (relative)")

    axes[1, 0].step(hist["t"], hist["cl"], where="post")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("$C_L$ [-]")
    axes[1, 0].set_title("Optimal lift coefficient")

    axes[1, 1].step(hist["t"], hist["throttle"], where="post")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("$\\delta_{throttle}$ [-]")
    axes[1, 1].set_title("Optimal throttle")

    axes[1, 2].step(hist["t"], hist["reward"], where="post")
    axes[1, 2].axhline(0, color="k", lw=0.8, ls="--")
    axes[1, 2].set_xlabel("Time [s]")
    axes[1, 2].set_ylabel("Step reward [m/Vs]")
    axes[1, 2].set_title("Instantaneous reward")

    fig.suptitle(
        f"2-DOF Recovery Trajectory with Thrust  (γ₀={np.rad2deg(gamma_0):.0f}°, V₀={vn_0:.1f}Vs)",
        fontsize=12,
    )
    fig.tight_layout()

    out = RESULTS_DIR / f"{prefix}_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"[+] Trajectory saved → {out}")


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------

def run_pipeline(level: int, plots: bool = True, retrain: bool = False) -> None:
    logger.info("=" * 60)
    logger.info(f"  2-DOF Symmetric Pullout with Thrust — Grid Level {level}")
    logger.info("=" * 60)

    env, states, actions, config = get_setup_for_level(level)
    pi = train_or_load(env, states, actions, config, retrain=retrain)

    prefix = "ReducedSymmetricPullout"

    if plots:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_altitude_loss(pi, prefix)
        plot_policy(pi, prefix)
        plot_trajectory(pi, prefix)

    env.close()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2-DOF Symmetric Pullout with Thrust — Policy Iteration"
    )
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4, 5, 6], default=1,
        help=(
            "Grid resolution: "
            "1=coarse (~1.6k states, 128 actions), "
            "2=medium (~6k, 231), "
            "3=fine (~24k, 496), "
            "4=very fine (~97k, 861), "
            "5=ultra (~385k, 1326), "
            "6=maximum (~1.5M, 1891)"
        ),
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Ignore cached policy and retrain from scratch",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip all plots",
    )
    args = parser.parse_args()
    run_pipeline(level=args.level, plots=not args.no_plots, retrain=args.retrain)
