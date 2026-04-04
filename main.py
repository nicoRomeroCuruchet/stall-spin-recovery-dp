"""
main.py — Pure Markovian DP Simulation
Executes the mathematically rigorous DP policy.
Includes persistent policy caching and precise physical altitude tracking.
"""
import logging
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

os.environ["NUMBA_THREADING_LAYER"] = "omp"

from aircraft.symmetric_stall import SymmetricStall  # noqa: E402
from PolicyIteration import (  # noqa: E402
    PolicyIterationStall,
    PolicyIterationStallConfig,
)
from utils.utils import get_optimal_action  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# ── Experiment Setup ─────────────────────────────────────────────────────


def setup_symmetric_stall_experiment() -> tuple[
    gym.Env, np.ndarray, np.ndarray, PolicyIterationStallConfig
]:
    """Configures the experiment with physically bounded state grids."""
    env = SymmetricStall()

    bins_space = {
        # 56 bins (~1.7° resolution)
        "flight_path_angle": np.linspace(
            np.deg2rad(-90), np.deg2rad(5), 56, dtype=np.float32
        ),
        # 41 bins
        "airspeed_norm": np.linspace(0.9, 2.0, 41, dtype=np.float32),
        # 36 bins (~0.97° resolution)
        "alpha": np.linspace(
            np.deg2rad(-14), np.deg2rad(20), 36, dtype=np.float32
        ),
        # 41 bins (~2.5°/s resolution)
        "pitch_rate": np.linspace(
            np.deg2rad(-50), np.deg2rad(50), 41, dtype=np.float32
        ),
    }
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack(
        [g.ravel() for g in grid]
    ).astype(np.float32).T

    # 21 x 7 = 147 actions
    de_vals = np.linspace(
        np.deg2rad(-25), np.deg2rad(15), 21, dtype=np.float32
    )
    dt_vals = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    action_grid = np.meshgrid(de_vals, dt_vals, indexing="ij")
    action_space = np.vstack(
        [a.ravel() for a in action_grid]
    ).astype(np.float32).T

    config = PolicyIterationStallConfig(
        gamma=1.0,
        theta=5e-6,
        n_steps=1000,
        log=False,
        log_interval=10,
        w_q_penalty=0.0,
        w_control_effort=60.0,
        w_alpha_barrier_pos=0.0,
        w_alpha_barrier_neg=0.0,
        w_crash_penalty=0.0,
        w_throttle_bonus=0.0,
    )
    return env, states_space, action_space, config


# ── Policy Training / Loading ────────────────────────────────────────────


def train_or_load_policy(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    config: PolicyIterationStallConfig,
    prefix: str,
) -> PolicyIterationStall:
    """Trains or loads pre-trained tensors from disk."""
    policy_filename = f"{env.unwrapped.__class__.__name__}_policy.npz"
    RESULTS_DIR.mkdir(exist_ok=True)
    policy_path = RESULTS_DIR / policy_filename

    if policy_path.exists():
        logger.info(f"[+] Existing policy found: {policy_filename}. Loading...")
        try:
            pi = PolicyIterationStall.load(policy_path, env=env)
            pi.states_space = states
            logger.info("[+] Policy loaded successfully. Skipping training.")
            return pi
        except Exception as e:
            logger.error(f"[-] Failed to load policy: {e}. Forcing retrain...")

    logger.info(f"[*] Training new policy for {prefix}...")
    pi = PolicyIterationStall(env, states, actions, config)
    pi.run()

    pi.save(policy_path)
    logger.info(f"[+] Policy cached to {policy_path.resolve()}")
    return pi


# ── DP Simulation ────────────────────────────────────────────────────────


def run_dp_simulation(
    pi: PolicyIterationStall,
    gamma_0_deg: float,
    v_norm_0: float,
    alpha_0_deg: float,
    q_0_deg: float,
    max_steps: int = 1500,
) -> dict[str, list]:
    """
    Runs a pure DP simulation and returns the full state history.

    All angular values in the returned history are in radians.
    """
    env = pi.env
    v_stall = env.airplane.STALL_AIRSPEED
    dt = env.airplane.TIME_STEP
    control_dt = 0.01
    steps_per_control = max(1, int(round(control_dt / dt)))

    obs, _ = env.specific_reset(
        np.deg2rad(gamma_0_deg),
        v_norm_0,
        np.deg2rad(alpha_0_deg),
        np.deg2rad(q_0_deg),
    )

    hist = {
        k: []
        for k in [
            "t", "gamma", "v_norm", "alpha", "q", "de", "dt_ctrl", "h"
        ]
    }

    t, h = 0.0, 0.0
    has_dived = False
    action = [0.0, 0.0]
    current_action_idx = None

    for step in range(max_steps):
        if step % steps_per_control == 0:
            action, _, current_action_idx = get_optimal_action(
                obs, pi, current_action_idx
            )

        hist["t"].append(t)
        hist["gamma"].append(obs[0])
        hist["v_norm"].append(obs[1])
        hist["alpha"].append(obs[2])
        hist["q"].append(obs[3])
        hist["de"].append(action[0])
        hist["dt_ctrl"].append(action[1])
        hist["h"].append(h)

        obs, _, terminated, _, _ = env.step(action)

        v_true = obs[1] * v_stall
        h += v_true * np.sin(obs[0]) * dt
        t += dt

        new_gamma = obs[0]
        if new_gamma < np.deg2rad(-2.0):
            has_dived = True

        if has_dived and new_gamma >= 0.0:
            # Append final recovered state
            hist["t"].append(t)
            hist["gamma"].append(new_gamma)
            hist["v_norm"].append(obs[1])
            hist["alpha"].append(obs[2])
            hist["q"].append(obs[3])
            hist["de"].append(action[0])
            hist["dt_ctrl"].append(action[1])
            hist["h"].append(h)
            logger.info(
                f"[+] Recovery success at {t:.2f}s, Δh={h:.2f}m"
            )
            break

        if terminated:
            if obs[2] >= np.deg2rad(40):
                logger.warning(f"[-] +Alpha limit at {t:.2f}s")
                break
            elif obs[2] <= np.deg2rad(-40):
                logger.warning(f"[-] -Alpha limit at {t:.2f}s")
                break
            elif obs[0] <= -np.pi + 0.05:
                logger.warning(f"[-] Catastrophic dive at {t:.2f}s")
                break

    return hist


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_time_response(hist: dict, prefix: str) -> None:
    """Generates the 7-panel time-response figure from a simulation history."""
    t = hist["t"]
    gamma_deg = np.rad2deg(hist["gamma"])
    v_norm = hist["v_norm"]
    alpha_deg = np.rad2deg(hist["alpha"])
    q_deg = np.rad2deg(hist["q"])
    de_deg = np.rad2deg(hist["de"])
    dt_ctrl = hist["dt_ctrl"]
    h = hist["h"]

    panels = [
        (gamma_deg, r"$\gamma$ (deg)", "#532C8A", "plot"),
        (v_norm, r"$V/V_s$", "#E377C2", "plot"),
        (alpha_deg, r"$\alpha$ (deg)", "#D62728", "plot"),
        (q_deg, r"$q$ (deg/s)", "#2CA02C", "plot"),
        (de_deg, r"$\delta_e$ (deg)", "#FF8C00", "step"),
        (dt_ctrl, r"$\delta_t$", "#17BECF", "step"),
        (h, "Altitude Loss (m)", "#1F77B4", "plot"),
    ]

    fig, axs = plt.subplots(len(panels), 1, figsize=(8, 16), sharex=True)
    fig.suptitle(
        r"Markov-Compliant DP Response (100Hz Control)", fontsize=14
    )

    for ax, (data, ylabel, color, style) in zip(axs, panels):
        if style == "step":
            ax.step(t, data, color=color, linewidth=2, where="post")
        else:
            ax.plot(t, data, color=color, linewidth=2)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="-", alpha=0.4)

    axs[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[5].set_ylim([-0.05, 1.05])
    axs[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{prefix}_trajectory.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[*] Plot saved to: {out_path.resolve()}")


def plot_heatmaps(pi: PolicyIterationStall, prefix: str) -> None:
    """Generates the 3x3 heatmap figure (elevator, throttle, altitude loss)."""
    logger.info("[*] Extracting 4D Tensors for heatmaps...")

    gamma_bins = np.unique(pi.states_space[:, 0])
    v_bins = np.unique(pi.states_space[:, 1])
    alpha_bins = np.unique(pi.states_space[:, 2])
    q_bins = np.unique(pi.states_space[:, 3])

    shape_4d = (
        len(gamma_bins), len(v_bins), len(alpha_bins), len(q_bins)
    )
    V_4D = pi.value_function.reshape(shape_4d)
    Pol_4D = pi.policy.reshape(shape_4d)

    q_idx = int(np.argmin(np.abs(q_bins - 0.0)))
    v_targets = [0.9, 1.0, 1.1]

    gamma_mask = (gamma_bins >= np.deg2rad(-90.1)) & (gamma_bins <= 0.01)
    alpha_mask = (
        (alpha_bins >= np.deg2rad(-5.1))
        & (alpha_bins <= np.deg2rad(20.1))
    )

    gamma_deg = np.rad2deg(gamma_bins[gamma_mask])
    alpha_deg = np.rad2deg(alpha_bins[alpha_mask])
    A_mesh, G_mesh = np.meshgrid(alpha_deg, gamma_deg, indexing="xy")

    fig, axes = plt.subplots(
        3, 3, figsize=(11, 8), sharex="col", sharey="row"
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.15, bottom=0.2)

    for i, v_target in enumerate(v_targets):
        v_idx = int(np.argmin(np.abs(v_bins - v_target)))
        v_slice = V_4D[:, v_idx, :, q_idx][gamma_mask][:, alpha_mask]
        p_slice = Pol_4D[:, v_idx, :, q_idx][gamma_mask][:, alpha_mask]

        de_slice = np.array([
            [np.rad2deg(pi.action_space[p_slice[g, a], 0])
             for a in range(p_slice.shape[1])]
            for g in range(p_slice.shape[0])
        ], dtype=np.float32)

        dt_slice = np.array([
            [pi.action_space[p_slice[g, a], 1]
             for a in range(p_slice.shape[1])]
            for g in range(p_slice.shape[0])
        ], dtype=np.float32)

        alt_loss_slice = -v_slice

        # Elevator
        axes[i, 0].pcolormesh(
            A_mesh, G_mesh, de_slice,
            cmap="plasma", vmin=-25, vmax=15, shading="gouraud",
        )
        # Throttle
        axes[i, 1].pcolormesh(
            A_mesh, G_mesh, dt_slice,
            cmap="plasma", vmin=0, vmax=1, shading="nearest",
        )
        # Altitude loss
        axes[i, 2].pcolormesh(
            A_mesh, G_mesh, alt_loss_slice,
            cmap="plasma", vmin=0, vmax=100, shading="gouraud",
        )

        axes[i, 2].text(
            1.05, 0.5, f"V/Vs = {v_target}",
            transform=axes[i, 2].transAxes,
            va="center", ha="left", fontsize=11,
        )

    titles = ["Policy for Elevator", "Policy for Throttle", "Altitude Loss"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, pad=10)

    for j in range(3):
        axes[2, j].set_xlabel(r"$\alpha$ (deg)")
        axes[2, j].set_xticks([0, 10, 20])

    for i in range(3):
        axes[i, 0].set_ylabel(r"$\gamma$ (deg)")
        axes[i, 0].set_yticks([0, -30, -60, -90])

    # Colorbars
    cbar_specs = [
        (0, r"$\delta_e$ (deg)", [-20, 0, 15]),
        (1, r"$\delta_t$", [0.0, 0.5, 1.0]),
        (2, "Altitude Loss (m)", [0, 100]),
    ]
    for j, (col, label, ticks) in enumerate(cbar_specs):
        cax = fig.add_axes([0.15 + 0.27 * j, 0.05, 0.2, 0.02])
        mappable = axes[2, col].collections[0]
        fig.colorbar(
            mappable, cax=cax, orientation="horizontal",
            label=label, ticks=ticks,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{prefix}_heatmaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[+] Heatmaps saved to {out_path.resolve()}")


# ── CasADi Validation ────────────────────────────────────────────────────


def validate_with_casadi(pi: PolicyIterationStall, prefix: str) -> None:
    """
    Compares the DP policy trajectory against a CasADi/IPOPT continuous-time
    optimal trajectory on the same 7-panel figure.
    """
    try:
        from casadi_stall_optimizer import CasadiStallOptimizer
    except ImportError:
        logger.error(
            "casadi_stall_optimizer.py not found — skipping CasADi."
        )
        return

    k_thrust = pi.env.airplane.THROTTLE_LINEAR_MAPPING

    gamma_0, v_norm_0, alpha_0, q_0 = 0.0, 0.95, 20.0, 0.0

    # 1. Run DP simulation
    logger.info("[*] Running DP simulation for CasADi seed...")
    hist = run_dp_simulation(pi, gamma_0, v_norm_0, alpha_0, q_0)
    dp_T = hist["t"][-1]
    logger.info(f"[+] DP: T={dp_T:.2f}s, Δh={hist['h'][-1]:.2f}m")

    # 2. Build warm-start seed resampled to CasADi node count
    n_nodes = 100
    t_orig = np.array(hist["t"])
    t_states = np.linspace(0, dp_T, n_nodes + 1)
    t_ctrls = np.linspace(0, dp_T, n_nodes)

    dp_seed = {
        "T": dp_T,
        "gamma": np.interp(t_states, t_orig, hist["gamma"]),
        "v_norm": np.interp(t_states, t_orig, hist["v_norm"]),
        "alpha": np.interp(t_states, t_orig, hist["alpha"]),
        "q": np.interp(t_states, t_orig, hist["q"]),
        "h": np.interp(t_states, t_orig, hist["h"]),
        "de": np.interp(t_ctrls, t_orig[:-1], hist["de"][:-1]),
        "dt_ctrl": np.interp(
            t_ctrls, t_orig[:-1], hist["dt_ctrl"][:-1]
        ),
    }

    # 3. Solve CasADi OCP
    optimizer = CasadiStallOptimizer(k_thrust=k_thrust)
    cas = optimizer.solve_trajectory(
        gamma_0=np.deg2rad(gamma_0),
        v_norm_0=v_norm_0,
        alpha_0=np.deg2rad(alpha_0),
        q_0=np.deg2rad(q_0),
        n_nodes=n_nodes,
        dp_seed=dp_seed,
    )

    # 4. Plot comparison
    _plot_dp_vs_casadi(hist, cas, prefix)


def _plot_dp_vs_casadi(
    hist: dict, cas: dict, prefix: str
) -> None:
    """Renders the 7-panel DP vs CasADi comparison figure."""
    dp_t = np.array(hist["t"])
    dp_h = hist["h"]

    status = "Converged" if cas["converged"] else "Infeasible (debug)"

    COLOR_DP = "#532C8A"
    COLOR_CAS = "#E8742A"
    LW = 2.0

    panels = [
        ("gamma", r"$\gamma$ (deg)", True),
        ("v_norm", r"$V/V_s$", False),
        ("alpha", r"$\alpha$ (deg)", True),
        ("q", r"$q$ (deg/s)", True),
        ("de", r"$\delta_e$ (deg)", True),
        ("dt_ctrl", r"$\delta_t$", False),
        ("h", "Altitude Loss (m)", False),
    ]

    fig, axs = plt.subplots(len(panels), 1, figsize=(9, 17), sharex=False)
    fig.suptitle(
        f"DP vs CasADi/IPOPT — Stall Recovery\n"
        f"DP: Δh={dp_h[-1]:.1f}m, T={dp_t[-1]:.2f}s   |   "
        f"CasADi ({status}): "
        f"Δh={cas['h'][-1]:.1f}m, T={cas['T']:.2f}s",
        fontsize=12,
    )

    for ax, (key, ylabel, to_deg) in zip(axs, panels):
        dp_data = np.rad2deg(hist[key]) if to_deg else hist[key]
        ax.plot(dp_t, dp_data, color=COLOR_DP, lw=LW, label="DP Policy")
        ax.plot(
            cas["t"], cas[key],
            color=COLOR_CAS, lw=LW, ls="--", label="CasADi NLP",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="-", alpha=0.35)
        ax.legend(loc="best", fontsize=8)

    axs[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[5].set_ylim([-0.05, 1.05])
    axs[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{prefix}_DP_vs_CasADi.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[+] Comparison plot saved to {out_path.resolve()}")


# ── Entry Point ──────────────────────────────────────────────────────────


def main():
    prefix = "symmetric_stall"
    env, states, actions, config = setup_symmetric_stall_experiment()
    pi = train_or_load_policy(env, states, actions, config, prefix)

    hist = run_dp_simulation(pi, 0.0, 0.95, 20.0, 0.0)
    plot_time_response(hist, prefix)
    plot_heatmaps(pi, prefix)
    # validate_with_casadi(pi, prefix)


if __name__ == "__main__":
    main()
