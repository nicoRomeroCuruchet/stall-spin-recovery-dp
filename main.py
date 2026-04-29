"""
main_banked_spin.py — 6-DOF Banked-Spin Recovery DP pipeline.

State:  (γ, V/Vs, α, μ, p, q)
Action: (δe, δa, δt)

Trains (or loads) a Policy Iteration optimal policy on the GPU,
runs a closed-loop DP simulation, and renders:
  - 10-panel time response
  - 4×4 heatmap grid (policy + value for several μ slices)
"""
import logging
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

os.environ["NUMBA_THREADING_LAYER"] = "omp"

from aircraft.banked_spin import BankedSpin  # noqa: E402
from PolicyIterationBankedSpin import (  # noqa: E402
    PolicyIterationBankedSpin,
    PolicyIterationBankedSpinConfig,
)
from utils.utils import get_optimal_action  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# ── Experiment Setup ─────────────────────────────────────────────────────


def setup_banked_spin_experiment(
    bins_gamma: int = 20,
    bins_v: int = 20,
    bins_alpha: int = 20,
    bins_mu: int = 20,
    bins_p: int = 15,
    bins_q: int = 15,
    bins_de: int = 21,
    bins_da: int = 11,
    bins_dt: int = 7,
) -> tuple[gym.Env, np.ndarray, np.ndarray, PolicyIterationBankedSpinConfig]:
    """Build the 6D state grid and 3D action grid.

    Defaults: ~1.8e7 states × 1617 actions. Fits comfortably under 2 GB
    of VRAM. Increase the bin counts to push closer to the 30·25·24·30·20·20
    target grid agreed in the planning phase.
    """
    env = BankedSpin()

    bins_space = {
        "flight_path_angle": np.linspace(
            np.deg2rad(-90), np.deg2rad(5), bins_gamma, dtype=np.float32),
        "airspeed_norm": np.linspace(
            0.9, 2.0, bins_v, dtype=np.float32),
        "alpha": np.linspace(
            np.deg2rad(-14), np.deg2rad(20), bins_alpha, dtype=np.float32),
        "bank_angle": np.linspace(
            np.deg2rad(-60), np.deg2rad(60), bins_mu, dtype=np.float32),
        "roll_rate": np.linspace(
            -2.0, 2.0, bins_p, dtype=np.float32),
        "pitch_rate": np.linspace(
            np.deg2rad(-50), np.deg2rad(50), bins_q, dtype=np.float32),
    }
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T

    de_vals = np.linspace(np.deg2rad(-25), np.deg2rad(15), bins_de, dtype=np.float32)
    da_vals = np.linspace(np.deg2rad(-15), np.deg2rad(15), bins_da, dtype=np.float32)
    dt_vals = np.linspace(0.0, 1.0, bins_dt, dtype=np.float32)
    action_grid = np.meshgrid(de_vals, da_vals, dt_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T

    config = PolicyIterationBankedSpinConfig(
        gamma=1.0,
        theta=5e-4,
        n_steps=50,
        maximum_iterations=8000,
        log=False,
        log_interval=10,
        # Longitudinal: same flat-altitude-loss formulation as 4DOF
        w_q_penalty=0.0,
        w_control_effort=60.0,
        w_alpha_barrier_pos=0.0,
        w_alpha_barrier_neg=0.0,
        w_crash_penalty=0.0,
        w_throttle_bonus=0.0,
        # Lateral (Markov-compliant shaping)
        w_p_penalty=0.01,
        w_mu_barrier=0.5,
        w_aileron_effort=0.001,
    )
    logger.info(
        f"Experiment grid: states={len(states_space):,} ({tuple(int(b) for b in [bins_gamma, bins_v, bins_alpha, bins_mu, bins_p, bins_q])}),"
        f" actions={len(action_space):,} ({bins_de}·{bins_da}·{bins_dt})"
    )
    return env, states_space, action_space, config


# ── Policy Training / Loading ────────────────────────────────────────────


def train_or_load_policy(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    config: PolicyIterationBankedSpinConfig,
    prefix: str,
) -> PolicyIterationBankedSpin:
    policy_filename = f"{env.unwrapped.__class__.__name__}_policy.npz"
    RESULTS_DIR.mkdir(exist_ok=True)
    policy_path = RESULTS_DIR / policy_filename

    if policy_path.exists():
        logger.info(f"[+] Existing policy found: {policy_filename}. Loading...")
        try:
            pi = PolicyIterationBankedSpin.load(policy_path, env=env)
            pi.states_space = states
            logger.info("[+] Policy loaded successfully. Skipping training.")
            return pi
        except Exception as e:
            logger.error(f"[-] Failed to load policy: {e}. Forcing retrain...")

    logger.info(f"[*] Training new policy for {prefix}...")
    pi = PolicyIterationBankedSpin(env, states, actions, config)
    pi.run()
    pi.save(policy_path)
    logger.info(f"[+] Policy cached to {policy_path.resolve()}")
    return pi


# ── DP Simulation ────────────────────────────────────────────────────────


def run_dp_simulation(
    pi: PolicyIterationBankedSpin,
    gamma_0_deg: float,
    v_norm_0: float,
    alpha_0_deg: float,
    mu_0_deg: float,
    p_0: float,
    q_0_deg: float,
    max_steps: int = 1500,
) -> dict[str, list]:
    env = pi.env
    v_stall = env.airplane.STALL_AIRSPEED
    dt = env.airplane.TIME_STEP

    obs, _ = env.specific_reset(
        flight_path_angle=np.deg2rad(gamma_0_deg),
        airspeed_norm=v_norm_0,
        alpha=np.deg2rad(alpha_0_deg),
        bank_angle=np.deg2rad(mu_0_deg),
        roll_rate=p_0,
        pitch_rate=np.deg2rad(q_0_deg),
    )

    keys = ["t", "gamma", "v_norm", "alpha", "mu", "p", "q",
            "de", "da", "dt_ctrl", "h"]
    hist = {k: [] for k in keys}

    t, h = 0.0, 0.0
    has_dived = False
    action = [0.0, 0.0, 0.0]
    current_action_idx = None

    for _ in range(max_steps):
        action, _, current_action_idx = get_optimal_action(
            obs, pi, current_action_idx
        )

        hist["t"].append(t)
        hist["gamma"].append(obs[0])
        hist["v_norm"].append(obs[1])
        hist["alpha"].append(obs[2])
        hist["mu"].append(obs[3])
        hist["p"].append(obs[4])
        hist["q"].append(obs[5])
        hist["de"].append(action[0])
        hist["da"].append(action[1])
        hist["dt_ctrl"].append(action[2])
        hist["h"].append(h)

        obs, _, terminated, _, _ = env.step(action)
        v_true = obs[1] * v_stall
        h += v_true * np.sin(obs[0]) * dt
        t += dt

        new_gamma = obs[0]
        if new_gamma < np.deg2rad(-2.0):
            has_dived = True

        if has_dived and new_gamma >= 0.0:
            for k, val in zip(
                keys[1:7],
                [new_gamma, obs[1], obs[2], obs[3], obs[4], obs[5]],
            ):
                hist[k].append(val)
            hist["t"].append(t)
            hist["de"].append(action[0])
            hist["da"].append(action[1])
            hist["dt_ctrl"].append(action[2])
            hist["h"].append(h)
            logger.info(f"[+] Recovery success at {t:.2f}s, Δh={h:.2f}m")
            break

        if terminated:
            cause = []
            if obs[2] >= np.deg2rad(40):  cause.append("+α")
            if obs[2] <= np.deg2rad(-40): cause.append("-α")
            if obs[0] <= -np.pi + 0.05:   cause.append("γ-dive")
            if abs(obs[3]) >= np.pi / 2:  cause.append("|μ|≥90°")
            if abs(obs[4]) >= 3.0:        cause.append("|p|≥3")
            logger.warning(f"[-] Terminated at {t:.2f}s ({' '.join(cause) or 'unknown'})")
            break

    return hist


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_time_response(hist: dict, prefix: str) -> None:
    """10-panel figure: γ, V/Vs, α, μ, p, q, δe, δa, δt, altitude_loss."""
    t = hist["t"]
    panels = [
        (np.rad2deg(hist["gamma"]),  r"$\gamma$ (deg)", "#532C8A", "plot"),
        (hist["v_norm"],             r"$V/V_s$",         "#E377C2", "plot"),
        (np.rad2deg(hist["alpha"]),  r"$\alpha$ (deg)",  "#D62728", "plot"),
        (np.rad2deg(hist["mu"]),     r"$\mu$ (deg)",     "#9467BD", "plot"),
        (np.rad2deg(hist["p"]),      r"$p$ (deg/s)",     "#8C564B", "plot"),
        (np.rad2deg(hist["q"]),      r"$q$ (deg/s)",     "#2CA02C", "plot"),
        (np.rad2deg(hist["de"]),     r"$\delta_e$ (deg)", "#FF8C00", "step"),
        (np.rad2deg(hist["da"]),     r"$\delta_a$ (deg)", "#BCBD22", "step"),
        (hist["dt_ctrl"],            r"$\delta_t$",       "#17BECF", "step"),
        (hist["h"],                  "Altitude Loss (m)", "#1F77B4", "plot"),
    ]

    fig, axs = plt.subplots(len(panels), 1, figsize=(8, 18), sharex=True)
    fig.suptitle(r"6-DOF Banked-Spin DP Response", fontsize=14)

    for ax, (data, ylabel, color, style) in zip(axs, panels):
        if style == "step":
            ax.step(t, data, color=color, linewidth=2, where="post")
        else:
            ax.plot(t, data, color=color, linewidth=2)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="-", alpha=0.4)

    axs[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[3].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[8].set_ylim([-0.05, 1.05])
    axs[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{prefix}_trajectory.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[*] Time-response plot saved to: {out_path.resolve()}")


def plot_heatmaps(pi: PolicyIterationBankedSpin, prefix: str,
                  v_target: float = 1.0) -> None:
    """4×4 grid: rows=μ values, cols=(δe, δa, δt, altitude_loss).

    Each cell shows a 2D slice in (α, γ) at fixed (V_target, q=0, p=0, μ_row).
    """
    logger.info(f"[*] Building 6D heatmaps at V/Vs={v_target} ...")

    gamma_bins = np.unique(pi.states_space[:, 0])
    v_bins     = np.unique(pi.states_space[:, 1])
    alpha_bins = np.unique(pi.states_space[:, 2])
    mu_bins    = np.unique(pi.states_space[:, 3])
    p_bins     = np.unique(pi.states_space[:, 4])
    q_bins     = np.unique(pi.states_space[:, 5])

    shape_6d = (len(gamma_bins), len(v_bins), len(alpha_bins),
                len(mu_bins), len(p_bins), len(q_bins))
    V_6D = pi.value_function.reshape(shape_6d)
    P_6D = pi.policy.reshape(shape_6d)

    v_idx = int(np.argmin(np.abs(v_bins - v_target)))
    p_idx = int(np.argmin(np.abs(p_bins - 0.0)))
    q_idx = int(np.argmin(np.abs(q_bins - 0.0)))

    # μ slices we want to display
    mu_targets_deg = [0.0, 15.0, 30.0, 45.0]
    mu_indices = [int(np.argmin(np.abs(mu_bins - np.deg2rad(d))))
                  for d in mu_targets_deg]

    gamma_mask = (gamma_bins >= np.deg2rad(-90.1)) & (gamma_bins <= 0.01)
    alpha_mask = ((alpha_bins >= np.deg2rad(-5.1))
                  & (alpha_bins <= np.deg2rad(20.1)))
    gamma_deg = np.rad2deg(gamma_bins[gamma_mask])
    alpha_deg = np.rad2deg(alpha_bins[alpha_mask])
    A_mesh, G_mesh = np.meshgrid(alpha_deg, gamma_deg, indexing="xy")

    fig, axes = plt.subplots(
        len(mu_targets_deg), 4, figsize=(13, 12),
        sharex="col", sharey="row",
    )
    fig.suptitle(
        rf"6-DOF Banked-Spin Policy & Value Heatmaps "
        rf"(V/Vs={v_target}, p=0, q=0)",
        fontsize=13,
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.18, bottom=0.12, top=0.93)

    for i, mu_idx in enumerate(mu_indices):
        v_slice = V_6D[:, v_idx, :, mu_idx, p_idx, q_idx][gamma_mask][:, alpha_mask]
        p_slice = P_6D[:, v_idx, :, mu_idx, p_idx, q_idx][gamma_mask][:, alpha_mask]

        de_slice = np.rad2deg(pi.action_space[p_slice, 0])
        da_slice = np.rad2deg(pi.action_space[p_slice, 1])
        dt_slice = pi.action_space[p_slice, 2]
        alt_loss_slice = -v_slice

        axes[i, 0].pcolormesh(A_mesh, G_mesh, de_slice,
                              cmap="plasma", vmin=-25, vmax=15, shading="gouraud")
        axes[i, 1].pcolormesh(A_mesh, G_mesh, da_slice,
                              cmap="coolwarm", vmin=-15, vmax=15, shading="gouraud")
        axes[i, 2].pcolormesh(A_mesh, G_mesh, dt_slice,
                              cmap="plasma", vmin=0, vmax=1, shading="nearest")
        axes[i, 3].pcolormesh(A_mesh, G_mesh, alt_loss_slice,
                              cmap="plasma", vmin=0, vmax=200, shading="gouraud")

        axes[i, 3].text(
            1.05, 0.5, f"μ = {mu_targets_deg[i]:.0f}°",
            transform=axes[i, 3].transAxes,
            va="center", ha="left", fontsize=11,
        )

    titles = [r"$\delta_e$", r"$\delta_a$", r"$\delta_t$", "Altitude Loss"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, pad=10)
    for j in range(4):
        axes[-1, j].set_xlabel(r"$\alpha$ (deg)")
    for i in range(len(mu_targets_deg)):
        axes[i, 0].set_ylabel(r"$\gamma$ (deg)")

    cbar_specs = [
        (0, r"$\delta_e$ (deg)", [-20, 0, 15], "plasma", -25, 15),
        (1, r"$\delta_a$ (deg)", [-15, 0, 15], "coolwarm", -15, 15),
        (2, r"$\delta_t$",         [0.0, 0.5, 1.0], "plasma", 0, 1),
        (3, "Alt. Loss (m)",       [0, 100, 200], "plasma", 0, 200),
    ]
    for j, (col, label, ticks, cmap, vmin, vmax) in enumerate(cbar_specs):
        cax = fig.add_axes([0.13 + 0.205 * j, 0.04, 0.16, 0.015])
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation="horizontal",
                     label=label, ticks=ticks)

    out_path = RESULTS_DIR / f"{prefix}_heatmaps.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[+] Heatmaps saved to {out_path.resolve()}")


# ── Entry Point ──────────────────────────────────────────────────────────


def main():
    prefix = "banked_spin"
    # Production-quality grid for an RTX 3090: ~97M states, ~4.3 GB VRAM,
    # ~80-120 min training. Bin allocation prioritises α (stall transition)
    # and γ (objective) at 30 bins each; V/Vs at 20 (narrow range);
    # μ at 24; p/q at 15 (most trajectories stay near zero).
    env, states, actions, config = setup_banked_spin_experiment(
        bins_gamma=30, bins_v=20, bins_alpha=30, bins_mu=24,
        bins_p=15, bins_q=15,
    )
    config.theta = 5e-4
    config.maximum_iterations = 8000

    pi = train_or_load_policy(env, states, actions, config, prefix)

    # Initial condition: stall recovery from a banked descent
    hist = run_dp_simulation(
        pi,
        gamma_0_deg=-15.0,
        v_norm_0=1.0,
        alpha_0_deg=18.0,
        mu_0_deg=30.0,
        p_0=0.0,
        q_0_deg=0.0,
    )
    plot_time_response(hist, prefix)
    plot_heatmaps(pi, prefix, v_target=1.0)


if __name__ == "__main__":
    main()
