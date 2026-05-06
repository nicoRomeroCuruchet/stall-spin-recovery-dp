"""
main.py — 8-DOF Spin Recovery DP pipeline.

State:  (γ, V/Vs, α, β, μ, p, q, r)
Action: (δe, δa, δt, δr)

Trains (or loads) a Policy Iteration optimal policy on the GPU,
runs a closed-loop DP simulation, and renders:
  - 14-panel time response (γ, V, α, β, μ, p, q, r, δe, δa, δt, δr, h, …)
  - 4x5 heatmap grid (μ slices × {δe, δa, δt, δr, altitude_loss})

Default grid (Level 1 friendly): 14·12·12·10·12·8·8·8 = ~158M states.
"""
import logging
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

os.environ["NUMBA_THREADING_LAYER"] = "omp"

from aircraft.spin import Spin                          # noqa: E402
from PolicyIterationSpin import (                        # noqa: E402
    PolicyIterationSpin,
    PolicyIterationSpinConfig,
)
from utils.utils import get_optimal_action               # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# ------------------------------------------------------------------
# Experiment setup
# ------------------------------------------------------------------

def setup_spin_experiment(
    bins_gamma: int = 14, bins_v: int = 12, bins_alpha: int = 12,
    bins_beta: int = 10, bins_mu: int = 12,
    bins_p: int = 8, bins_q: int = 8, bins_r: int = 8,
    bins_de: int = 21, bins_da: int = 11, bins_dt: int = 7, bins_dr: int = 9,
) -> tuple[gym.Env, np.ndarray, np.ndarray, PolicyIterationSpinConfig]:
    """Build the 8D state grid and 4D action grid for the spin DP."""
    env = Spin()

    bins = {
        "gamma":  np.linspace(np.deg2rad(-90), np.deg2rad(5),   bins_gamma, dtype=np.float32),
        "v_norm": np.linspace(0.9, 2.0,                          bins_v,     dtype=np.float32),
        "alpha":  np.linspace(np.deg2rad(-14), np.deg2rad(20),   bins_alpha, dtype=np.float32),
        "beta":   np.linspace(np.deg2rad(-25), np.deg2rad(25),   bins_beta,  dtype=np.float32),
        "mu":     np.linspace(np.deg2rad(-60), np.deg2rad(60),   bins_mu,    dtype=np.float32),
        "p":      np.linspace(-2.0, 2.0,                          bins_p,     dtype=np.float32),
        "q":      np.linspace(np.deg2rad(-50), np.deg2rad(50),   bins_q,     dtype=np.float32),
        "r":      np.linspace(-3.0, 3.0,                          bins_r,     dtype=np.float32),
    }
    grid = np.meshgrid(*bins.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T

    de_v = np.linspace(np.deg2rad(-25), np.deg2rad(15), bins_de, dtype=np.float32)
    da_v = np.linspace(np.deg2rad(-15), np.deg2rad(15), bins_da, dtype=np.float32)
    dt_v = np.linspace(0.0, 1.0, bins_dt, dtype=np.float32)
    dr_v = np.linspace(np.deg2rad(-25), np.deg2rad(25), bins_dr, dtype=np.float32)
    action_grid = np.meshgrid(de_v, da_v, dt_v, dr_v, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T

    config = PolicyIterationSpinConfig(
        gamma=1.0, theta=5e-4, n_steps=50, maximum_iterations=8000,
        log=False, log_interval=10,
        w_q_penalty=2.0, w_control_effort=10.0,
        w_alpha_barrier_pos=100.0, w_alpha_barrier_neg=10.0,
        w_crash_penalty=1000.0, w_throttle_bonus=0.2,
        w_p_penalty=0.1, w_r_penalty=0.1, w_beta_penalty=0.5,
        w_mu_barrier=0.5, w_aileron_effort=5.0, w_rudder_effort=5.0,
    )
    bin_tuple = (bins_gamma, bins_v, bins_alpha, bins_beta,
                 bins_mu, bins_p, bins_q, bins_r)
    logger.info(
        f"Experiment grid: states={len(states_space):,} {bin_tuple}, "
        f"actions={len(action_space):,} ({bins_de}·{bins_da}·{bins_dt}·{bins_dr})"
    )
    return env, states_space, action_space, config


# ------------------------------------------------------------------
# Policy training / loading
# ------------------------------------------------------------------

def train_or_load_policy(env, states, actions, config, prefix):
    policy_filename = f"{env.unwrapped.__class__.__name__}_policy.npz"
    RESULTS_DIR.mkdir(exist_ok=True)
    policy_path = RESULTS_DIR / policy_filename

    if policy_path.exists():
        logger.info(f"[+] Existing policy found: {policy_filename}. Loading...")
        try:
            pi = PolicyIterationSpin.load(policy_path, env=env)
            cached_n = int(np.prod(pi.grid_shape))
            requested_n = len(states)
            if cached_n != requested_n:
                logger.warning(
                    f"[!] Cached policy was trained on {cached_n:,} states "
                    f"({tuple(int(b) for b in pi.grid_shape)}) but the current "
                    f"experiment requests {requested_n:,}. Discarding stale cache."
                )
                policy_path.unlink()
            else:
                pi.states_space = states
                logger.info("[+] Policy loaded successfully. Skipping training.")
                return pi
        except Exception as e:
            logger.error(f"[-] Failed to load policy: {e}. Forcing retrain...")

    logger.info(f"[*] Training new policy for {prefix}...")
    pi = PolicyIterationSpin(env, states, actions, config)
    pi.run()
    pi.save(policy_path)
    logger.info(f"[+] Policy cached to {policy_path.resolve()}")
    return pi


# ------------------------------------------------------------------
# DP simulation (closed-loop rollout from a single IC)
# ------------------------------------------------------------------

def run_dp_simulation(
    pi, gamma_0_deg, v_norm_0, alpha_0_deg,
    beta_0_deg, mu_0_deg, p_0, q_0_deg, r_0,
    max_steps: int = 1500,
):
    env = pi.env
    v_stall = env.airplane.STALL_AIRSPEED
    dt = env.airplane.TIME_STEP

    obs, _ = env.specific_reset(
        gamma=np.deg2rad(gamma_0_deg),
        v_norm=v_norm_0,
        alpha=np.deg2rad(alpha_0_deg),
        beta=np.deg2rad(beta_0_deg),
        mu=np.deg2rad(mu_0_deg),
        p=p_0,
        q=np.deg2rad(q_0_deg),
        r=r_0,
    )

    keys = ["t", "gamma", "v_norm", "alpha", "beta", "mu",
            "p", "q", "r", "de", "da", "dt_ctrl", "dr", "h"]
    hist = {k: [] for k in keys}

    t, h = 0.0, 0.0
    has_dived = False
    action = [0.0, 0.0, 0.0, 0.0]
    current_action_idx = None

    for _ in range(max_steps):
        action, _, current_action_idx = get_optimal_action(obs, pi, current_action_idx)

        hist["t"].append(t)
        hist["gamma"].append(obs[0])
        hist["v_norm"].append(obs[1])
        hist["alpha"].append(obs[2])
        hist["beta"].append(obs[3])
        hist["mu"].append(obs[4])
        hist["p"].append(obs[5])
        hist["q"].append(obs[6])
        hist["r"].append(obs[7])
        hist["de"].append(action[0])
        hist["da"].append(action[1])
        hist["dt_ctrl"].append(action[2])
        hist["dr"].append(action[3])
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
                keys[1:9],
                [new_gamma, obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7]],
            ):
                hist[k].append(val)
            hist["t"].append(t)
            hist["de"].append(action[0])
            hist["da"].append(action[1])
            hist["dt_ctrl"].append(action[2])
            hist["dr"].append(action[3])
            hist["h"].append(h)
            logger.info(f"[+] Recovery success at {t:.2f}s, Δh={h:.2f}m")
            break

        if terminated:
            cause = []
            if obs[2] >= np.deg2rad(40):  cause.append("+α")
            if obs[2] <= np.deg2rad(-40): cause.append("-α")
            if obs[0] <= -np.pi + 0.05:   cause.append("γ-dive")
            if abs(obs[3]) >= np.deg2rad(30): cause.append("|β|≥30°")
            if abs(obs[4]) >= np.pi / 2:  cause.append("|μ|≥90°")
            if abs(obs[5]) >= 3.0:        cause.append("|p|≥3")
            if abs(obs[7]) >= 4.0:        cause.append("|r|≥4")
            logger.warning(f"[-] Terminated at {t:.2f}s ({' '.join(cause) or 'unknown'})")
            break

    return hist


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

STATE_COLOR = "#1F4E79"
ACTION_COLOR = "#D95F02"


def plot_time_response(hist: dict, prefix: str) -> None:
    """14-panel figure: γ, V/Vs, α, β, μ, p, q, r, δe, δa, δt, δr, h."""
    t = hist["t"]
    panels = [
        (np.rad2deg(hist["gamma"]),  r"$\gamma$ (deg)",   STATE_COLOR,  "plot"),
        (hist["v_norm"],             r"$V/V_s$",          STATE_COLOR,  "plot"),
        (np.rad2deg(hist["alpha"]),  r"$\alpha$ (deg)",   STATE_COLOR,  "plot"),
        (np.rad2deg(hist["beta"]),   r"$\beta$ (deg)",    STATE_COLOR,  "plot"),
        (np.rad2deg(hist["mu"]),     r"$\mu$ (deg)",      STATE_COLOR,  "plot"),
        (np.rad2deg(hist["p"]),      r"$p$ (deg/s)",      STATE_COLOR,  "plot"),
        (np.rad2deg(hist["q"]),      r"$q$ (deg/s)",      STATE_COLOR,  "plot"),
        (np.rad2deg(hist["r"]),      r"$r$ (deg/s)",      STATE_COLOR,  "plot"),
        (np.rad2deg(hist["de"]),     r"$\delta_e$ (deg)", ACTION_COLOR, "step"),
        (np.rad2deg(hist["da"]),     r"$\delta_a$ (deg)", ACTION_COLOR, "step"),
        (hist["dt_ctrl"],            r"$\delta_t$",       ACTION_COLOR, "step"),
        (np.rad2deg(hist["dr"]),     r"$\delta_r$ (deg)", ACTION_COLOR, "step"),
        (hist["h"],                  "Altitude Loss (m)", STATE_COLOR,  "plot"),
    ]
    fig, axs = plt.subplots(len(panels), 1, figsize=(8, 22), sharex=True)
    fig.suptitle("8-DOF Spin Recovery DP Response", fontsize=14)

    for ax, (data, ylabel, color, style) in zip(axs, panels):
        if style == "step":
            ax.step(t, data, color=color, linewidth=2, where="post")
        else:
            ax.plot(t, data, color=color, linewidth=2)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="-", alpha=0.4)

    axs[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[3].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[4].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[10].set_ylim([-0.05, 1.05])
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{prefix}_trajectory.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[*] Time-response plot saved to: {out_path.resolve()}")


def plot_heatmaps(pi: PolicyIterationSpin, prefix: str,
                  v_target: float = 1.0,
                  alpha_fix: float = 6.0,
                  beta_fix: float = 0.0) -> None:
    """4×5 heatmap grid: μ slices × (δe, δa, δt, δr, altitude_loss)."""
    logger.info(f"[*] Building 8D heatmaps at V/Vs={v_target}, α={alpha_fix}°, β={beta_fix}° ...")

    # Reshape policy/value to 8D grid
    gamma_b = np.unique(pi.states_space[:, 0])
    v_b     = np.unique(pi.states_space[:, 1])
    alpha_b = np.unique(pi.states_space[:, 2])
    beta_b  = np.unique(pi.states_space[:, 3])
    mu_b    = np.unique(pi.states_space[:, 4])
    p_b     = np.unique(pi.states_space[:, 5])
    q_b     = np.unique(pi.states_space[:, 6])
    r_b     = np.unique(pi.states_space[:, 7])

    shape = (len(gamma_b), len(v_b), len(alpha_b), len(beta_b),
             len(mu_b), len(p_b), len(q_b), len(r_b))
    V_8D = pi.value_function.reshape(shape)
    P_8D = pi.policy.reshape(shape)

    # Fixed indices
    v_idx     = int(np.argmin(np.abs(v_b - v_target)))
    alpha_idx = int(np.argmin(np.abs(alpha_b - np.deg2rad(alpha_fix))))
    beta_idx  = int(np.argmin(np.abs(beta_b - np.deg2rad(beta_fix))))
    p_idx     = int(np.argmin(np.abs(p_b - 0.0)))
    q_idx     = int(np.argmin(np.abs(q_b - 0.0)))
    r_idx     = int(np.argmin(np.abs(r_b - 0.0)))

    mu_targets_deg = [0.0, 15.0, 30.0, 45.0]
    mu_indices = [int(np.argmin(np.abs(mu_b - np.deg2rad(d)))) for d in mu_targets_deg]

    gamma_mask = (gamma_b >= np.deg2rad(-90.1)) & (gamma_b <= 0.01)
    alpha_mask = (alpha_b >= np.deg2rad(-5.1)) & (alpha_b <= np.deg2rad(20.1))
    gamma_deg = np.rad2deg(gamma_b[gamma_mask])
    alpha_deg = np.rad2deg(alpha_b[alpha_mask])
    A_mesh, G_mesh = np.meshgrid(alpha_deg, gamma_deg, indexing="xy")

    fig, axes = plt.subplots(
        len(mu_targets_deg), 5, figsize=(15, 12),
        sharex="col", sharey="row",
    )
    fig.suptitle(
        rf"8-DOF Spin Recovery — Policy & Value (V/Vs={v_target}, "
        rf"$\alpha_{{fix}}={alpha_fix}^\circ$, $\beta_{{fix}}={beta_fix}^\circ$, "
        rf"$p=q=r=0$)",
        fontsize=12,
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.18, bottom=0.12, top=0.93)

    for i, mu_idx in enumerate(mu_indices):
        # Slice: vary (γ, α), fix everything else.
        v_slice = V_8D[:, v_idx, :, beta_idx, mu_idx, p_idx, q_idx, r_idx][gamma_mask][:, alpha_mask]
        p_slice = P_8D[:, v_idx, :, beta_idx, mu_idx, p_idx, q_idx, r_idx][gamma_mask][:, alpha_mask]

        de_slice = np.rad2deg(pi.action_space[p_slice, 0])
        da_slice = np.rad2deg(pi.action_space[p_slice, 1])
        dt_slice = pi.action_space[p_slice, 2]
        dr_slice = np.rad2deg(pi.action_space[p_slice, 3])
        alt_loss = -v_slice

        axes[i, 0].pcolormesh(A_mesh, G_mesh, de_slice, cmap="plasma",
                              vmin=-25, vmax=15, shading="gouraud")
        axes[i, 1].pcolormesh(A_mesh, G_mesh, da_slice, cmap="coolwarm",
                              vmin=-15, vmax=15, shading="gouraud")
        axes[i, 2].pcolormesh(A_mesh, G_mesh, dt_slice, cmap="plasma",
                              vmin=0, vmax=1, shading="nearest")
        axes[i, 3].pcolormesh(A_mesh, G_mesh, dr_slice, cmap="coolwarm",
                              vmin=-25, vmax=25, shading="gouraud")
        axes[i, 4].pcolormesh(A_mesh, G_mesh, alt_loss, cmap="plasma",
                              vmin=0, vmax=200, shading="gouraud")

        axes[i, 4].text(1.05, 0.5, f"μ = {mu_targets_deg[i]:.0f}°",
                        transform=axes[i, 4].transAxes,
                        va="center", ha="left", fontsize=11)

    titles = [r"$\delta_e$", r"$\delta_a$", r"$\delta_t$",
              r"$\delta_r$", "Altitude Loss"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, pad=10)
    for j in range(5):
        axes[-1, j].set_xlabel(r"$\alpha$ (deg)")
    for i in range(len(mu_targets_deg)):
        axes[i, 0].set_ylabel(r"$\gamma$ (deg)")

    cbar_specs = [
        (0, r"$\delta_e$ (deg)",  [-20, 0, 15],   "plasma",   -25, 15),
        (1, r"$\delta_a$ (deg)",  [-15, 0, 15],   "coolwarm", -15, 15),
        (2, r"$\delta_t$",        [0.0, 0.5, 1.0],"plasma",     0,  1),
        (3, r"$\delta_r$ (deg)",  [-25, 0, 25],   "coolwarm", -25, 25),
        (4, "Alt. Loss (m)",      [0, 100, 200],  "plasma",     0, 200),
    ]
    for j, (col, label, ticks, cmap, vmin, vmax) in enumerate(cbar_specs):
        cax = fig.add_axes([0.10 + 0.165 * j, 0.04, 0.13, 0.012])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cax, orientation="horizontal", label=label, ticks=ticks)

    out_path = RESULTS_DIR / f"{prefix}_heatmaps.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"[+] Heatmaps saved to {out_path.resolve()}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    prefix = "spin"
    env, states, actions, config = setup_spin_experiment()
    pi = train_or_load_policy(env, states, actions, config, prefix)

    # Spin-recovery initial condition: banked descent with sideslip and
    # initial yaw rate (typical incipient-spin entry state).
    hist = run_dp_simulation(
        pi,
        gamma_0_deg=-15.0, v_norm_0=1.0, alpha_0_deg=18.0,
        beta_0_deg=5.0, mu_0_deg=30.0, p_0=0.0, q_0_deg=0.0, r_0=0.5,
    )
    plot_time_response(hist, prefix)
    plot_heatmaps(pi, prefix, v_target=1.0, alpha_fix=6.0, beta_fix=0.0)


if __name__ == "__main__":
    main()
