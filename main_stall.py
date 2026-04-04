"""
main_stall.py — Pure Markovian DP Simulation
Executes the mathematically rigorous DP policy.
Includes persistent policy caching and precise physical altitude tracking.
"""
import logging
import os
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

os.environ["NUMBA_THREADING_LAYER"] = "omp"

from airplane.symmetric_stall import SymmetricStall  # noqa: E402
from PolicyIterationStall import PolicyIterationStall, PolicyIterationStallConfig  # noqa: E402
from utils.utils import get_optimal_action  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_symmetric_stall_experiment() -> Tuple[
    gym.Env, np.ndarray, np.ndarray, PolicyIterationStallConfig
]:
    """Configures the experiment with physically bounded state grids."""
    env = SymmetricStall()

    bins_space = {
        # 56 bins (~1.7° resolution)
        "flight_path_angle": np.linspace(np.deg2rad(-90), np.deg2rad(5), 56, dtype=np.float32),
        # 41 bins
        "airspeed_norm": np.linspace(0.9, 2.0, 41, dtype=np.float32),
        # 36 bins (~0.97° resolution)
        "alpha": np.linspace(np.deg2rad(-14), np.deg2rad(20), 36, dtype=np.float32),
        # 41 bins (~2.5°/s resolution)
        "pitch_rate": np.linspace(np.deg2rad(-50), np.deg2rad(50), 41, dtype=np.float32),
    }
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T

    # 21 x 7 = 147 actions
    de_vals = np.linspace(np.deg2rad(-25), np.deg2rad(15), 21, dtype=np.float32)
    dt_vals = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    action_grid = np.meshgrid(de_vals, dt_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T

    # Coarse precision settings for quick verification (~2 min run)
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
        w_throttle_bonus=0.0
    )
    return env, states_space, action_space, config


def train_or_load_policy(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    config: PolicyIterationStallConfig,
    prefix: str
) -> PolicyIterationStall:
    """Handles training execution or loads pre-trained tensors from disk to avoid recalculation."""
    policy_filename = f"{env.unwrapped.__class__.__name__}_policy.npz"
    policy_path = Path(policy_filename)

    if policy_path.exists():
        logger.info(f"[+] Existing policy found: {policy_filename}. Loading...")
        try:
            pi = PolicyIterationStall.load(policy_path, env=env)
            pi.states_space = states
            logger.info("[+] Policy loaded successfully from disk. Skipping training.")
            return pi
        except Exception as e:
            logger.error(f"[-] Failed to load policy: {e}. Forcing retrain...")

    logger.info(f"[*] Training new policy for {prefix}...")
    pi = PolicyIterationStall(env, states, actions, config)
    pi.run()

    # Explicitly enforce caching to disk
    pi.save(policy_path)
    logger.info(f"[+] Policy cached successfully to {policy_path.resolve()}")
    return pi


def simulate_and_plot_pure_response(pi: PolicyIterationStall, prefix: str) -> None:
    """
    Executes a pure, unadulterated simulation of the discrete policy.
    Maintains strict 10Hz synchronization and accurate altitude tracking.
    """
    logger.info("[*] Running raw 100Hz DP simulation...")
    env = pi.env
    v_stall = env.airplane.STALL_AIRSPEED

    gamma_0 = 0.0
    v_norm_0 = 0.95
    alpha_0 = 20.0
    q_0 = 0.0

    obs, _ = env.specific_reset(
        np.deg2rad(gamma_0),
        v_norm_0,
        np.deg2rad(alpha_0),
        np.deg2rad(q_0)
    )

    history_t, history_gamma, history_alpha, history_q = [], [], [], []
    history_de, history_dt, history_v, history_h = [], [], [], []

    t = 0.0
    h = 0.0
    dt = env.airplane.TIME_STEP
    control_dt = 0.01  # 100Hz policy evaluation loop
    steps_per_control = max(1, int(round(control_dt / dt)))

    has_dived = False
    action = [0.0, 0.0]
    current_action_idx = None

    for step in range(1500):
        # 1. DP Brain queried exactly matching its native temporal horizon
        if step % steps_per_control == 0:
            action, _, current_action_idx = get_optimal_action(obs, pi, current_action_idx)

        current_gamma = np.rad2deg(obs[0])
        current_alpha = np.rad2deg(obs[2])
        current_q = np.rad2deg(obs[3])
        current_de = np.rad2deg(action[0])
        current_dt_val = action[1]
        current_v = obs[1]

        history_t.append(t)
        history_gamma.append(current_gamma)
        history_alpha.append(current_alpha)
        history_q.append(current_q)
        history_de.append(current_de)
        history_dt.append(current_dt_val)
        history_v.append(current_v)
        history_h.append(h)

        # 2. Step the physics
        obs, reward, terminated, truncated, _ = env.step(action)

        # 3. Integrate altitude loss kinematically
        v_true = obs[1] * v_stall
        h += v_true * np.sin(obs[0]) * dt
        t += dt

        # Extract the post-step Gamma directly for accurate termination parsing
        new_gamma = np.rad2deg(obs[0])

        if new_gamma < -2.0:
            has_dived = True

        # Recovery success: gamma returned to ~0 after diving
        # Stop at -0.5 deg to avoid policy instability at the terminal boundary
        if has_dived and new_gamma >= 0.0:
            history_t.append(t)
            history_gamma.append(new_gamma)
            history_alpha.append(np.rad2deg(obs[2]))
            history_q.append(np.rad2deg(obs[3]))
            history_de.append(current_de)
            history_dt.append(current_dt_val)
            history_v.append(obs[1])
            history_h.append(h)
            logger.info(f"[+] Recovery success at {t:.2f}s with {h:.2f}m altitude loss.")
            break

        if terminated:
            # Crash conditions
            if obs[2] >= np.deg2rad(40):
                logger.warning(f"[-] Structural failure (+Alpha limit) at {t:.2f}s")
                break
            elif obs[2] <= np.deg2rad(-40):
                logger.warning(f"[-] Structural failure (-Alpha limit) at {t:.2f}s")
                break
            elif obs[0] <= -np.pi + 0.05:
                logger.warning(f"[-] Catastrophic dive failure (-180 deg) at {t:.2f}s")
                break

    # === PLOT RAW DP RESPONSES ===
    fig, axs = plt.subplots(7, 1, figsize=(8, 16), sharex=True)
    fig.suptitle(r"Markov-Compliant DP Response (100Hz Control)", fontsize=14)

    axs[0].plot(history_t, history_gamma, color='#532C8A', linewidth=2, label=r"$\gamma$")
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axs[0].set_ylabel(r'$\gamma$ (deg)')

    axs[1].plot(history_t, history_v, color='#E377C2', linewidth=2, label=r"$V/V_s$")
    axs[1].set_ylabel(r'$V/V_s$')

    axs[2].plot(history_t, history_alpha, color='#D62728', linewidth=2, label=r"$\alpha$")
    axs[2].set_ylabel(r'$\alpha$ (deg)')

    axs[3].plot(history_t, history_q, color='#2CA02C', linewidth=2, label=r"$q$")
    axs[3].set_ylabel(r'$q$ (deg/s)')

    axs[4].step(history_t, history_de, color='#FF8C00', linewidth=2, where="post", label=r"$\delta_e$")
    axs[4].set_ylabel(r'$\delta_e$ (deg)')

    axs[5].step(history_t, history_dt, color='#17BECF', linewidth=2, where="post", label=r"$\delta_t$")
    axs[5].set_ylabel(r'$\delta_t$')
    axs[5].set_ylim([-0.05, 1.05])

    axs[6].plot(history_t, history_h, color='#1F77B4', linewidth=2, label=r"$\Delta h$")
    axs[6].set_ylabel('Altitude Loss (m)')
    axs[6].set_xlabel('Time (s)')

    for ax in axs:
        ax.grid(True, linestyle='-', alpha=0.4)
        ax.legend(loc="best")

    plt.tight_layout()

    out_path = Path("img") / f"{prefix}_Markovian_DP.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[*] Plot saved securely to: {out_path.resolve()}")


def generate_paper_heatmap_figures(pi: PolicyIterationStall, prefix: str) -> None:
    logger.info("[*] Extracting 4D Tensors for the Figure 6 Heatmaps...")

    gamma_bins = np.unique(pi.states_space[:, 0])
    v_bins = np.unique(pi.states_space[:, 1])
    alpha_bins = np.unique(pi.states_space[:, 2])
    q_bins = np.unique(pi.states_space[:, 3])

    V_4D = pi.value_function.reshape((len(gamma_bins), len(v_bins), len(alpha_bins), len(q_bins)))
    Pol_4D = pi.policy.reshape((len(gamma_bins), len(v_bins), len(alpha_bins), len(q_bins)))

    q_idx = int(np.argmin(np.abs(q_bins - 0.0)))
    v_targets = [0.9, 1.0, 1.1]

    gamma_mask = (gamma_bins >= np.deg2rad(-90.1)) & (gamma_bins <= 0.01)
    alpha_mask = (alpha_bins >= np.deg2rad(-5.1)) & (alpha_bins <= np.deg2rad(20.1))

    gamma_plot = gamma_bins[gamma_mask]
    alpha_plot = alpha_bins[alpha_mask]

    gamma_deg = np.rad2deg(gamma_plot)
    alpha_deg = np.rad2deg(alpha_plot)
    A_mesh, G_mesh = np.meshgrid(alpha_deg, gamma_deg, indexing="xy")

    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1, hspace=0.15, bottom=0.2)
    cmap_str = 'plasma'

    for i, v_target in enumerate(v_targets):
        v_idx = int(np.argmin(np.abs(v_bins - v_target)))

        v_slice = V_4D[:, v_idx, :, q_idx][gamma_mask][:, alpha_mask]
        p_slice = Pol_4D[:, v_idx, :, q_idx][gamma_mask][:, alpha_mask]

        de_slice = np.zeros_like(p_slice, dtype=np.float32)
        dt_slice = np.zeros_like(p_slice, dtype=np.float32)

        for g_i in range(p_slice.shape[0]):
            for a_i in range(p_slice.shape[1]):
                act_idx = p_slice[g_i, a_i]
                de_slice[g_i, a_i] = np.rad2deg(pi.action_space[act_idx, 0])
                dt_slice[g_i, a_i] = pi.action_space[act_idx, 1]

        alt_loss_slice = -v_slice

        # -- Elevator Plot --
        ax_de = axes[i, 0]
        pcm_de = ax_de.pcolormesh(A_mesh, G_mesh, de_slice, cmap=cmap_str, vmin=-25, vmax=15, shading='gouraud')
        if i == 0:
            ax_de.set_title('Policy for Elevator', pad=10)
        ax_de.set_ylabel(r'$\gamma$ (deg)')
        ax_de.set_yticks([0, -30, -60, -90])

        # -- Throttle Plot --
        ax_dt = axes[i, 1]
        pcm_dt = ax_dt.pcolormesh(A_mesh, G_mesh, dt_slice, cmap=cmap_str, vmin=0, vmax=1, shading='nearest')
        if i == 0:
            ax_dt.set_title('Policy for Throttle', pad=10)

        # -- Altitude Loss Plot --
        ax_al = axes[i, 2]
        pcm_al = ax_al.pcolormesh(A_mesh, G_mesh, alt_loss_slice, cmap=cmap_str, vmin=0, vmax=100, shading='gouraud')
        if i == 0:
            ax_al.set_title('Altitude Loss', pad=10)

        ax_al.text(1.05, 0.5, f'V/Vs = {v_target}', transform=ax_al.transAxes, va='center', ha='left', fontsize=11)

        if i == 2:
            ax_de.set_xlabel(r'$\alpha$ (deg)')
            ax_dt.set_xlabel(r'$\alpha$ (deg)')
            ax_al.set_xlabel(r'$\alpha$ (deg)')
            ax_de.set_xticks([0, 10, 20])
            ax_dt.set_xticks([0, 10, 20])
            ax_al.set_xticks([0, 10, 20])

    cbar_ax_de = fig.add_axes([0.15, 0.05, 0.2, 0.02])
    cbar_ax_dt = fig.add_axes([0.42, 0.05, 0.2, 0.02])
    cbar_ax_al = fig.add_axes([0.70, 0.05, 0.2, 0.02])

    fig.colorbar(pcm_de, cax=cbar_ax_de, orientation='horizontal', label=r'$\delta_e$ (deg)', ticks=[-20, 0, 15])
    fig.colorbar(pcm_dt, cax=cbar_ax_dt, orientation='horizontal', label=r'$\delta_t$', ticks=[0.0, 0.5, 1.0])
    fig.colorbar(pcm_al, cax=cbar_ax_al, orientation='horizontal', label='Altitude Loss (m)', ticks=[0, 100])

    img_dir = Path("img")
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"{prefix}_Fig6_Stall_Heatmaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"[+] Figure 6 (3x3 Heatmaps) securely saved to {out_path.resolve()}")


def validate_with_casadi(pi: PolicyIterationStall, prefix: str) -> None:
    """
    Compares the DP policy trajectory against a CasADi/IPOPT continuous-time
    optimal trajectory on the same 7-panel figure.

    The DP trajectory is used as a warm-start seed for IPOPT, so both solutions
    start from identical initial conditions and the comparison is fair.
    """
    try:
        from casadi_stall_optimizer import CasadiStallOptimizer
    except ImportError:
        logger.error("casadi_stall_optimizer.py not found — skipping CasADi validation.")
        return

    env = pi.env
    v_stall = env.airplane.STALL_AIRSPEED
    k_thrust = env.airplane.THROTTLE_LINEAR_MAPPING

    gamma_0 = 0.0
    v_norm_0 = 0.95
    alpha_0 = 20.0
    q_0 = 0.0

    # ── 1. Run DP simulation and collect full history ──────────────────────
    logger.info("[*] Running DP simulation for CasADi seed...")
    obs, _ = env.specific_reset(
        np.deg2rad(gamma_0), v_norm_0, np.deg2rad(alpha_0), np.deg2rad(q_0)
    )

    hist = {k: [] for k in ["t", "gamma", "v_norm", "alpha", "q", "de", "dt_ctrl", "h"]}
    t, h = 0.0, 0.0
    dt_phys = env.airplane.TIME_STEP
    control_dt = 0.01  # 100Hz
    steps_per_ctrl = max(1, int(round(control_dt / dt_phys)))

    has_dived = False
    action = [0.0, 0.0]
    current_action_idx = None

    for step in range(1500):
        if step % steps_per_ctrl == 0:
            action, _, current_action_idx = get_optimal_action(obs, pi, current_action_idx)

        hist["t"].append(t)
        hist["gamma"].append(obs[0])           # radians
        hist["v_norm"].append(obs[1])
        hist["alpha"].append(obs[2])           # radians
        hist["q"].append(obs[3])               # rad/s
        hist["de"].append(action[0])           # radians
        hist["dt_ctrl"].append(action[1])
        hist["h"].append(h)

        obs, _, terminated, _, _ = env.step(action)
        v_true = obs[1] * v_stall
        h += v_true * np.sin(obs[0]) * dt_phys
        t += dt_phys

        new_gamma = obs[0]
        if new_gamma < np.deg2rad(-2.0):
            has_dived = True

        if has_dived and new_gamma >= 0.0:
            hist["t"].append(t)
            hist["gamma"].append(new_gamma)
            hist["v_norm"].append(obs[1])
            hist["alpha"].append(obs[2])
            hist["q"].append(obs[3])
            hist["de"].append(action[0])
            hist["dt_ctrl"].append(action[1])
            hist["h"].append(h)
            break

        if terminated:
            logger.warning("[!] DP simulation terminated before recovery.")
            break

    dp_T = hist["t"][-1]
    logger.info(f"[+] DP simulation: T={dp_T:.2f}s, Δh={h:.2f}m")

    # ── 2. Build warm-start seed resampled to CasADi node count ───────────
    n_nodes = 100
    t_orig = np.array(hist["t"])
    t_states = np.linspace(0, dp_T, n_nodes + 1)
    t_ctrls = np.linspace(0, dp_T, n_nodes)

    dp_seed = {
        "T":       dp_T,
        "gamma":   np.interp(t_states, t_orig, hist["gamma"]),
        "v_norm":  np.interp(t_states, t_orig, hist["v_norm"]),
        "alpha":   np.interp(t_states, t_orig, hist["alpha"]),
        "q":       np.interp(t_states, t_orig, hist["q"]),
        "h":       np.interp(t_states, t_orig, hist["h"]),
        "de":      np.interp(t_ctrls,  t_orig[:-1], hist["de"][:-1]),
        "dt_ctrl": np.interp(t_ctrls,  t_orig[:-1], hist["dt_ctrl"][:-1]),
    }

    # ── 3. Solve CasADi OCP ────────────────────────────────────────────────
    optimizer = CasadiStallOptimizer(k_thrust=k_thrust)
    cas = optimizer.solve_trajectory(
        gamma_0=np.deg2rad(gamma_0),
        v_norm_0=v_norm_0,
        alpha_0=np.deg2rad(alpha_0),
        q_0=np.deg2rad(q_0),
        n_nodes=n_nodes,
        dp_seed=dp_seed,
    )

    # ── 4. Convert DP history to degrees for plotting ─────────────────────
    dp_t = np.array(hist["t"])
    dp_gamma = np.rad2deg(hist["gamma"])
    dp_v = hist["v_norm"]
    dp_alpha = np.rad2deg(hist["alpha"])
    dp_q = np.rad2deg(hist["q"])
    dp_de = np.rad2deg(hist["de"])
    dp_dt = hist["dt_ctrl"]
    dp_h = hist["h"]

    # ── 5. Comparison plot ────────────────────────────────────────────────
    fig, axs = plt.subplots(7, 1, figsize=(9, 17), sharex=False)
    status = "Converged" if cas["converged"] else "Infeasible (debug)"
    fig.suptitle(
        f"DP vs CasADi/IPOPT — Stall Recovery\n"
        f"DP: Δh={dp_h[-1]:.1f}m, T={dp_t[-1]:.2f}s   |   "
        f"CasADi ({status}): Δh={cas['h'][-1]:.1f}m, T={cas['T']:.2f}s",
        fontsize=12,
    )

    COLOR_DP = "#532C8A"
    COLOR_CAS = "#E8742A"
    LW = 2.0

    panels = [
        (dp_gamma, cas["gamma"], r"$\gamma$ (deg)", "γ"),
        (dp_v, cas["v_norm"], r"$V/V_s$", "V"),
        (dp_alpha, cas["alpha"], r"$\alpha$ (deg)", "α"),
        (dp_q, cas["q"], r"$q$ (deg/s)", "q"),
        (dp_de, cas["de"], r"$\delta_e$ (deg)", "δe"),
        (dp_dt, cas["dt_ctrl"], r"$\delta_t$", "δt"),
        (dp_h, cas["h"], "Altitude Loss (m)", "Δh"),
    ]

    for ax, (dp_data, cas_data, ylabel, _) in zip(axs, panels):
        ax.plot(dp_t,      dp_data,  color=COLOR_DP,  lw=LW,        label="DP Policy")
        ax.plot(cas["t"],  cas_data, color=COLOR_CAS, lw=LW, ls="--", label="CasADi NLP")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="-", alpha=0.35)
        ax.legend(loc="best", fontsize=8)

    axs[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axs[5].set_ylim([-0.05, 1.05])
    axs[6].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = Path("img") / f"{prefix}_DP_vs_CasADi.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[+] Comparison plot saved to {out_path.resolve()}")


def main():
    prefix = "symmetric_stall"
    env, states, actions, config = setup_symmetric_stall_experiment()
    pi = train_or_load_policy(env, states, actions, config, prefix)
    simulate_and_plot_pure_response(pi, prefix)
    generate_paper_heatmap_figures(pi, prefix)
    # validate_with_casadi(pi, prefix)


if __name__ == "__main__":
    main()
