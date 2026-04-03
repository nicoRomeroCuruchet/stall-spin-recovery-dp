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

from airplane.symmetric_stall import SymmetricStall
from PolicyIteration import PolicyIterationStall, PolicyIterationStallConfig
from utils.utils import get_optimal_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # 36 bins (~0.94° resolution)
        "alpha": np.linspace(np.deg2rad(-14), np.deg2rad(20), 36, dtype=np.float32),
        # 41 bins (~2.4°/s resolution over ±50°/s)
        "pitch_rate": np.linspace(np.deg2rad(-50), np.deg2rad(50), 41, dtype=np.float32),
    }
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T

    # 21 x 7 = 147 actions (unchanged)
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
        # Best Weights from Sweep (q6.0_ce6.0_ab100):
        w_q_penalty=6.0,
        w_control_effort=6.0,
        w_alpha_barrier_pos=100.0,
        w_alpha_barrier_neg=10.0,
        w_crash_penalty=1000.0,
        w_throttle_bonus=0.2
    )
    return env, states_space, action_space, config


def train_or_load_policy(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    config: PolicyIterationStallConfig,
    prefix: str
) -> PolicyIterationStall:
    """Loads pre-trained policy from disk or trains a new one and caches it."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    policy_filename = f"{env.unwrapped.__class__.__name__}_policy.npz"
    policy_path = results_dir / policy_filename

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
    Maintains strict 100Hz synchronization and accurate altitude tracking.
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
    has_dived = False
    action = [0.0, 0.0]

    for _ in range(1500):
        action, _ = get_optimal_action(obs, pi)

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
        obs, _, terminated, _, _ = env.step(action)

        # 3. Integrate altitude loss kinematically
        v_true = obs[1] * v_stall
        h += v_true * np.sin(obs[0]) * dt
        t += dt

        # Extract the post-step Gamma directly for accurate termination parsing
        new_gamma = np.rad2deg(obs[0])

        if new_gamma < -2.0:
            has_dived = True

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
    # Colors match AIAA 2023 Fig. 7: states=dark blue, controls=orange, altitude=red
    C_STATE = '#2C4B9E'
    C_CTRL  = '#E87C1E'
    C_ALT   = '#D62728'

    fig, axs = plt.subplots(7, 1, figsize=(8, 16), sharex=True)
    fig.suptitle(r"Symmetric Stall Recovery — DP Policy (100Hz)", fontsize=14)

    axs[0].plot(history_t, history_gamma, color=C_STATE, linewidth=2, label="DP Policy")
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axs[0].set_ylabel(r'$\gamma$ (deg)')

    axs[1].plot(history_t, history_v, color=C_STATE, linewidth=2, label="DP Policy")
    axs[1].set_ylabel(r'$V/V_s$')

    axs[2].plot(history_t, history_alpha, color=C_STATE, linewidth=2, label="DP Policy")
    axs[2].set_ylabel(r'$\alpha$ (deg)')

    axs[3].plot(history_t, history_q, color=C_STATE, linewidth=2, label="DP Policy")
    axs[3].set_ylabel(r'$q$ (deg/s)')

    axs[4].step(
        history_t, history_de, color=C_CTRL, linewidth=2, where="post", label="DP Policy"
    )
    axs[4].set_ylabel(r'$\delta_e$ (deg)')

    axs[5].step(
        history_t, history_dt, color=C_CTRL, linewidth=2, where="post", label="DP Policy"
    )
    axs[5].set_ylabel(r'$\delta_t$')
    axs[5].set_ylim([-0.05, 1.05])

    axs[6].plot(history_t, history_h, color=C_ALT, linewidth=2, label="DP Policy")
    axs[6].set_ylabel('Altitude Loss (m)')
    axs[6].set_xlabel('Time (s)')

    for ax in axs:
        ax.grid(True, linestyle='-', alpha=0.4)
        ax.legend(loc="best")

    plt.tight_layout()

    out_path = Path("results") / f"{prefix}_Markovian_DP.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"[*] Plot saved securely to: {out_path.resolve()}")


def generate_paper_heatmap_figures(pi: PolicyIterationStall, prefix: str) -> None:
    logger.info("[*] Extracting 4D Tensors for the Figure 6 Heatmaps...")

    gamma_bins = np.unique(pi.states_space[:, 0])
    v_bins     = np.unique(pi.states_space[:, 1])
    alpha_bins = np.unique(pi.states_space[:, 2])
    q_bins     = np.unique(pi.states_space[:, 3])

    shape_4d = (len(gamma_bins), len(v_bins), len(alpha_bins), len(q_bins))
    V_4D = pi.value_function.reshape(shape_4d)
    Pol_4D = pi.policy.reshape(shape_4d)

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
        pcm_de = ax_de.pcolormesh(
            A_mesh, G_mesh, de_slice, cmap=cmap_str, vmin=-25, vmax=15, shading='gouraud'
        )
        if i == 0:
            ax_de.set_title('Policy for Elevator', pad=10)
        ax_de.set_ylabel(r'$\gamma$ (deg)')
        ax_de.set_yticks([0, -30, -60, -90])

        # -- Throttle Plot --
        ax_dt = axes[i, 1]
        pcm_dt = ax_dt.pcolormesh(
            A_mesh, G_mesh, dt_slice, cmap=cmap_str, vmin=0, vmax=1, shading='nearest'
        )
        if i == 0:
            ax_dt.set_title('Policy for Throttle', pad=10)

        # -- Altitude Loss Plot --
        ax_al = axes[i, 2]
        pcm_al = ax_al.pcolormesh(
            A_mesh, G_mesh, alt_loss_slice,
            cmap=cmap_str, vmin=0, vmax=100, shading='gouraud'
        )
        if i == 0:
            ax_al.set_title('Altitude Loss', pad=10)

        ax_al.text(
            1.05, 0.5, f'V/Vs = {v_target}',
            transform=ax_al.transAxes, va='center', ha='left', fontsize=11
        )

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

    fig.colorbar(
        pcm_de, cax=cbar_ax_de, orientation='horizontal',
        label=r'$\delta_e$ (deg)', ticks=[-20, 0, 15]
    )
    fig.colorbar(
        pcm_dt, cax=cbar_ax_dt, orientation='horizontal',
        label=r'$\delta_t$', ticks=[0.0, 0.5, 1.0]
    )
    fig.colorbar(
        pcm_al, cax=cbar_ax_al, orientation='horizontal',
        label='Altitude Loss (m)', ticks=[0, 100]
    )

    out_path = Path("results") / f"{prefix}_Fig6_Stall_Heatmaps.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"[+] Figure 6 (3x3 Heatmaps) securely saved to {out_path.resolve()}")


def main():
    prefix = "symmetric_stall"
    env, states, actions, config = setup_symmetric_stall_experiment()
    pi = train_or_load_policy(env, states, actions, config, prefix)
    simulate_and_plot_pure_response(pi, prefix)
    generate_paper_heatmap_figures(pi, prefix)


if __name__ == "__main__":
    main()
