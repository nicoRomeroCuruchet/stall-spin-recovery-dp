"""
plotting.py
-----------
All visualization functions for the Banked Pullout Policy Iteration results.
Includes policy heatmaps, value function contours, and DP recovery
trajectories overlaid with the 6DOF branch's rollouts when available.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interpn

# Allow imports from the parent project directory

from analysis.interpolation import get_optimal_action

logger = logging.getLogger(__name__)


# =====================================================================
# COLORMAPS
# =====================================================================

def get_parula_cmap() -> LinearSegmentedColormap:
    """
    Create a highly accurate approximation of MATLAB's proprietary parula colormap.
    Used for scientific publication matching.
    """
    parula_anchors = [
        (0.2081, 0.1663, 0.5292), (0.0165, 0.4266, 0.8786),
        (0.0384, 0.6743, 0.7436), (0.4420, 0.7481, 0.5033),
        (0.8185, 0.7327, 0.3498), (0.9990, 0.7653, 0.2384),
        (0.9769, 0.9839, 0.0805),
    ]
    return LinearSegmentedColormap.from_list("parula_approx", parula_anchors)


# =====================================================================
# POLICY HEATMAPS
# =====================================================================

def plot_all_paper_style_policies(
    pi,
    prefix: str,
    show_mesh_lines: bool = False
) -> None:
    """
    Generates high-fidelity policy heatmaps overlaid with the exact Bellman
    switching boundaries extracted mathematically from the converged DP tensor.
    """
    if pi.n_dims != 3:
        logger.warning("Policy heatmaps require a 3D state space. Skipping.")
        return

    v_slices = [1.2, 4.0]

    gamma_training = np.unique(pi.states_space[:, 0])
    mu_training = np.unique(pi.states_space[:, 2])

    gamma_mask = (gamma_training >= np.deg2rad(-90.1)) & (gamma_training <= 0.01)
    mu_mask = (mu_training >= -0.01) & (mu_training <= np.deg2rad(180.1))

    gamma_plot = gamma_training[gamma_mask]
    mu_plot = mu_training[mu_mask]

    M_centers, G_centers = np.meshgrid(mu_plot, gamma_plot, indexing="ij")
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
    mesh_kwargs = {
        "cmap": "gray", "shading": "nearest",
        "edgecolors": "k" if show_mesh_lines else "none",
        "linewidth": 0.1 if show_mesh_lines else 0.0
    }

    for v_slice in v_slices:
        logger.info(f"[*] Extracting Bellman Boundaries at V/Vs = {v_slice}...")

        query_pts = np.column_stack([
            G_centers.ravel(), np.full_like(G_centers.ravel(), v_slice), M_centers.ravel()
        ]).astype(np.float32)

        cl_map = np.empty(query_pts.shape[0], dtype=np.float32)
        mu_dot_map = np.empty(query_pts.shape[0], dtype=np.float32)
        dt_map = np.empty(query_pts.shape[0], dtype=np.float32)

        for i, pt in enumerate(query_pts):
            act, _ = get_optimal_action(pt, pi)
            cl_map[i] = act[0]
            mu_dot_map[i] = np.rad2deg(act[1])
            dt_map[i] = act[2] if len(act) >= 3 else 0.0

        C_L = cl_map.reshape(M_centers.shape).T
        P_CMD = mu_dot_map.reshape(M_centers.shape).T
        D_T = dt_map.reshape(M_centers.shape).T

        gamma_deg = np.rad2deg(gamma_plot)
        mu_deg = np.rad2deg(mu_plot)

        # Plot: Commanded Lift Coefficient (C_L)
        fig_cl, ax_cl = plt.subplots(figsize=(10, 4.5))
        ax_cl.pcolormesh(mu_deg, gamma_deg, C_L, vmin=-0.5, vmax=1.0, **mesh_kwargs)

        ax_cl.set_title(f"Optimal policy for $C_L^*$ (V/$V_s$ = {v_slice})", fontsize=16, pad=15)
        ax_cl.set_ylabel("Flight path angle (deg)", fontsize=14)
        ax_cl.set_xlabel("Bank angle (deg)", fontsize=14)
        ax_cl.set_xlim([0, 180])
        ax_cl.set_ylim([-90, 0])
        ax_cl.set_xticks([0, 45, 90, 135, 180])
        ax_cl.set_yticks([-90, -60, -30, 0])

        ax_cl.text(45, -45, "$C_L^* = 1.0$", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_cl.text(145, -30, "$C_L^* = -0.5$", ha="center", va="center", bbox=bbox_props, fontsize=12)

        fig_cl.tight_layout()
        fig_cl.savefig(Path(f"results/{prefix}_policy_CL_V_{v_slice}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_cl)

        # Plot: Commanded Bank Rate (Mu_dot)
        fig_mu, ax_mu = plt.subplots(figsize=(10, 4.5))
        ax_mu.pcolormesh(mu_deg, gamma_deg, P_CMD, vmin=-30.0, vmax=30.0, **mesh_kwargs)

        ax_mu.set_title(f"Optimal policy for $\\dot{{\\mu}}_{{cmd}}^*$ (V/$V_s$ = {v_slice})", fontsize=16, pad=15)
        ax_mu.set_ylabel("Flight path angle (deg)", fontsize=14)
        ax_mu.set_xlabel("Bank angle (deg)", fontsize=14)
        ax_mu.set_xlim([0, 180])
        ax_mu.set_ylim([-90, 0])
        ax_mu.set_xticks([0, 45, 90, 135, 180])
        ax_mu.set_yticks([-90, -60, -30, 0])

        ax_mu.text(45, -35, "$\\dot{\\mu}^* = -30$ deg/s\n(roll back)", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_mu.text(145, -70, "$\\dot{\\mu}^* = 30$ deg/s\n(roll over)", ha="center", va="center", bbox=bbox_props, fontsize=12)

        fig_mu.tight_layout()
        fig_mu.savefig(Path(f"results/{prefix}_policy_MuDot_V_{v_slice}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_mu)

        # Plot: Commanded Throttle (delta_t)
        fig_dt, ax_dt = plt.subplots(figsize=(10, 4.5))
        ax_dt.pcolormesh(mu_deg, gamma_deg, D_T, vmin=0.0, vmax=1.0, **mesh_kwargs)

        ax_dt.set_title(f"Optimal policy for $\\delta_t^*$ (V/$V_s$ = {v_slice})", fontsize=16, pad=15)
        ax_dt.set_ylabel("Flight path angle (deg)", fontsize=14)
        ax_dt.set_xlabel("Bank angle (deg)", fontsize=14)
        ax_dt.set_xlim([0, 180])
        ax_dt.set_ylim([-90, 0])
        ax_dt.set_xticks([0, 45, 90, 135, 180])
        ax_dt.set_yticks([-90, -60, -30, 0])

        ax_dt.text(45, -45, "$\\delta_t^* = 1.0$ (full)", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_dt.text(145, -30, "$\\delta_t^* = 0$ (idle)", ha="center", va="center", bbox=bbox_props, fontsize=12)

        fig_dt.tight_layout()
        fig_dt.savefig(Path(f"results/{prefix}_policy_DeltaT_V_{v_slice}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_dt)


# =====================================================================
# VALUE FUNCTION CONTOURS
# =====================================================================

def plot_value_function_contours(pi, prefix: str) -> None:
    """
    Generate the optimal value function contour maps (Minimum Altitude Loss).
    """
    if pi.n_dims != 3:
        logger.warning("Contour plotting requires a 3D state space. Skipping.")
        return

    logger.info("[*] Generating Value Function contour maps (Figure 1)...")

    gamma_grid = np.unique(pi.states_space[:, 0])
    v_grid = np.unique(pi.states_space[:, 1])
    mu_grid = np.unique(pi.states_space[:, 2])

    V_3D = -pi.value_function.reshape((len(gamma_grid), len(v_grid), len(mu_grid)))

    if np.max(V_3D) > 0 and np.max(V_3D) < 50.0:
        logger.info("[*] Scaling normalized Value Function to physical meters.")
        v_stall = pi.env.airplane.STALL_AIRSPEED
        V_3D *= v_stall

    mu_lo = max(0.0, mu_grid[0])
    mu_hi = min(np.deg2rad(180), mu_grid[-1])
    gamma_lo = max(np.deg2rad(-90), gamma_grid[0])
    gamma_hi = min(0.0, gamma_grid[-1])

    eps = 1e-5
    mu_visual = np.linspace(mu_lo + eps, mu_hi - eps, 200)
    gamma_visual = np.linspace(gamma_lo + eps, gamma_hi - eps, 200)
    M_vis, G_vis = np.meshgrid(mu_visual, gamma_visual, indexing="ij")

    v_slices = [1.2, 2.0, 3.0, 4.0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    levels = np.arange(0, 300, 30)
    cmap = plt.get_cmap("jet")

    for idx, v_slice in enumerate(v_slices):
        ax = axes[idx]
        query_pts = np.column_stack([
            G_vis.ravel(),
            np.full_like(G_vis.ravel(), v_slice),
            M_vis.ravel()
        ])

        alt_loss_flat = interpn(
            (gamma_grid, v_grid, mu_grid), V_3D, query_pts,
            method="cubic", bounds_error=False, fill_value=None
        )
        alt_loss = alt_loss_flat.reshape(M_vis.shape).T

        ax.contourf(np.rad2deg(mu_visual), np.rad2deg(gamma_visual), alt_loss, levels=levels, cmap=cmap, extend="both")
        contour_lines = ax.contour(np.rad2deg(mu_visual), np.rad2deg(gamma_visual), alt_loss, levels=levels, colors='k', linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=9, fmt="%1.0f")

        ax.set_title(f"$V/V_s = {v_slice}$", fontsize=14)
        ax.set_xlim([0, 180])
        ax.set_ylim([-90, 0])
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks([-90, -60, -30, 0])

        if idx >= 2: ax.set_xlabel("Bank angle (deg)", fontsize=12)
        if idx % 2 == 0: ax.set_ylabel("Flight path angle (deg)", fontsize=12)

    plt.tight_layout()
    output_path = Path(f"results/{prefix}_value_function_contours.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[*] Value function contours saved to {output_path.resolve()}")


# =====================================================================
# CASADI VALIDATION
# =====================================================================

def validate_trajectories_with_casadi(pi, prefix: str) -> None:
    """
    Render DP recovery trajectories overlaid against the 6DOF branch's
    rollouts (when CSVs are available in results/comparison_csvs/).

    Three scenarios (Fig3, Fig4, Fig5) covering banked descents at
    gamma_0 = -30, -60, -45 deg. CasADi NLP curve removed; only
    the 3DOF DP and (optionally) the 6DOF DP are plotted.
    Initial bank angles restricted to mu_0 in {30, 60, 90} deg —
    higher banks are outside the 6DOF A.i envelope.
    """
    if pi.n_dims != 3:
        logger.warning("Trajectory plot requires the 3D environment. Skipping.")
        return

    env = pi.env
    dt = env.airplane.TIME_STEP
    mass = env.airplane.MASS
    wing_area = env.airplane.WING_SURFACE_AREA
    air_density = env.airplane.AIR_DENSITY
    v_stall = env.airplane.STALL_AIRSPEED

    # mu_0_list shrunk to {90, 60, 30} so every entry has a matching valid
    # 6DOF rollout (|mu| < 90 deg under simplification A.i).
    scenarios = [
        {
            "gamma_0_deg": -30.0,
            "v_0_norm": 1.2,
            "mu_0_list": [90.0, 60.0, 30.0],
            "x_offsets": [15.0, 45.0, 60.0],
            "fig_id": 3,
        },
        {
            "gamma_0_deg": -60.0,
            "v_0_norm": 1.2,
            "mu_0_list": [90.0, 60.0, 30.0],
            "x_offsets": [15.0, 45.0, 60.0],
            "fig_id": 4,
        },
        # Spiral dive scenario (single mu, V/Vs = 1.3, gamma_0 = -45)
        {
            "gamma_0_deg": -45.0,
            "v_0_norm": 1.3,
            "mu_0_list": [30.0],
            "x_offsets": [0.0],
            "fig_id": 5,
        },
    ]

    for scenario in scenarios:
        gamma_0_deg = scenario["gamma_0_deg"]
        v_0_norm = scenario["v_0_norm"]
        mu_0_list = scenario["mu_0_list"]
        x_offsets = scenario["x_offsets"]
        fig_id = scenario["fig_id"]
        logger.info(
            f"[*] Rendering DP recovery trajectories for "
            f"gamma_0 = {gamma_0_deg} deg, V/Vs = {v_0_norm}..."
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        for mu_0_deg, x_offset in zip(mu_0_list, x_offsets):
            gamma, v_norm, mu = np.deg2rad(gamma_0_deg), v_0_norm, np.deg2rad(mu_0_deg)
            x_dp, h_dp, xi_dp = 0.0, 0.0, 0.0

            s_hist = {"v": [v_norm], "gamma": [gamma], "mu": [mu], "h": [h_dp], "x": [x_dp], "xi": [xi_dp]}

            step_count = 0
            max_steps = 2000

            while gamma < 0.0 and step_count < max_steps:
                state_vector = np.array([gamma, v_norm, mu], dtype=np.float32)
                action, _ = get_optimal_action(state_vector, pi)
                c_lift = action[0]

                v_true = v_norm * v_stall
                lift_force = 0.5 * air_density * wing_area * (v_true ** 2) * c_lift

                h_dot = v_true * np.sin(gamma)
                cos_gamma = np.cos(gamma) if abs(np.cos(gamma)) > 1e-3 else 1e-3
                xi_dot = (lift_force * np.sin(mu)) / (mass * v_true * cos_gamma)
                x_dot = v_true * np.cos(gamma) * np.cos(xi_dp)

                h_dp += h_dot * dt
                xi_dp += xi_dot * dt
                x_dp += x_dot * dt

                env.state = np.atleast_2d(state_vector)
                next_state_matrix, _, _, _, _ = env.step(action)
                next_state = next_state_matrix.flatten()

                gamma, v_norm, mu = next_state[0], next_state[1], next_state[2]

                s_hist["v"].append(v_norm)
                s_hist["gamma"].append(gamma)
                s_hist["mu"].append(mu)
                s_hist["h"].append(h_dp)
                s_hist["x"].append(x_dp)
                s_hist["xi"].append(xi_dp)
                step_count += 1

            x_history_dp = np.array(s_hist["x"]) + x_offset
            h_history_dp = np.array(s_hist["h"])

            lbl_dp = "3DOF DP Policy Iteration" if mu_0_deg == mu_0_list[0] else ""

            ax.plot(x_history_dp, h_history_dp, color='darkred', linewidth=2.0,
                    linestyle='-', label=lbl_dp, zorder=2)

            mark_every = max(1, len(x_history_dp) // 10)
            marker_indices = list(range(0, len(x_history_dp), mark_every))
            for mi in marker_indices:
                if mi < len(x_history_dp) - 1:
                    dx = x_history_dp[mi + 1] - x_history_dp[mi]
                    dh = h_history_dp[mi + 1] - h_history_dp[mi]
                else:
                    dx = x_history_dp[mi] - x_history_dp[mi - 1]
                    dh = h_history_dp[mi] - h_history_dp[mi - 1]
                angle = np.rad2deg(np.arctan2(dh, dx))
                ax.annotate('✈', xy=(x_history_dp[mi], h_history_dp[mi]),
                            fontsize=14, ha='center', va='center',
                            rotation=angle, color='darkred', zorder=5)

            # ---------------------------------------------------------------
            # Optional 6DOF overlay (Camino A): if a CSV from the 6DOF branch
            # exists for this scenario, plot it as a third trajectory.
            # File layout: results/comparison_csvs/6dof_g{gamma0}_v{V0}_mu{mu0}.csv
            # Columns: t, gamma, v_norm, alpha, mu, p, q, x_position, h_loss
            # ---------------------------------------------------------------
            csv_path = Path(
                f"results/comparison_csvs/6dof_g{int(round(gamma_0_deg))}"
                f"_v{v_0_norm:.1f}_mu{int(round(mu_0_deg))}.csv"
            )
            if csv_path.exists():
                try:
                    csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                    # Filter out degenerate rollouts: the 6DOF declares
                    # immediate crash for any |mu_0| >= 90 deg under
                    # simplification A.i, producing a 1-2 row CSV. Skip those.
                    if csv_data.ndim < 2 or len(csv_data) < 50:
                        logger.info(
                            f"    [.] Skipping degenerate 6DOF rollout "
                            f"{csv_path.name} ({len(csv_data)} rows — "
                            f"likely outside A.i envelope |mu| >= 90 deg)"
                        )
                    else:
                        x_6dof = csv_data[:, 7] + x_offset   # x_position column
                        h_6dof = csv_data[:, 8]              # h_loss column
                        # Only label the first valid 6DOF curve so the legend
                        # has at most one '6DOF DP' entry per figure.
                        lbl_6dof = (
                            "6DOF DP (Riley aero, full lateral dynamics)"
                            if not any(
                                (line.get_label() or "").startswith("6DOF DP")
                                for line in ax.lines
                            )
                            else ""
                        )
                        ax.plot(
                            x_6dof, h_6dof,
                            color='darkgreen', linewidth=2.0, linestyle='-',
                            label=lbl_6dof, zorder=3, alpha=0.85,
                        )

                        # Plane glyphs along the 6DOF trajectory (same
                        # convention as the 3DOF red curve).
                        mark_every_6 = max(1, len(x_6dof) // 10)
                        marker_indices_6 = list(range(0, len(x_6dof), mark_every_6))
                        for mi in marker_indices_6:
                            if mi < len(x_6dof) - 1:
                                dx_6 = x_6dof[mi + 1] - x_6dof[mi]
                                dh_6 = h_6dof[mi + 1] - h_6dof[mi]
                            else:
                                dx_6 = x_6dof[mi] - x_6dof[mi - 1]
                                dh_6 = h_6dof[mi] - h_6dof[mi - 1]
                            angle_6 = np.rad2deg(np.arctan2(dh_6, dx_6))
                            ax.annotate(
                                '✈',
                                xy=(x_6dof[mi], h_6dof[mi]),
                                fontsize=14, ha='center', va='center',
                                rotation=angle_6, color='darkgreen',
                                zorder=5,
                            )

                        logger.info(
                            f"    [+] Overlaid 6DOF trajectory from {csv_path.name}"
                        )
                except Exception as e:
                    logger.warning(
                        f"    [!] Failed to read 6DOF CSV {csv_path.name}: {e}"
                    )
            else:
                logger.info(f"    [.] No 6DOF CSV at {csv_path} (skipping overlay)")

        ax.set_title(
            f"Recovery Trajectories: 3DOF DP vs 6DOF DP\n"
            f"Starting from $\\gamma_0$ = {gamma_0_deg:.0f} deg, "
            f"$V_0/V_s$ = {v_0_norm:.1f}",
            fontsize=15, pad=15
        )
        ax.set_xlabel("x-position (m)", fontsize=13)
        ax.set_ylabel("Altitude (m)", fontsize=13)
        ax.set_xlim([-200, 250])
        ax.set_ylim([-250, 50])
        ax.grid(True, which='both', linestyle='-', color='lightgray', linewidth=0.7)
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

        plt.tight_layout()
        output_path = Path(f"results/{prefix}_validation_guided_Fig{fig_id}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"[+] Validation plot saved to {output_path.resolve()}")
