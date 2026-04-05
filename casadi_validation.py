"""
casadi_validation.py
--------------------
Validates the DP policy (from PolicyIteration) against the continuous-time
NLP solver (CasADi + IPOPT) for the 2-DOF Symmetric Pullout with thrust.

For each test scenario the DP trajectory is simulated first, then used as a
warm start for the NLP.  Both trajectories are plotted on the same axes:

    • Spatial view  : x (horizontal distance) vs h (altitude change)
    • Time view     : γ(t), V/Vs(t), C_L(t), δ_throttle(t)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from casadi_optimizer import CasadiSymmetricPulloutOptimizer
from PolicyIteration import PolicyIteration

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# -----------------------------------------------------------------------
# DP trajectory simulation
# -----------------------------------------------------------------------

def _simulate_dp_trajectory(pi: PolicyIteration, v0_norm: float, gamma0_deg: float):
    """
    Simulate the greedy DP policy from a given initial condition.

    Returns dicts of state and control histories.
    """
    from aircraft.reduced_symmetric_pullout import ReducedSymmetricPullout

    env = ReducedSymmetricPullout()
    dt  = env.airplane.TIME_STEP
    Vs  = env.airplane.STALL_AIRSPEED

    ng, nv             = pi.grid_shape
    gamma_lo, gamma_hi = pi.bounds_low[0], pi.bounds_high[0]
    vn_lo,    vn_hi    = pi.bounds_low[1], pi.bounds_high[1]

    env.airplane.reset(np.deg2rad(gamma0_deg), v0_norm, 0.0)
    obs = env._get_obs()

    s = {"t": [], "gamma": [], "v_norm": [], "h": [], "x": []}
    c = {"c_lift": [], "delta_throttle": []}
    t, h, x = 0.0, 0.0, 0.0

    for _ in range(5000):
        gamma = obs[0]
        vn    = obs[1]

        g_idx = int(np.clip(
            (gamma - gamma_lo) / (gamma_hi - gamma_lo) * (ng - 1), 0, ng - 1
        ))
        v_idx = int(np.clip(
            (vn - vn_lo) / (vn_hi - vn_lo) * (nv - 1), 0, nv - 1
        ))
        a_idx    = int(pi.policy[g_idx * nv + v_idx])
        cl       = float(pi.action_space[a_idx, 0])
        throttle = float(pi.action_space[a_idx, 1])

        s["t"].append(t)
        s["gamma"].append(gamma)
        s["v_norm"].append(vn)
        s["h"].append(h)
        s["x"].append(x)
        c["c_lift"].append(cl)
        c["delta_throttle"].append(throttle)

        v_true = vn * Vs
        h += v_true * np.sin(gamma) * dt
        x += v_true * np.cos(gamma) * dt

        obs, _, terminated, _, _ = env.step([cl, throttle])
        t += dt

        if terminated:
            # Append final state
            s["t"].append(t)
            s["gamma"].append(obs[0])
            s["v_norm"].append(obs[1])
            s["h"].append(h)
            s["x"].append(x)
            break

    env.close()
    return (
        {k: np.array(v) for k, v in s.items()},
        {k: np.array(v) for k, v in c.items()},
    )


# -----------------------------------------------------------------------
# Warm-start builder
# -----------------------------------------------------------------------

def _build_dp_seed(s_hist, c_hist, n_nodes: int) -> dict:
    """Interpolate DP trajectory onto the NLP time grid."""
    dp_T  = s_hist["t"][-1]
    t_dp  = s_hist["t"]
    t_cas = np.linspace(0, dp_T, n_nodes + 1)
    t_ctrl = np.linspace(0, dp_T, len(c_hist["c_lift"]))
    t_cas_ctrl = np.linspace(0, dp_T, n_nodes)

    return {
        "T":               dp_T,
        "v_norm":          np.interp(t_cas,      t_dp,   s_hist["v_norm"]),
        "gamma":           np.interp(t_cas,      t_dp,   s_hist["gamma"]),
        "h":               np.interp(t_cas,      t_dp,   s_hist["h"]),
        "c_lift":          np.interp(t_cas_ctrl, t_ctrl, c_hist["c_lift"]),
        "delta_throttle":  np.interp(t_cas_ctrl, t_ctrl, c_hist["delta_throttle"]),
    }


# -----------------------------------------------------------------------
# Spatial validation plot  (x vs h)
# -----------------------------------------------------------------------

def _plot_spatial(scenarios_data: list, gamma0_deg: float, prefix: str, fig_id: int) -> None:
    """
    Plot x vs h for all initial speeds at a fixed gamma_0, comparing DP and NLP.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(scenarios_data)))

    for idx, (v0, s_hist, c_hist, nlp_res) in enumerate(scenarios_data):
        color = colors[idx]
        label_dp  = f"DP  V₀={v0:.1f}Vs"
        label_nlp = f"NLP V₀={v0:.1f}Vs"

        # DP
        ax.plot(s_hist["x"], s_hist["h"],
                color=color, lw=2.0, ls="-", label=label_dp, zorder=3)

        # Airplane markers along DP track
        n_marks = 8
        step = max(1, len(s_hist["x"]) // n_marks)
        for i in range(0, len(s_hist["x"]) - 1, step):
            dx = s_hist["x"][i + 1] - s_hist["x"][i]
            dh = s_hist["h"][i + 1] - s_hist["h"][i]
            angle = np.rad2deg(np.arctan2(dh, dx))
            ax.annotate("✈", xy=(s_hist["x"][i], s_hist["h"][i]),
                        fontsize=12, ha="center", va="center",
                        rotation=angle, color=color, zorder=5)

        # NLP
        if "x" in nlp_res and "h" in nlp_res:
            gamma_arr = nlp_res.get("gamma", np.array([-1.0]))
            cut = len(gamma_arr)
            recovered = np.where(gamma_arr >= 0.0)[0]
            if len(recovered):
                cut = recovered[0] + 1
            ax.plot(nlp_res["x"][:cut], nlp_res["h"][:cut],
                    color=color, lw=1.5, ls="--", label=label_nlp, zorder=4)

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Horizontal distance (m)", fontsize=13)
    ax.set_ylabel("Altitude change (m)", fontsize=13)
    ax.set_title(
        f"DP Policy vs DP-Guided NLP — γ₀ = {gamma0_deg:.0f}°\n"
        f"Solid: DP Policy Iteration  |  Dashed: CasADi IPOPT",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, ls="-", color="lightgray", lw=0.7)
    fig.tight_layout()

    out = RESULTS_DIR / f"{prefix}_validation_spatial_Fig{fig_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[+] Spatial validation saved → {out}")


# -----------------------------------------------------------------------
# Time-domain detail plot for a single scenario
# -----------------------------------------------------------------------

def _plot_time_detail(s_hist, c_hist, nlp_res, v0_norm, gamma0_deg, prefix: str) -> None:
    """
    Four-panel time plot comparing DP and NLP for one initial condition.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    t_dp = s_hist["t"]
    t_ctrl = np.linspace(0, t_dp[-1], len(c_hist["c_lift"]))

    has_nlp = "t" in nlp_res and len(nlp_res.get("gamma", [])) > 1

    # γ(t)
    axes[0, 0].plot(t_dp, np.rad2deg(s_hist["gamma"]), label="DP", color="darkred", lw=2)
    if has_nlp:
        axes[0, 0].plot(nlp_res["t"], nlp_res["gamma"], label="NLP", color="cyan", lw=1.5, ls="--")
    axes[0, 0].axhline(0, color="k", lw=0.7, ls=":")
    axes[0, 0].set_ylabel("γ [°]")
    axes[0, 0].set_title("Flight-path angle")
    axes[0, 0].legend(fontsize=9)

    # V/Vs(t)
    axes[0, 1].plot(t_dp, s_hist["v_norm"], label="DP", color="darkred", lw=2)
    if has_nlp:
        axes[0, 1].plot(nlp_res["t"], nlp_res["v_norm"], label="NLP", color="cyan", lw=1.5, ls="--")
    axes[0, 1].axhline(1.0, color="k", lw=0.7, ls=":", label="Vs")
    axes[0, 1].set_ylabel("V/Vs [-]")
    axes[0, 1].set_title("Normalised airspeed")
    axes[0, 1].legend(fontsize=9)

    # C_L(t)
    axes[1, 0].step(t_ctrl, c_hist["c_lift"], where="post", label="DP", color="darkred", lw=2)
    if has_nlp and "c_lift" in nlp_res:
        t_nlp_ctrl = nlp_res["t"][:-1]
        axes[1, 0].step(t_nlp_ctrl, nlp_res["c_lift"], where="post",
                        label="NLP", color="cyan", lw=1.5, ls="--")
    axes[1, 0].set_ylabel("$C_L$ [-]")
    axes[1, 0].set_title("Lift coefficient")
    axes[1, 0].legend(fontsize=9)

    # δ_throttle(t)
    axes[1, 1].step(t_ctrl, c_hist["delta_throttle"], where="post", label="DP", color="darkred", lw=2)
    if has_nlp and "delta_throttle" in nlp_res:
        axes[1, 1].step(nlp_res["t"][:-1], nlp_res["delta_throttle"], where="post",
                        label="NLP", color="cyan", lw=1.5, ls="--")
    axes[1, 1].set_ylabel("$\\delta_t$ [-]")
    axes[1, 1].set_title("Throttle")
    axes[1, 1].legend(fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(True, ls="-", color="lightgray", lw=0.7)

    fig.suptitle(
        f"DP vs NLP — γ₀ = {gamma0_deg:.0f}°,  V₀/Vs = {v0_norm:.2f}",
        fontsize=13,
    )
    fig.tight_layout()

    tag = f"g{abs(int(gamma0_deg)):02d}_v{v0_norm:.1f}".replace(".", "p")
    out = RESULTS_DIR / f"{prefix}_validation_detail_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[+] Detail plot saved → {out}")


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def validate_with_casadi(pi: PolicyIteration, prefix: str) -> None:
    """
    Run DP + NLP validation for the 2-DOF symmetric pullout.

    Scenarios:
        - Two gamma_0 values: -30° and -60°
        - Three initial speeds per scenario: 1.0, 1.4, 2.0 Vs
        - One detail plot for (gamma_0=-60°, v0=1.4)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        nlp = CasadiSymmetricPulloutOptimizer()
    except Exception as e:
        logger.error(f"CasADi not available: {e}")
        return

    scenarios_config = [
        {"gamma0_deg": -30.0, "fig_id": 1},
        {"gamma0_deg": -60.0, "fig_id": 2},
    ]
    v0_list = [1.0, 1.4, 2.0]
    n_nodes = 150

    # Detail plot for this scenario
    detail_scenario = {"gamma0_deg": -60.0, "v0_norm": 1.4}

    for sc in scenarios_config:
        gamma0_deg = sc["gamma0_deg"]
        scenarios_data = []

        for v0 in v0_list:
            logger.info(f"[*] γ₀={gamma0_deg:.0f}°, V₀/Vs={v0:.1f}")

            s_hist, c_hist = _simulate_dp_trajectory(pi, v0, gamma0_deg)
            dp_seed        = _build_dp_seed(s_hist, c_hist, n_nodes)
            nlp_res        = nlp.solve_trajectory(
                v0_norm=v0, gamma0_deg=gamma0_deg,
                n_nodes=n_nodes, dp_seed=dp_seed,
            )
            scenarios_data.append((v0, s_hist, c_hist, nlp_res))

            # Detail plot for the reference scenario
            if (abs(gamma0_deg - detail_scenario["gamma0_deg"]) < 1e-3
                    and abs(v0 - detail_scenario["v0_norm"]) < 1e-3):
                _plot_time_detail(s_hist, c_hist, nlp_res, v0, gamma0_deg, prefix)

        _plot_spatial(scenarios_data, gamma0_deg, prefix, sc["fig_id"])
