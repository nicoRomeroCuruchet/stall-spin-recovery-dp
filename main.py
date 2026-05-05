"""
main.py
-------
Entry point for the Banked Glider Policy Iteration pipeline.

Usage:
    python main.py --level 1               # paper grid, load policy if exists
    python main.py --level 1 --retrain     # force retrain even if policy exists
    python main.py --level 2 --no-plots    # train + profile only, no plots
    python main.py --level 3               # high fidelity
    python main.py --level 4               # maximum resolution (24 GB VRAM)

All outputs (policy .npz, plots, profiling reports) are saved under results/.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path


import gymnasium as gym
import numpy as np

from solver.policy_iteration import PolicyIteration, PolicyIterationConfig
from analysis.experiments import get_setup_for_level, run_profiling
from analysis.plotting import (
    plot_all_paper_style_policies,
    plot_value_function_contours,
    validate_trajectories_with_casadi,
    plot_final_mu_heatmap,
    plot_final_mu_heatmap_at_v,
    plot_final_gamma_heatmap_at_v,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# =====================================================================
# CORE LOGIC
# =====================================================================

def train_or_load_policy(
    env: gym.Env,
    states: np.ndarray,
    actions: np.ndarray,
    config: PolicyIterationConfig,
    prefix: str,
    retrain: bool = False,
) -> tuple[PolicyIteration, bool]:
    """
    Load an existing policy from results/ or train a new one.

    Returns:
        (pi, trained) — trained=True if a new policy was trained,
                        trained=False if an existing policy was loaded.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    policy_filename = f"{env.unwrapped.__class__.__name__}_policy.npz"
    policy_path = RESULTS_DIR / policy_filename

    if not retrain and policy_path.exists():
        logger.info(f"[+] Loading existing policy from {policy_path} (use --retrain to force retrain)")
        try:
            pi = PolicyIteration.load(policy_path, env=env)
            pi.states_space = states
            logger.info("[+] Policy loaded successfully.")
            return pi, False
        except Exception as e:
            logger.error(f"[-] Failed to load policy: {e}. Retraining...")

    if retrain and policy_path.exists():
        logger.info(f"[!] --retrain flag set. Ignoring existing policy at {policy_path}")

    logger.info(f"[*] Training new policy for {prefix}...")
    pi = PolicyIteration(env, states, actions, config)
    pi.run()  # saves to cwd by default

    # Move the auto-saved file from cwd into results/
    cwd_policy = Path.cwd() / policy_filename
    if cwd_policy.exists():
        shutil.move(str(cwd_policy), str(policy_path))
        logger.info(f"[+] Policy saved to {policy_path.resolve()}")

    return pi, True


# =====================================================================
# PIPELINE
# =====================================================================

def run_pipeline(level: int, plots: bool = True, retrain: bool = False) -> None:
    """
    Full pipeline: setup → train/load → profile → plots.

    Args:
        level:   Grid resolution level 1–4.
        plots:   If False, skip all matplotlib output.
        retrain: If True, ignore cached policy and train from scratch.
    """
    setup_func = get_setup_for_level(level)
    prefix = f"banked_pullout_L{level}"

    logger.info(f"{'=' * 55}")
    logger.info(f"  Banked Glider — Grid Level {level}")
    logger.info(f"{'=' * 55}")

    env, states, actions, config = setup_func()
    pi, trained = train_or_load_policy(env, states, actions, config, prefix, retrain=retrain)

    # GPU profiling only when a new policy was trained
    if trained:
        logger.info(f"[*] Running GPU profiling for level {level}...")
        run_profiling(setup_func)
    else:
        logger.info("[*] Skipping profiling (policy was loaded from cache, use --retrain to re-profile)")

    # Plots
    if plots and pi.n_dims == 3:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_all_paper_style_policies(pi, prefix)
        plot_value_function_contours(pi, prefix)
        validate_trajectories_with_casadi(pi, prefix)
        # Final-mu heatmaps: where does the policy leave the bank angle
        # for each starting point?
        # Sweep over (gamma_0, V_0) at fixed mu_0:
        plot_final_mu_heatmap(pi, prefix, mu_0_deg=30.0)
        plot_final_mu_heatmap(pi, prefix, mu_0_deg=60.0)
        # Sweep over (gamma_0, mu_0) at fixed V_0:
        plot_final_mu_heatmap_at_v(pi, prefix, v_slice=1.2)
        plot_final_mu_heatmap_at_v(pi, prefix, v_slice=4.0)
        # Sweep over (gamma_0, mu_0) at fixed V_0, colour = gamma_final:
        plot_final_gamma_heatmap_at_v(pi, prefix, v_slice=1.2)
        plot_final_gamma_heatmap_at_v(pi, prefix, v_slice=4.0)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banked Glider Policy Iteration")
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4], default=1,
        help=(
            "Grid discretization level: "
            "1=paper Bunge 2018 Table 1 (~53k states), "
            "2=2x refinement (~409k), "
            "3=high fidelity (~4M), "
            "4=maximum (~32M, needs 24GB VRAM)"
        )
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if a saved policy exists in results/"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip all plots (training + profiling only)"
    )
    args = parser.parse_args()

    run_pipeline(level=args.level, plots=not args.no_plots, retrain=args.retrain)
