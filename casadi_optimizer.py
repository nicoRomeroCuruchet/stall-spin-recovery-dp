"""
casadi_optimizer.py
-------------------
Nonlinear Trajectory Optimizer for the 2-DOF Symmetric Pullout with thrust.

Uses CasADi + IPOPT with Multiple Shooting to solve the continuous-time
Free Final Time Optimal Control Problem:

    min   -h(T_f)
    s.t.  EOM with thrust,  γ(T_f) = 0,  CL ∈ [-0.5, 1.0],  δ_t ∈ [0, 1]

A DP-guided warm start can be injected to steer IPOPT toward the global optimum.
"""

import logging
from typing import Dict, Optional

import casadi as ca
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CasadiSymmetricPulloutOptimizer:
    """
    Free-final-time OCP for the 2-DOF symmetric pullout with thrust.

    States  : (v_norm, γ, h)
    Controls: (C_L, δ_throttle)
    """

    def __init__(self) -> None:
        # Physical constants — identical to grumman.py
        self.MASS              = 697.18
        self.WING_SURFACE_AREA = 9.1147
        self.AIR_DENSITY       = 1.225
        self.GRAVITY           = 9.81

        self.CL_0      = 0.41
        self.CL_ALPHA  = 4.6983
        self.CD_0      = 0.0525
        self.CD_ALPHA  = 0.2068
        self.CD_ALPHA2 = 1.8712

        self.ALPHA_STALL   = np.deg2rad(15)
        self.CL_REF        = self.CL_0 + self.CL_ALPHA * self.ALPHA_STALL
        self.STALL_AIRSPEED = np.sqrt(
            (self.MASS * self.GRAVITY) /
            (0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * self.CL_REF)
        )

        # Throttle mapping: full throttle = drag at Vmax = 2*Vs
        v_max    = 2.0 * self.STALL_AIRSPEED
        cl_cruise = (2.0 * self.MASS * self.GRAVITY) / (
            self.AIR_DENSITY * self.WING_SURFACE_AREA * v_max ** 2
        )
        alpha_cruise = (cl_cruise - self.CL_0) / self.CL_ALPHA
        cd_cruise    = (self.CD_0 + self.CD_ALPHA * alpha_cruise
                        + self.CD_ALPHA2 * alpha_cruise ** 2)
        self.THROTTLE_MAPPING = (
            0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * v_max ** 2 * cd_cruise
        )

        # Control bounds
        self.CL_MIN       = -0.5
        self.CL_MAX       =  1.0
        self.THROTTLE_MIN =  0.0
        self.THROTTLE_MAX =  1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_trajectory(
        self,
        v0_norm:   float,
        gamma0_deg: float,
        n_nodes:   int = 150,
        dp_seed:   Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Solve the OCP from (v0_norm, gamma0_deg).

        Args:
            v0_norm:    Initial normalised airspeed V/Vs.
            gamma0_deg: Initial flight-path angle [deg].
            n_nodes:    Number of multiple-shooting intervals.
            dp_seed:    Optional warm-start dict from DP simulation.

        Returns:
            Dict with keys: t, v_norm, gamma, h, x, c_lift, delta_throttle.
        """
        opti = ca.Opti()

        # ── Decision variables ─────────────────────────────────────────
        T_f  = opti.variable()                        # free final time

        X    = opti.variable(3, n_nodes + 1)
        v_n  = X[0, :]
        gam  = X[1, :]
        h    = X[2, :]

        U       = opti.variable(2, n_nodes)
        c_lift  = U[0, :]
        delta_t = U[1, :]

        # ── Objective ─────────────────────────────────────────────────
        # Maximise final altitude (= minimise altitude loss)
        opti.minimize(-h[-1])

        # ── Dynamics ──────────────────────────────────────────────────
        f = self._dynamics_function()
        dt = T_f / n_nodes
        for k in range(n_nodes):
            xk  = X[:, k]
            uk  = U[:, k]
            k1  = f(xk,              uk)
            k2  = f(xk + dt/2 * k1, uk)
            k3  = f(xk + dt/2 * k2, uk)
            k4  = f(xk + dt   * k3, uk)
            opti.subject_to(X[:, k + 1] == xk + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4))

        # ── Boundary conditions ────────────────────────────────────────
        opti.subject_to(v_n[0]  == v0_norm)
        opti.subject_to(gam[0]  == np.deg2rad(gamma0_deg))
        opti.subject_to(h[0]    == 0.0)
        opti.subject_to(gam[-1] == 0.0)           # terminal: level flight

        # ── Path constraints ──────────────────────────────────────────
        opti.subject_to(opti.bounded(self.CL_MIN,       c_lift,  self.CL_MAX))
        opti.subject_to(opti.bounded(self.THROTTLE_MIN, delta_t, self.THROTTLE_MAX))
        opti.subject_to(opti.bounded(0.5, T_f, 60.0))
        opti.subject_to(v_n >= 0.1)

        # ── Warm start ────────────────────────────────────────────────
        if dp_seed is not None:
            logger.info("    [+] Injecting DP global optimum as warm start …")
            opti.set_initial(T_f,    dp_seed["T"])
            opti.set_initial(v_n,    dp_seed["v_norm"])
            opti.set_initial(gam,    dp_seed["gamma"])
            opti.set_initial(h,      dp_seed["h"])
            opti.set_initial(c_lift, dp_seed["c_lift"])
            opti.set_initial(delta_t,dp_seed["delta_throttle"])
        else:
            opti.set_initial(T_f,    10.0)
            opti.set_initial(v_n,    np.linspace(v0_norm, v0_norm * 1.3, n_nodes + 1))
            opti.set_initial(gam,    np.linspace(np.deg2rad(gamma0_deg), 0.0, n_nodes + 1))
            opti.set_initial(c_lift, self.CL_MAX)
            opti.set_initial(delta_t, 0.5)

        # ── Solver ────────────────────────────────────────────────────
        opti.solver("ipopt", {"expand": True, "print_time": False}, {
            "max_iter": 5000,
            "print_level": 0,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 15,
        })

        logger.info(
            f"Solving OCP: γ₀={gamma0_deg:.0f}°, V₀/Vs={v0_norm:.2f} …"
        )
        try:
            sol = opti.solve()
            logger.info("    [+] Optimal trajectory found.")

            opt_T   = float(sol.value(T_f))
            v_val   = sol.value(v_n)
            gam_val = sol.value(gam)
            h_val   = sol.value(h)
            cl_val  = sol.value(c_lift)
            dt_val  = sol.value(delta_t)

            # Reconstruct horizontal distance (symmetric → straight track)
            t_nodes = np.linspace(0, opt_T, n_nodes + 1)
            dt_step = opt_T / n_nodes
            x_val   = np.zeros(n_nodes + 1)
            for i in range(n_nodes):
                v_true  = v_val[i] * self.STALL_AIRSPEED
                x_val[i + 1] = x_val[i] + v_true * np.cos(gam_val[i]) * dt_step

            return {
                "t":               t_nodes,
                "v_norm":          v_val,
                "gamma":           np.rad2deg(gam_val),
                "h":               h_val,
                "x":               x_val,
                "c_lift":          cl_val,
                "delta_throttle":  dt_val,
            }

        except Exception as e:
            logger.error(f"    [-] Solver failed: {e}")
            return {
                "t":    np.linspace(0, float(opti.debug.value(T_f)), n_nodes + 1),
                "h":    opti.debug.value(h),
                "x":    np.zeros(n_nodes + 1),
            }

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _dynamics_function(self) -> ca.Function:
        """Compiled CasADi function for the 2-DOF symmetric EOM with thrust."""
        v_n    = ca.MX.sym("v_n")
        gamma  = ca.MX.sym("gamma")
        h      = ca.MX.sym("h")
        state  = ca.vertcat(v_n, gamma, h)

        cl      = ca.MX.sym("c_lift")
        delta_t = ca.MX.sym("delta_throttle")
        ctrl    = ca.vertcat(cl, delta_t)

        alpha = (cl - self.CL_0) / self.CL_ALPHA
        cd    = self.CD_0 + self.CD_ALPHA * alpha + self.CD_ALPHA2 * alpha ** 2

        v_true = v_n * self.STALL_AIRSPEED
        dyn    = 0.5 * self.AIR_DENSITY * (self.WING_SURFACE_AREA / self.MASS)
        thrust = self.THROTTLE_MAPPING * delta_t / self.MASS   # [m/s²]

        v_n_dot  = (
            -self.GRAVITY * ca.sin(gamma)
            - dyn * v_true ** 2 * cd
            + thrust
        ) / self.STALL_AIRSPEED

        v_safe   = ca.fmax(v_true, 0.1)
        gam_dot  = dyn * v_true * cl - (self.GRAVITY / v_safe) * ca.cos(gamma)
        h_dot    = v_true * ca.sin(gamma)

        rhs = ca.vertcat(v_n_dot, gam_dot, h_dot)
        return ca.Function("dynamics", [state, ctrl], [rhs])
