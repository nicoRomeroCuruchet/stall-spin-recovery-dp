import logging
from typing import Dict

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

# Configure professional logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CasadiPulloutOptimizer:
    """
    Nonlinear Trajectory Optimizer for aircraft pullout maneuvers.
    Uses CasADi and IPOPT with Multiple Shooting to solve the continuous-time
    Free Final Time Optimal Control Problem.
    """

    def __init__(self) -> None:
        # 1. Aerodynamic & Physical Constants
        self.MASS = 697.18
        self.WING_SURFACE_AREA = 9.1147
        self.AIR_DENSITY = 1.225
        self.GRAVITY = 9.81
        
        self.CL_0 = 0.41
        self.CL_ALPHA = 4.6983
        self.CD_0 = 0.0525
        self.CD_ALPHA = 0.2068
        self.CD_ALPHA2 = 1.8712
        
        self.ALPHA_STALL = np.deg2rad(15)
        self.CL_REF = self.CL_0 + self.CL_ALPHA * self.ALPHA_STALL
        self.STALL_AIRSPEED = np.sqrt(
            (self.MASS * self.GRAVITY) / 
            (0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * self.CL_REF)
        )
        
        # 2. Control constraints
        self.CL_MIN = -0.5
        self.CL_MAX = 1.0
        self.MU_DOT_MAX = np.deg2rad(30.0)

    def solve_trajectory(
        self, 
        v0_norm: float, 
        gamma0_deg: float, 
        mu0_deg: float, 
        n_nodes: int = 150,
        dp_seed: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        opti = ca.Opti()

        # ---------------------------------------------------------------------
        # 1. Optimization Variables (REDUCED MODEL)
        # Kinematic states (x, xi) removed to perfectly decouple the singularity
        # ---------------------------------------------------------------------
        T = opti.variable() 
        
        X = opti.variable(4, n_nodes + 1)
        v_norm = X[0, :]
        gamma  = X[1, :]
        mu     = X[2, :]
        h      = X[3, :]
        
        U = opti.variable(2, n_nodes)
        c_lift = U[0, :]
        mu_dot = U[1, :]

        # ---------------------------------------------------------------------
        # 2. Objective Function
        # ---------------------------------------------------------------------
        cost = 0.0
        dt = T / n_nodes
        for k in range(n_nodes):
            cost += 0.01 * (mu_dot[k] ** 2) * dt
        opti.minimize(-h[-1] + cost)

        # ---------------------------------------------------------------------
        # 3. Dynamic Constraints
        # ---------------------------------------------------------------------
        f_dynamics = self._build_dynamics_function()
        for k in range(n_nodes):
            state_k = X[:, k]
            ctrl_k = U[:, k]
            k1 = f_dynamics(state_k, ctrl_k)
            k2 = f_dynamics(state_k + (dt / 2.0) * k1, ctrl_k)
            k3 = f_dynamics(state_k + (dt / 2.0) * k2, ctrl_k)
            k4 = f_dynamics(state_k + dt * k3, ctrl_k)
            x_next = state_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)

        # ---------------------------------------------------------------------
        # 4. Boundary Conditions & Constraints
        # ---------------------------------------------------------------------
        opti.subject_to(v_norm[0] == v0_norm)
        opti.subject_to(gamma[0] == np.deg2rad(gamma0_deg))
        opti.subject_to(mu[0] == np.deg2rad(mu0_deg))
        opti.subject_to(h[0] == 0.0)
        
        opti.subject_to(gamma[-1] == 0.0)
        opti.subject_to(opti.bounded(self.CL_MIN, c_lift, self.CL_MAX))
        opti.subject_to(opti.bounded(-self.MU_DOT_MAX, mu_dot, self.MU_DOT_MAX))
        opti.subject_to(opti.bounded(0.1, T, 30.0))  
        opti.subject_to(v_norm >= 0.1)  

        # ---------------------------------------------------------------------
        # 5. Initialization (DP-Guided Warm Start)
        # ---------------------------------------------------------------------
        if dp_seed is not None:
            logger.info("    [+] Injecting Dynamic Programming global optimum as seed...")
            opti.set_initial(T, dp_seed["T"])
            opti.set_initial(v_norm, dp_seed["v_norm"])
            opti.set_initial(gamma, dp_seed["gamma"])
            opti.set_initial(mu, dp_seed["mu"])
            opti.set_initial(h, dp_seed["h"])
            opti.set_initial(c_lift, dp_seed["c_lift"])
            opti.set_initial(mu_dot, dp_seed["mu_dot"])
        else:
            opti.set_initial(T, 8.0)
            opti.set_initial(v_norm, np.linspace(v0_norm, v0_norm * 1.5, n_nodes + 1))  
            opti.set_initial(gamma, np.linspace(np.deg2rad(gamma0_deg), 0.0, n_nodes + 1))
            opti.set_initial(mu, np.linspace(np.deg2rad(mu0_deg), 0.0, n_nodes + 1))
            opti.set_initial(c_lift, self.CL_MAX)

        # ---------------------------------------------------------------------
        # 6. Solver Configuration
        # ---------------------------------------------------------------------
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 5000, 
            "print_level": 0, 
            "tol": 1e-4, 
            "acceptable_tol": 1e-2,
            "acceptable_iter": 15
        }
        opti.solver("ipopt", p_opts, s_opts)

        logger.info(f"Solving OCP for mu0 = {mu0_deg} deg...")
        try:
            sol = opti.solve()
            logger.info("    [+] Optimal trajectory found.")
            
            # =================================================================
            # 7. POST-OPTIMIZATION KINEMATIC RECONSTRUCTION
            # Forward integrate the decoupled states (x, xi) safely in Python
            # without triggering CasADi gradient singularities.
            # =================================================================
            opt_T = sol.value(T)
            dt_val = opt_T / n_nodes
            v_val = sol.value(v_norm)
            gam_val = sol.value(gamma)
            mu_val = sol.value(mu)
            cl_val = sol.value(c_lift)
            
            x_val = np.zeros(n_nodes + 1)
            xi_val = np.zeros(n_nodes + 1)
            
            for i in range(n_nodes):
                v_true = v_val[i] * self.STALL_AIRSPEED
                lift = 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * (v_true ** 2) * cl_val[i]
                
                # Protect division by zero purely algebraically (no NLP gradients affected)
                cos_g = max(abs(np.cos(gam_val[i])), 1e-4) * np.sign(np.cos(gam_val[i]))
                if cos_g == 0: cos_g = 1e-4
                
                xi_dot = (lift * np.sin(mu_val[i])) / (self.MASS * v_true * cos_g)
                x_dot = v_true * np.cos(gam_val[i]) * np.cos(xi_val[i])
                
                xi_val[i+1] = xi_val[i] + xi_dot * dt_val
                x_val[i+1] = x_val[i] + x_dot * dt_val

            return {
                "t": np.linspace(0, opt_T, n_nodes + 1),
                "v_norm": v_val,
                "gamma": np.rad2deg(gam_val),
                "mu": np.rad2deg(mu_val),
                "h": sol.value(h),
                "x": x_val,  # Reconstructed X successfully
                "c_lift": cl_val,
                "mu_dot": np.rad2deg(sol.value(mu_dot)),
            }
        except Exception as e:
            logger.error(f"    [-] Solver failed: {e}")
            return {
                "t": np.linspace(0, opti.debug.value(T), n_nodes + 1),
                "h": opti.debug.value(h),
                "x": np.zeros(n_nodes + 1),
            }

    def _build_dynamics_function(self) -> ca.Function:
        """
        Creates a compiled CasADi function representing the continuous-time dynamics.
        Reduced strictly to aerodynamic states to prevent singular Hessian inversion.
        """
        v_n = ca.MX.sym("v_n")
        gam = ca.MX.sym("gamma")
        mu  = ca.MX.sym("mu")
        h   = ca.MX.sym("h")
        state = ca.vertcat(v_n, gam, mu, h)
        
        cl  = ca.MX.sym("c_lift")
        mu_d = ca.MX.sym("mu_dot")
        ctrl = ca.vertcat(cl, mu_d)

        alpha = (cl - self.CL_0) / self.CL_ALPHA
        cd = self.CD_0 + self.CD_ALPHA * alpha + self.CD_ALPHA2 * (alpha ** 2)
        
        v_true = v_n * self.STALL_AIRSPEED
        dyn_pressure = 0.5 * self.AIR_DENSITY * (self.WING_SURFACE_AREA / self.MASS)
        
        v_n_dot = (-self.GRAVITY * ca.sin(gam) - dyn_pressure * (v_true ** 2) * cd) / self.STALL_AIRSPEED
        gam_dot = dyn_pressure * v_true * cl * ca.cos(mu) - (self.GRAVITY / v_true) * ca.cos(gam)
        h_dot = v_true * ca.sin(gam)
        
        rhs = ca.vertcat(v_n_dot, gam_dot, mu_d, h_dot)
        
        return ca.Function("dynamics", [state, ctrl], [rhs])


def plot_casadi_validation() -> None:
    pass

if __name__ == "__main__":
    pass