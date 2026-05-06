import numpy as np


class Grumman:
    # ##################################
    # ## Grumman American AA-1 Yankee ##
    # ##################################
    """Base class for airplane parameters."""

    def __init__(self):
        # ####################
        # ## Sim parameters ##
        # ####################
        self.TIME_STEP = 0.01
        self.GRAVITY = 9.81
        self.AIR_DENSITY = 1.225  # Density (ρ) [kg/m3]

        # #########################
        # ## Airplane parameters ##
        # #########################
        # Aerodynamic model: CL coefficients (linear approximation, kept for reference)
        self.CL_0 = 0.41
        self.CL_ALPHA = 4.6983
        self.CL_ELEVATOR = 0.361
        self.CL_QHAT = 2.42

        # Riley (1985) Table III(a) — CL_o and CL_q vs alpha, CT=0 (power-off)
        _alpha_deg = np.array(
            [-10., -5., 0., 5., 10., 12., 14., 16., 18., 20., 25., 30., 35., 40.])
        self._CL_O_ALPHA_RAD = np.deg2rad(_alpha_deg)
        # CL_o: linear up to ~12°, flat-top 14°–18°, gradual post-stall drop
        self._CL_O_TABLE = np.array(
            [-0.41, -0.01, 0.41, 0.84, 1.16, 1.23,
             1.26, 1.26, 1.26, 1.25, 1.22, 1.17, 1.13, 1.08])
        # CL_q: pitch-rate damping — nearly 2x larger at stall angles than cruise
        self._CL_Q_TABLE = np.array(
            [2.41, 2.41, 2.42, 2.46, 2.59, 2.96,
             3.72, 4.73, 5.29, 5.16, 5.05, 5.06, 5.08, 5.08])

        # CD_o nonlinear table — Riley (1985) Table III(b), CT=0 (power-off)
        # Same 14 breakpoints as CL_o (-10 deg to 40 deg)
        self._CD_O_TABLE = np.array([
            0.0666, 0.0486, 0.0526, 0.0846, 0.1456,
            0.1856, 0.2446, 0.3136, 0.3786, 0.4486,
            0.6186, 0.7786, 0.9255, 1.0636])

        # Aerodynamic model: Cm coefficients
        # CM_o nonlinear table — Riley (1985) Table III(c), CT=0 (power-off)
        # Same 14 breakpoints as CL_o (-10 deg to 40 deg)
        self._CM_O_TABLE = np.array([0.2700, 0.1580, 0.0760, 0.0020, -0.0800,
                                     -0.1180, -0.1670, -0.2250, -0.2770, -0.3160,
                                     -0.4080, -0.4800, -0.5560, -0.6060])
        self.CM_ELEVATOR = -1.0313
        # CM_q nonlinear table — Riley (1985) Table III(c), CT=0 (power-off)
        # Same 14 breakpoints as CL_o (-10 deg to 40 deg)
        self._CM_Q_TABLE = np.array([-7.0000, -7.0000, -7.0400, -7.1500, -7.5200,
                                     -8.6200, -10.8000, -13.7300, -15.3800, -15.0000,
                                     -14.6600, -14.7100, -14.7700, -14.7700])

        # CT breakpoints
        self._CT_BREAKPOINTS = np.array([0.0, 0.5], dtype=np.float32)

        # CT=0.5 tables (Riley 1985, Table III — power-on, thrust included)
        self._CL_O_TABLE_CT05 = np.array(
            [-0.67, -0.14, 0.41, 0.97, 1.42, 1.54,
             1.62, 1.67, 1.72, 1.76, 1.85, 1.92, 1.99, 2.05],
            dtype=np.float32)
        self._CL_Q_TABLE_CT05 = np.array(
            [3.012, 3.012, 3.029, 3.222, 3.594, 4.351,
             6.072, 6.382, 6.988, 6.833, 6.561, 6.127, 5.966, 5.811],
            dtype=np.float32)
        self._CD_O_TABLE_CT05 = np.array(
            [-0.3273, -0.3499, -0.3474, -0.3139, -0.2483,
             -0.2057, -0.1435, -0.0709, -0.0018,
             0.0727, 0.2561, 0.4322, 0.5979, 0.7572],
            dtype=np.float32)
        self._CM_O_TABLE_CT05 = np.array(
            [0.27, 0.158, 0.076, 0.002, -0.08, -0.118, -0.167, -0.225,
             -0.277, -0.316, -0.408, -0.48, -0.556, -0.606],
            dtype=np.float32)
        self._CM_Q_TABLE_CT05 = np.array(
            [-8.75, -8.75, -8.80, -9.36, -10.44, -12.64, -17.64, -18.54,
             -20.30, -19.85, -19.06, -17.80, -17.33, -16.88],
            dtype=np.float32)

        # CL_de and CM_de: alpha and CT dependent (Riley Table III, converted to /rad)
        self._CL_DE_TABLE_CT0 = np.array(
            [0.0062, 0.0063, 0.0062, 0.0058, 0.0053, 0.0051, 0.0050, 0.0049,
             0.0048, 0.0047, 0.0044, 0.0042, 0.0039, 0.0037],
            dtype=np.float32) * 57.2958
        self._CL_DE_TABLE_CT05 = np.array(
            [0.0139, 0.0134, 0.0131, 0.0123, 0.0109, 0.0104, 0.0101, 0.0098,
             0.0094, 0.0090, 0.0080, 0.0073, 0.0061, 0.0055],
            dtype=np.float32) * 57.2958
        self._CM_DE_TABLE_CT0 = np.array(
            [-0.0193, -0.0193, -0.0193, -0.0180, -0.0165, -0.0164, -0.0163, -0.0162,
             -0.0162, -0.0162, -0.0162, -0.0150, -0.0130, -0.0100],
            dtype=np.float32) * 57.2958
        self._CM_DE_TABLE_CT05 = np.array(
            [-0.0374, -0.0393, -0.0394, -0.0395, -0.0384, -0.0360, -0.0334, -0.0311,
             -0.0288, -0.0269, -0.0226, -0.0213, -0.0190, -0.0153],
            dtype=np.float32) * 57.2958

        # Aerodynamic model: Cl coefficients (linear approximation, kept for reference)
        self.Cl_BETA = -0.1089
        self.Cl_PHAT = -0.52
        self.Cl_RHAT = 0.19
        self.Cl_AILERON = -0.1031
        self.Cl_RUDDER = 0.0143

        # Riley (1985) Table III(f) — Rolling-moment coefficients, CT=0 and CT=0.5
        # Same 14 alpha breakpoints as CL_o (-10° to 40°)
        # Under A.i (β=0, r=0, δr=0) only Cl_o, Cl_p̂ and Cl_δa enter the dynamics;
        # Cl_β, Cl_r̂ and Cl_δr would only matter if r/β/δr were active (future A.iii).

        # Cl_o: asymmetric base rolling moment (dimensionless)
        self._CL_ROLL_O_TABLE_CT0 = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             -0.0025, -0.0050, -0.0075, -0.0075,
             -0.0075, -0.0075, -0.0075, -0.0075],
            dtype=np.float32)
        self._CL_ROLL_O_TABLE_CT05 = np.array(
            [0.0060, 0.0040, 0.0020, 0.0, 0.0, 0.0,
             -0.0025, -0.0050, -0.0075, -0.0075,
             -0.0075, -0.0095, -0.0115, -0.0135],
            dtype=np.float32)

        # Cl_p̂: roll-damping derivative — multiplies p_hat = p·b/(2V) (dimensionless)
        # At α=0, CT=0 → -0.52, matches the legacy linear constant Cl_PHAT.
        self._CL_ROLL_PHAT_TABLE_CT0 = np.array(
            [-0.5200, -0.5200, -0.5200, -0.5200, -0.4000, -0.3100,
             -0.2200, -0.1300, -0.0400,  0.0500,
              0.0000, -0.0500, -0.1000, -0.1500],
            dtype=np.float32)
        self._CL_ROLL_PHAT_TABLE_CT05 = np.array(
            [-0.5200, -0.5200, -0.5200, -0.5200, -0.4000, -0.3100,
             -0.2200, -0.1300, -0.0400,  0.0500,
              0.0000, -0.0500, -0.1000, -0.1500],
            dtype=np.float32)

        # Cl_δa: aileron control derivative (Riley table is CT-independent; per deg → per rad)
        self._CL_ROLL_DA_TABLE = np.array(
            [-0.001040, -0.001040, -0.001040, -0.001000, -0.000920, -0.000880,
             -0.000840, -0.000790, -0.000740, -0.000690,
             -0.000600, -0.000500, -0.000400, -0.000330],
            dtype=np.float32) * 57.2958

        # Cl_β: dihedral effect (per deg → per rad). Riley Table III(f).
        self._CL_ROLL_BETA_TABLE_CT0 = np.array(
            [-0.00140, -0.00115, -0.00115, -0.00190, -0.00315, -0.00365,
             -0.00400, -0.00420, -0.00435, -0.00450,
             -0.00450, -0.00420, -0.00400, -0.00390],
            dtype=np.float32) * 57.2958
        self._CL_ROLL_BETA_TABLE_CT05 = np.array(
            [-0.00273, -0.00215, -0.00182, -0.00190, -0.00239, -0.00265,
             -0.00267, -0.00237, -0.00212, -0.00183,
             -0.00217, -0.00353, -0.00467, -0.00523],
            dtype=np.float32) * 57.2958

        # Cl_δr: rudder cross-coupling on roll (per deg → per rad), CT-indep
        self._CL_ROLL_DR_TABLE = np.array(
            [0.00025, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025,
             0.00025, 0.00025, 0.00025, 0.00025,
             0.00013, 0.0, 0.0, 0.0],
            dtype=np.float32) * 57.2958

        # Cl_r̂: yaw-rate cross effect on roll (per rad)
        self._CL_ROLL_RHAT_TABLE_CT0 = np.array(
            [0.1000, 0.1300, 0.1600, 0.1900, 0.1400, 0.1300,
             0.1200, 0.1100, 0.1000, 0.0900,
             0.0700, 0.0700, 0.0700, 0.1000],
            dtype=np.float32)
        self._CL_ROLL_RHAT_TABLE_CT05 = np.array(
            [0.1150, 0.1450, 0.1750, 0.2050, 0.1540, 0.1450,
             0.1290, 0.1140, 0.1040, 0.0950,
             0.0760, 0.0750, 0.0740, 0.1040],
            dtype=np.float32)

        # =====================================================================
        # Riley (1985) Table III(d) — Side-force coefficients (Cy)
        # Used in the 8-DOF model when β ≠ 0 (full lateral dynamics).
        # =====================================================================

        # Cy_o: asymmetric base side-force (dimensionless)
        self._CY_O_TABLE_CT0 = np.zeros(14, dtype=np.float32)
        self._CY_O_TABLE_CT05 = np.array(
            [0.0810, 0.0540, 0.0270, 0.0, -0.0270, -0.0378,
             -0.0486, -0.0540, -0.0540, -0.0540,
             -0.0540, -0.0540, -0.0540, -0.0540],
            dtype=np.float32)

        # Cy_β: side-force per sideslip (per deg → per rad)
        self._CY_BETA_TABLE_CT0 = np.array(
            [-0.01300, -0.01250, -0.01180, -0.01100, -0.01090, -0.01080,
             -0.00980, -0.00880, -0.00820, -0.00780,
             -0.00670, -0.00600, -0.00620, -0.00680],
            dtype=np.float32) * 57.2958
        self._CY_BETA_TABLE_CT05 = np.array(
            [-0.02260, -0.02260, -0.02260, -0.02260, -0.02260, -0.02260,
             -0.02210, -0.02130, -0.02100, -0.02080,
             -0.02030, -0.02020, -0.02100, -0.02220],
            dtype=np.float32) * 57.2958

        # Cy_δa: aileron cross effect on side-force (per deg → per rad), CT-indep
        self._CY_DA_TABLE = np.array(
            [-0.000100, -0.000080, -0.000090, -0.000100, -0.000140, -0.000150,
             -0.000160, -0.000130, -0.000110, -0.000100,
             -0.000080, -0.000100, 0.0, 0.0],
            dtype=np.float32) * 57.2958

        # Cy_δr: rudder side-force (per deg → per rad), CT-indep
        self._CY_DR_TABLE = np.array(
            [-0.0140, -0.0040, 0.0060, 0.0160, 0.0260, 0.0300,
             0.0340, 0.0380, 0.0420, 0.0460,
             0.0560, 0.0660, 0.0330, 0.0],
            dtype=np.float32) * 57.2958

        # Cy_p̂: roll-rate cross effect on side-force (per rad)
        self._CY_PHAT_TABLE_CT0 = np.array(
            [0.00244, 0.00263, 0.00282, 0.00295, 0.00307, 0.00295,
             0.00282, 0.00267, 0.00255, 0.00242,
             0.00189, 0.00137, 0.00093, 0.00053],
            dtype=np.float32)
        self._CY_PHAT_TABLE_CT05 = np.array(
            [0.00589, 0.00629, 0.00674, 0.00722, 0.00773, 0.00775,
             0.00777, 0.00777, 0.00779, 0.00779,
             0.00665, 0.00558, 0.00425, 0.00295],
            dtype=np.float32)

        # Cy_r̂: yaw-rate cross effect on side-force (per rad)
        self._CY_RHAT_TABLE_CT0 = np.array(
            [0.8000, 0.9000, 1.0000, 1.1000, 0.8000, 0.6000,
             0.4000, 0.2000, 0.0, -0.2500,
             -0.2400, -0.1200, 0.0, 0.0],
            dtype=np.float32)
        self._CY_RHAT_TABLE_CT05 = np.array(
            [1.0110, 1.1110, 1.2110, 1.3110, 1.0010, 0.8020,
             0.5290, 0.2490, 0.0560, -0.1870,
             -0.1680, -0.0450, 0.0610, 0.0520],
            dtype=np.float32)

        # =====================================================================
        # Riley (1985) Table III(e) — Yawing-moment coefficients (Cn)
        # =====================================================================

        # Cn_o: asymmetric base yawing moment (dimensionless)
        self._CN_O_TABLE_CT0 = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             -0.0010, -0.0010, -0.0010, -0.0010,
             -0.0010, -0.0010, -0.0010, -0.0010],
            dtype=np.float32)
        self._CN_O_TABLE_CT05 = np.array(
            [-0.0166, -0.0166, -0.0166, -0.0166, -0.0166, -0.0166,
             -0.0142, -0.0118, -0.0094, -0.0070,
             -0.0010, -0.0040, -0.0070, -0.0100],
            dtype=np.float32)

        # Cn_β: weathercock stability (per deg → per rad). Positive = stable.
        self._CN_BETA_TABLE_CT0 = np.array(
            [0.00250, 0.00220, 0.00192, 0.00175, 0.00142, 0.00128,
             0.00110, 0.00090, 0.00080, 0.00070,
             0.00032, -0.00002, -0.00025, -0.00038],
            dtype=np.float32) * 57.2958
        self._CN_BETA_TABLE_CT05 = np.array(
            [0.00327, 0.00304, 0.00292, 0.00287, 0.00265, 0.00256,
             0.00242, 0.00227, 0.00221, 0.00216,
             0.00190, 0.00167, 0.00156, 0.00154],
            dtype=np.float32) * 57.2958

        # Cn_δa: aileron adverse-yaw (per deg → per rad), CT-indep
        self._CN_DA_TABLE = np.array(
            [0.000090, 0.000070, 0.000050, 0.000030, 0.000010, 0.0,
             -0.000030, -0.000060, -0.000100, -0.000150,
             -0.000090, -0.000040, 0.0, 0.000030],
            dtype=np.float32) * 57.2958

        # Cn_δr: rudder yawing moment (per rad — Riley's value is large; per-rad
        # interpretation matches plausible rudder authority for AA-1 Yankee).
        self._CN_DR_TABLE_CT0 = np.array(
            [-0.2000, -0.2000, -0.2000, -0.2000, -0.2000, -0.2000,
             -0.1300, -0.0500, -0.0600, -0.0700,
             -0.1000, -0.1000, -0.1000, -0.1000],
            dtype=np.float32)
        self._CN_DR_TABLE_CT05 = np.array(
            [-0.2900, -0.2900, -0.2900, -0.2900, -0.2870, -0.2860,
             -0.1850, -0.0710, -0.0840, -0.0970,
             -0.1350, -0.1320, -0.1260, -0.1220],
            dtype=np.float32)

        # Cn_p̂: roll-rate cross effect on yaw (per rad)
        self._CN_PHAT_TABLE_CT0 = np.array(
            [-0.0300, -0.0400, -0.0500, -0.0600, -0.0700, -0.0600,
             -0.0300, 0.0, 0.0300, 0.0400,
             0.0150, 0.0150, 0.0150, 0.0150],
            dtype=np.float32)
        self._CN_PHAT_TABLE_CT05 = np.array(
            [-0.0480, -0.0580, -0.0680, -0.0780, -0.0880, -0.0770,
             -0.0470, -0.0160, 0.0140, 0.0400,
             0.0150, 0.0150, 0.0100, 0.0070],
            dtype=np.float32)

        # Cn_r̂: yaw damping (per rad). Negative = stable.
        self._CN_RHAT_TABLE_CT0 = np.array(
            [-0.00116, -0.00125, -0.00134, -0.00140, -0.00146, -0.00140,
             -0.00134, -0.00127, -0.00121, -0.00115,
             -0.00090, -0.00065, -0.00044, -0.00025],
            dtype=np.float32)
        self._CN_RHAT_TABLE_CT05 = np.array(
            [-0.00280, -0.00299, -0.00320, -0.00343, -0.00367, -0.00368,
             -0.00369, -0.00369, -0.00370, -0.00370,
             -0.00316, -0.00265, -0.00202, -0.00140],
            dtype=np.float32)

        # Physical model — Riley Table I (AA-1 Yankee)
        self.MASS = 715.21    # [kg]      — 1577 lb × 0.453592
        self.WING_SURFACE_AREA = 9.1147   # [m2]
        self.CHORD = 1.22                 # [m]
        self.WING_SPAN = 8.066            # [m]      — 26.46 ft × 0.3048
        self.I_XX = 808.06                # [kg·m^2] — 596 slug·ft²
        self.I_YY = 1000.60               # [kg·m^2] — 738 slug·ft²
        self.I_ZZ = 1719.18               # [kg·m^2] — 1268 slug·ft²
        self.I_XZ = 0.0                   # [kg·m^2] — Riley Table I (zero for AA-1)
        # Propeller gyroscopic
        self.I_P = 1.559                  # [kg·m^2] — 1.15 slug·ft² (Riley Table I)
        # Engine: 2600 rpm full = 272.27 rad/s; n(δt) = N_MAX_RAD_S · δt (linear approx)
        self.N_MAX_RAD_S = 272.27         # [rad/s]

        # Stall angle of attack (αs) [rad] — flat-top onset per Riley Table III
        self.ALPHA_STALL = np.deg2rad(14)

        # Negative stall angle of attack (αs) [rad]
        self.ALPHA_NEGATIVE_STALL = np.deg2rad(-10)

        self.CL_STALL = 1.26   # Max CL from Riley Table III (flat-top 14°–18°, CT=0)
        self.CL_REF = self.CL_STALL

        # Stall air speed (Vs) [m/s]
        self.STALL_AIRSPEED = np.sqrt(
            (self.MASS * self.GRAVITY) /
            (0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * self.CL_REF)
        )

        # Maximum air speed (Vs) [m/s]
        self.MAX_CRUISE_AIRSPEED = 2 * self.STALL_AIRSPEED

        # Throttle model
        self.THROTTLE_LINEAR_MAPPING = None
        self._initialize_throttle_model()

    def _update_state_from_derivative(self, value_to_update, value_derivative):
        value_to_update += self.TIME_STEP * value_derivative
        return value_to_update

    def _rk4_update(self, value_to_update: np.ndarray, f, dt: float) -> np.ndarray:
        """
        Perform a Runge-Kutta 4th-order integration step.

        Parameters
        ----------
        - value_to_update: np.ndarray — current state
        - f: function(value) -> np.ndarray — function returning derivative
        - dt: float — time step

        Returns
        -------
        - np.ndarray — updated state
        """
        k1 = f(value_to_update)
        k2 = f(value_to_update + 0.5 * dt * k1)
        k3 = f(value_to_update + 0.5 * dt * k2)
        k4 = f(value_to_update + dt * k3)
        return value_to_update + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _alpha_from_cl(self, c_lift):
        alpha = (c_lift - self.CL_0) / self.CL_ALPHA
        return alpha

    def _cl_from_lift_force_and_speed(self, lift_force, airspeed):
        cl = (2 * lift_force) / (
            self.AIR_DENSITY * self.WING_SURFACE_AREA * airspeed ** 2
        )
        return cl

    def _cl_from_alpha_bkp(self, alpha, elevator, q_hat):
        # TODO: review model
        if alpha <= self.ALPHA_NEGATIVE_STALL:
            c_lift = self.CL_0 + self.CL_ALPHA * self.ALPHA_NEGATIVE_STALL
        elif alpha >= self.ALPHA_STALL:
            # Stall model: Lift saturation
            c_lift = self.CL_0 + self.CL_ALPHA * self.ALPHA_STALL
            # Stall model: Lift reduction with opposite slope
            # c_lift = (
            #     -self.CL_ALPHA * alpha
            #     + self.CL_0
            #     + 2 * self.CL_ALPHA * self.ALPHA_STALL
            # )
        else:
            c_lift = (
                self.CL_0
                + self.CL_ALPHA * alpha
                + self.CL_ELEVATOR * elevator
                + self.CL_QHAT * q_hat
            )
        return c_lift

    def _compute_ct(self, throttle: float, airspeed: float) -> float:
        vt = max(float(airspeed), 0.1)
        q_bar = 0.5 * self.AIR_DENSITY * vt * vt
        thrust = self.THROTTLE_LINEAR_MAPPING * throttle
        return float(np.clip(thrust / (q_bar * self.WING_SURFACE_AREA), 0.0, 0.5))

    def _bilinear_interp(self, alpha, ct, table_ct0, table_ct05):
        t_ct = np.clip(ct / 0.5, 0.0, 1.0)
        v0 = np.interp(alpha, self._CL_O_ALPHA_RAD, table_ct0)
        v05 = np.interp(alpha, self._CL_O_ALPHA_RAD, table_ct05)
        return v0 + t_ct * (v05 - v0)

    def _cl_from_alpha(self, alpha, elevator, q_hat, ct=0.0):
        """
        Calculate lift coefficient (C_L) using bilinear table lookup for CL_o,
        based on Riley (1985) NASA-TM-86309, Table III(a/b), CT=0 and CT=0.5.

        At CT=0 (power-off): flat-top plateau from 14° to 18° (CL_max = 1.26).
        At CT=0.5 (power-on): CL_max = 1.72 at 18°, thrust effects embedded.
        Outside the alpha range np.interp clamps to boundary values.
        """
        alpha, elevator, q_hat = np.broadcast_arrays(
            np.asarray(alpha, dtype=np.float32),
            np.asarray(elevator, dtype=np.float32),
            np.asarray(q_hat, dtype=np.float32),
        )

        cl_o = self._bilinear_interp(alpha, ct, self._CL_O_TABLE, self._CL_O_TABLE_CT05)
        cl_q = self._bilinear_interp(alpha, ct, self._CL_Q_TABLE, self._CL_Q_TABLE_CT05)
        cl_de = self._bilinear_interp(
            alpha, ct, self._CL_DE_TABLE_CT0, self._CL_DE_TABLE_CT05)
        return cl_o + cl_de * elevator + cl_q * q_hat

    def _lift_force_at_speed_and_cl(self, airspeed, lift_coefficient):
        return (
            0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA
            * airspeed ** 2 * lift_coefficient
        )

    def _cd_from_alpha(self, alpha, ct=0.0):
        return self._bilinear_interp(alpha, ct, self._CD_O_TABLE, self._CD_O_TABLE_CT05)

    def _cd_from_cl(self, c_lift):
        c_drag = self._cd_from_alpha(self._alpha_from_cl(c_lift))
        return c_drag

    def _drag_force_at_speed_and_cd(self, airspeed, drag_coefficient):
        return (
            0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA
            * airspeed ** 2 * drag_coefficient
        )

    def _drag_force_at_cruise_speed(self, airspeed):
        cruise_lift_force = self.MASS * self.GRAVITY
        cruise_cl = self._cl_from_lift_force_and_speed(cruise_lift_force, airspeed)
        alpha = self._alpha_from_cl(cruise_cl)
        cruise_cd = self._cd_from_alpha(alpha)
        drag_force = self._drag_force_at_speed_and_cd(airspeed, cruise_cd)
        return drag_force

    def _rolling_moment_coefficient(self, beta, p_hat, r_hat, aileron, rudder):
        c_rolling_moment = (
            self.Cl_BETA * beta
            + self.Cl_PHAT * p_hat
            + self.Cl_RHAT * r_hat
            + self.Cl_AILERON * aileron
            + self.Cl_RUDDER * rudder
        )
        return c_rolling_moment

    def _rolling_moment_coefficient_riley(self, alpha, p_hat, aileron, ct=0.0):
        """
        Rolling moment coefficient under A.i (β=0, r=0, δr=0).

            Cl_b = Cl_o(α, CT) + Cl_p̂(α, CT) · p̂ + Cl_δa(α) · δa

        p_hat = p · b / (2V) is dimensionless; aileron is in radians.
        Tables come from Riley (1985) NASA-TM-86309 Table III(f).
        """
        alpha, p_hat, aileron = np.broadcast_arrays(
            np.asarray(alpha, dtype=np.float32),
            np.asarray(p_hat, dtype=np.float32),
            np.asarray(aileron, dtype=np.float32),
        )
        cl_o = self._bilinear_interp(
            alpha, ct, self._CL_ROLL_O_TABLE_CT0, self._CL_ROLL_O_TABLE_CT05)
        cl_p = self._bilinear_interp(
            alpha, ct, self._CL_ROLL_PHAT_TABLE_CT0, self._CL_ROLL_PHAT_TABLE_CT05)
        cl_da = np.interp(alpha, self._CL_O_ALPHA_RAD, self._CL_ROLL_DA_TABLE)
        return cl_o + cl_p * p_hat + cl_da * aileron

    def _rolling_moment_coefficient_full(
        self, alpha, beta, p_hat, r_hat, aileron, rudder, ct=0.0,
    ):
        """
        Full rolling moment coefficient (8-DOF model — Riley III(f) all terms).

            Cl_b = Cl_o + Cl_β·β + Cl_p̂·p̂ + Cl_r̂·r̂ + Cl_δa·δa + Cl_δr·δr
        """
        cl_o = self._bilinear_interp(
            alpha, ct, self._CL_ROLL_O_TABLE_CT0, self._CL_ROLL_O_TABLE_CT05)
        cl_b = self._bilinear_interp(
            alpha, ct, self._CL_ROLL_BETA_TABLE_CT0, self._CL_ROLL_BETA_TABLE_CT05)
        cl_p = self._bilinear_interp(
            alpha, ct, self._CL_ROLL_PHAT_TABLE_CT0, self._CL_ROLL_PHAT_TABLE_CT05)
        cl_r = self._bilinear_interp(
            alpha, ct, self._CL_ROLL_RHAT_TABLE_CT0, self._CL_ROLL_RHAT_TABLE_CT05)
        cl_da = np.interp(alpha, self._CL_O_ALPHA_RAD, self._CL_ROLL_DA_TABLE)
        cl_dr = np.interp(alpha, self._CL_O_ALPHA_RAD, self._CL_ROLL_DR_TABLE)
        return (cl_o + cl_b * beta + cl_p * p_hat + cl_r * r_hat
                + cl_da * aileron + cl_dr * rudder)

    def _side_force_coefficient(
        self, alpha, beta, p_hat, r_hat, aileron, rudder, ct=0.0,
    ):
        """
        Side-force coefficient (Riley Table III(d)).

            Cy = Cy_o + Cy_β·β + Cy_p̂·p̂ + Cy_r̂·r̂ + Cy_δa·δa + Cy_δr·δr
        """
        cy_o = self._bilinear_interp(
            alpha, ct, self._CY_O_TABLE_CT0, self._CY_O_TABLE_CT05)
        cy_b = self._bilinear_interp(
            alpha, ct, self._CY_BETA_TABLE_CT0, self._CY_BETA_TABLE_CT05)
        cy_p = self._bilinear_interp(
            alpha, ct, self._CY_PHAT_TABLE_CT0, self._CY_PHAT_TABLE_CT05)
        cy_r = self._bilinear_interp(
            alpha, ct, self._CY_RHAT_TABLE_CT0, self._CY_RHAT_TABLE_CT05)
        cy_da = np.interp(alpha, self._CL_O_ALPHA_RAD, self._CY_DA_TABLE)
        cy_dr = np.interp(alpha, self._CL_O_ALPHA_RAD, self._CY_DR_TABLE)
        return (cy_o + cy_b * beta + cy_p * p_hat + cy_r * r_hat
                + cy_da * aileron + cy_dr * rudder)

    def _yawing_moment_coefficient(
        self, alpha, beta, p_hat, r_hat, aileron, rudder, ct=0.0,
    ):
        """
        Yawing-moment coefficient (Riley Table III(e)).

            Cn = Cn_o + Cn_β·β + Cn_p̂·p̂ + Cn_r̂·r̂ + Cn_δa·δa + Cn_δr·δr
        """
        cn_o = self._bilinear_interp(
            alpha, ct, self._CN_O_TABLE_CT0, self._CN_O_TABLE_CT05)
        cn_b = self._bilinear_interp(
            alpha, ct, self._CN_BETA_TABLE_CT0, self._CN_BETA_TABLE_CT05)
        cn_p = self._bilinear_interp(
            alpha, ct, self._CN_PHAT_TABLE_CT0, self._CN_PHAT_TABLE_CT05)
        cn_r = self._bilinear_interp(
            alpha, ct, self._CN_RHAT_TABLE_CT0, self._CN_RHAT_TABLE_CT05)
        cn_dr = self._bilinear_interp(
            alpha, ct, self._CN_DR_TABLE_CT0, self._CN_DR_TABLE_CT05)
        cn_da = np.interp(alpha, self._CL_O_ALPHA_RAD, self._CN_DA_TABLE)
        return (cn_o + cn_b * beta + cn_p * p_hat + cn_r * r_hat
                + cn_da * aileron + cn_dr * rudder)

    def _rolling_moment_at_speed_and_cl(self, airspeed, rolling_moment_coefficient):
        return (
            0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA
            * self.WING_SPAN * airspeed ** 2 * rolling_moment_coefficient
        )

    def _pitching_moment_coefficient(self, alpha, elevator, q_hat, ct=0.0):
        cm_o = self._bilinear_interp(alpha, ct, self._CM_O_TABLE, self._CM_O_TABLE_CT05)
        cm_q = self._bilinear_interp(alpha, ct, self._CM_Q_TABLE, self._CM_Q_TABLE_CT05)
        cm_de = self._bilinear_interp(
            alpha, ct, self._CM_DE_TABLE_CT0, self._CM_DE_TABLE_CT05)
        return cm_o + cm_de * elevator + cm_q * q_hat

    def _pitching_moment_at_speed_and_cm(self, airspeed, pitching_moment_coefficient):
        return (
            0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA
            * self.CHORD * airspeed ** 2 * pitching_moment_coefficient
        )

    def _initialize_throttle_model(self):
        # Throttle model: Thrust force = Kt * δ_throttle
        # Max Thrust -> Kt * 1 = Drag(V=Vmax) -> Kt = 0.5 ρ S (Vmax)^2 CD
        # δ_throttle = 1.0 -> Max Cruise speed: V' = Vmax -> V_dot = 0 = Thrust - Drag
        self.THROTTLE_LINEAR_MAPPING = self._drag_force_at_cruise_speed(
            self.MAX_CRUISE_AIRSPEED
        )

    def _thrust_force_at_throttle(self, throttle):
        thrust_force = self.THROTTLE_LINEAR_MAPPING * throttle
        return thrust_force
