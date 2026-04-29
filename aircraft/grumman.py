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

        # Physical model
        self.MASS = 715.21  # Mass (m) [kg] — Riley Table I: 1577 lb × 0.453592
        self.WING_SURFACE_AREA = 9.1147  # Wing surface area (S) [m2]
        self.CHORD = 1.22  # Chord (c) [m]
        self.WING_SPAN = 8.066  # Wing Span (b) [m] — Riley Table I: 26.46 ft × 0.3048
        self.I_XX = 808.06   # Inertia [Kg.m^2]
        self.I_YY = 1000.60  # Inertia [Kg.m^2] — Riley Table I: 738 slug·ft² × 1.35582
        self.I_ZZ = 1719.18  # Inertia [Kg.m^2] — Riley Table I: 1268 slug·ft² × 1.35582

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
