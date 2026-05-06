import numpy as np

from aircraft.extended_grumman import ExtendedGrumman


class SpinGrumman(ExtendedGrumman):
    """
    8-DOF spin-recovery dynamics in stability-axis representation.

    State:  (γ, V/Vs, α, β, μ, p, q, r)
    Action: (δe, δa, δt, δr)

    Aerodynamic model: Riley (1985) NASA TM-86309 wind-tunnel tables
    (Tables III a-c for longitudinal CL/CD/Cm, III d-f for lateral
    Cy/Cn/Cl), all bilinear-interpolated in (α, CT).

    Equations of motion follow the standard stability-axis derivation
    (Stengel "Flight Dynamics", Phillips "Mechanics of Flight"). The
    three rotational equations (ṗ, q̇, ṙ) are taken directly from
    Riley Appendix B body-axis form with I_xz = 0 (per Riley Table I)
    and the propeller gyroscopic correction `Ip·n` retained on q̇ and ṙ.

    With β = 0, r = 0, δr = 0 forced and the 6DOF longitudinal/lateral
    states pinned to the same regime, this collapses to the 6-DOF
    banked-spin model under simplification A.i.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def command_airplane(self, elevator, aileron, throttle, rudder):
        """
        RK4 integration of the 8 stability-axis states.
        """
        dt = self.TIME_STEP
        v_stall = self.STALL_AIRSPEED

        self.last_elevator = elevator
        self.last_aileron = aileron
        self.last_throttle = throttle
        self.last_rudder = rudder

        gamma = self.flight_path_angle
        vn = self.airspeed_norm
        alpha = self.alpha
        beta = self.beta
        mu = self.bank_angle
        p = self.roll_rate
        q = self.pitch_rate
        r = self.yaw_rate

        def _derivs(g, v, a, b, m, pr, qr, rr):
            return self._derivatives(
                g, v, a, b, m, pr, qr, rr,
                elevator, aileron, throttle, rudder, v_stall,
            )

        # --- RK4 over 8 states ---
        k1 = _derivs(gamma, vn, alpha, beta, mu, p, q, r)
        s2 = tuple(s + 0.5 * dt * k for s, k in zip(
            (gamma, vn, alpha, beta, mu, p, q, r), k1))
        k2 = _derivs(*s2)
        s3 = tuple(s + 0.5 * dt * k for s, k in zip(
            (gamma, vn, alpha, beta, mu, p, q, r), k2))
        k3 = _derivs(*s3)
        s4 = tuple(s + dt * k for s, k in zip(
            (gamma, vn, alpha, beta, mu, p, q, r), k3))
        k4 = _derivs(*s4)

        sixth_dt = dt / 6.0
        self.flight_path_angle += sixth_dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self.airspeed_norm     += sixth_dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.alpha             += sixth_dt * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        self.beta              += sixth_dt * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        self.bank_angle        += sixth_dt * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
        self.roll_rate         += sixth_dt * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
        self.pitch_rate        += sixth_dt * (k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
        self.yaw_rate          += sixth_dt * (k1[7] + 2*k2[7] + 2*k3[7] + k4[7])
        self.airspeed = self.airspeed_norm * v_stall

    def reset(self, flight_path_angle, airspeed_norm, alpha, beta,
              bank_angle, roll_rate, pitch_rate, yaw_rate):
        super().reset(
            flight_path_angle, airspeed_norm, alpha, beta,
            bank_angle, roll_rate, pitch_rate, yaw_rate,
        )

    # ------------------------------------------------------------------
    # Derivative computation (8 ODEs at one state-control point)
    # ------------------------------------------------------------------

    def _derivatives(
        self, gamma, vn, alpha, beta, mu, p, q, r,
        elevator, aileron, throttle, rudder, v_stall,
    ):
        """
        Return (γ̇, V̇/Vs, α̇, β̇, μ̇, ṗ, q̇, ṙ) at the supplied state and controls.
        """
        vt = max(vn * v_stall, 0.1)
        q_hat = q * self.CHORD / (2.0 * vt)
        p_hat = p * self.WING_SPAN / (2.0 * vt)
        r_hat = r * self.WING_SPAN / (2.0 * vt)

        ct = self._compute_ct(throttle, vt)

        # --- Longitudinal aero (Riley III(a-c)) ---
        cl_o = self._bilinear_interp(alpha, ct, self._CL_O_TABLE, self._CL_O_TABLE_CT05)
        cl_q = self._bilinear_interp(alpha, ct, self._CL_Q_TABLE, self._CL_Q_TABLE_CT05)
        cl_de = self._bilinear_interp(alpha, ct, self._CL_DE_TABLE_CT0, self._CL_DE_TABLE_CT05)
        cl = cl_o + cl_de * elevator + cl_q * q_hat

        cd = self._bilinear_interp(alpha, ct, self._CD_O_TABLE, self._CD_O_TABLE_CT05)

        cm_o = self._bilinear_interp(alpha, ct, self._CM_O_TABLE, self._CM_O_TABLE_CT05)
        cm_q = self._bilinear_interp(alpha, ct, self._CM_Q_TABLE, self._CM_Q_TABLE_CT05)
        cm_de = self._bilinear_interp(alpha, ct, self._CM_DE_TABLE_CT0, self._CM_DE_TABLE_CT05)
        cm = cm_o + cm_de * elevator + cm_q * q_hat

        # --- Lateral aero (Riley III(d-f) full) ---
        cy = self._side_force_coefficient(alpha, beta, p_hat, r_hat, aileron, rudder, ct)
        cl_roll = self._rolling_moment_coefficient_full(
            alpha, beta, p_hat, r_hat, aileron, rudder, ct)
        cn = self._yawing_moment_coefficient(alpha, beta, p_hat, r_hat, aileron, rudder, ct)

        # --- Forces and moments ---
        qS = 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * vt * vt
        L = qS * cl
        D = qS * cd
        Y = qS * cy
        My = qS * self.CHORD * cm
        Lr = qS * self.WING_SPAN * cl_roll
        Nr = qS * self.WING_SPAN * cn

        # Aliases for brevity
        m = self.MASS
        g_acc = self.GRAVITY
        Ix = self.I_XX
        Iy = self.I_YY
        Iz = self.I_ZZ
        Ip_n = self.I_P * self.N_MAX_RAD_S * float(throttle)  # n(δt) = n_max·δt

        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        cos_m = np.cos(mu)
        sin_m = np.sin(mu)
        tan_g = np.tan(gamma)
        tan_b = np.tan(beta)
        sec_b = 1.0 / max(cos_b, 1e-3)

        # --- Stability-axis EOM (Stengel/Phillips/Etkin standard form) ---
        # V̇ = (1/m)·[−D·cos β + Y·sin β] − g·sin γ      (thrust embedded in CT tables)
        v_dot = ((-D * cos_b + Y * sin_b) / m - g_acc * sin_g) / v_stall

        # γ̇: vertical component of resultant force
        gamma_dot = (
            (L * cos_m - Y * sin_m * cos_b) / (m * vt)
            - (g_acc / vt) * cos_g
        )

        # α̇: pitch rate − γ̇ projected
        alpha_dot = (
            q
            - tan_b * (p * cos_a + r * sin_a)
            + (-L + m * g_acc * cos_g * cos_m) / (m * vt * max(cos_b, 1e-3))
        )

        # β̇: side-slip kinematics
        beta_dot = (
            p * sin_a - r * cos_a
            + (Y * cos_b + m * g_acc * cos_g * sin_m) / (m * vt)
        )

        # μ̇: bank-angle kinematics in stability axis
        mu_dot = (
            sec_b * (p * cos_a + r * sin_a)
            + sin_m * tan_g * L / (m * vt)
        )

        # Body-axis rotational dynamics (Riley Appendix B, I_xz = 0).
        # Propeller gyroscopic terms retained on q̇ and ṙ.
        p_dot = (Iy - Iz) / Ix * q * r + Lr / Ix
        q_dot = (Iz - Ix) / Iy * p * r + My / Iy - (Ip_n / Iy) * r
        r_dot = (Ix - Iy) / Iz * p * q + Nr / Iz + (Ip_n / Iz) * q

        return (gamma_dot, v_dot, alpha_dot, beta_dot, mu_dot,
                p_dot, q_dot, r_dot)
