import numpy as np

from aircraft.extended_grumman import ExtendedGrumman


class BankedSpinGrumman(ExtendedGrumman):
    """
    6-DOF banked-spin dynamics under simplification A.i (β = 0, r = 0, δr = 0).

    State:  (γ, V/Vs, α, μ, p, q)
    Action: (δe, δa, δt)

    Longitudinal aero (CL, CD, CM) from Riley (1985) Table III(a-c);
    rolling moment Cl_b from Table III(f). Side-force (d) and yawing-
    moment (e) tables are deliberately not included: with β=0 and r=0
    they would not couple into the dynamics, and ṙ is not integrated.

    With μ=0, p=0 and δa=0 the equations collapse to SymmetricFullGrumman.
    """

    def __init__(self):
        super().__init__()

    def command_airplane(self, elevator, aileron, throttle):
        """
        RK4 integration of 6 states. Mirrors the structure of
        SymmetricFullGrumman.command_airplane and is intended to
        match the 6-DOF CUDA kernel byte-for-byte.
        """
        dt = self.TIME_STEP
        v_stall = self.STALL_AIRSPEED

        self.last_elevator = elevator
        self.last_aileron = aileron
        self.last_throttle = throttle

        gamma = self.flight_path_angle
        vn = self.airspeed_norm
        alpha = self.alpha
        mu = self.bank_angle
        p = self.roll_rate
        q = self.pitch_rate

        def _derivs(g, v, a, m, pr, qr):
            vt = max(v * v_stall, 0.1)
            q_hat = qr * self.CHORD / (2.0 * vt)
            p_hat = pr * self.WING_SPAN / (2.0 * vt)

            ct = self._compute_ct(throttle, vt)

            # --- Longitudinal aero (Riley III(a-c)) ---
            cl_o = float(self._bilinear_interp(
                a, ct, self._CL_O_TABLE, self._CL_O_TABLE_CT05))
            cl_q = float(self._bilinear_interp(
                a, ct, self._CL_Q_TABLE, self._CL_Q_TABLE_CT05))
            cl_de = float(self._bilinear_interp(
                a, ct, self._CL_DE_TABLE_CT0, self._CL_DE_TABLE_CT05))
            cl = cl_o + cl_de * elevator + cl_q * q_hat

            cd = float(self._bilinear_interp(
                a, ct, self._CD_O_TABLE, self._CD_O_TABLE_CT05))

            cm_o = float(self._bilinear_interp(
                a, ct, self._CM_O_TABLE, self._CM_O_TABLE_CT05))
            cm_q = float(self._bilinear_interp(
                a, ct, self._CM_Q_TABLE, self._CM_Q_TABLE_CT05))
            cm_de = float(self._bilinear_interp(
                a, ct, self._CM_DE_TABLE_CT0, self._CM_DE_TABLE_CT05))
            cm = cm_o + cm_de * elevator + cm_q * q_hat

            # --- Lateral rolling moment (Riley III(f), A.i: β=0, r=0, δr=0) ---
            cl_roll_o = float(self._bilinear_interp(
                a, ct, self._CL_ROLL_O_TABLE_CT0, self._CL_ROLL_O_TABLE_CT05))
            cl_roll_p = float(self._bilinear_interp(
                a, ct, self._CL_ROLL_PHAT_TABLE_CT0, self._CL_ROLL_PHAT_TABLE_CT05))
            cl_roll_da = float(np.interp(
                a, self._CL_O_ALPHA_RAD, self._CL_ROLL_DA_TABLE))
            cl_roll = cl_roll_o + cl_roll_p * p_hat + cl_roll_da * aileron

            # --- Forces & moments ---
            qS = 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * vt * vt
            L = qS * cl
            D = qS * cd
            My = qS * self.CHORD * cm
            Lr = qS * self.WING_SPAN * cl_roll

            cos_g = np.cos(g)
            sin_g = np.sin(g)
            cos_m = np.cos(m)
            sin_m = np.sin(m)
            tan_g = np.tan(g)
            cos_a = np.cos(a)

            # Stability-axis EOM (A.i). Thrust embedded in CT-dependent tables.
            g_dot = (L * cos_m) / (self.MASS * vt) - (self.GRAVITY / vt) * cos_g
            v_dot = (-self.GRAVITY * sin_g - D / self.MASS) / v_stall
            a_dot = qr - L / (self.MASS * vt) + (self.GRAVITY / vt) * cos_g * cos_m
            m_dot = pr * cos_a + sin_m * tan_g * L / (self.MASS * vt)
            p_dot = Lr / self.I_XX
            q_dot = My / self.I_YY

            return g_dot, v_dot, a_dot, m_dot, p_dot, q_dot

        # --- RK4 over 6 states ---
        k1_g, k1_v, k1_a, k1_m, k1_p, k1_q = _derivs(gamma, vn, alpha, mu, p, q)

        k2_g, k2_v, k2_a, k2_m, k2_p, k2_q = _derivs(
            gamma + 0.5 * dt * k1_g, vn + 0.5 * dt * k1_v,
            alpha + 0.5 * dt * k1_a, mu + 0.5 * dt * k1_m,
            p + 0.5 * dt * k1_p, q + 0.5 * dt * k1_q,
        )

        k3_g, k3_v, k3_a, k3_m, k3_p, k3_q = _derivs(
            gamma + 0.5 * dt * k2_g, vn + 0.5 * dt * k2_v,
            alpha + 0.5 * dt * k2_a, mu + 0.5 * dt * k2_m,
            p + 0.5 * dt * k2_p, q + 0.5 * dt * k2_q,
        )

        k4_g, k4_v, k4_a, k4_m, k4_p, k4_q = _derivs(
            gamma + dt * k3_g, vn + dt * k3_v,
            alpha + dt * k3_a, mu + dt * k3_m,
            p + dt * k3_p, q + dt * k3_q,
        )

        self.flight_path_angle += (dt / 6.0) * (k1_g + 2 * k2_g + 2 * k3_g + k4_g)
        self.airspeed_norm     += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        self.alpha             += (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        self.bank_angle        += (dt / 6.0) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m)
        self.roll_rate         += (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        self.pitch_rate        += (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        self.airspeed = self.airspeed_norm * v_stall

    def reset(self, flight_path_angle, airspeed_norm, alpha,
              bank_angle, roll_rate, pitch_rate):
        super().reset(
            flight_path_angle, airspeed_norm, alpha, 0,
            bank_angle, roll_rate, pitch_rate, 0,
        )
