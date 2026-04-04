import numpy as np
from aircraft.extended_grumman import ExtendedGrumman


class SymmetricFullGrumman(ExtendedGrumman):
    def __init__(self):
        super().__init__()

    def command_airplane(self, elevator, throttle):
        """
        RK4 integration matching the CUDA PolicyIteration kernel exactly.
        Operates on normalized velocity (V/Vs) to ensure identical dynamics.
        """
        dt = self.TIME_STEP
        v_stall = self.STALL_AIRSPEED

        self.last_elevator = elevator
        self.last_throttle = throttle

        gamma = self.flight_path_angle
        vn = self.airspeed_norm
        alpha = self.alpha
        q = self.pitch_rate

        def _derivs(g, v, a, qr):
            vt = max(v * v_stall, 0.1)
            q_hat = qr * self.CHORD / (2.0 * vt)

            ct = self._compute_ct(throttle, vt)

            # Bilinear interp between CT=0 and CT=0.5 tables (Riley 1985, Table III)
            cl_o = float(self._bilinear_interp(a, ct, self._CL_O_TABLE, self._CL_O_TABLE_CT05))
            cl_q = float(self._bilinear_interp(a, ct, self._CL_Q_TABLE, self._CL_Q_TABLE_CT05))
            cl_de = float(self._bilinear_interp(a, ct, self._CL_DE_TABLE_CT0, self._CL_DE_TABLE_CT05))
            cl = cl_o + cl_de * elevator + cl_q * q_hat

            cd = float(self._bilinear_interp(a, ct, self._CD_O_TABLE, self._CD_O_TABLE_CT05))
            cm_o = float(self._bilinear_interp(a, ct, self._CM_O_TABLE, self._CM_O_TABLE_CT05))
            cm_q = float(self._bilinear_interp(a, ct, self._CM_Q_TABLE, self._CM_Q_TABLE_CT05))
            cm_de = float(self._bilinear_interp(a, ct, self._CM_DE_TABLE_CT0, self._CM_DE_TABLE_CT05))
            cm = cm_o + cm_de * elevator + cm_q * q_hat

            qS = 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * vt * vt
            L = qS * cl
            D = qS * cd
            My = qS * self.CHORD * cm

            # Thrust is embedded in CT tables — no separate thrust term
            g_dot = L / (self.MASS * vt) - (self.GRAVITY / vt) * np.cos(g)
            v_dot = (-self.GRAVITY * np.sin(g) - D / self.MASS) / v_stall
            a_dot = qr - g_dot
            q_dot = My / self.I_YY

            return g_dot, v_dot, a_dot, q_dot

        # --- RK4 ---
        k1_g, k1_v, k1_a, k1_q = _derivs(gamma, vn, alpha, q)

        k2_g, k2_v, k2_a, k2_q = _derivs(
            gamma + 0.5 * dt * k1_g, vn + 0.5 * dt * k1_v,
            alpha + 0.5 * dt * k1_a, q + 0.5 * dt * k1_q,
        )

        k3_g, k3_v, k3_a, k3_q = _derivs(
            gamma + 0.5 * dt * k2_g, vn + 0.5 * dt * k2_v,
            alpha + 0.5 * dt * k2_a, q + 0.5 * dt * k2_q,
        )

        k4_g, k4_v, k4_a, k4_q = _derivs(
            gamma + dt * k3_g, vn + dt * k3_v,
            alpha + dt * k3_a, q + dt * k3_q,
        )

        self.flight_path_angle += (dt / 6.0) * (k1_g + 2 * k2_g + 2 * k3_g + k4_g)
        self.airspeed_norm += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        self.alpha += (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        self.pitch_rate += (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        self.airspeed = self.airspeed_norm * v_stall

    def reset(self, flight_path_angle, airspeed_norm, alpha, pitch_rate):
        super().reset(flight_path_angle, airspeed_norm, alpha, 0, 0, 0, pitch_rate, 0)
