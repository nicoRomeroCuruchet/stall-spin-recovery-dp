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
        thrust = self.THROTTLE_LINEAR_MAPPING * throttle

        self.last_elevator = elevator
        self.last_throttle = throttle

        gamma = self.flight_path_angle
        vn = self.airspeed_norm
        alpha = self.alpha
        q = self.pitch_rate

        def _derivs(g, v, a, qr):
            vt = max(v * v_stall, 0.1)
            q_hat = qr * self.CHORD / (2.0 * vt)

            # CL with stall saturation (matches CUDA kernel exactly)
            if a >= self.ALPHA_STALL:
                cl = self.CL_0 + self.CL_ALPHA * self.ALPHA_STALL
            elif a <= self.ALPHA_NEGATIVE_STALL:
                cl = self.CL_0 + self.CL_ALPHA * self.ALPHA_NEGATIVE_STALL
            else:
                cl = (self.CL_0 + self.CL_ALPHA * a
                      + self.CL_ELEVATOR * elevator + self.CL_QHAT * q_hat)

            cd = self.CD_0 + self.CD_ALPHA * a + self.CD_ALPHA2 * a * a
            cm = (self.CM_0 + self.CM_ALPHA * a
                  + self.CM_ELEVATOR * elevator + self.CM_QHAT * q_hat)

            qS = 0.5 * self.AIR_DENSITY * self.WING_SURFACE_AREA * vt * vt
            L = qS * cl
            D = qS * cd
            My = qS * self.CHORD * cm

            g_dot = (
                (L + thrust * np.sin(a)) / (self.MASS * vt)
                - (self.GRAVITY / vt) * np.cos(g)
            )
            v_dot = (
                (-self.GRAVITY * np.sin(g) - (D - thrust * np.cos(a)) / self.MASS) / v_stall
            )
            a_dot = qr - g_dot
            q_dot = My / self.I_YY

            return g_dot, v_dot, a_dot, q_dot

        # --- RK4 ---
        k1_g, k1_v, k1_a, k1_q = _derivs(gamma, vn, alpha, q)

        k2_g, k2_v, k2_a, k2_q = _derivs(
            gamma + 0.5 * dt * k1_g, vn + 0.5 * dt * k1_v,
            alpha + 0.5 * dt * k1_a, q  + 0.5 * dt * k1_q,
        )

        k3_g, k3_v, k3_a, k3_q = _derivs(
            gamma + 0.5 * dt * k2_g, vn + 0.5 * dt * k2_v,
            alpha + 0.5 * dt * k2_a, q  + 0.5 * dt * k2_q,
        )

        k4_g, k4_v, k4_a, k4_q = _derivs(
            gamma + dt * k3_g, vn + dt * k3_v,
            alpha + dt * k3_a, q  + dt * k3_q,
        )

        self.flight_path_angle += (dt / 6.0) * (k1_g + 2 * k2_g + 2 * k3_g + k4_g)
        self.airspeed_norm     += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        self.alpha             += (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        self.pitch_rate        += (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        self.airspeed = self.airspeed_norm * v_stall

    def reset(self, flight_path_angle, airspeed_norm, alpha, pitch_rate):
        super().reset(flight_path_angle, airspeed_norm, alpha, 0, 0, 0, pitch_rate, 0)
