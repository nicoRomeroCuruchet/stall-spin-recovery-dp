import numpy as np
from gymnasium import spaces

from aircraft.airplane_env import AirplaneEnv
from aircraft.spin_grumman import SpinGrumman


# Markov-compliant reward weights (per-step, dt-scaled)
W_P_PENALTY      = 0.1     # roll-rate damping incentive
W_R_PENALTY      = 0.1     # yaw-rate damping incentive
W_BETA_PENALTY   = 0.5     # sideslip damping (encourages coordinated flight)
W_MU_BARRIER     = 0.5     # soft barrier on |μ| past 60°
W_AILERON_EFFORT = 5.0     # δa²
W_RUDDER_EFFORT  = 5.0     # δr²

MU_BARRIER  = np.deg2rad(60.0)
MU_CRASH    = np.pi / 2.0
P_CRASH     = 3.0
R_CRASH     = 4.0
BETA_CRASH  = np.deg2rad(30.0)
ALPHA_CRASH = np.deg2rad(40.0)
GAMMA_CRASH = -np.pi + 0.05


class Spin(AirplaneEnv):
    """
    Pure Markovian Environment for 8-DOF Spin Recovery.

    State (8): (γ, V/Vs, α, β, μ, p, q, r)
    Action (4): (δe, δa, δt, δr)

    Aero from Riley (1985) Table III complete (a-f, including side-force
    and yawing-moment terms). Stability-axis EOM derived from Stengel/
    Phillips with `I_xz = 0` (Riley Table I) and propeller gyroscopic
    `Ip · n` retained on q̇ and ṙ.
    """

    def __init__(self, render_mode=None):
        self.airplane = SpinGrumman()
        super().__init__(self.airplane)

        self.action_space = spaces.Box(
            np.array([np.deg2rad(-25), np.deg2rad(-15), 0.0, np.deg2rad(-25)], np.float32),
            np.array([np.deg2rad( 15), np.deg2rad( 15), 1.0, np.deg2rad( 25)], np.float32),
            shape=(4,), dtype=np.float32,
        )

    def _get_obs(self):
        return np.array([
            self.airplane.flight_path_angle,
            self.airplane.airspeed_norm,
            self.airplane.alpha,
            self.airplane.beta,
            self.airplane.bank_angle,
            self.airplane.roll_rate,
            self.airplane.pitch_rate,
            self.airplane.yaw_rate,
        ], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # Spawn over the spin-recovery envelope.
        min_spawn = [-np.pi / 3, 0.9, np.deg2rad(14),
                     np.deg2rad(-10), 0.0, -0.2, -0.2, -0.5]
        max_spawn = [-0.1,        1.5, np.deg2rad(20),
                     np.deg2rad( 10), np.deg2rad(60), 0.2, 0.2, 0.5]

        (gamma, v_norm, alpha, beta, mu, p, q, r) = np.random.uniform(min_spawn, max_spawn)
        self.airplane.reset(gamma, v_norm, alpha, beta, mu, p, q, r)
        return self._get_obs(), self._get_info()

    def specific_reset(self, gamma, v_norm, alpha, beta, mu, p, q, r):
        self.airplane.reset(gamma, v_norm, alpha, beta, mu, p, q, r)
        return self._get_obs(), self._get_info()

    def step(self, action):
        elevator, aileron, throttle, rudder = (
            action[0], action[1], action[2], action[3]
        )

        self.airplane.command_airplane(elevator, aileron, throttle, rudder)

        fpa  = self.airplane.flight_path_angle
        v    = self.airplane.airspeed_norm
        alpha = self.airplane.alpha
        beta = self.airplane.beta
        mu   = self.airplane.bank_angle
        p    = self.airplane.roll_rate
        r    = self.airplane.yaw_rate
        dt   = self.airplane.TIME_STEP
        v_stall = self.airplane.STALL_AIRSPEED

        # Base reward: signed altitude change (m)
        reward = dt * v * np.sin(fpa) * v_stall

        # Markov shaping (state-only and action-magnitude)
        reward -= dt * W_P_PENALTY * p * p
        reward -= dt * W_R_PENALTY * r * r
        reward -= dt * W_BETA_PENALTY * beta * beta
        mu_excess = max(0.0, abs(mu) - MU_BARRIER)
        reward -= dt * W_MU_BARRIER * mu_excess * mu_excess
        reward -= dt * W_AILERON_EFFORT * aileron * aileron
        reward -= dt * W_RUDDER_EFFORT * rudder * rudder

        # Terminal evaluation
        fpa_success = (fpa >= 0.0)

        alpha_crash = (alpha >= ALPHA_CRASH) or (alpha <= -ALPHA_CRASH)
        fpa_crash   = (fpa <= GAMMA_CRASH)
        mu_crash    = (abs(mu) >= MU_CRASH)
        p_crash     = (abs(p) >= P_CRASH)
        r_crash     = (abs(r) >= R_CRASH)
        beta_crash  = (abs(beta) >= BETA_CRASH)

        failure = (
            fpa_crash or alpha_crash or mu_crash
            or p_crash or r_crash or beta_crash
        )
        terminated = fpa_success or failure

        if failure:
            reward = -1000.0 * v_stall

        return self._get_obs(), reward, terminated, False, self._get_info()

    def terminal(self, states: np.ndarray):
        """
        Vectorised terminal check for GPU Policy Iteration initialisation.
        Expects (N, 8) ordered (γ, V/Vs, α, β, μ, p, q, r).
        """
        fpa   = np.asarray(states[:, 0])
        alpha = np.asarray(states[:, 2])
        beta  = np.asarray(states[:, 3])
        mu    = np.asarray(states[:, 4])
        p     = np.asarray(states[:, 5])
        r     = np.asarray(states[:, 7])

        fpa_success = (fpa >= 0.0)

        alpha_crash = (alpha >= ALPHA_CRASH) | (alpha <= -ALPHA_CRASH)
        fpa_crash   = (fpa <= GAMMA_CRASH)
        mu_crash    = (np.abs(mu) >= MU_CRASH)
        p_crash     = (np.abs(p) >= P_CRASH)
        r_crash     = (np.abs(r) >= R_CRASH)
        beta_crash  = (np.abs(beta) >= BETA_CRASH)

        failure = (
            fpa_crash | alpha_crash | mu_crash
            | p_crash | r_crash | beta_crash
        )
        terminate = fpa_success | failure

        rewards = np.zeros_like(fpa)
        rewards[failure] = -1000.0 * self.airplane.STALL_AIRSPEED
        return terminate, rewards
