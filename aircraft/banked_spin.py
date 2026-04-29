import numpy as np
from gymnasium import spaces

from aircraft.airplane_env import AirplaneEnv
from aircraft.banked_spin_grumman import BankedSpinGrumman


# Reward-shaping coefficients (Markov-compliant, all multiplied by dt
# so they live on the same scale as the base height-loss reward).
W_P_PENALTY = 0.01       # penalises p² to discourage fast roll oscillations
W_MU_BARRIER = 0.5       # soft quadratic barrier when |μ| exceeds μ_BARRIER
W_AILERON_EFFORT = 5.0   # δa² quadratic penalty (symmetric with W_CONTROL_EFFORT · δe²)

MU_BARRIER = np.deg2rad(60.0)  # soft barrier kicks in past 60°
MU_CRASH = np.pi / 2.0         # |μ| ≥ 90° = inverted, hard crash
P_CRASH = 3.0                  # |p| ≥ 3 rad/s = unrecoverable rotation
ALPHA_CRASH = np.deg2rad(40.0)
GAMMA_CRASH = -np.pi + 0.05


class BankedSpin(AirplaneEnv):
    """
    Pure Markovian Environment for 6-DOF Banked-Spin Recovery (A.i).

    State (6): (γ, V/Vs, α, μ, p, q)
    Action (3): (δe, δa, δt)

    Mirrors SymmetricStall but adds bank-angle, roll-rate dynamics and a
    rolling-moment from Riley (1985) Table III(f). Reward extends the base
    height-loss term with Markov-compliant penalties on p², |μ|, and |δa|.
    """

    def __init__(self, render_mode=None):
        self.airplane = BankedSpinGrumman()
        super().__init__(self.airplane)
        self.action_space = spaces.Box(
            np.array([np.deg2rad(-25), np.deg2rad(-15), 0.0], np.float32),
            np.array([np.deg2rad(15),  np.deg2rad(15),  1.0], np.float32),
            shape=(3,), dtype=np.float32,
        )

    def _get_obs(self):
        return np.array([
            self.airplane.flight_path_angle,
            self.airplane.airspeed_norm,
            self.airplane.alpha,
            self.airplane.bank_angle,
            self.airplane.roll_rate,
            self.airplane.pitch_rate,
        ], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # Spawn over the banked-stall envelope (legacy banked_spin ranges,
        # widened on γ and α to match SymmetricStall).
        min_spawn_state = [-np.pi / 3, 0.9, np.deg2rad(14),
                           0.0, -0.2, -0.2]
        max_spawn_state = [-0.1,        1.5, np.deg2rad(20),
                           np.deg2rad(60), 0.2, 0.2]

        (flight_path_angle, airspeed_norm, alpha,
         bank_angle, roll_rate, pitch_rate) = np.random.uniform(
            min_spawn_state, max_spawn_state)

        self.airplane.reset(flight_path_angle, airspeed_norm, alpha,
                            bank_angle, roll_rate, pitch_rate)
        return self._get_obs(), self._get_info()

    def specific_reset(self, flight_path_angle, airspeed_norm, alpha,
                       bank_angle, roll_rate, pitch_rate):
        self.airplane.reset(flight_path_angle, airspeed_norm, alpha,
                            bank_angle, roll_rate, pitch_rate)
        return self._get_obs(), self._get_info()

    def step(self, action):
        elevator, aileron, delta_throttle = action[0], action[1], action[2]

        # 1. Integrate physical kinematics (RK4, 6 states)
        self.airplane.command_airplane(elevator, aileron, delta_throttle)

        # 2. Retrieve new physical state
        fpa = self.airplane.flight_path_angle
        v_norm = self.airplane.airspeed_norm
        alpha = self.airplane.alpha
        mu = self.airplane.bank_angle
        p = self.airplane.roll_rate
        dt = self.airplane.TIME_STEP
        v_stall = self.airplane.STALL_AIRSPEED

        # 3. Base physical reward: signed altitude change in metres
        reward = dt * v_norm * np.sin(fpa) * v_stall

        # 4. Markov-compliant shaping (all per-step, dt-scaled)
        reward -= dt * W_P_PENALTY * p * p
        mu_excess = max(0.0, abs(mu) - MU_BARRIER)
        reward -= dt * W_MU_BARRIER * mu_excess * mu_excess
        reward -= dt * W_AILERON_EFFORT * aileron * aileron

        # 5. Terminal evaluation
        fpa_success = (fpa >= 0.0)

        alpha_crash = (alpha >= ALPHA_CRASH) or (alpha <= -ALPHA_CRASH)
        fpa_crash = (fpa <= GAMMA_CRASH)
        mu_crash = (abs(mu) >= MU_CRASH)
        p_crash = (abs(p) >= P_CRASH)

        failure = fpa_crash or alpha_crash or mu_crash or p_crash
        terminated = fpa_success or failure

        if failure:
            reward = -1000.0 * v_stall

        return self._get_obs(), reward, terminated, False, self._get_info()

    def terminal(self, states: np.ndarray):
        """
        Vectorised terminal check for GPU Policy Iteration initialisation.
        Expects a 2D array of states (N, 6) ordered (γ, V/Vs, α, μ, p, q).
        """
        fpa = np.asarray(states[:, 0])
        alpha = np.asarray(states[:, 2])
        mu = np.asarray(states[:, 3])
        p = np.asarray(states[:, 4])

        fpa_success = (fpa >= 0.0)

        alpha_crash = (alpha >= ALPHA_CRASH) | (alpha <= -ALPHA_CRASH)
        fpa_crash = (fpa <= GAMMA_CRASH)
        mu_crash = (np.abs(mu) >= MU_CRASH)
        p_crash = (np.abs(p) >= P_CRASH)

        failure = fpa_crash | alpha_crash | mu_crash | p_crash
        terminate = fpa_success | failure

        rewards = np.zeros_like(fpa)
        rewards[failure] = -1000.0 * self.airplane.STALL_AIRSPEED

        return terminate, rewards
