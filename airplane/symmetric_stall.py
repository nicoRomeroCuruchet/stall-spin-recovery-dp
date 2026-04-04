import numpy as np
from gymnasium import spaces

from airplane.symmetric_full_grumman import SymmetricFullGrumman
from airplane.airplane_env import AirplaneEnv


class SymmetricStall(AirplaneEnv):
    """
    Pure Markovian Environment for Symmetric Stall Recovery.
    Matches the exact physics and rewards of the PPO literature.
    """
    def __init__(self, render_mode=None):
        self.airplane = SymmetricFullGrumman()
        super().__init__(self.airplane)
        self.action_space = spaces.Box(
            np.array([np.deg2rad(-25), 0.0], np.float32),
            np.array([np.deg2rad(15),  1.0], np.float32),
            shape=(2,), dtype=np.float32
        )

    def _get_obs(self):
        """Standard 1D observation array for Gym compliance."""
        return np.array([
            self.airplane.flight_path_angle,
            self.airplane.airspeed_norm,
            self.airplane.alpha,
            self.airplane.pitch_rate
        ], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        min_spawn_state = [-1.5, 0.9, np.deg2rad(14), -0.2]
        max_spawn_state = [-0.1, 2.0, np.deg2rad(20), 0.2]

        flight_path_angle, airspeed_norm, alpha, pitch_rate = np.random.uniform(
            min_spawn_state, max_spawn_state
        )
        self.airplane.reset(flight_path_angle, airspeed_norm, alpha, pitch_rate)

        return self._get_obs(), self._get_info()

    def specific_reset(self, flight_path_angle, airspeed_norm, alpha, pitch_rate):
        """Forces the environment into a specific mathematical state."""
        self.airplane.reset(flight_path_angle, airspeed_norm, alpha, pitch_rate)
        return self._get_obs(), self._get_info()

    def step(self, action: list):
        """
        Pure step function: No history, no hidden filters.
        Strict compliance with the Markov property.
        """
        elevator = action[0]
        delta_throttle = action[1]

        # 1. Integrate physical kinematics
        self.airplane.command_airplane(elevator, delta_throttle)

        # 2. Retrieve new physical state
        fpa = self.airplane.flight_path_angle
        v_norm = self.airplane.airspeed_norm
        alpha = self.airplane.alpha

        # 3. Base Physical Reward: True physical height loss in meters
        reward = (self.airplane.TIME_STEP * v_norm * np.sin(fpa) * self.airplane.STALL_AIRSPEED)

        # 4. Evaluate specific terminal conditions
        fpa_success = (fpa >= 0.0)

        alpha_crash = (alpha >= np.deg2rad(40)) or (alpha <= np.deg2rad(-40))
        fpa_crash = (fpa <= -np.pi + 0.05)

        failure = fpa_crash or alpha_crash
        terminated = fpa_success or failure

        # Apply catastrophic penalty only if boundaries are violated
        if failure:
            reward = -1000.0 * self.airplane.STALL_AIRSPEED

        return self._get_obs(), reward, terminated, False, self._get_info()

    def terminal(self, states: np.ndarray):
        """
        Vectorized terminal check exclusively for GPU Policy Iteration initialization.
        Expects a 2D array of states (N, 4).
        """
        fpa = np.asarray(states[:, 0])
        alpha = np.asarray(states[:, 2])

        fpa_success = (fpa >= 0.0)

        alpha_crash = (alpha >= np.deg2rad(40)) | (alpha <= np.deg2rad(-40))
        fpa_crash = (fpa <= -np.pi + 0.05)

        failure = fpa_crash | alpha_crash
        terminate = fpa_success | failure

        rewards = np.zeros_like(fpa)

        # FIX: Align GPU initialization penalty with the step function
        rewards[failure] = -1000.0 * self.airplane.STALL_AIRSPEED

        return terminate, rewards
