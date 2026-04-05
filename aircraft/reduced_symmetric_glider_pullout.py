import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aircraft.reduced_grumman import ReducedGrumman


class ReducedSymmetricGliderPullout(gym.Env):

    def __init__(self, render_mode=None):
        super().__init__()
        self.airplane = ReducedGrumman()

        # Observation space: Flight Path Angle (γ), Air Speed (V/Vs)
        self.observation_space = spaces.Box(
            np.array([-np.pi, 0.9], dtype=np.float32),
            np.array([0.0,    4.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        # Action space: Lift Coefficient
        self.action_space = spaces.Box(-0.5, 1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        return np.array(
            [self.airplane.flight_path_angle, self.airplane.airspeed_norm],
            dtype=np.float32,
        )

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Choose the initial agent's state uniformly
        [flight_path_angle, airspeed_norm] = np.random.uniform(
            self.observation_space.low, self.observation_space.high
        )
        self.airplane.reset(flight_path_angle, airspeed_norm, 0)
        return self._get_obs(), {}

    def step(self, action: list):
        c_lift = action[0]
        self.airplane.command_airplane(c_lift, 0, 0)
        # Reward: height change (negative = altitude loss)
        reward = (
            self.airplane.TIME_STEP
            * self.airplane.airspeed_norm
            * np.sin(self.airplane.flight_path_angle)
        )
        terminated = self.termination()
        return self._get_obs(), reward, terminated, False, self._get_info()

    def termination(self):
        return (
            self.airplane.flight_path_angle >= 0.0
            or self.airplane.flight_path_angle <= -np.pi
        )

    def terminal(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised terminal check for Policy Iteration.

        Args:
            states: (N, 2) array of (γ, V/Vs) pairs.

        Returns:
            (is_terminal, terminal_rewards) — both shape (N,).
        """
        gamma = states[:, 0]
        is_terminal = (gamma >= 0.0) | (gamma <= -np.pi)
        terminal_rewards = np.zeros(len(states), dtype=np.float32)
        return is_terminal.astype(bool), terminal_rewards
