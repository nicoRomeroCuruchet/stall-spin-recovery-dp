import numpy as np
from gymnasium import spaces

from airplane.airplane_env import AirplaneEnv
from airplane.reduced_grumman import ReducedGrumman


class ReducedSymmetricPullout(AirplaneEnv):
    """Environment for the powered symmetric pullout maneuver."""

    def __init__(self, render_mode=None):
        self.airplane = ReducedGrumman()
        super().__init__(self.airplane, render_mode=render_mode)

        # Observation space: Flight Path Angle (γ), Air Speed (V)
        low_obs = np.array([-np.pi, 0.9], dtype=np.float32)
        high_obs = np.array([0.0, 4.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=(2,), dtype=np.float32
        )

        # Action space: Lift Coefficient (CL), Throttle Setting (δ_throttle)
        low_action = np.array([-0.5, 0.0], dtype=np.float32)
        high_action = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=low_action, high=high_action, shape=(2,), dtype=np.float32
        )

        # Initialize modern random number generator
        self.np_random = np.random.default_rng()

    def _get_obs(self) -> np.ndarray:
        """Retrieve the current observation from the airplane state."""
        return np.vstack(
            [self.airplane.flight_path_angle, self.airplane.airspeed_norm],
            dtype=np.float32,
        ).T

    def _get_info(self) -> dict:
        """Retrieve additional environment information."""
        return {}

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to a random initial state."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        flight_path_angle, airspeed_norm = self.np_random.uniform(
            self.observation_space.low, self.observation_space.high
        )
        self.airplane.reset(flight_path_angle, airspeed_norm, 0)

        observation = self._get_obs()
        observation = np.clip(
            observation, self.observation_space.low, self.observation_space.high
        )

        # Initialize the internal state of the environment for the first step
        self.state = observation.copy()

        return observation.flatten(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        """Execute one time step within the environment without artificial state clipping."""
        # Restore environment state for the integration step
        """Execute time steps for one or multiple aircraft states simultaneously."""
        # Restore internal state batch
        self.airplane.flight_path_angle = self.state[:, 0].copy()
        self.airplane.airspeed_norm = self.state[:, 1].copy()

        # Dynamic action unpacking (Handles scalar CL or [CL, Throttle] batch)
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)
        
        if action_clipped.ndim == 2:  # If batch of actions provided
            c_lift = action_clipped[:, 0]
            # Check if throttle exists (Powered vs Glider)
            delta_throttle = action_clipped[:, 1] if action_clipped.shape[1] > 1 else 0.0
        else:  # Single action provided
            c_lift = action_clipped[0] if self.action_space.shape[0] > 1 else action_clipped
            delta_throttle = action_clipped[1] if self.action_space.shape[0] > 1 else 0.0
        
        
        # Evaluate termination on the pre-step state
        init_terminal, _ = self.terminal(self.state)

        # Integrate dynamics natively (NO STATE CLIPPING)
        self.airplane.command_airplane(c_lift, 0.0, delta_throttle)

        # Calculate reward: Altitude loss mapped dynamically from airspeed
        reward = (
            self.airplane.TIME_STEP
            * self.airplane.airspeed_norm
            * np.sin(self.airplane.flight_path_angle)
            * self.airplane.STALL_AIRSPEED
        )

        # Retrieve new observation and evaluate post-step termination
        obs = self._get_obs()
        terminated, _ = self.terminal(obs)
        terminated |= init_terminal

        # Zero out rewards for states that were already terminal
        reward = np.where(init_terminal, 0.0, reward)

        # Update the internal state array for the next integration step
        self.state = obs.copy()

        return obs, reward, terminated, False, self._get_info()

    def terminal(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine if the state is terminal based on successful leveling or boundary limits.

        Terminal conditions:
        1. Success: Flight path angle (gamma) >= 0.0 AND airspeed recovered (V >= 1.0).
        2. Boundary/Failure: Flight path angle (gamma) <= -pi (-180 degrees).
        3. Out of Bounds: Airspeed exceeds maximum grid limit (V > 4.0).
        """
        gamma = state[:, 0]
        v_norm = state[:, 1]

        # Extract grid upper boundary dynamically from observation space
        v_max = self.observation_space.high[1]

        # Logically group the physical bounds and out-of-bounds limits
        is_terminal = (
            ((gamma >= 0.0) & (v_norm >= 1.0))  # Success
            | (gamma <= -np.pi)                 # Failure: Vertical dive exceeded
            | (v_norm > v_max)                  # Failure: Overspeed (Out of Bounds)
        )

        # Cast to boolean for Numba and C-contiguous operations compatibility
        terminate = is_terminal.astype(np.bool_)

        # Ensure terminal nodes return exactly 0 to halt Bellman value accumulation
        terminal_rewards = np.zeros_like(terminate, dtype=np.float32)

        return terminate, terminal_rewards

    def terminal(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine if the state is terminal based on successful leveling or boundary limits.

        Terminal conditions:
        1. Success: Flight path angle (gamma) >= 0.0 AND airspeed recovered (V >= 1.0).
        2. Boundary/Failure: Flight path angle (gamma) <= -pi (-180 degrees).

        Args:
            state: A 2D NumPy array (N, 2) where state[:, 0] is gamma (radians)
                   and state[:, 1] is relative airspeed.

        Returns
        -------
            A tuple containing a boolean termination mask and a zero-reward array.
        """
        gamma = state[:, 0]
        v_norm = state[:, 1]

        # Logically group the physical bounds
        is_terminal = ((gamma >= 0.0) & (v_norm >= 1.0)) | (gamma <= -np.pi)

        # Cast to boolean for Numba and C-contiguous operations compatibility
        terminate = is_terminal.astype(np.bool_)

        # Ensure terminal nodes return exactly 0 to halt Bellman value accumulation
        terminal_rewards = np.zeros_like(terminate, dtype=np.float32)

        return terminate, terminal_rewards