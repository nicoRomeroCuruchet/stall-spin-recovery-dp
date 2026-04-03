import numpy as np
from gymnasium import spaces

from airplane.airplane_env import AirplaneEnv
from airplane.reduced_grumman import ReducedGrumman


class ReducedSymmetricGliderPullout(AirplaneEnv):
    """Environment for the reduced symmetric glider pullout maneuver."""

    def __init__(self, render_mode=None):
        self.airplane = ReducedGrumman()
        super().__init__(self.airplane)

        # Split array creation to comply with 88-character limit
        low_bounds = np.array([np.deg2rad(-180), 0.7], dtype=np.float32)
        high_bounds = np.array([np.deg2rad(0), 4.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=low_bounds, 
            high=high_bounds, 
            shape=(2,), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(-0.5, 1.0, shape=(1,), dtype=np.float32)
        
        # Initialize modern random number generator
        self.np_random = np.random.default_rng()

    def _get_obs(self):
        """Retrieve the current observation from the airplane state."""
        return np.vstack(
            [self.airplane.flight_path_angle, self.airplane.airspeed_norm], 
            dtype=np.float32
        ).T

    def _get_info(self):
        """Retrieve additional environment information."""
        return {}

    def reset(self, seed=None, options=None):
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
        
        # BUG FIX 2: Initialize the internal state of the environment
        self.state = observation.copy()
        
        return observation.flatten(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        """
        Execute one or more time steps using vectorized operations.
        
        This implementation supports both single-step execution and high-performance
        batch simulation for heatmap generation.
        """
        # Standardize input as a 2D batch for consistent matrix operations
        # This handles both a single action [cl] and a batch of actions [[cl1], [cl2], ...]
        action_batch = np.atleast_2d(action)
        
        # Synchronize physics engine with the current environment state batch
        self.airplane.flight_path_angle = self.state[:, 0].copy() 
        self.airplane.airspeed_norm = self.state[:, 1].copy() 

        # Extract and clip the lift coefficient from the action batch
        c_lift = np.clip(
            action_batch[:, 0], 
            self.action_space.low[0], 
            self.action_space.high[0]
        )
        
        # Evaluate terminal conditions before the integration step
        init_terminal, _ = self.terminal(self.state)
        
        # Command the aircraft (Throttle and Aileron are 0 for the glider)
        self.airplane.command_airplane(c_lift, 0.0, 0.0)

        # Vectorized altitude loss calculation (Reward)
        # Reward = dt * V * sin(gamma) * Vs
        reward = (
            self.airplane.TIME_STEP 
            * self.airplane.airspeed_norm 
            * np.sin(self.airplane.flight_path_angle) 
            * self.airplane.STALL_AIRSPEED
        )
    
        # Capture new observations and evaluate post-step termination
        obs = self._get_obs()
        terminated, _ = self.terminal(obs) 
        terminated |= init_terminal
        
        # Zero out rewards for trajectories that were already terminal
        reward = np.where(init_terminal, 0.0, reward)
        
        # Update the internal state buffer for the next iteration
        self.state = obs.copy()

        # Compatibility layer for standard Gymnasium API (Non-vectorized calls)
        if np.asarray(action).ndim < 2:
            return obs.flatten(), reward.flatten(), terminated.flatten(), False, self._get_info()

        return obs, reward, terminated, False, self._get_info()

    def step(self, action: float):
        """Execute one time step within the environment."""
        self.airplane.flight_path_angle = self.state[:, 0].copy() 
        self.airplane.airspeed_norm = self.state[:, 1].copy() 

        # BUG FIX 3: Preventive action clipping
        c_lift = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        
        init_terminal, _ = self.terminal(self.state)
        
        self.airplane.command_airplane(c_lift, 0.0, 0.0)

        # Break down the reward calculation to avoid long lines
        reward = (
            self.airplane.TIME_STEP 
            * self.airplane.airspeed_norm 
            * np.sin(self.airplane.flight_path_angle) 
            * self.airplane.STALL_AIRSPEED
        )
    
        obs = self._get_obs()
        terminated, _ = self.terminal(obs) 
        terminated |= init_terminal
        reward = np.where(init_terminal, 0.0, reward)
        
        # Update the internal state for the next step
        self.state = obs.copy()

        return obs, reward, terminated, False, self._get_info()

    def terminal(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine if the state is terminal based on successful leveling or limits.
        
        Terminal conditions:
        1. Success: Flight path angle (gamma) >= 0.0 (leveled flight).
        2. Boundary/Failure: Flight path angle (gamma) <= -pi (-180 degrees).
        
        Args:
            state: A 2D NumPy array (N, 2) where state[:, 0] is gamma (radians).
                
        Returns
        -------
            A tuple of (boolean_mask, zero_reward_array).
        """
        # Extract flight path angle (gamma) for all states in the batch
        gamma = state[:, 0]
        v_norm = state[:, 1]
        
        # 1. Logic: Terminate if leveled (>= 0) OR if we hit the -180 limit (<= -pi).
        # Parentheses are strictly required here because '|' has higher precedence.
        is_terminal = ((gamma >= 0.0) & (v_norm >= 1.0)) | (gamma <= -np.pi)
        
        # Cast to boolean for explicit compatibility with Numba/C-wrappers
        terminate = is_terminal.astype(np.bool_)
        
        # In Policy Iteration with gamma=1.0, terminal states must return 0 reward
        # to stop cumulative altitude loss calculation at that exact moment.
        terminal_rewards = np.zeros_like(terminate, dtype=np.float32)
        
        return terminate, terminal_rewards