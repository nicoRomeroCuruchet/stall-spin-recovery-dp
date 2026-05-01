import numpy as np
from gymnasium import spaces

from envs.base import AirplaneEnv
from aircraft.reduced_grumman import ReducedGrumman


class ReducedBankedGliderPullout(AirplaneEnv):
    """
    Environment for the asymmetric (banked) glider pullout maneuver.
    Matched exactly to the original reference implementation.
    """

    def __init__(self, render_mode=None):
        self.airplane = ReducedGrumman()
        super().__init__(self.airplane, render_mode=render_mode)
        
        low_obs = np.array([-np.pi, 0.9, np.deg2rad(-20)], dtype=np.float32)
        high_obs = np.array([0.0, 4.0, np.deg2rad(200)], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=(3,), dtype=np.float32
        )
        
        low_action = np.array([-0.5, np.deg2rad(-30), 0.0], dtype=np.float32)
        high_action = np.array([1.0, np.deg2rad(30), 1.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action, high=high_action, shape=(3,), dtype=np.float32
        )

        self.np_random = np.random.default_rng()

    def _get_obs(self) -> np.ndarray:
        return np.vstack([
            self.airplane.flight_path_angle, 
            self.airplane.airspeed_norm, 
            self.airplane.bank_angle
        ], dtype=np.float32).T

    def _get_info(self) -> dict:
        return {}

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        flight_path, airspeed, bank_angle = self.np_random.uniform(
            self.observation_space.low, self.observation_space.high
        )
        self.airplane.reset(flight_path, airspeed, bank_angle)

        observation = self._get_obs()
        self.state = observation.copy()
        
        return observation.flatten(), self._get_info()

    def step(self, action: np.ndarray) -> tuple:
        action_batch = np.atleast_2d(action)
        
        self.airplane.flight_path_angle = self.state[:, 0].copy()
        self.airplane.airspeed_norm = self.state[:, 1].copy()
        self.airplane.bank_angle = self.state[:, 2].copy()

        action_clipped = np.clip(action_batch, self.action_space.low, self.action_space.high)

        c_lift = action_clipped[:, 0]
        bank_rate = action_clipped[:, 1]
        throttle = action_clipped[:, 2]

        init_terminal, _ = self.terminal(self.state)

        self.airplane.command_airplane(c_lift, bank_rate, throttle)

        # Reward matched to the CUDA kernel and identical to the original
        # idle-power branch: r = v_true * sin(gamma) * dt - 0.01 * mu_dot^2 * dt.
        # No throttle term — the only structural delta vs the idle branch
        # is that delta_t is part of the action space and enters V_dot.
        v_true = self.airplane.airspeed_norm * self.airplane.STALL_AIRSPEED
        h_dot = v_true * np.sin(self.airplane.flight_path_angle)
        dt = self.airplane.TIME_STEP
        reward = h_dot * dt - 0.01 * bank_rate**2 * dt

        obs = self._get_obs()
        terminated, _ = self.terminal(obs) 
        terminated |= init_terminal
        
        reward = np.where(init_terminal, 0.0, reward)
        self.state = obs.copy()

        if self.state.shape[0] == 1:
            return obs.flatten(), float(reward[0]), bool(terminated[0]), False, self._get_info()
            
        return obs, reward, terminated, False, self._get_info()
    
    def terminal(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Terminación original simplificada: Solo revisar Gamma."""
        gamma = state[:, 0]

        is_terminal = (gamma >= 0.0) | (gamma <= -np.pi)
        
        terminate = is_terminal.astype(np.bool_)
        terminal_rewards = np.zeros_like(terminate, dtype=np.float32)

        return terminate, terminal_rewards