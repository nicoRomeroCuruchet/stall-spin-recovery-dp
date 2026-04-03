import numpy as np
from airplane.grumman import Grumman


class ReducedGrumman(Grumman):
    """
    Class for simplified airplane state and dynamics.
    Refactored to support dynamic batch sizes for highly scalable vectorized environments.
    """

    def __init__(self) -> None:
        super().__init__()
        
        ##########################
        ### Airplane variables ###
        ##########################
        
        # Initialize dynamically. The exact shape (scalar or N-dimensional batch) 
        # will be inferred and overwritten dynamically at runtime by the environment.
        self.flight_path_angle = np.array([0.0], dtype=np.float32)
        self.airspeed_norm = np.array([1.0], dtype=np.float32)
        self.bank_angle = np.array([0.0], dtype=np.float32)
        
        # Previous commands
        self.last_c_lift = 0.0
        self.last_bank_rate = 0.0
        self.last_throttle = 0.0

    def command_airplane(self, c_lift: float, bank_rate: float, delta_throttle: float) -> None:
        """
        Executes control commands and updates the coupled kinematic state using RK4 integration.
        Uses numpy broadcasting to handle heterogeneous input shapes automatically.
        """
        self.last_c_lift = c_lift
        self.last_bank_rate = bank_rate
        self.last_throttle = delta_throttle
        
        # 1. FIX: Broadcast arrays to ensure they all have the exact same shape 
        # (scalar or N-dimensional) before packing them into a single state vector.
        v_norm, gamma, mu = np.broadcast_arrays(
            self.airspeed_norm, 
            self.flight_path_angle, 
            self.bank_angle
        )
        
        # Now NumPy guarantees they are identically shaped
        state_vec = np.array([v_norm, gamma, mu], dtype=np.float32)
        
        # 2. Simultaneous RK4 update to respect physical coupling
        new_state = self._rk4_update(state_vec, self._coupled_dynamics, self.TIME_STEP)
        
        # 3. Unpack dynamically sized state
        self.airspeed_norm = new_state[0]
        self.flight_path_angle = new_state[1]
        self.bank_angle = new_state[2]

    def _coupled_dynamics(self, state_vec: np.ndarray) -> np.ndarray:
        """
        Calculates the simultaneous derivatives for the Runge-Kutta 4 integrator.
        Optimized for memory locality and strict vectorization.
        """
        vn = state_vec[0]
        gamma = state_vec[1]
        mu = state_vec[2]
        
        c_drag = self._cd_from_cl(self.last_c_lift)
        v_true = vn * self.STALL_AIRSPEED
        
        # Airspeed derivative (V_dot / Vs)
        vn_dot = (
            - self.GRAVITY * np.sin(gamma) 
            - 0.5 * self.AIR_DENSITY * (self.WING_SURFACE_AREA / self.MASS) * (v_true ** 2) * c_drag 
            + self.THROTTLE_LINEAR_MAPPING * self.last_throttle / self.MASS
        ) / self.STALL_AIRSPEED
        
        # Flight path angle derivative (gamma_dot)
        gamma_dot = (
            0.5 * self.AIR_DENSITY * (self.WING_SURFACE_AREA / self.MASS) * v_true * self.last_c_lift * np.cos(mu) 
            - (self.GRAVITY / v_true) * np.cos(gamma)
        )
        
        # Bank angle derivative (mu_dot) maintains the shape of the input array
        mu_dot = np.full_like(mu, self.last_bank_rate, dtype=np.float32)
        
        return np.array([vn_dot, gamma_dot, mu_dot], dtype=np.float32)

    def reset(self, flight_path_angle: np.ndarray, airspeed_norm: np.ndarray, bank_angle: np.ndarray) -> None:
        """
        Resets the physical state of the airplane.
        """
        self.flight_path_angle = flight_path_angle
        self.airspeed_norm = airspeed_norm
        self.bank_angle = bank_angle