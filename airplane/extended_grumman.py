from airplane.grumman import Grumman


class ExtendedGrumman(Grumman):
    ####################################
    ### Grumman American AA-1 Yankee ###
    ####################################
    """Class for complete airplane state and dynamics."""

    def __init__(self):
        super().__init__()
        ##########################
        ### Airplane variables ###
        ##########################
        self.flight_path_angle = 0.0  # Flight path Angle (γ) [rad]
        self.airspeed = self.STALL_AIRSPEED  # Air Speed (V) [m/s]
        self.airspeed_norm = 1.0  # Air Speed (V/Vs) [1]
        self.alpha = 0.0  # Angle of attack (α) [rad]
        self.beta = 0.0  
        self.bank_angle = 0.0  
        self.roll_rate = 0.0  
        self.pitch_rate = 0.0  # Pitch rate (q) [rad/s]
        self.yaw_rate = 0.0  

        #######################
        ### Airplane forces ###
        #######################
        self.lift_force = None
        self.drag_force = None
        self.side_force = None
        self.thrust_force = None
        self.rolling_moment = None
        self.pitching_moment = None
        self.yawing_moment = None

        # Previous commands
        self.last_elevator = 0.0
        self.last_aileron = 0.0
        self.last_throttle = 0.0
        self.last_rudder = 0.0

    def _command_airplane(self, elevator, aileron, throttle, rudder):
        """Process control inputs and update the airplane state derivatives."""
        self.last_elevator = elevator
        self.last_aileron = aileron
        self.last_throttle = throttle
        self.last_rudder = rudder

        self.pre_hook(elevator, aileron, throttle, rudder) 
        
        flight_path_angle_dot = self.compute_flight_path_angle_dot()
        airspeed_dot = self.compute_airspeed_dot()
        alpha_dot = self.compute_alpha_dot()
        beta_dot = self.compute_beta_dot()
        bank_angle_dot = self.compute_bank_angle_dot()
        roll_rate_dot = self.compute_roll_rate_dot()
        pitch_rate_dot = self.compute_pitch_rate_dot()
        yaw_rate_dot = self.compute_yaw_rate_dot()

        self.flight_path_angle = self._update_state_from_derivative(
            self.flight_path_angle, flight_path_angle_dot
        )
        self.airspeed = self._update_state_from_derivative(
            self.airspeed, airspeed_dot
        )
        self.airspeed_norm = self.airspeed / self.STALL_AIRSPEED
        
        self.alpha = self._update_state_from_derivative(self.alpha, alpha_dot)
        self.beta = self._update_state_from_derivative(self.beta, beta_dot)
        
        self.bank_angle = self._update_state_from_derivative(
            self.bank_angle, bank_angle_dot
        )
        self.roll_rate = self._update_state_from_derivative(
            self.roll_rate, roll_rate_dot
        )
        self.pitch_rate = self._update_state_from_derivative(
            self.pitch_rate, pitch_rate_dot
        )
        self.yaw_rate = self._update_state_from_derivative(
            self.yaw_rate, yaw_rate_dot
        )

    def pre_hook(self, elevator, aileron, throttle, rudder):
        """Execute logic before applying control commands."""
        pass

    def compute_flight_path_angle_dot(self):
        """Compute the derivative of the flight path angle."""
        return 0

    def compute_airspeed_dot(self):
        """Compute the derivative of the airspeed."""
        return 0

    def compute_alpha_dot(self):
        """Compute the derivative of the angle of attack."""
        return 0

    def compute_beta_dot(self):
        """Compute the derivative of the sideslip angle."""
        return 0

    def compute_bank_angle_dot(self):
        """Compute the derivative of the bank angle."""
        return 0

    def compute_roll_rate_dot(self):
        """Compute the derivative of the roll rate."""
        return 0

    def compute_pitch_rate_dot(self):
        """Compute the derivative of the pitch rate."""
        return 0

    def compute_yaw_rate_dot(self):
        """Compute the derivative of the yaw rate."""
        return 0

    def reset(
        self, 
        flight_path_angle, 
        airspeed_norm, 
        alpha, 
        beta, 
        bank_angle, 
        roll_rate, 
        pitch_rate, 
        yaw_rate
    ):
        """Reset the airplane to a specified initial state."""
        self.flight_path_angle = flight_path_angle
        self.airspeed_norm = airspeed_norm
        self.airspeed = self.airspeed_norm * self.STALL_AIRSPEED
        self.alpha = alpha
        self.beta = beta
        self.bank_angle = bank_angle
        self.roll_rate = roll_rate
        self.pitch_rate = pitch_rate
        self.yaw_rate = yaw_rate