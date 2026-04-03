import numpy as np
from gymnasium import Env


class AirplaneEnv(Env):
    """Base environment class for the airplane simulation."""

    metadata = {"render_modes": ["human", "ascii", "ansi"], "render_fps": 60}
    # TODO(gtorre): use render fps

    def __init__(self, airplane, render_mode=None):
        self.airplane = airplane

        self.visualiser = None
        self.render_mode = render_mode
        
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
            
        self.window = None
        self.clock = None
        self.np_random = np.random.default_rng()

    def seed(self, seed=None):
        """Set the seed for this environment's random number generator."""
        self.np_random = np.random.default_rng(seed)

    def render(self, mode: str | None = "ascii"):
        """
        Render the environment.

        :param mode: str, the mode to render with:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """
        if mode == "human":
            pass
            # if not self.visualiser:
            #     self.visualiser = Visualizer(self.airplane)
            # self.visualiser.plot()

        else:  # ANSI or ASCII
            # TODO: Add additional observations
            fpa_deg = np.rad2deg(self.airplane.flight_path_angle)
            print(f"\u001b[34m Flight Path Angle (deg): {fpa_deg:.2f}\u001b[37m")
            
            # TODO: Proper stall prediction
            if self.airplane.flight_path_angle > 0.7:
                print("\u001b[35m -- STALL --\u001b[37m")

    def close(self):
        """Close the environment and perform cleanup operations."""
        if self.window is not None:
            # close
            raise NotImplementedError