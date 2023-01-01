from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import plotly
import plotly.express as px

@dataclass
class ForcingData:
    t_motion_base: np.array
    x_motion_base: np.array
    y_motion_base: np.array
    external_forcing: np.array


class Pendulum:

    def __init__(self, length: float, g: float, time_step: float, duration: float, init: np.ndarray,
                 forcing: ForcingData):
        """
        Class that serves as wrapper for solving the non-linear equation of motion of a pendulum with moving base in 2D.

        Eq. of motion:

        L * cos(phi) * phi'' + X'' * cos(phi) + (Y''+g) * sin(phi) = 0

        eom: theta
                   ^ y
                   |
                   0----> x
                 _______
                   | \
                   |  \
                   |   \
                   |    \ (L)
                   |(phi)\
                   |      \
                   |     [ M ]
                   |


        """
        self.length = length
        self.time_step = time_step
        self.duration = duration
        self.init = init
        self.g = g
        self.t = np.arange(0, self.duration, dt)
        self.sol = None
        self.forcing = forcing

    @property
    def eigen_period(self) -> float:
        """"
        Eigenperiod of the pendulum [s]
        """
        return 2 * np.pi * np.sqrt(self.g / self.length)

    @property
    def eigen_frequency(self) -> float:
        """"
        Eigenfrequency of the pendulum in [rad/s]
        """
        return np.sqrt(self.length / self.g)

    @staticmethod
    def ode_derivatives(dydt: np.ndarray, t: float, g: float, length: float,
                        x_acceleration_func: interp1d, y_acceleration_func: interp1d) -> np.ndarray:
        """
        d2/dt2 L theta (t) + (d2/dt2 X(t)) * cos(theta(t)) + (d2/dt2 * Y(t) + g) * sin(theta(t)) = 0

        solves ode for 2D-pendulum in x-y plane with moving support
        phi = theta
        phi_dot = d theta/dt
        phi = [phi, phi_dot]

        """
        # load results of prev iteration
        phi, phi_dot = dydt[0], dydt[1]

        # moving support accelerations
        x_acc_t = x_acceleration_func(t)
        y_acc_t = y_acceleration_func(t)

        # external forcing
        forcing = 0

        # give derivatives for equation in x-y plane
        theta_dot = phi_dot
        theta_dot_dot = (- x_acc_t * np.cos(phi) - (g + y_acc_t) * np.sin(phi)) / (length * np.cos(phi)) + forcing

        # give single derivatives vector as output
        derivatives = np.array([theta_dot, theta_dot_dot])

        return derivatives

    def interpolate_forcing(self) -> Tuple[np.array, np.array]:
        """
        method that interpolates the motion of the base and derives it w.r.t time twice to obtain the accelerations at
        the base
        Returns
        -------
        x_acc: interpolation of the x-accelerations
        y_acc: interpolation of the y-accelerations

        """
        # differentiate twice
        dt_data = self.forcing.t_motion_base[1]-self.forcing.t_motion_base[0]
        x_acc_data = np.gradient(np.gradient(self.forcing.x_motion_base, dt_data), dt_data)
        y_acc_data = np.gradient(np.gradient(self.forcing.y_motion_base, dt_data), dt_data)

        # signal becomes shorter 2 time steps because of differentiation
        x_acc = interp1d(self.forcing.t_motion_base, x_acc_data)
        y_acc = interp1d(self.forcing.t_motion_base, y_acc_data)

        return x_acc, y_acc

    def solve(self) -> np.ndarray:
        """
        method that solves the differential equations
        Returns
        -------
        sol: odeint solution

        """

        x_accelerations, y_accelerations = self.interpolate_forcing()

        self.sol = np.array(
            odeint(
                self.ode_derivatives,
                self.init,
                self.t,
                args=(self.g, self.length, x_accelerations, y_accelerations),
                full_output=False)
        )

        return self.sol

    def animate(self, sol: np.array, animation_time_scaler: float = 10) -> go.Figure:

        """
        Method that plots an animation of the pendulum motion with the plotly library

        Returns
        -------
        fig: go.Figure() instance

        """

        theta, theta_dot = sol[:, 0], sol[:, 1]

        x_motion = interp1d(self.forcing.t_motion_base, self.forcing.x_motion_base)
        y_motion = interp1d(self.forcing.t_motion_base, self.forcing.y_motion_base)

        mass_x = self.length * np.sin(theta) + x_motion(self.t)
        mass_y = -self.length * np.cos(theta) + y_motion(self.t)

        # resample results at specified frame rate
        resampled_time = self.t[0::animation_time_scaler]
        nb_frames = len(resampled_time)
        resampled_x_base = x_motion(resampled_time)
        resampled_y_base = y_motion(resampled_time)
        resampled_x = mass_x[0::animation_time_scaler]
        resampled_y = mass_y[0::animation_time_scaler]

        # make figure
        fig_dict = {
            "data":[go.Scatter(x=[x_motion(0), mass_x[0]], y=[y_motion(0), mass_y[0]])],
            "layout": {},
            "frames": [
                    go.Frame(data=go.Scatter(x=[resampled_x_base[k], resampled_x[k]], y=[resampled_y_base[k], resampled_y[k]]),
                             name=str(k)) for k in range(nb_frames)
                ]
        }

        # fill in most of layout
        fig_dict["layout"]["xaxis"] = dict(range=[np.min(mass_x)-1, np.max(mass_x)+1], autorange=False)
        fig_dict["layout"]["yaxis"] = dict(range=[np.min(mass_y)-1, 1], autorange=False, scaleanchor="x", scaleratio=1)
        fig_dict["layout"]["updatemenus"] = [
                    {
                        "buttons":
                            [
                             {"args": [None, {"frame": {"duration": 1}, "mode": "immediate",
                                              "fromcurrent": True}
                                       ],
                              "label": "Play",
                              "method": "animate"},
                             {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate",
                                                "fromcurrent": True}
                                       ],
                              "label": "Pause",
                              "method": "animate"}
                            ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0
                    }
                ]

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Time:",
                "visible": True,
                "xanchor": "right"
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }

        # make frames
        for i, t_step in enumerate(resampled_time):
            slider_step = {"args": [
                [i],
                {"frame": {"duration": 1, "redraw": False},
                 "mode": "immediate"}
            ],
                "label": np.round(t_step,3),
                "method": "animate"}
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]
        fig = go.Figure(fig_dict)

        fig.show()

        return fig


if __name__ == "__main__":
    # input parameters
    L = 5  # m
    dt = 0.1  # s
    dur = 150  # s
    initial_conditions = np.array([0.2, 0])  # init
    gravity = 9.81

    motion_data = pd.read_csv("sample_data_1.csv")

    # forcing parameters
    forcing_parameters = ForcingData(
        t_motion_base=np.array(motion_data["Time (s)"]),
        x_motion_base=np.array(motion_data["X (m)"]),
        y_motion_base=np.array(motion_data["Y (m)"]),
        external_forcing=np.zeros_like(np.array(motion_data["Y (m)"]))
    )

    pendulum = Pendulum(length=L, g=gravity, time_step=dt, duration=dur, init=initial_conditions,
                        forcing=forcing_parameters)
    pendulum.solve()
    fig_animation = pendulum.animate(pendulum.sol, animation_time_scaler=3)
