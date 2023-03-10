from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from exceptions import InvalidInputForcing


@dataclass
class ForcingData:
    """"
    Parameters
    ----------
    t_motion_base : np.ndarray
        time array of base motions

    x_motion_base : np.ndarray
        x-direction array of base motions

    y_motion_base : np.ndarray
        y-direction array of base motions

    external_forcing : np.ndarray
        external forcing applied on the pendulum (NOT IMPLEMENTED)

    """
    t_motion_base: np.ndarray
    x_motion_base: np.ndarray
    y_motion_base: np.ndarray
    external_forcing: np.ndarray


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

        Parameters
        ----------
        length: float
            length of pendulum

        g: float
            gravitational acceleration

        time_step: float
            time step for solution array

        duration: float
            duration of simulation

        init: np.ndarray,
            initial conditions of the angle and angular velocity

        forcing: ForcingData
            data class containing motions of the pendulum base and external forcing on the pendulum
        """

        self.length = length
        self.time_step = time_step
        self.duration = duration
        self.init = init
        self.g = g
        self.t = np.arange(0, self.duration, self.time_step)
        self.forcing = forcing
        self.sol = None

        if self.duration >= self.forcing.t_motion_base[-1]:
            raise InvalidInputForcing("Duration of forcing is shorter than given duration of simulation!"
                                      " No forcing can be applied.")

    @property
    def eigen_period(self) -> float:
        """"
        Eigen period of the pendulum [s]
        """
        return 2 * np.pi * np.sqrt(self.g / self.length)

    @property
    def eigen_frequency(self) -> float:
        """"
        Eigen frequency of the pendulum in [rad/s]
        """
        return np.sqrt(self.length / self.g)

    @staticmethod
    def ode_derivatives(dydt: np.ndarray, t: float, g: float, length: float,
                        x_acceleration_func: interp1d, y_acceleration_func: interp1d) -> np.ndarray:
        """
        Solves ode for 2D-pendulum in x-y plane with moving support. State-space solution for the equation of motion:

        d2/dt2 L theta (t) + (d2/dt2 X(t)) * cos(theta(t)) + (d2/dt2 * Y(t) + g) * sin(theta(t)) = 0

        phi = theta
        phi_dot = d theta/dt
        phi = [phi, phi_dot]

        Parameters
        ----------
        dydt: np.ndarray
            Array of the angle and angular velocity for each solver time-step.

        t: float
            Solver time-step.

        g: float
            Gravitational acceleration.

        length: float
            Length of the pendulum.

        x_acceleration_func: interp1d,
            Interpolation function of the x-accelerations of the pendulum base.

        y_acceleration_func: interp1d
            Interpolation function of the y-accelerations of the pendulum base.

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
        Method that interpolates the motion of the base and derives it w.r.t time twice to obtain the accelerations at
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
        Method that solves the differential equations
        Returns
        -------
        sol: odeint solution

        """

        # get accelerations of the base of the pendulum
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

    def animate(self, sol: np.array, animation_time_scaler: int = 10) -> go.Figure:

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
            "data": [go.Scatter(x=[x_motion(0), mass_x[0]], y=[y_motion(0), mass_y[0]])],
            "layout": {},
            "frames": [
                    go.Frame(data=go.Scatter(x=[resampled_x_base[k], resampled_x[k]], y=[resampled_y_base[k],
                                                                                         resampled_y[k]]),
                             name=str(k)) for k in range(nb_frames)
                ]
        }

        # fill in most of layout
        fig_dict["layout"]["xaxis"] = dict(range=[np.min(mass_x)-1, np.max(mass_x)+1], autorange=False)
        fig_dict["layout"]["yaxis"] = dict(range=[np.min(mass_y)-1, 1], autorange=False, scaleanchor="x", scaleratio=1)
        fig_dict["layout"]["title"] = "Pendulum with moving base"
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

        # make slider values
        for i, t_step in enumerate(resampled_time):
            slider_step = {"args": [
                [i],
                {"frame": {"duration": 1, "redraw": False},
                 "mode": "immediate"}
            ],
                "label": np.round(t_step, 3),
                "method": "animate"}
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]
        fig = go.Figure(fig_dict)
        fig.show()

        return fig