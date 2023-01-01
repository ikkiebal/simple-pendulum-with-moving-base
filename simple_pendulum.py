from typing import Tuple

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def ode_derivatives(dydt: np.ndarray, t: float, g: float, length: float) -> np.ndarray:
    """
    d2/dt2 L theta (t) + (d2/dt2 X(t)) * cos(theta(t)) + (d2/dt2 * Y(t) + g) * sin(theta(t)) = 0

    solves ode for 2D-pendulum in x-z plane with moving support
    phi = theta
    phi_dot = d theta/dt
    phi = [phi, phi_dot]

    """
    # load results of prev iteration
    phi, phi_dot = dydt[0], dydt[1]

    # moving support accelerations
    x_acc = 1.0 * np.cos(1 * t) * 0.15
    y_acc = 1.0 * np.cos(0.1 * t) * 0
    forcing = 0

    # give derivatives for equation in x-z plane
    theta_dot = phi_dot
    theta_dot_dot = (- x_acc * np.cos(phi) - (g + y_acc) * np.sin(phi)) / (length * np.cos(phi)) + forcing

    # give single derivatives vector as output
    derivatives = np.array([theta_dot, theta_dot_dot])

    return derivatives


class Pendulum:

    def __init__(self, length: float, g: float, time_step: float, duration: float, init: np.ndarray):
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

    @property
    def eigen_period(self) -> float:
        return 2 * np.pi * np.sqrt(self.g / self.length)  # s

    @property
    def eigen_frequency(self) -> float:
        return np.sqrt(self.length / self.g)  # rad/s

    def solve(self):
        """
        method that solves the differential equations
        Returns
        -------
        sol: odeint solution

        """
        self.sol = odeint(
            ode_derivatives,
            self.init,
            self.t,
            args=(self.g, self.length),
            full_output=False
        )

        return self.sol

    def plot_results(self):
        plt.figure()
        plt.plot(self.t, self.sol[:, 0])
        plt.plot(self.t, self.sol[:, 1])
        plt.grid()

    @staticmethod
    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    def interpolate(self) -> Tuple[np.array, np.array, np.array]:
        """
        method that interpolates the motion of the base and derives it w.r.t time twice to obtain the accelerations at
        the base
        Returns
        -------
        x_acc: interpolation of the x-accelerations
        y_acc: interpolation of the y-accelerations
        z_acc: interpolation of the z-accelerations

        """
        # differentiate twice
        dt_data = self.t_motion[1]-self.t_motion[0]
        x_acc_data = np.gradient(np.gradient(self.x_motion, dt_data), dt_data)
        y_acc_data = np.gradient(np.gradient(self.y_motion, dt_data), dt_data)
        z_acc_data = np.gradient(np.gradient(self.z_motion, dt_data), dt_data)

        # signal becomes shorter 2 time steps because of differentiation
        x_acc = interp1d(self.t_motion, x_acc_data)
        y_acc = interp1d(self.t_motion, y_acc_data)
        z_acc = interp1d(self.t_motion, z_acc_data)

        return x_acc, y_acc, z_acc


    def animate(self, sol: np.array):
        theta, theta_dot = sol[:, 0], sol[:, 1]
        mass_x = np.sin(theta)*self.length
        mass_y = -self.length*np.cos(theta)

        nb_frames = int(self.duration/self.time_step)

        fig = go.Figure(
            data=go.Scatter(x=[0, mass_x[0]], y=[0, mass_y[0]]),
            layout=go.Layout(
                xaxis=dict(range=[np.min(mass_x)-1, np.max(mass_x)+1], autorange=False),
                yaxis=dict(range=[np.min(mass_y)-1, 1], autorange=False, scaleanchor="x", scaleratio=1),
                title="Pendulum with Moving Base",
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, self.frame_args(1)],
                                "label": "&#9654;",  # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], self.frame_args(0)],
                                "label": "&#9724;",  # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ]),
            frames=[
                go.Frame(data=go.Scatter(x=[0.0, mass_x[k]], y=[0.0, mass_y[k]]), name=str(k)) for k in range(nb_frames)
            ]
        )

        fig.show()

        return fig


if __name__ == "__main__":
    # input parameters
    L = 3  # m
    dt = 0.01  # s
    dur = 50  # s
    init = np.array([0.5, 0])  # init
    gravity = 9.81
    pendulum = Pendulum(length=L, g=gravity, time_step=dt, duration=dur, init=init)
    pendulum.solve()
    pendulum.animate(pendulum.sol)




