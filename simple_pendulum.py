import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def ode_derivs(phi: np.ndarray, t: float, g: float, length: float) -> np.ndarray:
    """
    d2/dt2 L theta (t) + (d2/dt2 X(t)) * cos(theta(t)) + (d2/dt2 * Y(t) + g) * sin(theta(t)) = 0

    solves ode for 2D-pendulum in x-z plane with moving support
    phi1 = theta
    phi2 = d theta/dt
    phi = [phi1, phi2]

    """
    # load results of prev iteration
    phi1, phi2 = phi[0], phi[1]

    # moving support accelerations
    x_acc = 1.0 * np.cos(1 * t) * 0.15
    y_acc = 1.0 * np.cos(0.1 * t) * 0
    forcing = 0

    # give derivatives for equation in x-z plane
    theta_dot = phi2
    theta_dot_dot = (- x_acc * np.cos(phi1) - (g + y_acc) * np.sin(phi1)) / length + forcing

    # give single derivatives vector as output
    derivatives = np.array([theta_dot, theta_dot_dot])

    return derivatives


class Pendulum:
    def __init__(self, length: float, g: float, dt: float, dur: float, init: np.ndarray):
        """
        Class that serves as wrapper for solving the non-linear equation of motion of a pendulum with moving base in 2D.

        Eq. of motion:

        L * theta'' + X'' * cos(theta) + (Y''+g) * sin(theta) = 0

        eom: theta
                   ^ y
                   |
                   0----> x
                 _______
                   | \
                   |  \
                   |   \
                L  | theta
                   |
                   |
                 [ M ]


        """
        self.length = length
        self.dt = dt
        self.dur = dur
        self.init = init
        self.g = g
        self.t = np.arange(0, self.dur, dt)
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
        sol_x: odeint solution for motion in x-z plane

        """
        self.sol = odeint(
            ode_derivs,
            self.init,
            self.t,
            args=(self.g, self.length),
            full_output=False
        )

        return self.sol


if __name__ == "__main__":
    # input parameters
    L = 15  # m
    dt = 0.01  # s
    dur = 50  # s
    init = np.array([0.16, -0.15])  # init
    gravity = 9.81
    pendulum = Pendulum(length=L, g=gravity, dt=dt, dur=dur, init=init)
    print(pendulum.eigen_period)
    sol = pendulum.solve()
    print(sol)
    plt.plot(pendulum.t, pendulum.sol[:, 0])
    plt.plot(pendulum.t, pendulum.sol[:, 1])



