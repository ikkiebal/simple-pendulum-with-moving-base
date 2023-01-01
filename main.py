import numpy as np
import pandas as pd
from pendulum import ForcingData, Pendulum

# input parameters
L = 5  # m
dt = 0.1  # s
dur = 190  # s
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
fig_animation = pendulum.animate(pendulum.sol, animation_time_scaler=4)

print(pendulum.eigen_period)