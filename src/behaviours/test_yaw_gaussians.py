#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from random import randint

plt.ion()

GAUSSIAN_VARIANCES = [0.1, 0.6, 1.0, 2.0, 3.0]

def gaussian1D(x, x0, mu, sigma):
    """
    :param x: north position
    :param x0: centre of Normal distribution in North
    :param mu: Height of normal distribution
    :param sigma: variance
    :return: value of point x wrt normal distribution centred on x0
    """
    x_term = (x - x0)**2 / (2 * sigma**2)
    return (mu * np.exp(-x_term))

def func(step, variance):
    if variance > 0.0:
        desired_velocity = 1 - gaussian1D(step, 0.0, 1.0, variance)
        if step > 0.0:
            return -desired_velocity
        else:
            return desired_velocity
    elif variance < 0.0:
        desired_velocity = -(1 - gaussian1D(step, 0.0, 1.0, variance))
        if step > 0.0:
            return -desired_velocity
        else:
            return desired_velocity

color_switch = 0
for variance in GAUSSIAN_VARIANCES:
    interval = np.linspace(-3.14, 3.14, 100)

    res = [func(step, variance) for step in interval]
    plt.plot(interval, res)
    plt.hold(True)
    # if color_switch == 0:
    #     plt.plot(interval, res, 'r')
    # elif color_switch == 1:
    #     plt.plot(interval, res, 'g')

    color_switch += 1

plt.xlabel("angle_to_goal_wrt_vehicle_heading")
plt.ylabel("velocity")
plt.legend(["0.1", "0.6", "1.0", "2.0", "3.0"])
plt.show()

input("Enter something ...")