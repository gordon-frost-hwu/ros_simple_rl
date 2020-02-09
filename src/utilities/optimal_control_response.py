#! /usr/bin/env python
import numpy as np


def optimal_control_response():
    response = np.zeros([350, 2])
    for step in range(350):
        response[step, :] = [step, func(step)]
    return response

def func(x):
    amplitude = 0.55  # np.pi / 2.0
    y_offset = -0.55  # -(np.pi / 2.0)
    result = (amplitude * (1 / (1 + np.exp(-0.07 * (x - 40))))) + y_offset
    return result
