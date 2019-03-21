#! /usr/bin/env python

# ---------------------
# Author: Gordon Frost
# Date: 09/09/15
# Purpose: Sliding window class and another that differentiates a variable against the sample number. i.e. rate of
#          change w.r.t.time
# ---------------------

import sys
import numpy as np
import matplotlib.pyplot as plt

from collections import deque


class SlidingWindow(object):
    def __init__(self, window_size):
        """ Create a FIFO optimized queue with a max size. Upon append, elements are shifted left by 1 """
        self.first_val = True
        self._window_size = window_size
        self._reset()
    def _reset(self):
        # Initialize or reset the current window
        self.window = deque([0.0 for i in range(self._window_size)], maxlen=self._window_size)
        self.first_val = True
    def getWindow(self, val):
        """ return: current window of length, window_size """
        if self.first_val:
            for i in range(self._window_size):
                self.window.append(val)
            self.first_val = False
        else:
            self.window.append(val)
        return list(self.window)

class DifferentiateVariable(object):
    """ Class which is used for differientating, i.e. rate of change, of a real valued variable. The window size
        determines how many samples in the past the gradient is taken by (taken as average over window size). Therefore,
        aliasing comes into affect the larger the window size
    """
    _print_debug_statements = False
    def __init__(self, window_size=2):
        self.sliding_window = SlidingWindow(window_size=window_size)
    def print_if_enabled(self, to_print):
        if self._print_debug_statements:
            print(to_print)
        else:
            pass
    def diffs_for_window(self, window):
        """ Calculate the difference in the latest value compared to the previous one through a list
         returns a list of length window minus one"""
        # return [window[window.index(item) + 1] - item for item in window if item != window[-1]]
        return [j-i for i, j in zip(window[:-1], window[1:])]

    def gradient_in_list(self, vals):
        gradients = []
        for val in vals:
            window = self.sliding_window.getWindow(val)
            self.print_if_enabled("DifferentiateVariable: Window - {0}".format(window))
            diffs_in_window = self.diffs_for_window(window)
            if len(diffs_in_window) != 0:
                gradients.append(float(sum(diffs_in_window) / len(diffs_in_window)))
            else:
                gradients.append(0.0)
        return gradients


    def gradient(self, val):
        # Calculate the rate of change of the variables newest value given the window_size number of samples in the past
        window = self.sliding_window.getWindow(val)
        self.print_if_enabled("DifferentiateVariable: Window - {0}".format(window))
        diffs_in_window = self.diffs_for_window(window)
        self.print_if_enabled("DifferentiateVariable: Diffs  - {0}".format(diffs_in_window))
        if len(diffs_in_window) != 0:
            return float(sum(diffs_in_window) / len(diffs_in_window))
        return 0.0

    def reset_window(self):
        self.sliding_window._reset()

if __name__ == '__main__':
    x = np.linspace(0, 2*np.pi)
    print(len(x))
    f = np.cos(x) + np.sin(2*x)

    plt.hold(True)
    plt.figure(1)
    plt.plot(f, "b")

    win_sizes = [2, 10, 20]
    for win_size in win_sizes:
        dv = DifferentiateVariable(window_size=win_size)
        diffs = []
        for xx in f:
            g = dv.gradient(xx)
            diffs.append(g)
        print("Diffs: {0}".format(diffs))
        plt.plot(diffs, "r")

    plt.grid()
    plt.show()