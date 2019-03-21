""" Class which takes as input a value which is to be scaled to within a given range
    The scaling can be done statically if the output range is given in instantiation,
    or dynamically if it is not
     author: Gordon Frost; gwf2@hw.ac.uk
     date: 16/02/15
"""
import numpy as np
LOWER_IDX = 0
UPPER_IDX = 1

class DynamicNormalizer(object):
    dynamic = False
    def __init__(self, input_range=[0.0, 1.0], output_range=[0.0, 1.0]):
        self.last_value = None
        self.prev_value = None  # only required if variable statistics are desired (run/calculated in another thread?)

        # Convert to floats to make sure of float division
        if input_range is not None:
            input_range = [float(i) for i in input_range]
            output_range = [float(o) for o in output_range]
        self.out_limits = np.array(output_range)
        self.in_limits = np.array(input_range)

        if input_range is None:
            self.dynamic = True
            self.in_limits = np.array([0.0, 0.0])

    def scale_value(self, value):
        value = float(value)    # make sure that input value is converted to float for calculations
        if self.dynamic:
            # update the input range limits according to whether the
            if value < self.in_limits[LOWER_IDX]:
                self.in_limits[LOWER_IDX] = value
            elif value > self.in_limits[UPPER_IDX]:
                self.in_limits[UPPER_IDX] = value

        # map the values to range [0, 1]
        scaled_value = ((value - self.in_limits[LOWER_IDX]) /
                        (self.in_limits[UPPER_IDX] - self.in_limits[LOWER_IDX]))  # - 1/2 to map to range [-0.5, 0.5]
        # then map from range [0, 1] (above) to output range, self.out_limits
        scaled_value = (self.out_limits[LOWER_IDX]) + (scaled_value * (self.out_limits[UPPER_IDX] - self.out_limits[LOWER_IDX]))

        # Saturate the variable to the outer bounds
        scaled_value = np.clip(scaled_value, self.out_limits[LOWER_IDX], self.out_limits[UPPER_IDX])

        return scaled_value

    def scale_list(self, lst):
        return [self.scale_value(value) for value in lst]


## ---------- Testing ---------------- ##
if __name__ == '__main__':

    normalizer = DynamicNormalizer(None, [-1, 4])

    inputs = [-5, -1, 0, 1, 5, 2, 3, 1, 0]

    returns = [normalizer.scale_value(i) for i in inputs]

    print(returns)