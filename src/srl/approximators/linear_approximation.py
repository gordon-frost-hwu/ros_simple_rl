#! /usr/bin/python

import numpy as np
from copy import deepcopy
import time
from utilities.useful_math_functions import isclose

# import matplotlib.pyplot as plt
# plt.ion()
# fig = plt.figure(1)
# fig, ((ax_sv_features, ax_sv_weights), (ax_adv_features, ax_adv_weights), (ax_pol_features, ax_pol_weights)) = \
#     plt.subplots(nrows=3, ncols=2)
# ax_adv_features.set_title("adv_features")
# ax_adv_weights.set_title("adv_weights")
# ax_sv_features.set_title("sv_features")
# ax_sv_weights.set_title("sv_weights")
# ax_pol_features.set_title("pol_features")
# ax_pol_weights.set_title("pol_weights")

def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer

class LinearApprox(object):
    def __init__(self, config, basis_functions):
        self.name = "LinearApprox"
        self.alpha = config["alpha"]
        self._minimize = config["minimise"]
        self.compute_features = basis_functions.computeFeatures
        self.num_features_per_dim = basis_functions.resolution
        self.num_input_dims = config["num_input_dims"]
        self.initial_value = config["initial_value"]
        # self.num_features = len(self.compute_features([0.5 for i in range(self.num_input_dims)]))
        self.lower_bound, self.upper_bound = config["approximator_boundaries"]
        self.weights = None
        self.drawn_stuff_to_remove = []
        self.h = 0.001
        self.plotFeatures = False
    def setParams(self, weights):
        self.setWeights(weights)
    def getParams(self):
        return self.weights

    def setWeights(self, weights_list):
        self.weights = weights_list
        # print("weights set to: {0}".format(weights_list))

    def decayAlpha(self, decay_by_amount):
        if self.alpha - decay_by_amount > 0.0:
            self.alpha -= decay_by_amount

    def computeOutput(self, state_features):
        # If the number of state dimensions given is less than the configuration dict says there should be, add some
        # some zeroes. This padding is to allow the same state and weight vector dimensions for state value and action
        # if len(state_raw) < self.num_input_dims:
        #     state_raw.extend([0.0 for i in range(self.num_input_dims - len(state_raw))])
        # Extend state_raw again if there are additional features to be used (e.g. for the advantage fnct approximation
        # global ax_sv_features, ax_sv_weights, ax_adv_features, ax_adv_weights
        # if self.drawn_stuff_to_remove is not None:
        #     if len(self.drawn_stuff_to_remove) == 1:
        #         for item in self.drawn_stuff_to_remove:
        #             item.remove()
        #         self.drawn_stuff_to_remove = []
        # drawn_feature, = ax_adv_features.plot(state_features, "k"); self.drawn_stuff_to_remove.append(drawn_feature)
        # plt.draw()

        # Initialise the weights of the approximator now that we know the size of the feature vector
        if self.weights is None:
            # _features = self.compute_features([0.5 for i in range(self.num_input_dims)])
            avg = self.initial_value / sum(state_features)
            print("avg for weights: {0}".format(avg))
            self.weights = np.array([avg for i in range(len(state_features))])

        # print("{0}->feature size: {1}".format(self.name, state_features.shape))
        # print("{0}->weights size: {1}".format(self.name, self.weights.shape))

        # print("size of state_features: {0}".format(len(state_features)))
        # print("size of weights: {0}".format(len(self.weights)))
        return np.dot(state_features, self.weights)

    def updateWeights(self, features_for_update, gradient_vector=None):
        # Bound the TD Error and reverse it. Why it needs to be inverted, who knows!

        # If the number of state dimensions given is less than the configuration dict says there should be, add some
        # some zeroes. This padding is to allow the same state and weight vector dimensions for state value and action
        # if len(state_raw) < self.num_input_dims:
        #     state_raw.extend([0.0 for i in range(self.num_input_dims - len(state_raw))])

        # print("Updating ACTOR --------------Mwhahahahahah")
        feature_vector = gradient_vector

        # if self.plotFeatures:
        #     global ax_pol_features, ax_pol_weights
        #     if self.drawn_stuff_to_remove is not None:
        #         for item in self.drawn_stuff_to_remove:
        #             item.remove()
        #         self.drawn_stuff_to_remove = []
        #     drawn_feature, = ax_pol_features.plot(gradient_vector, "k"); self.drawn_stuff_to_remove.append(drawn_feature)
        #     drawn_weights, = ax_pol_weights.plot(self.weights, "k"); self.drawn_stuff_to_remove.append(drawn_weights)
        #     plt.draw()

        # print("updateWeights-> shape of weights: {0}".format(self.weights.shape))
        # print("updateWeights-> shape of feature: {0}".format(feature_vector.shape))
        # check that the update requested does not put the Approximators output outside it's bounds
        old_weights = deepcopy(self.weights)

        if self._minimize:
            tmp_updated_weights = old_weights - self.alpha * feature_vector
        else:
            tmp_updated_weights = old_weights + self.alpha * feature_vector

        output_tmp = np.dot(tmp_updated_weights, features_for_update)
        output = np.dot(self.weights, features_for_update)
        # print("updateParameters: {0} before tmp update: {1}".format(self.name, output))
        # print("updateParameters: {0} after tmp update: {1}".format(self.name, output_tmp))

        # When using the natural gradient vector, you don't use the TD Error in the weight update
        if (output_tmp > output and output_tmp < self.upper_bound) or (output_tmp < output and output_tmp > self.lower_bound):
            # print("updating weights IDEAL")
            if self._minimize:
                self.weights -= self.alpha * feature_vector
            else:
                self.weights += self.alpha * feature_vector
        elif (output_tmp < self.lower_bound and output_tmp > output) or (output_tmp > self.upper_bound and output_tmp < output):
            if self._minimize:
                self.weights -= self.alpha * feature_vector
            else:
                self.weights += self.alpha * feature_vector
                # print("updating weights TOWARDS RANGE")
        else:
            # print("NOT UPDATING as out of output range")
            pass

    def calculateGradient(self, state=None, basis_functions=None):
        """
        :return: the gradient of the output of the approximator wrt to its parameters
        """
        orig_parameter_vector = deepcopy(self.weights)
        # orig_feature_vector = basis_functions.computeFeatures(state)
        if state is None:
            print("Must provide state variable to calculateGradient method!!!!")
            exit(0)
        else:
            state_for_gradient_calc = list(state)
        orig_output = self.computeOutput(state_for_gradient_calc)

        # preallocate list of correct size for performance gains opposed to append metho
        gradient = [None] * len(state_for_gradient_calc)
        # print("max feature -----------: {0}".format(max(state_for_gradient_calc)))
        # print("state_for_gradient_calc: {0}".format(state_for_gradient_calc))
        # active_feature_idxs = [state_for_gradient_calc.index(feature) for feature in state_for_gradient_calc
        #                        if not isclose(feature, 0.0, abs_tol=0.0001)]

        for idx in range(len(orig_parameter_vector)):
            # if idx in active_feature_idxs:
            new_param_vector = deepcopy(orig_parameter_vector)
            new_param_vector[idx] += self.h

            self.setWeights(new_param_vector)

            new_output = self.computeOutput(state_for_gradient_calc)
            gradient[idx] = (new_output - orig_output) / self.h
            # else:
            #     gradient.append(0.0)

        # Reset the weights to what they were before the gradient calculation
        self.setWeights(orig_parameter_vector)
        return np.array(gradient)

if __name__=='__main__':
    from srl.basis_functions.simple_basis_functions import RBFBasisFunctions as BasisFunctions

    actor_config = {
        "approximator_name": "policy",
        "initial_value": 0.0,
        "alpha": 0.01,
        "random_weights": False,
        "minimise": False,
        "approximator_boundaries": [-200.0, 200.0],
        "num_input_dims": 4,
        # "basis_functions": Fourier(FourierDomain(), order=5)
        # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
        "basis_functions": BasisFunctions()
        }
    # Profiling procedure
    NUM_CALCS = 5
    lfa = LinearApprox(actor_config)
    basis = actor_config["basis_functions"]
    start_time = time.time()
    print("Start Time: {0}".format(start_time))
    for i in xrange(NUM_CALCS):
        lfa.computeOutput(basis.computeFeatures([0.0, 0.0, 0.0, 0.0]))
        lfa.calculateGradient(basis.computeFeatures([0.0, 0.0, 0.0, 0.0]))
        time.sleep(1.0/40.0)
    print("40 Hz calcs took: {0}".format(time.time() - start_time))
    print("Avg calc time: {0}".format((time.time() - start_time) / NUM_CALCS))