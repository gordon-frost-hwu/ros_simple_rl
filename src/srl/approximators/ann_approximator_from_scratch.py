#! /usr/bin/python

# Source modified for RL from: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import numpy as np
from math import exp
from copy import deepcopy
from variable_normalizer import DynamicNormalizer

scale = DynamicNormalizer(input_range=[0.0, 1.0], output_range=[-1.0, 1.0])
# Sigmoid activation function for hidden layer nuerons
def sigmoid(X):
    R = []
    for x in X[0]:
        R.append(scale.scale_value(1.0/(1.0+exp(-x))))
    return np.array(R)

class ANNApproximator(object):
    def __init__(self, num_input_dim, nn_hdim, hlayer_activation_func="tanh"):
        self.name = "scratch_ann"
        self.nn_input_dim = num_input_dim # input layer dimensionality
        self.nn_output_dim = 1 # output layer dimensionality

        # Gradient descent parameters (I picked these by hand)
        self.alpha = 0.01 # learning rate for gradient descent
        self.h = 0.001
        self.reg_lambda = 0.01 # regularization strength
        self.build_model(nn_hdim, hlayer_activation_func)
        self.last_input = None
        self.num_params = None
        self.params_minus_one = self.getParams()
        self.params_plus_one = None

    def build_model(self, nn_hdim, hlayer_activation_func):

        # Initialize the parameters to random values. We need to learn these.
        # print("nn_input_dim: {0}".format(self.nn_input_dim))
        # print("nn_hdim: {0}".format(nn_hdim))

        self.W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, nn_hdim))
        self.W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        self.b2 = np.zeros((1, self.nn_output_dim))

        if hlayer_activation_func == "tanh":
            self.hlayer_activation_func = np.tanh
        elif hlayer_activation_func == "sigmoid":
            self.hlayer_activation_func = sigmoid

        # This is what we return at the end
        # Assign new parameters to the model
        # model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        # return model
    def printParams(self, param="All"):
        if param == "All":
            params = [self.W1, self.b1, self.W2, self.b2]
            for param in params:
                print(param)
                print(param.flatten().shape)
        else:
            pass

    def computeOutputThetaMinusOne(self, inpt):
        params_t = self.getParams()
        self.setParams(self.params_minus_one)
        res = self.computeOutput(inpt)
        self.setParams(params_t)
        return res

    def computeOutput(self, inpt):
        """
        :param inpt: must be a (1, N) np array/vector
        :return:
        """
        # print("inpt shape: {0}".format(inpt))
        # print("inpt weights shape: {0}".format(self.W1.shape))
        inpt = np.array(inpt)
        # Forward propagation
        z1 = inpt.dot(self.W1) + self.b1
        a1 = self.hlayer_activation_func(z1)    # <---hidden layer activation function
        z2 = a1.dot(self.W2) + self.b2
        self.last_input = inpt
        return z2[0][0]

        # Backpropagation
        # delta3 = probs
        # delta3[range(num_examples), y] -= 1
        # dW2 = (a1.T).dot(delta3)
        # db2 = np.sum(delta3, axis=0, keepdims=True)
        # delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        # dW1 = np.dot(X.T, delta2)
        # db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        # dW2 += self.reg_lambda * W2
        # dW1 += self.reg_lambda * W1

    def getParams(self):
        flat_param_vector = np.r_[self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()]
        self.num_params = max(flat_param_vector.shape)
        return flat_param_vector

    def setParams(self, flat_param_vector):
        """
        :param flat_param_vector: size of W1 + b1 + W2 + b2
        :return:
        """
        self.params_minus_one = self.getParams()
        self.params_plus_one = flat_param_vector
        param_vector_copies = [deepcopy(x) for x in [self.W1, self.b1, self.W2, self.b2]]
        flattened_W1_shape = np.prod(param_vector_copies[0].shape)
        flattened_b1_shape = np.prod(param_vector_copies[1].shape)
        flattened_W2_shape = np.prod(param_vector_copies[2].shape)
        flattened_b2_shape = np.prod(param_vector_copies[3].shape)
        res = np.split(flat_param_vector, [flattened_W1_shape,
                                          flattened_W1_shape + flattened_b1_shape,
                                          flattened_W1_shape + flattened_b1_shape + flattened_W2_shape,
                                          flattened_W1_shape + flattened_b1_shape + flattened_W2_shape + flattened_b2_shape])
        self.W1, self.b1, self.W2, self.b2 = res[0].reshape(self.W1.shape), res[1].reshape(self.b1.shape),\
                                             res[2].reshape(self.W2.shape), res[3].reshape(self.b2.shape)

    def calculateGradient(self, state=None):
        """
        :return: the gradient of the output of the approximator wrt to its parameters
        """
        assert self.last_input is not None, "computeOutput method has not been called yet---no point calculating gradient then"
        orig_parameter_vector = self.getParams()
        if state is None:
            state_for_gradient_calc = deepcopy(self.last_input)
        else:
            state_for_gradient_calc = state

        orig_output = self.computeOutput(state_for_gradient_calc)
        gradient = []
        for idx in range(len(orig_parameter_vector)):
            new_param_vector = deepcopy(orig_parameter_vector)
            new_param_vector[idx] += self.h

            self.setParams(new_param_vector)

            new_output = self.computeOutput(state_for_gradient_calc)
            gradient.append((new_output - orig_output) / self.h)

        # Reset the weights to what they were before the gradient calculation
        self.setParams(orig_parameter_vector)
        return np.array(gradient)



    def approximatorParameterUpdate(self, shifted_param_vector):
        # Gradient descent parameter update
        # W1 += -self.alpha * dW1
        # b1 += -self.alpha * db1
        # W2 += -self.alpha * dW2
        # b2 += -self.alpha * db2
        pass

# ---------- Testing Script --------------
if __name__=='__main__':
    approx = ANNApproximator(12, "tanh")

    test_input = np.array([0.1, 0.2, -.02, 1.0])
    orig_output = approx.computeOutput(test_input)
    print("Original output: {0}".format(orig_output))
    # approx.printParams("All")

    params = approx.getParams()
    print("Flattened Parameter vector: {0}".format(params.shape))
    for i in range(10):
        params = approx.getParams()
        g = approx.calculateGradient()
        approx.setParams(params + (0.001 * g))
    # approx.printParams("All")

    new_output = approx.computeOutput(test_input)
    print("New output: {0}".format(new_output))

    g = approx.calculateGradient()
    print("Gradient: {0}".format(g))
    print("Gradient size: {0}".format(len(g)))
