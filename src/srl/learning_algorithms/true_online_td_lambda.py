__author__ = 'https://github.com/EllaBot/true-online-td-lambda'

import numpy as np
import math
from random import random


class TrueOnlineTDLambda(object):
    def __init__(self, basis_functions, critic_config, lmda=0.9, gamma=0.99, init_val=0.0):

        """
        :param numfeatures: the number of items in a state-action pair
        :param ranges: an array of tuples representing the closed intervals of
                         features
        :param alpha: the learning rate
        :param lmda: the rate at which to decay the traces
        :param gamma: the discount factor governing how future rewards
                          are weighted with respect to the immediate reward
        """
        assert 0 <= lmda <= 1
        self.name = "true online"
        self.alpha = critic_config["alpha"]
        self.lmbda = lmda
        self.gamma = gamma
        self.init_value = init_val
        self.basis = basis_functions    # object which has a computeFeatures method
        self.stateprime = None
        self.state = None
        self.theta = None
        self.params = self.theta
        self.vs = 0
        self.traces = np.zeros(len(self.basis.computeFeatures([0.0 for _ in range(critic_config["num_input_dims"])])))

    def start(self, state):
        """Provide the learner with the first state.
        """
        self.stateprime = state
        # print("start: {0}".format(len(state)))
        self.vs = self.getStateValue(state)
    def getParams(self):
        return self.theta
    def setParams(self, p):
        self.theta = np.array(p)
    def getTraces(self):
        return self.traces


    def step(self, reward, state):
        """ Perform the typical update
        See "True Online TD(lambda)" section 4.1
        """
        # print("step shape: {0}".format(len(state)))
        # Rotate the states back
        self.state = self.stateprime    # x_t
        self.stateprime = state         # x_t+1

        assert self.state is not None and self.stateprime is not None
        phi_t = self.basis.computeFeatures(self.state)      # phi_t-1
        vsprime = self.getStateValue(self.stateprime)       # V_t+1

        self.updateTrace(phi_t)
        delta = self.updateweights(phi_t, reward, self.vs, vsprime)

        self.vs = vsprime
        return delta

    def updateTrace(self, phi_t):
        """Compute the e vector.
        :param phi_t: vector of values of the features of the initial state.
                        Parameterized so it can be cached between calls.
        """
        if self.traces is None:
            self.traces = np.zeros(self.basis.computeFeatures(phi_t))
        termone = self.gamma * self.lmbda * self.traces
        termtwo = self.alpha * (1 - self.gamma * self.lmbda * np.dot(self.traces, phi_t)) * phi_t
        self.traces = termone + termtwo

    def updateweights(self, phi_t, reward, vs, vsprime):
        """Compute the theta vector.
        :param phi_t: vector of values of the features of the initial state.
        :param reward: the reward signal received
        :param vs: the value of the state
        :param vsprime: the value of the resulting state
        """
        # print("updateWeights: vs: {0}".format(vs))
        # print("updateWeights: vsprime: {0}".format(vsprime))
        delta = reward + (self.gamma * vsprime) - vs
        self.theta += delta * self.traces + self.alpha * (vs - np.dot(self.theta, phi_t)) * phi_t
        return delta

    def end(self, reward):
        """ Receive the reward from the final action.
        This action does not produce an additional state, so we update a
        little differently.
        """
        if self.state is None:
            # If we're ending before we have a state, we
            # don't have enough data to perform an update. Just reset.
            self.episodeReset()
            return
        phi_t = self.basis.computeFeatures(self.state)
        # There is no phi_tp because there is no second state, so we'll
        # set the value of the second state to zero.
        vsprime = 0.0

        self.updateTrace(phi_t)
        delta = self.updateweights(phi_t, reward, self.vs, vsprime)

        # Clear episode specific learning artifacts
        self.episodeReset()
        return delta

    def getStateValue(self, state_action):
        """Compute the value with the current weights.
        :param state_action:
        :return: The value as a float.
        """
        if self.theta is None:
            features = self.basis.computeFeatures(state_action)
            # print("getStateValue->features: {0}".format(features.shape))
            self.theta = np.ones(len(features)) * (self.init_value / sum(features))
        # print("Theta Size: {0}".format(self.theta.shape))
        # print("Basis function Size: {0}".format(self.basis.computeFeatures(state_action).shape))
        value = np.dot(self.theta, self.basis.computeFeatures(state_action))

        # If we've diverged, just crash
        assert not math.isnan(value) and not math.isinf(value)
        return value

    def episodeReset(self):
        self.state = None
        self.stateprime = None
        self.vs = None
        self.traces.fill(0.0)