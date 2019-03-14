#!/usr/bin/python
"""
Author: Jeremy M. Stober
Program: TD.PY
Date: Thursday, April 26 2011
Description: TD V function approximation using simple tabular state representation.
"""

import numpy as np

class TD(object):
    """
    Discrete value function approximation via temporal difference learning.
    """

    def __init__(self, nstates, alpha, gamma, ld, init_val = 0.0):
        self.V = np.ones(nstates) * init_val
        self.traces = np.zeros(nstates)
        self.nstates = nstates
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.ld = ld # lambda

    def getStateValue(self, state):
        return self.V[state]

    def delta(self, pstate, reward, state):
        """
        This is the core TD error calculation. Note that if the value
        function is perfectly accurate then this returns zero since by
        definition value(pstate) = gamma * value(state) + reward.
        """
        return reward + (self.gamma * self.getStateValue(state)) - self.getStateValue(pstate)

    def train(self, pstate, reward, state):
        """
        A single step of reinforcement learning.
        """

        delta = self.delta(pstate, reward, state)

        self.traces[pstate] += 1.0

        #for s in range(self.nstates):
        self.V += self.alpha * delta * self.traces
        self.traces *= (self.gamma * self.ld)

        return delta

    def learn(self, nepisodes, env, policy, verbose = True):
        # learn for niters episodes with resets
        for i in range(nepisodes):
            self.reset()
            t = env.single_episode(policy) # includes env reset
            for (previous, action, reward, state, next_action) in t:
                self.train(previous, reward, state)
            if verbose:
                print i

    def reset(self):
        self.traces = np.zeros(self.nstates)


class TDLinear(TD):
    """
    A more general linear value function representation.
    """

    def __init__(self, nfeatures, alpha, gamma, ld, init_val=0.0):
        self.name = "trad online"
        self.V = np.ones(nfeatures) * init_val
        self.params = np.ones(nfeatures) * init_val
        self.traces = np.zeros(nfeatures)
        self.nfeatures = nfeatures
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.ld = ld # lambda

    def getStateValue(self,features):
        return np.dot(self.params,features)

    def train(self, pfeatures, reward, features):
        delta = self.delta(pfeatures, reward, features)
        # print("UPDATED")
        # print(self.ld)
        # print(self.gamma)
        # print(type(pfeatures))
        # print(type(self.traces))
        self.traces = self.ld * self.traces + pfeatures
        self.params = self.params + self.alpha * delta * self.getTraces()
        return delta

    def end(self, pfeatures, reward):
        delta = reward + (self.gamma * self.getStateValue(pfeatures))
        self.traces = self.ld * self.traces + pfeatures
        self.params = self.params + self.alpha * delta * self.getTraces()
        print("-->end: delta: {0}".format(delta))
        return delta

    def getTraces(self):
        return self.traces

    def getParams(self):
        return self.params

    # def end(self, reward, features_t):
    #     # When s_t+1 is a terminal state, V(s_t+1) == 0
    #     delta = reward - self.getStateValue(features_t)
    #     self.traces = self.gamma * self.ld * self.traces + features_t
    #     self.params = self.params + self.alpha * delta * self.traces
    #     return delta

    def episodeReset(self):
        self.traces = np.zeros(self.nfeatures)