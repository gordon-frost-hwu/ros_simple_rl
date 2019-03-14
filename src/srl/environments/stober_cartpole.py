#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: __INIT__.PY
Date: Saturday, February 25 2012
Description: A cartpole implementation based on Rich Sutton's code.
"""

import math
from numpy import clip
import random as pr

class CartPole(object):

    def __init__(self, x = 0.0, xdot = 0.0, theta = 0.0, thetadot = 0.0):
        self.x = x
        self.xdot = xdot
        self.theta = theta
        self.thetadot = thetadot

        # some constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5		  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02		  # seconds between state updates
        self.fourthirds = 1.3333333333333

    def getSensors(self):
        return [self.theta, self.thetadot, self.x, self.xdot]

    def failure(self):
        twelve_degrees = 0.2094384
        if (not -2.4 < self.x < 2.4) or (not -twelve_degrees < self.theta < twelve_degrees):
            return True
        else:
            return False

    def reset(self):
        self.x,self.xdot,self.theta,self.thetadot = (0.05,0.0,-0.05,0.0)

    def random_policy(self, *args):
        return pr.choice([0,1])

    def single_episode(self, policy = None):
        self.reset()
        if policy is None: policy = self.random_policy

        trace = []
        next_action = policy(self.x,self.xdot,self.theta,self.thetadot)
        while not self.failure():
            pstate, paction, reward, state = self.move(next_action)
            next_action = policy(self.x,self.xdot,self.theta,self.thetadot)
            trace.append([pstate, paction, reward, state, next_action])

        return trace

    def state(self): # get boxed version of the state as per the original code

        one_degree = 0.0174532
        six_degrees = 0.1047192
        twelve_degrees = 0.2094384
        fifty_degrees = 0.87266

        if (not -2.4 < self.x < 2.4) or (not -twelve_degrees < self.theta < twelve_degrees):
            return -1

        box = 0

        if self.x < -0.8:
            box = 0
        elif self.x < 0.8:
            box = 1
        else:
            box = 2

        if self.xdot < -0.5:
            pass
        elif self.xdot < 0.5:
            box += 3
        else:
            box += 6

        if self.theta < -six_degrees:
            pass
        elif self.theta < -one_degree:
            box += 9
        elif self.theta < 0:
            box += 18
        elif self.theta < one_degree:
            box += 27
        elif self.theta < six_degrees:
            box += 36
        else:
            box += 45

        if self.thetadot < -fifty_degrees:
            pass
        elif self.thetadot < fifty_degrees:
            box += 54
        else:
            box += 108

        return box;

    def getReward(self):
        if self.failure():
            return 0.0
        else:
            return 1.0

    def performAction(self, action): # binary L/R action
        force = clip(action, -10.0, 10.0)

        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)

        tmp = (force + self.polemass_length * (self.thetadot ** 2) * sintheta) / self.total_mass;
        thetaacc = (self.gravity * sintheta - costheta * tmp) / (self.length * (self.fourthirds - self.masspole * costheta ** 2 / self.total_mass))
        xacc = tmp - self.polemass_length * thetaacc * costheta / self.total_mass

        (px,pxdot,ptheta,pthetadot) = (self.x, self.xdot, self.theta, self.thetadot)
        pstate = self.state()

        self.x += self.tau * self.xdot
        self.xdot += self.tau * xacc
        self.theta += self.tau * self.thetadot
        self.thetadot += self.tau * thetaacc

        return [px,pxdot,ptheta,pthetadot],action,self.getReward(),[self.x,self.xdot, self.theta, self.thetadot]

if __name__ == '__main__':

    cp = CartPole()