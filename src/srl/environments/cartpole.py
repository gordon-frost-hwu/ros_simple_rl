__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from matplotlib.mlab import rk4
from math import sin, cos
import time
from numpy import clip
from scipy import eye, matrix, random, asarray

class CartPoleEnvironment(object):
    """ This environment implements the cart pole balancing benchmark, as stated in:
        Riedmiller, Peters, Schaal: "Evaluation of Policy Gradient Methods and
        Variants on the Cart-Pole Benchmark". ADPRL 2007.
        It implements a set of differential equations, solved with a 4th order
        Runge-Kutta method.
    """

    indim = 1
    outdim = 4

    # some physical constants --- same as constants used in "Using continuous action spaces to solve discrete problems"
    g = 9.81
    l = 1.0
    mp = 0.1
    mc = 1.0
    dt = 0.02

    randomInitialization = True

    def __init__(self, polelength=None):
        if polelength != None:
            self.l = polelength

        # initialize the environment (randomly)
        self.reset()
        self.action = 0.0
        self.delay = False

    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return
            vector has 4 elements: theta, theta', s, s' (s being the distance from the
            origin).
        """
        return asarray(self.sensors)

    def performAction(self, action):
        """ stores the desired action for the next runge-kutta step.
        """
        self.action = action
        self._step()

    def getReward(self):
        episode_ended = self.episodeEnded()
        #print("penalty: {0}".format(penalty))
        if episode_ended:
            return -1.0
        else:
            return 1.0

    def episodeEnded(self):
        if abs(self.sensors[0]) > 0.20944 or abs(self.sensors[2]) > 2.4:
            #print("POLE FALLEN or CART POSITION WRONG")
            return True
        else:
            return False

    def _step(self):
        """
        perform a single step within the episode and update the rendering
        """
        self.sensors = rk4(self._derivs, self.sensors, [0, self.dt])
        self.sensors = self.sensors[-1]

    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        if self.randomInitialization:
            angle = random.uniform(-0.0523599, 0.0523599)
            # pos = random.uniform(-0.05, 0.05)
            pos = random.uniform(-0.5, 0.5)
            #pos = 0.0
        else:
            angle = -0.05
            pos = 0.1
        self.sensors = (angle, 0.0, pos, 0.0)

    def _derivs(self, x, t):
        """ This function is needed for the Runge-Kutta integration approximation method. It calculates the
            derivatives of the state variables given in x. for each variable in x, it returns the first order
            derivative at time t.
        """
        F = clip(self.action, -10.0, 10.0)
        (theta, theta_, _s, s_) = x
        u = theta_
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        mp = self.mp
        mc = self.mc
        l = self.l
        u_ = (self.g * sin_theta * (mc + mp) - (F + mp * l * theta_ ** 2 * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * cos_theta ** 2)
        v = s_
        v_ = (F - mp * l * (u_ * cos_theta - (theta_ ** 2 * sin_theta))) / (mc + mp)
        return (u, u_, v, v_)

    def getPoleAngles(self):
        """ auxiliary access to just the pole angle(s), to be used by BalanceTask """
        return [self.sensors[0]]

    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[2]



class CartPoleLinEnvironment(CartPoleEnvironment):
    """ This is a linearized implementation of the cart-pole system, as described in
    Peters J, Vijayakumar S, Schaal S (2003) Reinforcement learning for humanoid robotics.
    Polelength is fixed, the order of sensors has been changed to the above."""

    tau = 1. / 60.   # sec

    def __init__(self, **kwargs):
        CartPoleEnvironment.__init__(self, **kwargs)
        nu = 13.2 #  sec^-2
        tau = self.tau

        # linearized movement equations
        self.A = matrix(eye(4))
        self.A[0, 1] = tau
        self.A[2, 3] = tau
        self.A[1, 0] = nu * tau
        self.b = matrix([0.0, nu * tau / 9.80665, 0.0, tau])


    def step(self):
        self.sensors = random.normal(loc=self.sensors * self.A + self.action * self.b, scale=0.001).flatten()
        if self.hasRenderer():
            self.getRenderer().updateData(self.sensors)
            if self.delay:
                time.sleep(self.tau)

    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        self.sensors = random.normal(scale=0.1, size=4)

    def getSensors(self):
        return self.sensors.flatten()

    def getPoleAngles(self):
        """ auxiliary access to just the pole angle(s), to be used by BalanceTask """
        return [self.sensors[0]]

    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[2]


