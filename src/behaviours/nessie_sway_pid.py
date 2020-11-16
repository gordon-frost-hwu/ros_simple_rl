#! /usr/bin/env python
"""
    Module contains the TranslationalBehaviour class which is ROS based taking as input: goal position and nav.
    The behaviour outputs a force or velocity based on the distance to goal
    author: Gordon Frost; email: gwf2@hw.ac.uk
    date: 23/01/15
"""
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from behaviours.behaviour import Behaviour
import numpy as np
from vehicle_interface.msg import PilotRequest, Vector6
from srv_msgs.srv import ChangeParam, ChangeParamResponse
from utilities.PID import PID

GAUSSIAN_VARIANCE = 3.0     # Open parameter which has a MASSIVE effect on resulting velocity request
DISTANCE_OFFSET = 2.5

def gaussian2D(x, y, x0, y0, mu, sigma):
    """
    :param x: north position
    :param y: east position
    :param x0: centre of Normal distribution in North
    :param y0: centre of Normal distribution in East
    :param mu: Height of normal distribution
    :param sigma: 2 dimensional list of form [x_sigma, y_sigma]
    :return: value of point x,y wrt normal distribution centred on x0,y0
    """
    x_term = (x - x0)**2 / (2 * sigma[0]**2)
    y_term = (y - y0)**2 / (2 * sigma[1]**2)
    res = mu * np.exp( -(x_term + y_term))
    return res

def gaussian1D(x, x0, mu, sigma):
    x_term = (x - x0)**2 / (2 * sigma**2)
    res = mu * np.exp( -x_term)
    return res

def modifiedGaussian1D(x, x0, mu, sigma):
    """
    :param x: north position
    :param x0: centre of Normal distribution in North
    :param mu: Height of normal distribution
    :param sigma: variance
    :return: value of point x,y wrt normal distribution centred on x0,y0
    """
    x_term = (x - x0)**2 / (2 * sigma**2)
    if x >= x0:
        res = mu * np.exp(-x_term)
    else:
        res = -(mu * np.exp(-x_term))
    return res

class TranslationalBehaviour(Behaviour):
    _params = {}
    LIMITED_SPEED = np.array([1.0, 0.6, 0.6, 0.0, 2.0, 2.0])        # max speed (m/s and rad/s)
    def __init__(self):
        Behaviour.__init__(self, "sway")

        # Set default parameters
        self._params['gaussian_variance'] = GAUSSIAN_VARIANCE
        self._params['reverseAction'] = False

        self.pid = PID(gains=self.pid_coeffs["sway"])

    def navCallback(self, msg):
        """
        :param msg: navigation msg
        method overrides parent class navCallack method
        """
        self._nav = (msg.position.north, msg.position.east)

    def doWork(self):
        # rospy.loginfo("Computing behaviours output ...")
        # print(self.goal_position)
        if self.ip_position is not None and self._nav is not None:
            distance_to_goal = np.sqrt(sum((np.array(self.ip_position[0:2]) - np.array(self._nav))**2))

            print("Distance to target: {0}".format(self.distance_to_ip))
            print("Angle to target: {0}".format(self.raw_angle_to_ip))

            pos_err = self.distance_to_ip * np.sign(self.raw_angle_to_ip)
            # if abs(self.raw_angle_to_wp) > 1.57:
            #     pos_err = -pos_err
            pid_output = self.pid.GenOut(pos_err)
            # Limit the velocities by the maximum possible for the vehicle if required.
            scaled_velocity = np.array([0, pid_output, 0, 0, 0, 0]) * self.LIMITED_SPEED

            # Send the result msg to the coordinator
            self.publishResult(Vector6(values=scaled_velocity))

# Testing/usage Script for above class
if __name__ == '__main__':
    # Set Node Name and Spin Rate
    rospy.init_node('sway_behaviour')
    ros_rate = rospy.Rate(8)

    behaviour = TranslationalBehaviour()
    # behaviour.metric_goal = 0  # we want distance between goal position and nav to be zero
    # behaviour.setGoalPosition((9, 0, 0, 0, 0, 0))
    while not rospy.is_shutdown():
        behaviour.doWork()
        ros_rate.sleep()

    rospy.spin()
