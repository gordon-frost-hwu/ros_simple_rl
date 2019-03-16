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
        Behaviour.__init__(self, "surge")

        # Set default parameters
        self._params['gaussian_variance'] = GAUSSIAN_VARIANCE
        self._params['reverseAction'] = False

    def navCallback(self, msg):
        """
        :param msg: navigation msg
        method overrides parent class navCallack method
        """
        self._nav = (msg.position.north, msg.position.east)

    def doWork(self):
        # rospy.loginfo("Computing behaviours output ...")
        # print(self.goal_position)
        if self.goal_position is not None and self._nav is not None:
            distance_to_goal = np.sqrt(sum((np.array(self.goal_position[0:2]) - np.array(self._nav))**2))

            # calculate the velocity curve which is in effect based on the distance to the goal, through a
            # gaussian distribution
            # resultant_velocity = 1.0 - gaussian2D(self._nav[0], self._nav[1], self.goal_position[0], self.goal_position[1],
            #                               1.0, [self._params['gaussian_variance'], self._params['gaussian_variance']])

            # if self.distance_to_ip < DISTANCE_OFFSET and abs(self.raw_angle_to_wp) < 1.57 or\
            #                 abs(self.raw_angle_to_wp) < 2.5:    # + 2.5 metres to allow for momentum to slow
            #     rospy.loginfo("@@@@@@@@@@@@@@@@@@@")
            #     rospy.loginfo("going to offset ....")
            #     desired_surge_velocity = (1.0 - gaussian1D(self.distance_to_ip, DISTANCE_OFFSET,
            #                                         1.0,self._params['gaussian_variance'])) * np.cos(self.raw_angle_to_wp)
            #     # desired_surge_velocity = modifiedGaussian1D(self.distance_to_ip, DISTANCE_OFFSET, 1.0,self._params['gaussian_variance'])
            # else:
            # if self.distance_to_wp < 1.0:
            #     desired_surge_velocity = (1.0 - gaussian1D(self.distance_to_wp, 0.0,
            #                                              1.0,1.0))  # * np.cos(self.raw_angle_to_wp)
            #     if abs(self.raw_angle_to_wp) > 1.57:
            #         desired_surge_velocity = - desired_surge_velocity
            # elif self.distance_to_ip < 10:
            #     desired_surge_velocity = (1.0 - gaussian1D(self.distance_to_ip, DISTANCE_OFFSET,
            #                                              1.0,self._params['gaussian_variance'])) * np.cos(self.raw_angle_to_wp)
            # else:
            #     desired_surge_velocity = (1.0 - gaussian1D(self.distance_to_ip, 0.0,
            #                                              1.0,self._params['gaussian_variance'])) * np.cos(self.raw_angle_to_wp)
            #
            #
            # print("Distance to Goal:    {0}".format(distance_to_goal))
            # print("Resulting Surge Velocity:  {0}".format(desired_surge_velocity))

            # Limit the velocities by the maximum possible for the vehicle if required.
            scaled_velocity = np.array([self._params['gaussian_variance'], 0, 0, 0, 0, 0]) * self.LIMITED_SPEED

            # Send the result msg to the coordinator
            self.publishResult(Vector6(values=scaled_velocity))

# Testing/usage Script for above class
if __name__ == '__main__':
    # Set Node Name and Spin Rate
    rospy.init_node('surge_behaviour')
    ros_rate = rospy.Rate(8)

    behaviour = TranslationalBehaviour()
    # behaviour.metric_goal = 0  # we want distance between goal position and nav to be zero
    # behaviour.setGoalPosition((9, 0, 0, 0, 0, 0))
    while not rospy.is_shutdown():
        behaviour.doWork()
        ros_rate.sleep()

    rospy.spin()