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

GAUSSIAN_VARIANCE = 0.5     # Open parameter which has a MASSIVE effect on resulting velocity request

def gaussian1D(x, x0, mu, sigma):
    """
    :param x: north position
    :param x0: centre of Normal distribution in North
    :param mu: Height of normal distribution
    :param sigma: variance
    :return: value of point x wrt normal distribution centred on x0
    """
    x_term = (x - x0)**2 / (2 * sigma**2)
    return (mu * np.exp(-x_term))

class PitchBehaviour(Behaviour):
    _params = {}
    def __init__(self):
        Behaviour.__init__(self, "depth")

        # Set default parameters
        self._params['gaussian_variance'] = GAUSSIAN_VARIANCE
        self._params['reverseAction'] = False

    def navCallback(self, msg):
        """
        :param msg: navigation msg
        method overrides parent class navCallack method
        """
        self._nav = msg.position.depth

    def doWork(self):
        # rospy.loginfo("Computing behaviours output ...")
        # print(self.goal_position)
        if self.goal_position is not None and self._nav is not None:

            # calculate the velocity curve which is in effect based on the distance to the goal, through a
            # gaussian distribution
            desired_velocity = -(1.0 - gaussian1D(self._nav, self.goal_position[2],
                                          1.0, self._params['gaussian_variance']))

            print("Resulting Depth Velocity:  {0}".format(desired_velocity))

            # Limit the velocities by the maximum possible for the vehicle if required.
            scaled_velocity = np.array([0, 0, desired_velocity, 0, 0, 0]) * self.MAX_SPEED

            # Send the result msg to the coordinator
            self.publishResult(Vector6(values=scaled_velocity))

# Testing/usage Script for above class
if __name__ == '__main__':
    # Set Node Name and Spin Rate
    rospy.init_node('depth_behaviour')
    ros_rate = rospy.Rate(8)

    behaviour = PitchBehaviour()

    while not rospy.is_shutdown():
        behaviour.doWork()
        ros_rate.sleep()

    rospy.spin()