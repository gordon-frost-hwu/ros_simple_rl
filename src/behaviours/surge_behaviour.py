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

GAUSSIAN_VARIANCE = 1.0     # Open parameter which has a MASSIVE effect on resulting velocity request
GAUSSIAN_MEAN_DEFAULT = 0.0

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

class TranslationalBehaviour(Behaviour):
    _params = {}
    def __init__(self):
        Behaviour.__init__(self, "surge")

        # Set default parameters
        self._params['gaussian_variance'] = GAUSSIAN_VARIANCE
        self._params['gaussian_mean'] = GAUSSIAN_MEAN_DEFAULT
        self._params['reverseAction'] = False

    def _changeParamCallback(self, req):
        self._params["reverseAction"] = req.reverse
        print("Value request: {0}".format(req.value))
        print("Max Action Value: {0}".format(req.max_action_value))
        if req.name == "gaussian_variance":
            try:
                self._params[req.name] = (req.max_action_value + 0.05) - req.value
                print("Value modified: {0}".format(self._params[req.name]))
                return ChangeParamResponse(success = True, sigma = self._params[req.name])
            except KeyError:
                # rospy.loginfo("Parameter Index Error: param, {0}, does not exist!".format(req.name))
                return ChangeParamResponse(success = False, sigma = self._params[req.name])
        return ChangeParamResponse(success = True, sigma = self._params[req.name])

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
            desired_velocity = 1.0 - gaussian2D(self._nav[0], self._nav[1], self.goal_position[0], self.goal_position[1],
                                          1.0, [self._params['gaussian_variance'], self._params['gaussian_variance']])  #TODO: width of distribution adjusted by original distance to goal??

            # rospy.loginfo("Distance to Goal:    {0}".format(distance_to_goal))
            # rospy.loginfo("Resulting Surge Velocity:  {0}".format(desired_velocity))

            # Limit the velocities by the maximum possible for the vehicle if required.
            scaled_velocity = np.array([desired_velocity, 0, 0, 0, 0, 0]) * self.MAX_SPEED

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