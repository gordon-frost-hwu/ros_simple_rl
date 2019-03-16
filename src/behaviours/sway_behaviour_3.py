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
from my_msgs.msg import Goals

GAUSSIAN_VARIANCE = 0.5     # Open parameter which has a MASSIVE effect on resulting velocity request

def isApproximatelyEqualTo(a, b, e):
    # is a equal to b within tolerance e
    return a < b + e and a > b - e

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
    LIMITED_SPEED = np.array([0.5, 2.0, 0.6, 0.0, 2.0, 2.0])        # max speed (m/s and rad/s)
    def __init__(self):
        Behaviour.__init__(self, "sway")

        # ros_env_sub = rospy.Subscriber('/goal_points_data', Goals, self.rosEnvCallback)

        # Set default parameters
        self._params['gaussian_variance'] = GAUSSIAN_VARIANCE
        self._params['reverseAction'] = False

    # def rosEnvCallback(self, msg):
    #     for point in msg.goals:
    #         # Get the angle angle to the point-of-interest; 1 = waypoint_goal, 2 = inspection_goal, 3 = fear_goal
    #         if point.point_type == 1.0:
    #             self.raw_angle_to_goal = point.angle_wrt_vehicle
    #             self.distance_to_wp = point.distance

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

            # calculate the desired resultant velocity (vector) towards the waypoint goal
            resultant_velocity = 1.0 - gaussian2D(self._nav[0], self._nav[1], self.goal_position[0], self.goal_position[1],
                                          1.0, [self._params['gaussian_variance'], self._params['gaussian_variance']])


            is_goal_behind_ip = isApproximatelyEqualTo(self.raw_angle_to_ip, self.raw_angle_to_wp, 0.3)
            # if distance_to_goal < 5.0 and is_goal_behind_ip or distance_to_goal > 5.0:

            # theta_to_ip should be == 0
            # resultant velocity heading != global angle to goal
            desired_sway_velocity = resultant_velocity * np.sin(self.raw_angle_to_wp)

            # if self.raw_angle_to_ip - 0.3 < self.raw_angle_to_wp < self.raw_angle_to_ip + 0.3:
            #     if self.raw_angle_to_wp >= 0.0:
            #         desired_sway_velocity = abs(resultant_velocity * np.cos(self.raw_angle_to_ip))
            #     else:
            #         desired_sway_velocity = -(abs(resultant_velocity * np.cos(self.raw_angle_to_ip)))
            if self.distance_to_ip < 2.5 and abs(self.raw_angle_to_wp) < 1.57:  # and not is_goal_behind_ip:
                # theta_to_ip == 0
                # resultant velocity heading == global angle to goal
                desired_sway_velocity = resultant_velocity * np.cos(self.raw_angle_to_ip)
                if self.raw_angle_to_wp < 0.0:
                    desired_sway_velocity = - desired_sway_velocity



            print("Distance to Goal:    {0}".format(distance_to_goal))
            print("Resulting Sway Velocity:  {0}".format(desired_sway_velocity))

            # Limit the velocities by the maximum possible for the vehicle if required.
            scaled_velocity = np.array([0, desired_sway_velocity, 0, 0, 0, 0]) * self.LIMITED_SPEED

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