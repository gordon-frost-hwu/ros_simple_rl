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
import math
from vehicle_interface.msg import PilotRequest, Vector6

NORTH = 0
EAST = 1
YAW = 2
# was 1.0
GAUSSIAN_VARIANCE_DEFAULT = 1.0     # Open parameter which has a MASSIVE effect on resulting velocity request
GAUSSIAN_MEAN_DEFAULT = 0.0

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

def _G(theta, reverse, mean, sigma):
    val = 1 - gaussian1D(theta, mean, 1.0, sigma)
    if reverse:
        return val
    else:
        return -val
def _F(theta, reverse, mean, sigma):
    if theta < 0.0:
        return -gaussian1D(theta, mean, 1.0, sigma)
    else:
        return gaussian1D(theta, mean, 1.0, sigma)

class YawBehaviour(Behaviour):
    _params = {}
    def __init__(self):
        Behaviour.__init__(self, "yaw")

        # Enter main loop of behaviour
        #self.doWork()

        # Set default parameters
        self._params['gaussian_variance'] = GAUSSIAN_VARIANCE_DEFAULT
        self._params['gaussian_mean'] = GAUSSIAN_MEAN_DEFAULT
        self._params['reverseAction'] = False

    def navCallback(self, msg):
        """
        :param msg: navigation msg
        method overrides parent class navCallack method
        """
        self._nav = (msg.position.north, msg.position.east, msg.orientation.yaw)

    def doWork(self):
        # rospy.loginfo("Computing behaviours output ...")

        if self.goal_position is not None and self._nav is not None:
            # Seperate out north, east and yaw as these are the only DOFs of interest
            #goal_position is of form: [north, east, depth, roll, pitch, yaw]
            goal = tuple([self.goal_position[0], self.goal_position[1], self.goal_position[-1]])
            delta_nav = np.array(self._nav) - np.array( goal )

            # Calculate the heading of the goal position from the current position
            heading_wrt_north = (math.atan2(delta_nav[EAST], delta_nav[NORTH]) + math.pi)
            print("heading_wrt_north: {0}".format(heading_wrt_north))

            goal_angle_wrt_vehicle_heading = heading_wrt_north - self._nav[-1]
            if goal_angle_wrt_vehicle_heading > math.pi:
                goal_angle_wrt_vehicle_heading = -(2 * math.pi - goal_angle_wrt_vehicle_heading)

            print("goal_angle_wrt_vehicle_heading: {0}".format(goal_angle_wrt_vehicle_heading))
            # rospy.loginfo("**Debug**:Raw Delta Yaw: {0}".format(goal_angle_wrt_vehicle_heading))

            # make sure that variance is above zero
            assert (self._params['gaussian_variance'] >= 0.0)

            # calculate the velocity curve which is in effect based on the distance to the goal, through a
            # gaussian distribution
            G = _G(goal_angle_wrt_vehicle_heading, self._params['reverseAction'],
                                  self._params['gaussian_mean'], self._params['gaussian_variance'])
            if self.distance_to_danger < 5.0:
                F = _F(self.raw_angle_to_danger, False, 0.0, 0.3)
                desired_velocity = (G + F) / 2.0
            else:
                F = 0.0
                desired_velocity = G + F
            # print("G")

            # if self._params['reverseAction']:
            #     desired_velocity = 1 - gaussian1D(goal_angle_wrt_vehicle_heading, self._params['gaussian_mean'],
            #                                       1.0, self._params['gaussian_variance'])
            # else:
            #     # print("IM REVERSING MY OUTPUT --- LOOK OUT!!!")
            #     desired_velocity = -(1 - gaussian1D(goal_angle_wrt_vehicle_heading, self._params['gaussian_mean'],
            #                                         1.0, self._params['gaussian_variance']))

            # rospy.loginfo("Angle to Goal (Deg): {0}".format(goal_angle_wrt_vehicle_heading * (180.0/math.pi)))
            # rospy.loginfo("Current Yaw (Deg):   {0}".format(self._nav[YAW] * (180.0/math.pi)))
            # rospy.loginfo("desired_velocity (normalized):   {0}".format(desired_velocity))

            # Here we set the correct sign of the velocity depending on which "side" the goal is with respect
            # to the vehicles current heading
            # TODO: make a little smarter. ie. rotational distance including wrapped 3.14/-3.14 instead of just sign
            if goal_angle_wrt_vehicle_heading > 0.0:
                desired_velocity = -desired_velocity

            # Limit the velocities by the maximum possible for the vehicle if required.
            scaled_velocity = np.array([0, 0, 0, 0, 0, desired_velocity]) * self.MAX_SPEED

            # rospy.loginfo("Resulting Velocity:  {0}".format(scaled_velocity[-1]))

            # Send the result msg to the coordinator

            # self.publishResult(Vector6(values=scaled_velocity))
            self.publishResult(Vector6(values=scaled_velocity))

# Testing/usage Script for above class
if __name__ == '__main__':
    # Set Node Name and Spin Rate
    rospy.init_node('yaw_behaviour')
    ros_rate = rospy.Rate(8)

    # Instantiate the Behaviour, set the goal and then run the behaviours doWork method
    behaviour = YawBehaviour()
    # behaviour.metric_goal = 0  # we want distance between goal position and nav to be zero
    # behaviour.goal_position = (9, 9, 0.57)

    while not rospy.is_shutdown():
        behaviour.doWork()
        ros_rate.sleep()

    rospy.spin()