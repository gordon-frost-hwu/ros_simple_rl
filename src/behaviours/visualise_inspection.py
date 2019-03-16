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



# Testing/usage Script for above class
if __name__ == '__main__':
    # Set Node Name and Spin Rate
    rospy.init_node('surge_behaviour')
    ros_rate = rospy.Rate(8)


    while not rospy.is_shutdown():
        ros_rate.sleep()

    rospy.spin()