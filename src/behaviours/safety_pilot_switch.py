#! /usr/bin/env python
"""
    Module contains the TranslationalBehaviour class which is ROS based taking as input: goal position and nav.
    The behaviour outputs a force or velocity based on the distance to goal
    author: Gordon Frost; email: gwf2@hw.ac.uk
    date: 23/01/15
"""
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
import numpy as np
from copy import deepcopy
from std_msgs.msg import Empty
from srv_msgs.srv import DisableOutput
from utilities.nav_class import Nav
from vehicle_interface.msg import PilotRequest, Vector6

SPIN_RATE = 4 #hz

class SafetySwitch(Nav):
    def __init__(self):
        super(SafetySwitch, self).__init__()
        self.timer = 0
        sub = rospy.Subscriber("/safety_latch", Empty, self.latchCallback)
        self.pilot_request = rospy.Publisher("/pilot/position_req", PilotRequest)
        self.coordinatorDisable = rospy.ServiceProxy('/behaviours/coordinator/disable_output', DisableOutput)
        self.disabled = False
        self.orig_position = None
    def latchCallback(self, msg):
        self.timer = 0
    def loop(self):
        if self.timer >= SPIN_RATE:
            print("Disabling Coordinator and keeping position ...")
            if self.timer < 3 * SPIN_RATE:
                self.coordinatorDisable(True)
                self.disabled = True
            if self.orig_position is None:
                self.orig_position = [deepcopy(self._nav.position.north),
                                      deepcopy(self._nav.position.east),
                                      0.0, #deepcopy(self._nav.position.depth),
                                      0.0, 0.0,
                                      deepcopy(self._nav.orientation.yaw)]
            self.pilot_request.publish(PilotRequest(position=self.orig_position,
                                                    disable_axis=[0,0,0,0,0,0], priority=127))
        else:
            print("running ok")
            if self.disabled:
                self.coordinatorDisable(False)
                self.disabled = False
                self.orig_position = None
        self.timer += 1

# Testing/usage Script for above class
if __name__ == '__main__':
    # Set Node Name and Spin Rate
    rospy.init_node('safety_switch')
    ros_rate = rospy.Rate(SPIN_RATE)

    switch = SafetySwitch()

    while not rospy.is_shutdown():
        switch.loop()
        ros_rate.sleep()
