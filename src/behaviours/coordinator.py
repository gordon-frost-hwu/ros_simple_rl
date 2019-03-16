#! /usr/bin/env python
"""
    Module handles the coordination of the outputs from Behaviour derived classes such as TransitionalBehaviour and
    YawBehaviour.
    author: Gordon Frost; email: gwf2@hw.ac.uk
    date: 23/01/15
"""
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from copy import deepcopy
import numpy as np
from vehicle_interface.msg import PilotRequest, Vector6
from srv_msgs.srv import DisableOutput, DisableOutputResponse


# TODO: output of Actor module influences the balance between the TransitionalBehaviour and YawBehaviour outputs

class Coordinator(object):
    _tresult = None
    _yresult = None

    _disableOutput = True       # By Default, output is disabled!!!

    def __init__(self):
        sub = rospy.Subscriber("/behaviours/surge/output", Vector6, self.translationalCallback)
        sub = rospy.Subscriber("/behaviours/yaw/output", Vector6, self.yawCallback)
        self.pilot_vel_pub = rospy.Publisher("/pilot/velocity_req", PilotRequest)
        self.pilot_pos_pub = rospy.Publisher("/pilot/position_req", PilotRequest)

        service = rospy.Service('/behaviours/coordinator/disable_output', DisableOutput, self._disableOutputCallback)

    def _disableOutputCallback(self, req):
        self._disableOutput = req.disable
        return DisableOutputResponse()

    def translationalCallback(self, msg):
        # print("....")
        self._tresult = msg.values

    def yawCallback(self, msg):
        # print("@@@@@@")
        self._yresult = msg.values

    def copyResults(self):
        """
        method used to synchronise the inputs of this module, which are the outputs of the individual behaviours
        :return:tuple containing the results from each behaviour. Deepcopy is used so that the values do not change
        inside the timestep.
        """
        return np.array(deepcopy(self._tresult)), np.array(deepcopy(self._yresult))

    def coordinationLoop(self):
        """
        Loop which continuously runs coordinating the outputs of all active behaviours
        """
        if (self._yresult is not None and self._tresult is not None):

            # Combine the behaviours outputs
            # Just plain adding them just now #TODO: weighted combination of behaviours outputs
            r_results, y_results = self.copyResults()

            desired_velocities = r_results + y_results
            # desired_velocities = np.array(deepcopy(self._yresult))

            if not self._disableOutput:
                print("Sending Surge Request: {0}".format(desired_velocities[0]))
                print("Sending  Yaw  Request: {0}".format(desired_velocities[-1]))
                self.pilot_vel_pub.publish(PilotRequest(velocity = desired_velocities))
            else:
                pass
                # self.pilot_pos_pub.publish(PilotRequest(position = [0, 0, 0, 0, 0, 1.0]))
        else:
            rospy.loginfo("Waiting for all actions to have outputted something ...")


if __name__ == '__main__':
    # Set the Node Name and Spin Rate -- this spin rate should be lower than all of the behaviours
    rospy.init_node('coordinator')
    ros_rate = rospy.Rate(5)

    coordinator = Coordinator()

    # Publish the desired waypoint goal to all active behaviours
    # goal_pub = rospy.Publisher("/behaviours/goal", Vector6)
    # goal_pub.publish(Vector6(values=[10, 0, 0, 0, 0, 0]))

    while not rospy.is_shutdown():
        coordinator.coordinationLoop()
        ros_rate.sleep()