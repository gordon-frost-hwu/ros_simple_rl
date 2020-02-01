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

SEND_VEL_REQUEST = True

class Coordinator(object):
    _surgeresult = None
    _yawresult = None
    _swayresult = None
    _pitchresult = None
    _depthresult = None

    _disableOutput = True       # By Default, output is disabled!!!

    def __init__(self):
        sub = rospy.Subscriber("/behaviours/surge/output", Vector6, self.surgeCallback)
        sub = rospy.Subscriber("/behaviours/sway/output", Vector6, self.swayCallback)
        sub = rospy.Subscriber("/behaviours/yaw/output", Vector6, self.yawCallback)
        sub = rospy.Subscriber("/behaviours/pitch/output", Vector6, self.pitchCallback)
        sub = rospy.Subscriber("/behaviours/depth/output", Vector6, self.depthCallback)
        self.pilot_vel_pub = rospy.Publisher("/pilot/velocity_req", PilotRequest)
        self.pilot_pos_pub = rospy.Publisher("/pilot/position_req", PilotRequest)

        service = rospy.Service('/behaviours/coordinator/disable_output', DisableOutput, self._disableOutputCallback)

    def _disableOutputCallback(self, req):
        self._disableOutput = req.disable
        return DisableOutputResponse()

    def surgeCallback(self, msg):
        self._surgeresult = msg.values
    def swayCallback(self, msg):
        self._swayresult = msg.values
    def pitchCallback(self, msg):
        self._pitchresult = msg.values
    def depthCallback(self, msg):
        self._depthresult = msg.values
    def yawCallback(self, msg):
        self._yawresult = msg.values


    def copyResults(self):
        """
        method used to synchronise the inputs of this module, which are the outputs of the individual behaviours
        :return:tuple containing the results from each behaviour. Deepcopy is used so that the values do not change
        inside the timestep.
        """
        return np.array(deepcopy(self._surgeresult)), np.array(deepcopy(self._yawresult)), \
               np.array(deepcopy(self._swayresult)), np.array(deepcopy(self._pitchresult)), \
                np.array(deepcopy(self._depthresult))

    def coordinationLoop(self):
        """
        Loop which continuously runs coordinating the outputs of all active behaviours
        """

        if None not in [self._surgeresult, self._swayresult, self._depthresult,
                                    self._pitchresult, self._yawresult]:

            su_results, y_results, sw_results, p_results, d_results = self.copyResults()

            # Combine the behaviours outputs
            # Just plain adding them just now
            desired_velocities = su_results + sw_results + d_results + y_results    # TODO reinsert Surge and pitch!!!!!!

            if not self._disableOutput:
                print("Sending Request: [su        sw        dep       rol       pit       yaw]")
                print("                 [{0:.2f}    {1:.2f}     {2:.2f}     {3:.2f}     {4:.2f}     {5:.2f}]".format(
                                                                                             desired_velocities[0],
                                                                                             desired_velocities[1],
                                                                                             desired_velocities[2],
                                                                                             desired_velocities[3],
                                                                                             desired_velocities[4],
                                                                                             desired_velocities[5]))

                if SEND_VEL_REQUEST:
                    # Send a velocity request msg to the vehicle pilot
                    self.pilot_vel_pub.publish(PilotRequest(velocity=desired_velocities))
                else:
                    # Send a force request - easiest to implement new subscriber for this in node_pilot.py??
                    pass
            else:
                pass
                # ToDo: Change this to zero velocity request for Nessie real world tests?
                # self.pilot_pos_pub.publish(PilotRequest(position=[0, 0, 0, 0, 0, 1.0]))
        else:
            print("Waiting for all actions to have outputted something ...")


if __name__ == '__main__':
    # Set the Node Name and Spin Rate -- this spin rate should be lower than all of the behaviours
    rospy.init_node('coordinator')
    # Spin at 5Hz
    ros_rate = rospy.Rate(5)

    coordinator = Coordinator()

    # Publish the desired waypoint goal to all active behaviours
    # goal_pub = rospy.Publisher("/behaviours/goal", Vector6)
    # goal_pub.publish(Vector6(values=[10, 0, 0, 0, 0, 0]))

    while not rospy.is_shutdown():
        coordinator.coordinationLoop()
        ros_rate.sleep()
