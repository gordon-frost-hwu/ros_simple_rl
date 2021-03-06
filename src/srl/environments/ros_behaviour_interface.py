import roslib;

roslib.load_manifest("ros_simple_rl")
import rospy
from srv_msgs.srv import DisableOutput, ChangeParam, DoHover, LogNav, DoHoverResponse
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool
from auv_msgs.msg import NED, RPY, NavSts
from vehicle_interface.msg import Vector6, PilotRequest
from vehicle_interface.srv import BooleanService, BooleanServiceRequest
from utilities.nav_class import Nav
import random
import numpy as np
import utils


class ROSBehaviourInterface(Nav):
    """ Class that encapsulates the ROS services needed to use the Nessie platform in a RL environment using the
    behavioural architecture given by 'roslaunch behaviours behaviours_thruster_underactuated.launch'
    """

    def __init__(self):
        self.ManualEndRun = False
        # Services
        self.disable_behaviours = rospy.ServiceProxy('/behaviours/coordinator/disable_output', DisableOutput)
        self.pilot_srv = rospy.ServiceProxy('/do_hover', DoHover)
        self.yaw_ros_action = rospy.ServiceProxy('/behaviours/yaw/params', ChangeParam)
        self.nav_reset = rospy.ServiceProxy("/nav/reset", Empty)
        self.logNav = rospy.ServiceProxy("/nav/log_nav", LogNav)
        self.disable_pilot_srv = rospy.ServiceProxy('/pilot/switch', BooleanService)

        self.service = rospy.Service('do_hover', DoHover, self._do_hover_callback)

        terminate_run_sub = rospy.Subscriber("/terminate_run", Bool, self._terminateRunCallback)

        self.pilot_position_request = rospy.Publisher("/pilot/position_req", PilotRequest)

        Nav.__init__(self)
        self.position = self._nav.position
        self.orientation = self._nav.orientation
        self.orientation_rate = self._nav.orientation_rate

    def pilotPublishPositionRequest(self, pose):
        request = PilotRequest()
        request.position = pose
        self.pilot_position_request.publish(request)

    def _terminateRunCallback(self, msg):
        self.ManualEndRun = msg.data

    def pilot(self, position_request_as_list):
        """
        :param req_list: 6 element list which is used in the Vector6 msg type of the pilot request
        :return: nothing
        """
        assert type(
            position_request_as_list) == list, "ROSBehaviourInterface.pilot -> request type must be list of 6 length"
        self.pilot_srv(Vector6(values=position_request_as_list))

    def performAction(self, param_name, param_value, reverse_flag=False, offset=1.0):
        self.yaw_ros_action(param_name, param_value, reverse_flag, offset)

    def reset(self, disable_behaviours=True):
        if disable_behaviours:
            self.disable_behaviours(True)
        # while (np.abs(self.orientation.yaw - 1.57) > 0.1)
        start_yaw = 1.57  # random.randrange(-30.0, 30.0, 5) / 10.0
        self.pilot([0, 0, 0, 0, 0, start_yaw])
        rospy.sleep(2)
        #    print("Sending to start position")
        if disable_behaviours:
            self.disable_behaviours(False)

    def enable_pilot(self, req):
        srv_req = BooleanServiceRequest()
        srv_req.request = req
        self.disable_pilot_srv(srv_req)

    def getReward(self, state, action):
        angle = abs(state["angle"])
        angle_dt = state["angle_deriv"]
        if angle_dt < 0:
            neg_dt = abs(angle_dt)
        else:
            neg_dt = 0

        if angle_dt >= 0:
            pos_dt = angle_dt
        else:
            pos_dt = 0
        s = np.array([angle, neg_dt, pos_dt, np.abs(action)])
        # was [-20.0, -30.0, 15.0, -0.5]
        weights = np.array([-20.0, 0.0, 0.0, -2.0])
        reward = np.dot(s, weights)
        # reward = -10.0 + (10.0 * (1.0 - angle))
        return reward
    
    def _do_hover_callback(self, req):
        goal = req.goal.values
        rospy.loginfo(
            "DoHover: received goal [{0}, {1}, {2}, {4}, {5}]".format(goal[0], goal[1], goal[2], goal[3], goal[4],
                                                                      goal[5]))
        action_start_time = rospy.get_time()
        timeoutReached = False
        # take the received goal and push it to the motor controls semantics
        # get a publisher
        pub = rospy.Publisher("/pilot/position_req", PilotRequest)

        goalNED = NED();
        goalNED.north, goalNED.east, goalNED.depth = goal[0], goal[1], goal[2]
        goalRPY = RPY();
        goalRPY.roll, goalRPY.pitch, goalRPY.yaw = goal[3], goal[4], goal[5]

        # repeatedly call world waypoint req to move the robot to the goal
        # while the robot is not at teh desired location
        while not (rospy.is_shutdown() or
                   self._nav != None and
                   utils.epsilonEqualsNED(self._nav.position, goalNED, 0.5, depth_e=0.6) and
                   utils.epsilonEqualsY(self._nav.orientation, goalRPY, .2)
                # we compare on just pitch and yaw for 5 axis robot
        ):

            # publish the goal repeatedly
            pilotMsg = PilotRequest()
            pilotMsg.position = list(goal)
            pub.publish(pilotMsg)
            # rospy.loginfo("Sent Goal!!")
            rospy.sleep(0.5)

            if timeoutReached:
                return DoHoverResponse(False)
        print("Sleeping for a while ...")
        rospy.sleep(4)
        # pub and subscriber will be removed at the end of executes context
        rospy.loginfo("DoHover: complete")
        return DoHoverResponse(True)
