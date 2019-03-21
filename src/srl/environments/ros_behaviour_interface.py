
import roslib;roslib.load_manifest("ros_simple_rl")
import rospy
from srv_msgs.srv import DisableOutput, ChangeParam, DoHover, LogNav
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool
from vehicle_interface.msg import Vector6
from nav_class import Nav

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

        terminate_run_sub = rospy.Subscriber("/terminate_run", Bool, self._terminateRunCallback)

        Nav.__init__(self)
        self.position = self._nav.position
        self.orientation = self._nav.orientation
        self.orientation_rate = self._nav.orientation_rate

    def _terminateRunCallback(self, msg):
        self.ManualEndRun = msg.data

    def pilot(self, position_request_as_list):
        """
        :param req_list: 6 element list which is used in the Vector6 msg type of the pilot request
        :return: nothing
        """
        assert type(position_request_as_list) == list, "ROSBehaviourInterface.pilot -> request type must be list of 6 length"
        self.pilot_srv(Vector6(values=position_request_as_list))

    def performAction(self, param_name, param_value, reverse_flag=False, offset=1.0):
        self.yaw_ros_action(param_name, param_value, reverse_flag, offset)
