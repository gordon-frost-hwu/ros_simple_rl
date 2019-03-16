#! /usr/bin/env python
""" This node takes a text file with list of points as input and sequentially sends these as waypoint requests
"""
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from auv_msgs.msg import NavSts, WorldWaypointReq, VehiclePose
from vehicle_interface.msg import PilotRequest, Vector6
from srv_msgs.srv import LogNav
from std_srvs.srv import Empty
from my_msgs.msg import Goals, BehavioursGoal
from vehicle_interface.srv import DictionaryService
from diagnostic_msgs.msg import KeyValue
import utils
import math

TIME_BETWEEN_WAYPOINTS = 5	# The time to sleep for between sending waypoints
PUBLISH_RATE = 0.2

class EnvironmentInfo(object):
    def __init__(self):
        ros_env_sub = rospy.Subscriber('/goal_points_data', Goals, self.rosEnvCallback)
        self.raw_angle_to_goal = 0.0
        self.distance_to_goal = 0.0

    def rosEnvCallback(self, msg):
        """
        :param msg: Goals msg which contains the environment data such as angle to goal wrt. to vehicle heading
        """
        for point in msg.goals:
            # Get the angle angle to the point-of-interest; 1 = waypoint_goal, 2 = inspection_goal, 3 = fear_goal
            if point.point_type == 1.0:
                self.raw_angle_to_goal = point.angle_wrt_vehicle
                self.distance_to_goal = point.distance

class Thrusters(object):
    def __init__(self):
        self.srv_fault = rospy.ServiceProxy('/thrusters/faults', DictionaryService)

    def inject_surge_thruster_fault(self, new_surge_health, index_of_fault):
        fault_type = KeyValue(key='fault_type',value='hard_limit')
        print("Injecting Fault of {0} into thruster {1}".format(new_surge_health, index_of_fault))
        if index_of_fault == 0:
            th_min = KeyValue(key='th_min', value='-{0}, -85, -85, -85, -85, -85'.format(str(new_surge_health)))
            th_max = KeyValue(key='th_max', value='{0}, 85, 85, 85, 85, 85'.format(str(new_surge_health)))
        elif index_of_fault == 1:
            th_min = KeyValue(key='th_min', value='-85, -{0}, -85, -85, -85, -85'.format(str(new_surge_health)))
            th_max = KeyValue(key='th_max', value='85, {0}, 85, 85, 85, 85'.format(str(new_surge_health)))

        thruster_fault_response = self.srv_fault(request=[fault_type, th_min, th_max])

        last_thruster_failure_index = index_of_fault
        last_thruster_health = new_surge_health
        print("Thruster Health changed to:\n{0}".format(thruster_fault_response))
        print(thruster_fault_response.response[-1].value)


def navCallback(msg):
    global _stats
    _stats = msg

if __name__ == '__main__':
    global _stats
    _stats = None
    rospy.init_node('lawnmower_sender')
    pub_wp = rospy.Publisher("/pilot/position_req", PilotRequest)
    pub_behaviour_goal = rospy.Publisher("/behaviours/goal", BehavioursGoal)
    nav = rospy.Subscriber("/nav/nav_sts", NavSts, navCallback)
    logNav = rospy.ServiceProxy("/nav/log_nav", LogNav)
    nav_reset = rospy.ServiceProxy("/nav/reset", Empty)

    environment = EnvironmentInfo()
    thrusters = Thrusters()
    thrusters.inject_surge_thruster_fault(17, 1)
    # wps = []
    # with open("/home/gordon/ros_workspace/auv_learning/behaviours/data/waypoints_2d.csv") as f:
    #     wps_raw = f.read().splitlines()
    # wps = [point.split(",") for point in wps_raw]

    # Wait for Nav
    while not rospy.is_shutdown() and _stats == None:
        rospy.sleep(0.5)
        print "Waiting for nav"

    wps = [[0, 0, 2.35], [10, 0, 0.0, 0.0]]
    nav_reset()
    rospy.sleep(2)

    logNav(log_nav=True, dir_path="/tmp/", file_name_descriptor=str(0))

    for wp in wps:
        wp_msg = PilotRequest()
        wp_msg.disable_axis = [0, 1, 0, 0, 0, 0]
        goal_msg = BehavioursGoal()
        cnt = 0
        while not rospy.is_shutdown():

            if wp != wps[0]:
                angle = math.atan2(wp[1] - _stats.position.east, wp[0] - _stats.position.north)
                # new_heading = environment.raw_angle_to_goal + _stats.orientation.yaw
                wp_msg.position = [wp[0], wp[1], 0, 0, 0, angle]
            else:
                wp_msg.position = [wp[0], wp[1], 0, 0, 0, wp[2]]

            goal_msg.goal_position = Vector6(values=[wp[0], wp[1], 0, 0, 0, wp[2]])

            pub_wp.publish(wp_msg)
            pub_behaviour_goal.publish(goal_msg)

            rospy.sleep(PUBLISH_RATE)

            print("cnt: {0}\ndistance: {1}".format(cnt, environment.distance_to_goal))

            if abs(_stats.orientation.yaw - wp[2]) < 0.1:
                cnt += 1

            if cnt > 10 * PUBLISH_RATE and environment.distance_to_goal < 1.0:
                break

            if rospy.is_shutdown():
                logNav(log_nav=False, dir_path="/tmp/", file_name_descriptor=str(0))
                break
        rospy.sleep(TIME_BETWEEN_WAYPOINTS)

        print "-------> moving on to next waypoint"

    logNav(log_nav=False, dir_path="/tmp/", file_name_descriptor=str(0))

    print "Finished the Lawn Mower Pattern"
    print "-------------END---------------"
