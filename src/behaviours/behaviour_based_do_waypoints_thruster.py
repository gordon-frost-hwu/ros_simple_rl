#! /usr/bin/env python
""" This node takes a text file with list of points as input and sequentially sends these as waypoint requests
"""
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from auv_msgs.msg import NavSts, WorldWaypointReq
from my_msgs.msg import Goals, BehavioursGoal
from vehicle_interface.msg import Vector6
from srv_msgs.srv import ChangeParam, DoHover, DisableOutput, LogNav
from vehicle_interface.srv import DictionaryService
from diagnostic_msgs.msg import KeyValue
import utils
import sys
from rospkg import RosPack
rospack = RosPack()
from utilities.nav_class import Nav
TIME_BETWEEN_WAYPOINTS = 5    # The time to sleep for between sending waypoints
GOAL_PRECISION = 1.0

def navCallback(msg):
    global _stats
    _stats = msg

if __name__ == '__main__':
    args = sys.argv
    print(args)
    if len(args) > 1:
        # print("Using path: {0}".format(args[1]))
        option = args[2]
        if option == "c":
            path = "/data/clockwise_loop.csv"
        elif option == "cc":
            path = "/data/anti_clockwise_loop.csv"
        elif option == "e":
            path = "/data/eggtimer.csv"
        elif option == "o":
            path = "/data/original_curve_pattern_waypoints.csv"
    else:
        path = "/data/waypoints_6d.csv"
    print("Using Trajectory File: {0}".format(path.split("/")[-1]))
    global _stats
    _stats = None
    rospy.init_node('lawnmower_sender')
    #pub_wp = rospy.Publisher("/pilot/world_waypoint_req", WorldWaypointReq)
    nav = rospy.Subscriber("/nav/nav_sts", NavSts, navCallback)
    disable_behaviours = rospy.ServiceProxy('/behaviours/coordinator/disable_output', DisableOutput)
    disable_behaviours(True)

    # Make sure that the vehicle starts from the origin
    print("Sending vehicle to origin ...")
    pilot_request = rospy.ServiceProxy('/do_hover', DoHover)
    pilot_request(Vector6(values=[0, 0, 0, 0, 0, 0.0]))

    # Send the goal point location to the behaviour-based module
    goal_pub = rospy.Publisher("/behaviours/goal", BehavioursGoal)
    
    srv_fault = rospy.ServiceProxy('/thrusters/faults', DictionaryService)

    # Service to enable logging of the navigation data (saves to /tmp/)
    logNav = rospy.ServiceProxy("/nav/log_nav", LogNav)

    # Read in the desired ordered waypoints from file
    behaviours_pkg_path = rospack.get_path('behaviours')
    waypoints_file_path = behaviours_pkg_path + path
    wps = []
    with open(waypoints_file_path, 'r') as f:
        wps_raw = f.read().splitlines()
    wps = [point.split(",") for point in wps_raw]

    # Wait for Nav
    while not rospy.is_shutdown() and _stats == None:
        rospy.sleep(0.5)
        print "Waiting for nav"

    # enable the behaviour based coordinators output
    disable_behaviours(False)
    healths = [85, 17]
    for health in healths:
        fault_type = KeyValue(key='fault_type',value='hard_limit')
        th_min = KeyValue(key='th_min', value='-{0}, -85, -85, -85, -85, -85'.format(str(-health)))
        th_max = KeyValue(key='th_max', value='{0}, 85, 85, 85, 85, 85'.format(str(health)))
        thruster_fault_response = srv_fault(request=[fault_type, th_min, th_max])

        # Start logging the navigation data
        logNav(log_nav=True, file_name_descriptor="0")
        idx = 1
        for wp in wps:
            north = float(wp[0])
            east  = float(wp[1])
            wp_msg = WorldWaypointReq()
            print wp[0]
            wp_msg.position.north = north
            wp_msg.position.east  = east
            wp_msg.position.depth = 0
            wp_msg.goal.id = idx

            # Wait until the waypoint has been reached
            # depth error set to 10.0 atm due to depth not being controlled currently
            if _stats != None and wp_msg != None:
                while not(  rospy.is_shutdown() and utils.epsilonEqualsNED(_stats.position, wp_msg.position, GOAL_PRECISION, depth_e=10.0)):
                    check = utils.epsilonEqualsNED(_stats.position, wp_msg.position, GOAL_PRECISION, depth_e=10.0)
                    print("Waypoint Achieved: {0}".format(check))
                    print("Sending Goal Point: [{0}, {1}]".format(wp_msg.position.north, wp_msg.position.east))
                    goal_pub.publish(Vector6(values=[north, east, 0, 0, 0, 0]))
                    rospy.sleep(1)
                    if check == True:
                        break
                    if rospy.is_shutdown():
                        break
            else:
                print "something was None!!"
                print _stats
                print wp_msg.position
            idx = idx + 1
            print "-------> moving on to next waypoint"

        # Stop the logging of navigation data
        logNav(log_nav=False, file_name_descriptor="0")
        # Disable the behaviour based output in case any waypoint requests wish to be made
        disable_behaviours(True)
        print "Finished the List of Waypoints"
        print "-------------END---------------"
