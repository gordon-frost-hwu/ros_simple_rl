#! /usr/bin/env python
""" This node takes a text file with list of points as input and sequentially sends these as waypoint requests
"""
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from auv_msgs.msg import NavSts, WorldWaypointReq
from vehicle_interface.msg import Vector6
from srv_msgs.srv import ChangeParam, DoHover, DisableOutput, LogNav
from std_srvs.srv import Empty, EmptyResponse
import utils
from utilities.nav_class import Nav
TIME_BETWEEN_WAYPOINTS = 5	# The time to sleep for between sending waypoints

def navCallback(msg):
    global _stats
    _stats = msg

if __name__ == '__main__':
    global _stats
    _stats = None
    rospy.init_node('lawnmower_sender')
    #pub_wp = rospy.Publisher("/pilot/world_waypoint_req", WorldWaypointReq)
    nav = rospy.Subscriber("/nav/nav_sts", NavSts, navCallback)
    disable_behaviours = rospy.ServiceProxy('/behaviours/coordinator/disable_output', DisableOutput)
    disable_behaviours(True)

    reset_nav = rospy.ServiceProxy('/nav/reset', Empty)

    # Make sure that the vehicle starts from the origin
    print("Sending vehicle to origin ...")
    pilot_request = rospy.ServiceProxy('/do_hover', DoHover)

    reset_nav()
    pilot_request(Vector6(values=[0, 0, 0, 0, 0, 0.0]))

    # Send the goal point location to the behaviour-based module
    goal_pub = rospy.Publisher("/behaviours/goal", Vector6)

    # Service to enable logging of the navigation data (saves to /tmp/)
    logNav = rospy.ServiceProxy("/nav/log_nav", LogNav)

    # Read in the desired ordered waypoints from file
    wps = []
    with open("/home/gordon/ros_workspace/auv_learning/behaviours/data/waypoints_6d.csv") as f:
        wps_raw = f.read().splitlines()
    wps = [point.split(",") for point in wps_raw]

    # Wait for Nav
    while not rospy.is_shutdown() and _stats == None:
        rospy.sleep(0.5)
        print "Waiting for nav"

    # enable the behaviour based coordinators output
    # disable_behaviours(False)

    # Start logging the navigation data
    print("starting to log nav data")

    idx = 0
    validation_set = [0.0, 0.0, 0.35, 0.35, 1.15, 1.15, 1.57, 1.57, 2.1, 2.1, 2.6, 2.6, 3.13, 3.13]
    for wp in validation_set:
        print("Starting from Yaw: {0}".format(wp))
        reset_nav()
        pilot_request(Vector6(values=[0, 0, 0, 0, 0, wp]))
        rospy.sleep(5)
        logNav(log_nav=True, file_name_descriptor=str(idx))
        disable_behaviours(False)
        # north = float(wp[0])
        # east  = float(wp[1])
        north = 10.0
        east = 10.0
        wp_msg = WorldWaypointReq()
        # print wp[0]
        wp_msg.position.north = north
        wp_msg.position.east  = east
        wp_msg.position.depth = 0
        wp_msg.goal.id = idx

        # Wait until the waypoint has been reached
        if _stats != None and wp_msg != None:
            while not(  rospy.is_shutdown() and utils.epsilonEqualsNED(_stats.position, wp_msg.position, 0.5, depth_e=10.0)):
                check = utils.epsilonEqualsNED(_stats.position, wp_msg.position, 0.5, depth_e=10.0)
                print("Waypoint Achieved: {0}".format(check))
                print("Sending Goal Point: [{0}, {1}]".format(wp_msg.position.north, wp_msg.position.east))
                goal_pub.publish(Vector6(values=[north, east, 0, 0, 0, wp]))
                rospy.sleep(1)
                if check == True:
                    break
                if rospy.is_shutdown():
                    break
        else:
            print "something was None!!"
            print _stats
            print wp_msg.position
        disable_behaviours(True)
        logNav(log_nav=False, file_name_descriptor=str(idx))
        idx = idx + 1
        print "-------> moving on to next waypoint"

    # Stop the logging of navigation data
    # logNav(log_nav=False, file_name_descriptor="0")
    # Disable the behaviour based output in case any waypoint requests wish to be made
    disable_behaviours(True)
    print "Finished the List of Waypoints"
    print "-------------END---------------"
