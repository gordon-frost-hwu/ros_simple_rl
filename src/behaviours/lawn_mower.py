#! /usr/bin/env python
""" This node takes a text file with list of points as input and sequentially sends these as waypoint requests
"""
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from auv_msgs.msg import NavSts, WorldWaypointReq, VehiclePose
import utils

TIME_BETWEEN_WAYPOINTS = 0.5	# The time to sleep for between sending waypoints

def navCallback(msg):
    global _stats
    _stats = msg

if __name__ == '__main__':
    global _stats
    _stats = None
    rospy.init_node('lawnmower_sender')
    pub_wp = rospy.Publisher("/pilot/world_waypoint_req", WorldWaypointReq)
    nav = rospy.Subscriber("/nav/nav_sts", NavSts, navCallback)

    wps = []
    with open("/home/gordon/ros_workspace/auv_learning/behaviours/data/lawnmower.txt") as f:
        wps_raw = f.read().splitlines()
    wps = [point.split(",") for point in wps_raw]

    # Wait for Nav
    while not rospy.is_shutdown() and _stats == None:
        rospy.sleep(0.5)
        print "Waiting for nav"

    #wps = [[0, 0], [5, 5]]
    idx = 1
    for wp in wps:
        wp_msg = WorldWaypointReq()
        print wp[0]
        wp_msg.position.north = float(wp[0])
        wp_msg.position.east = float(wp[1])
        wp_msg.position.depth = 0
        wp_msg.goal.id = idx

        # Wait until the waypoint has been reached
        if _stats != None and wp_msg != None:
            while not(  rospy.is_shutdown() and utils.epsilonEqualsNED(_stats.position, wp_msg.position, 0.5, depth_e=0.6)):
                check = utils.epsilonEqualsNED(_stats.position, wp_msg.position, 0.5, depth_e=0.6)
                print("Waypoint Achieved: %s" % utils.epsilonEqualsNED(_stats.position, wp_msg.position, 0.5, depth_e=0.6))
                print("Sending point: N: %s, E: %s" % (wp_msg.position.north, wp_msg.position.east))
                pub_wp.publish(wp_msg)
                rospy.sleep(1)
                if check == True:
                    break
                if rospy.is_shutdown():
                    break
            rospy.sleep(TIME_BETWEEN_WAYPOINTS)
        else:
            print "something was None!!"
            print _stats
            print wp_msg.position
        idx = idx + 1
        print "-------> moving on to next waypoint"

    print "Finished the Lawn Mower Pattern"
    print "-------------END---------------"
