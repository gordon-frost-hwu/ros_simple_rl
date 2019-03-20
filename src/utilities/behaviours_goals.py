#! /usr/bin/env python
import roslib; roslib.load_manifest("ros_simple_rl")
from my_msgs.msg import BehavioursGoal
from vehicle_interface.msg import Vector6
import rospy
import sys

class BehavioursGoalPublisher(object):
    def __init__(self):
        self.pub = rospy.Publisher("/behaviours/goal", BehavioursGoal, queue_size=1)
    def publish(self, position, velocity=None):
        # Method to keep legacy code working by just changing the Pulisher
        # statement to an instantiation of this class instead
        assert len(position) == 6, "Goal Position must be a 6 element list"
        assert len(velocity) == 6, "Goal Velocity must be a 6 element list"
        self.pub.publish(BehavioursGoal(goal_position=Vector6(position), goal_velocity=Vector6(velocity)))

if __name__ == '__main__':
    rospy.init_node("goal_publisher")
    args = sys.argv[1:]

    if "-h" in args:
        # print help info on terminal usage
        print("USAGE: python behaviours_goals.py north east depth roll pitch yaw [-v] OR [velocities in Vector6]")
        sys.exit(0)

    behaviours_goal_publisher = BehavioursGoalPublisher()

    if "-wp" in args:
        wp_index = args.index("-wp")
        position_req_str = args[wp_index+1:wp_index+7]
        position_req = [float(arg) for arg in position_req_str]
    else:
        position_req = [1000, 1000, 1000, 1000, 1000, 1000]

    if "-ip" in args:
        ip_index = args.index("-ip")
        ip_req_str = args[ip_index+1:ip_index+7]
        ip_req = [float(arg) for arg in ip_req_str]
    else:
        ip_req = [1000, 1000, 1000, 1000, 1000, 1000]

    while not rospy.is_shutdown():
        print("Sending to Behaviours:")
        print("     WP Position: {0}".format(position_req))
        print("     IP Position: {0}".format(ip_req))
        behaviours_goal_publisher.publish(position_req, ip_req)

        rospy.sleep(0.5)

    rospy.spin()