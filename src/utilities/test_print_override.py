#! /usr/bin/env python
from __future__ import print_function
import __builtin__

import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from std_msgs.msg import String

def print(*args, **kwargs):
    # __builtin__.print('Prefixed:', *args, **kwargs)
    rospy.loginfo(*args, **kwargs)

if __name__ == '__main__':
    rospy.init_node("test_print_override")
    pub = rospy.Publisher('chatter', String, queue_size=10)

    while not rospy.is_shutdown():
        # print("Sending to Behaviours:")
        # rospy.loginfo("Sending to Behaviours:")
        hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        pub.publish(hello_str)
        print("Sending to Behaviours: {0}".format(rospy.get_time()))
        print("     WP Position: na")
        rospy.sleep(1)

    rospy.spin()