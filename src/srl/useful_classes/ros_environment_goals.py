#! /usr/bin/python
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy

from my_msgs.msg import Goals

class EnvironmentInfo(object):
    def __init__(self):
        ros_env_sub = rospy.Subscriber('/goal_points_data', Goals, self.rosEnvCallback)
        self.raw_angle_to_goal = 0.0
        self.distance_to_goal = 1000000.0

    def rosEnvCallback(self, msg):
        """
        :param msg: Goals msg which contains the environment data such as angle to goal wrt. to vehicle heading
        """
        for point in msg.goals:
            # Get the angle angle to the point-of-interest; 1 = waypoint_goal, 2 = inspection_goal, 3 = fear_goal
            if point.point_type == 1.0:
                self.raw_angle_to_goal = point.angle_wrt_vehicle
                self.distance_to_goal = point.distance
