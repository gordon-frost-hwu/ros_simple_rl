#! /usr/bin/env python
#
# ROS Node which calculates the distances to an N element list of 2D points and determines whether the points are
# in view of a sythentic forward looking sonar.
# All angles/headings are in the range: [-3.14, 3.14] radians
# Outputs the heading of the points, w.r.t  to the vehicles heading
# and the euclidean distance to the points
# Author: Gordon Frost, date: 02/02/2015
#
import roslib; roslib.load_manifest('ros_simple_rl')
import rospy
import numpy as np
import math
from utilities.load_goals import load_goals
from utilities.cartesian_distance_fov import CartesianDistanceFov
from rospkg import RosPack
rospack = RosPack()

from auv_msgs.msg import NavSts
from my_msgs.msg import Goals, Goal, BehavioursGoal
from vehicle_interface.msg import Vector6

# List of goals to calculate distances etc... [North, East, Depth, Class]
# "Class" of goal determines whether the point is of interest or should be feared etc...
# Read the Goal Positions from a file (as they are required across multiple ROS nodes)
GOAL_POSITIONS = load_goals()

class NAVIGATION(object):
    def __init__(self):
        # Subscribe to the vehicle position
        sub = rospy.Subscriber("/nav/nav_sts", NavSts, self.navCallback)
        #sub = rospy.Subscriber("/synthetic_nav", SynthNav, self.synth_navCallback)
        self.vehicle_position = np.array([0, 0, 0])
        self.vehicle_orientation = np.array([0, 0, 0])

    def navCallback(self, msg):
        self.vehicle_position = [msg.position.north, msg.position.east, msg.position.depth]
        if msg.orientation.yaw < 0:
            yaw = (msg.orientation.yaw) #*(180/math.pi))+360
        else:
            yaw = msg.orientation.yaw #*(180/math.pi)
        self.vehicle_orientation = [msg.orientation.roll, msg.orientation.pitch, yaw]

    def synth_navCallback(self, msg):
        self.vehicle_position = [msg.north, msg.east, msg.depth]
        if msg.yaw < 0:
            yaw = (msg.yaw*(180/math.pi))+360
        else:
            yaw = msg.yaw*(180/math.pi)
        self.vehicle_orientation = [0, 0, yaw]

def goalCallback(behaviours_goal_msg):
    global new_goal
    global new_ip
    new_goal = behaviours_goal_msg.goal_position
    new_ip = behaviours_goal_msg.goal_velocity

if __name__=='__main__':
    rospy.init_node("perception_simulation_module")
    pub = rospy.Publisher("/goal_points_data", Goals, queue_size=1)
    global new_goal
    global new_ip
    new_goal = Vector6(values=[10, 10, 0, 0, 0, 0])
    new_ip = Vector6(values=[0, 0, 0, 0, 0, 0])
    goal_sub = rospy.Subscriber("/behaviours/goal", BehavioursGoal, goalCallback)

    # Instantiate an object to keep track of navigation info
    nav = NAVIGATION()

    #distance_to_goal = DistanceMeasure()
    # goal_objects = [DistanceMeasure(goal) for goal in GOAL_POSITIONS] #won't work with goal subscriber
    goal_objects = [CartesianDistanceFov(goal) for goal in GOAL_POSITIONS]

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        print "--------------"
        print(GOAL_POSITIONS)
        # Create a list of Goals msg types, 1 for each goal position
        #goal_data = [Goals() for goal in GOAL_POSITIONS]
        goalsMsg = Goals()
        goalsMsg_list = []

        for goal_object in goal_objects:
            print("Point Type: {0}".format(goal_object.goal_type))
            if goal_object.goal_type == 1:
                goal_object.updateGoalPosition(new_goal)
            elif goal_object.goal_type == 2:
                goal_object.updateGoalPosition(new_ip)
            goalMsg = Goal()

            goal_object.updateNav(nav)
            # Save the goals position and class into the msg to be sent out (used for emotional processing)
            goalMsg.position.north = goal_object.goal_position[0]
            goalMsg.position.east  = goal_object.goal_position[1]
            goalMsg.position.depth = goal_object.goal_position[2]
            goalMsg.orientation.pitch = goal_object.goal_position[-2]
            goalMsg.orientation.yaw = goal_object.goal_position[-1]
            goalMsg.point_type = goal_object.goal_type

            # Calculate the distance to the goal
            distance_to_goal = goal_object.calcDistance()

            # Now calculate whether the goal is in the FOV of the robot/agent
            goal_in_view = goal_object.can_seeGoal(distance_to_goal)

            # Put the distance and isVisible results into the msg to publish
            goalMsg.distance = distance_to_goal
            goalMsg.isVisible = goal_in_view
            goalMsg.angle_wrt_vehicle = goal_object.getAngleWrtVehicle()

            # Append the Goal msg type to the
            goalsMsg_list.append(goalMsg)


            if not goal_in_view:
                distance_to_goal = 0
                print("GOAL NOT IN VIEW!!")
            else:
                print "Distance to Goal Object: ", distance_to_goal
        # rospy.sleep(1)
        rate.sleep()
        goalsMsg.goals = goalsMsg_list
        print goalsMsg
        # Publish the goal points data (position, distances and type) to ROS
        pub.publish(goalsMsg)
        print "@@@@@@@@@@@@@@"

    rospy.spin()
