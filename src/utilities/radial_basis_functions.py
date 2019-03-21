#! /usr/bin/env python
import roslib;roslib.load_manifest("ros_simple_rl")
import rospy
from utilities.load_goals import load_goals
from utilities.gaussian import gaussian2D
from vehicle_interface.msg import Vector6
from srv_msgs.srv import Features, FeaturesResponse
from std_msgs.msg import Float32
from my_msgs.msg import BehavioursGoal

# List of goals to calculate distances etc... [North, East, Depth, Class]
# "Class" of goal determines whether the point is of interest or should be feared etc...

GOAL_POSITIONS = load_goals()
POINT_TYPE_IDX = -1    # minus one due to indexing of last element

def radialBasisFunctions4(req):
    """ function which creates a Gaussian X distance away from the goal point in 4 Directions to be used as features """
    north, east = req.position.values[0], req.position.values[1]
    cnt = 0
    while cnt < len(GOAL_POSITIONS):
        if GOAL_POSITIONS[cnt][POINT_TYPE_IDX] == 1:
            x0 = GOAL_POSITIONS[cnt][0]; y0 = GOAL_POSITIONS[cnt][1]
            variance_x = GOAL_POSITIONS[cnt][-3]; variance_y = GOAL_POSITIONS[cnt][-2]
        cnt += 1
    step_size = 5.0
    rbf1 = gaussian2D(north, east, x0 + step_size, y0, 1.0, [4.0, 4.0])
    rbf2 = gaussian2D(north, east, x0 - step_size, y0, 1.0, [4.0, 4.0])
    rbf3 = gaussian2D(north, east, x0, y0 + step_size, 1.0, [4.0, 4.0])
    rbf4 = gaussian2D(north, east, x0, y0 - step_size, 1.0, [4.0, 4.0])
    return FeaturesResponse(features=[Float32(rbf1), Float32(rbf2), Float32(rbf3), Float32(rbf4)])

def radialBasisFunctionsCircle(req):
    """ function which creates a Gaussian X distance away from the goal point in 4 Directions to be used as features """
    north, east = req.position.values[0], req.position.values[1]
    cnt = 0
    while cnt < len(GOAL_POSITIONS):
        if GOAL_POSITIONS[cnt][POINT_TYPE_IDX] == 1:
            x0 = GOAL_POSITIONS[cnt][0]; y0 = GOAL_POSITIONS[cnt][1]
            variance_x = GOAL_POSITIONS[cnt][-3]; variance_y = GOAL_POSITIONS[cnt][-2]
        cnt += 1
    step_size = 5.0
    rbf1 = gaussian2D(north, east, x0 + step_size, y0, 1.0, [4.0, 4.0])
    rbf2 = gaussian2D(north, east, x0 - step_size, y0, 1.0, [4.0, 4.0])
    rbf3 = gaussian2D(north, east, x0, y0 + step_size, 1.0, [4.0, 4.0])
    rbf4 = gaussian2D(north, east, x0, y0 - step_size, 1.0, [4.0, 4.0])
    return FeaturesResponse(features=[Float32(rbf1), Float32(rbf2), Float32(rbf3), Float32(rbf4)])

def updateGoalPositionCallback(msg):
    global GOAL_POSITIONS

    cnt = 0
    while cnt < len(GOAL_POSITIONS):
        if GOAL_POSITIONS[cnt][POINT_TYPE_IDX] == 1:
            GOAL_POSITIONS[cnt][0:6] = msg.goal_position.values
        cnt += 1
    # print("Updated Goal:\n{0}".format(GOAL_POSITIONS))


if __name__ == '__main__':
    global GOAL_POSITIONS
    rospy.init_node("dynamic_basis_functions")
    sub = rospy.Subscriber("/behaviours/goal", BehavioursGoal, updateGoalPositionCallback)
    srv = rospy.Service("/compute_rbfs", Features, radialBasisFunctions4)

    rospy.spin()
