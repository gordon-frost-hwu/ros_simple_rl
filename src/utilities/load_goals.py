#! /usr/bin/env python
""" This module provides a function which takes the sum of N gaussian distributions """
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
import numpy as np
from rospkg import RosPack
import os
rospack = RosPack()

# List of goals to calculate distances etc... [North, East, Depth, Class]
# "Class" of goal determines whether the point is of interest or should be feared etc...
# Read the Goal Positions from a file (as they are required across multiple ROS nodes)
# emotion_model_pkg_path = rospack.get_path('emotion_model')
# print("GOAL_POSITIONS shape: [%s, %s]" % GOAL_POSITIONS.shape)
# print("Idx of goal point type: %s" % POINT_TYPE_IDX)
#def load_goals():
#	goals = np.zeros([1,6])
#	return goals
	
def load_goals():
    """ Wrapper function to maintain compatability with legacy code """
    print("@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Loading Data for simulation TYPE: ")
    try:
        scenario = rospy.get_param('/simulation_type')
        print("Simulation Type @@@@@@@@@@: {0}".format(scenario))
        GOAL_POSITIONS = load_goals_by_type(scenario)
    except KeyError:
        # no parameter set so use the default data file
        print("WARNING: /simulation_type param not set - using default data files")
        GOAL_POSITIONS = load_goals_by_type()
    print("------------------------")
    return GOAL_POSITIONS

def load_goals_using_path(path):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Loading Data for simulation from path: ")
    abs_path = os.path.abspath(path)
    filename = path.split("/")[-1]
    directory_path = abs_path.replace(filename, "")
    goal_data = np.loadtxt(directory_path + "default_goal_location.csv", comments='#', delimiter=' ')
    other_points_data = np.loadtxt(directory_path + filename, comments='#', delimiter=' ')

    if len(goal_data.shape) == 1:
        goal_data = goal_data.reshape([1, goal_data.shape[0]])
    if len(other_points_data.shape) == 1:
        other_points_data = other_points_data.reshape([1, other_points_data.shape[0]])

    GOAL_POSITIONS = np.zeros([goal_data.shape[0] + other_points_data.shape[0], goal_data.shape[1]])
    # print(GOAL_POSITIONS.shape)
    GOAL_POSITIONS[0, :] = goal_data
    GOAL_POSITIONS[1:, :] = other_points_data
    print("GOAL_POSITIONS:\n{0}".format(GOAL_POSITIONS))
    print("------------------------")
    return GOAL_POSITIONS

def load_goals_by_type(load_type=None):
    emotion_model_pkg_path = rospack.get_path('ros_simple_rl')
    f = open(emotion_model_pkg_path + '/config/default_goal_location.csv', 'r')
    columns = f.readline()
    columns = columns.strip('#\n').split(' ')
    f.close()
    goal_data = np.loadtxt(emotion_model_pkg_path + '/config/default_goal_location.csv', comments='#', delimiter=' ')
    # GOAL_POSITIONS.append(list(goal_data))
    if load_type is not None:
        if load_type == "avoidance":
            other_points_data = np.loadtxt(emotion_model_pkg_path + '/config/points_of_interest_avoidance.csv', comments='#', delimiter=' ')
        elif load_type == "thruster":
            other_points_data = np.loadtxt(emotion_model_pkg_path + '/config/points_of_interest_thruster.csv', comments='#', delimiter=' ')
        elif load_type == "inspect":
            other_points_data = np.loadtxt(emotion_model_pkg_path + '/config/points_of_interest_inspect.csv', comments='#', delimiter=' ')
    else:
        other_points_data = np.loadtxt(emotion_model_pkg_path + '/config/points_of_interest_default.csv', comments='#', delimiter=' ')

    # GOAL_POSITIONS.append(list(other_points_data))
    if len(goal_data.shape) == 1:
        goal_data = goal_data.reshape([1, goal_data.shape[0]])
    if len(other_points_data.shape) == 1:
        other_points_data = other_points_data.reshape([1, other_points_data.shape[0]])

    # print("goal data shape: {0}".format((goal_data.shape)))
    # print("other points shape: {0}".format((other_points_data.shape)))
    # print(goal_data.shape[0])
    # print(goal_data.shape[1])
    GOAL_POSITIONS = np.zeros([goal_data.shape[0] + other_points_data.shape[0], goal_data.shape[1]])
    # print(GOAL_POSITIONS.shape)
    GOAL_POSITIONS[0, :] = goal_data
    GOAL_POSITIONS[1:, :] = other_points_data
    print("GOAL_POSITIONS:\n{0}".format(GOAL_POSITIONS))
    return GOAL_POSITIONS
