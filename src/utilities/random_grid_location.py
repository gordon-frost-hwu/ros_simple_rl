#! /usr/bin/python

import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from random import randint
from auv_msgs.msg import NED, RPY, VehiclePose
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler
from numpy import pi, deg2rad, rad2deg, sqrt, arctan
import sys
from utilities.nav_class import Nav
WORLD_FRAME = "/map"

def random_location_in_grid(corners, spacing=1.0):
    north_max, north_min = corners[0][0], corners[2][0]
    east_max, east_min = corners[1][1], corners[0][1]
    return [randint(north_min, north_max), randint(east_min, east_max)]

def convert_input_string(str_list):
    list_of_string_coords = str_list.replace("[", "").replace("]", "").split(",")   # elements of type string
    list_of_coords = [float(item) for item in list_of_string_coords]    # elements of type float
    return list_of_coords

def help_information():
    print("Usage:")
    print("python random_grid_location.py [coord-1] [coord-2] [coord-3] [coord-4]")
    print("    1----2")
    print("    |    |")
    print("    |    |")
    print("    3----4")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if "-h" in sys.argv:
            help_information()
        else:
            corners = [convert_input_string(corner) for corner in sys.argv[1:len(sys.argv)]]
            print("Corners Given: {0}".format(corners))

            # calculate a random position within the given area (grid)
            point = random_location_in_grid(corners)
            print("Random Location Chosen: {0}".format(point))
    else:
        # run as a ros node to publish the calculated resultant velocity vector to Rviz
        print("Please provide arguments to script ....")
        help_information()