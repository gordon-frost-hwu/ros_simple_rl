#! /usr/bin/env python

import numpy as np
import math
POINT_TYPE_IDX = -1
NESSIE_FOV = 120    # in degrees (obviously!). It is converted to radians in below class

class CartesianDistanceFov(object):
    def __init__(self, goal_position):
        """
        :param goal_position: contains North, East, Depth, and point type (integer)
        :return: Instantiated class object holding properties of a single goal point wrt to vehicle curr pos
        """
        self.vehicle_position = np.array([0, 0, 0])
        self.vehicle_orientation = np.array([0, 0, 0])
        self.goal_position = np.array(goal_position[0:6])
        self.goal_type = goal_position[POINT_TYPE_IDX]
        self.theta_wrt_vehicle_heading = None

    def updateGoalPosition(self, goal):
        # print("Goal: {0}".format(goal))
        self.goal_position = np.array(goal.values)

    def getAngleWrtVehicle(self):
        return self.theta_wrt_vehicle_heading

    def updateNav(self, nav):
        self.vehicle_position = nav.vehicle_position
        self.vehicle_orientation = nav.vehicle_orientation

    def calcDistance(self):
        dist_squared = (self.vehicle_position[0:2] - self.goal_position[0:2])**2
        return math.sqrt(dist_squared.sum())

    def can_seeGoal(self, distance_to_goal):
        """ Method which computes whether the goal is in FFS FOV """
        isVisible = False
        delta_north = (self.vehicle_position[0] - self.goal_position[0])
        delta_east = (self.vehicle_position[1] - self.goal_position[1])
        angle_to_goal = (math.atan2(delta_east, delta_north) + (math.pi))#*(180/math.pi) # 0 < angle-to-goal < +-2*pi
        print("Stage 1: {0}".format(angle_to_goal))
        #~ if angle_to_goal > 180:
            #~ angle_to_goal = angle_to_goal - 180
        vehicle_heading = self.vehicle_orientation[2]       # 0 < vehicle-heading < +- 2*pi radians
        
        tmp_angle_wrt_vehicle_heading = (angle_to_goal - vehicle_heading)
        if tmp_angle_wrt_vehicle_heading > math.pi:
            self.theta_wrt_vehicle_heading = -(2*math.pi - tmp_angle_wrt_vehicle_heading)
        else:
            self.theta_wrt_vehicle_heading = tmp_angle_wrt_vehicle_heading
        print("Stage 2: {0}".format(self.theta_wrt_vehicle_heading))

        #~ if (angle_to_goal < (vehicle_heading + NESSIE_FOV/2)) and (angle_to_goal > (vehicle_heading - NESSIE_FOV/2)):
            #~ isVisible = True
        if (self.theta_wrt_vehicle_heading > ((-(NESSIE_FOV/2) * (math.pi/180.0)))) or \
                (self.theta_wrt_vehicle_heading < ((NESSIE_FOV/2) * (math.pi/180.0))):
            isVisible = True


        print("Vehicle Position: {0}".format(self.vehicle_position))
        print("Vehicle Heading: {0}".format(vehicle_heading))
        print("Global Angle to Goal Object is: {0}".format(angle_to_goal))
        print("Target wrt to Vehicle Heading: {0}".format(self.theta_wrt_vehicle_heading))
        print("See Goal?: {0}".format(isVisible))
        return isVisible

    def loop(self):
        print "--------------"
        distance_to_goal = self.calcDistance()

        # Check to see whether the goal is in the FOV of the FFS
        goal_in_view = self.can_seeGoal(distance_to_goal)

        if not goal_in_view:
            distance_to_goal = 0
            print("GOAL NOT IN VIEW!!")
        else:
            print "Distance to Goal Object: ", distance_to_goal

        # Publish the distances and visible data to anyone who is interested
        #self.pub.publish(distance_to_goal)
        print "@@@@@@@@@@@@@@"
