#! /usr/bin/env python
"""
    Module contains base class, Behaviour, that any specific ROS behaviour class inherits
    author: Gordon Frost; email: gwf2@hw.ac.uk
    date: 23/01/15
"""
import rospy
from auv_msgs.msg import NavSts
from vehicle_interface.msg import Vector6, PilotRequest
from numpy import array
from my_msgs.msg import Goals, BehavioursGoal
from copy import deepcopy
from srv_msgs.srv import ChangeParam, ChangeParamResponse

USE_NESSIE_PID_CONFIG = False

def abstractMethod():
    raise Exception("Method not implemented ...")

class Behaviour(object):
    result_pub = None
    _nav = None
    metric_metric = None # goal value for the behaviour
    goal_position = None
    ip_position = None
    raw_angle_to_wp = 0.0
    raw_angle_to_ip = 0.0
    raw_angle_to_danger = 0.0
    distance_to_wp = 0.0
    distance_to_ip = 100.0
    distance_to_danger = 100.0
    # min_dist_to_ip = 100.0
    # min_dist_to_danger = 100.0

    MAX_SPEED = array([1.0, 0.6, 0.6, 0.0, 1.0, 2.0])        # max speed (m/s and rad/s)
    
    def __init__(self, name):
        """
        :param name: string containing an unique identifer to put in result topic name
        creates a publisher for result and subscriber for navigation input
        """
        self.result_pub = rospy.Publisher("/behaviours/{0}/output".format(name), Vector6, queue_size=1)
        # self.result_pub = rospy.Publisher("/pilot/velocity_req", PilotRequest)
        nav_sub = rospy.Subscriber("/nav/nav_sts", NavSts, self.navCallback)
        # goal_sub = rospy.Subscriber("/behaviours/{0}/input_goal".format(name), Vector6, self.setGoalPosition)
        goal_sub = rospy.Subscriber("/behaviours/goal", BehavioursGoal, self.setGoalPosition)
        ros_env_sub = rospy.Subscriber('/goal_points_data', Goals, self.rosEnvCallback)

        service = rospy.Service("/behaviours/{0}/params".format(name), ChangeParam, self._changeParamCallback)

        # Use either Nessies pilot controller params or my own
        if USE_NESSIE_PID_CONFIG:
            print("Using Nessie pilots ..")
            self.pid_coeffs = {"surge": rospy.get_param("/pilot/controller/pos_x"),
                               "sway":  rospy.get_param("/pilot/controller/pos_y"),
                               "depth": rospy.get_param("/pilot/controller/pos_z"),
                               "pitch": rospy.get_param("/pilot/controller/pos_m"),
                               "yaw":   rospy.get_param("/pilot/controller/pos_n")
            }
        else:
            print("Using PID gains from behaviour.py")
            self.pid_coeffs = {"surge":     {"kp": 0.05, "kd": 0.0, "ki": 0.0, "lim": 0.0, "offset": 0.0},
                               "sway":      {"kp": 0.75, "kd": 0.1, "ki": 0.0, "lim": 0.0, "offset": 0.0},
                               "depth":     {"kp": 0.75, "kd": 0.0, "ki": 0.01, "lim": 0.1, "offset": 0.0},
                               "pitch":     {"kp": 0.75, "kd": 0.5, "ki": 0.0, "lim": 1.0, "offset": 0.1},
                               "yaw":       {"kp": 0.15, "kd": 0.0, "ki": 0.0, "lim": 0.0, "offset": 0.0}
            }

    def publishResult(self, res):
        """
        :param res: msg object to send out in result publisher
        """
        self.result_pub.publish(res)

    def _changeParamCallback(self, req):
        self._params["reverseAction"] = req.reverse
        # print("Value request: {0}".format(req.value))
        # print("Max Action Value: {0}".format(req.max_action_value))
        if req.name == "gaussian_variance":
            try:
                #self._params[req.name] = (req.max_action_value + 0.05) - req.value
                self._params[req.name] = req.value
                # print("Value modified: {0}".format(self._params[req.name]))
                # rospy.loginfo("Param change successful, will take affect in next iteration")
                return ChangeParamResponse(success = True, sigma = self._params[req.name])
            except KeyError:
                # rospy.loginfo("Parameter Index Error: param, {0}, does not exist!".format(req.name))
                return ChangeParamResponse(success = False, sigma = self._params[req.name])
        elif req.name == "gaussian_mean":
            self._params[req.name] = req.value
            return ChangeParamResponse(success = True, sigma = self._params[req.name])
        return ChangeParamResponse(success = True, sigma = self._params[req.name])

    def rosEnvCallback(self, msg):
        updateDanger = False
        min_dist_to_ip = 100.0
        # min_dist_to_danger = 100.0
        for point in msg.goals:
            # Get the angle angle to the point-of-interest; 1 = waypoint_goal, 2 = inspection_goal, 3 = fear_goal
            if point.point_type == 1.0:
                # point is a waypoint for vehicle to pass through
                self.raw_angle_to_ip = point.angle_wrt_vehicle
                self.distance_to_ip = point.distance
                self.goal_position = [point.position.north, point.position.east, point.position.depth,
                                      0.0, 0.0, point.orientation.yaw]
            elif point.point_type == 2.0:
                # point of interest, i.e. inspection point - a point the vehicle should look at
                # set the variables to the closest IP's properties (euclidean distance-wise)
                if point.distance <= min_dist_to_ip:
                    self.raw_angle_to_ip = point.angle_wrt_vehicle
                    self.distance_to_ip = point.distance
                    self.ip_position = [point.position.north, point.position.east, point.position.depth,
                                      0.0, point.orientation.pitch, point.orientation.yaw]
                    min_dist_to_ip = deepcopy(self.distance_to_ip)
            elif point.point_type == 3.0:
                # point of danger - signifies an area the vehicle should avoid
                self.raw_angle_to_danger = point.angle_wrt_vehicle
                # if point.distance < self.min_dist_to_danger:
                self.distance_to_danger = point.distance
                    # self.min_dist_to_danger = point.distance

    def setGoalPosition(self, goal):
        """
        :param goal: 6DOF tuple, list, or numpy array. within code, np.array is used
        """
        assert len(goal.goal_position.values) == 6, 'Goal must be 6DOF tuple, array, or list ...'
        print("New GOAL POSITION!!!!")
        if sum(goal.goal_position.values) < 1000:
            self.goal_position = goal.goal_position.values
        if sum(goal.goal_velocity.values) < 1000:
            self.ip_position = goal.goal_velocity.values

    def navCallback(self, msg):
        """
        :param msg: navigation msg of the vehicle/agent
        method to be overridden in child class for custom navigation data desired
        """
        abstractMethod()
        
    def doWork(self):
        """
         Abstract method which should be overridden in child classes
        """
        abstractMethod()
