#! /usr/bin/python
import roslib;
import numpy as np
import rospy
import time

from copy import deepcopy
from srl.useful_classes.ros_environment_goals import EnvironmentInfo
from srl.useful_classes.ros_thruster_wrapper import Thrusters
from vehicle_interface.msg import FloatArray
from utilities.optimal_control_response import optimal_control_response
from utilities.variable_normalizer import DynamicNormalizer
from utilities.moving_differentiator import SlidingWindow

CONFIG = {
    "run_time": 30,
}

class PilotPidProcess(object):
    def __init__(self, env, results_parent_dir):
        self.results_dir = results_parent_dir
        self.position_normaliser = DynamicNormalizer([-2.4, 2.4], [-1.0, 1.0])
        self.position_deriv_normaliser = DynamicNormalizer([-1.75, 1.75], [-1.0, 1.0])
        self.angle_normaliser = DynamicNormalizer([-3.14, 3.14], [-1.0, 1.0])
        self.angle_deriv_normaliser = DynamicNormalizer([-0.02, 0.02], [-1.0, 1.0])

        self.angle_dt_moving_window = SlidingWindow(5)
        self.last_150_episode_returns = SlidingWindow(150)

        self.thrusters = Thrusters()
        self.env = env
        self.environment_info = EnvironmentInfo()

        sub_pilot_position_controller_output = rospy.Subscriber("/pilot/position_pid_output", FloatArray,
                                                                self.positionControllerCallback)
        self.prev_action = 0.0
        self.pos_pid_output = np.zeros(6)

        self.baseline_response = optimal_control_response()

    def positionControllerCallback(self, msg):
        self.pos_pid_output = msg.values

    def update_state_t(self):
        raw_angle = deepcopy(self.environment_info.raw_angle_to_goal)
        # print("raw angle:")
        # raw_angle_dt = raw_angle - self.prev_angle_dt_t
        # print("raw angle dt: {0}".format(raw_angle_dt))
        self.state_t = {
            "angle": self.angle_normaliser.scale_value(raw_angle),
            "angle_deriv": self.prev_angle_dt_t
        }
        self.prev_angle_dt_t = deepcopy(raw_angle)

    def update_state_t_p1(self):
        raw_angle = deepcopy(self.environment_info.raw_angle_to_goal)
        angle_tp1 = self.angle_normaliser.scale_value(raw_angle)
        angle_t = self.state_t["angle"]

        abs_angle_tp1 = np.abs(angle_tp1)
        abs_angle_t = np.abs(angle_t)
        if abs_angle_tp1 > abs_angle_t:
            sign = -1
        else:
            sign = 1
        angle_change = sign * abs(abs_angle_tp1 - abs_angle_t)

        # print("angle t: {0}".format(abs_angle_t))
        # print("angle tp1: {0}".format(abs_angle_tp1))
        # print("angle change: {0}".format(angle_change))

        tmp_angle_change = sum(self.angle_dt_moving_window.getWindow(angle_change)) / 5.0
        self.state_t_plus_1 = {
            "angle": self.angle_normaliser.scale_value(raw_angle),
            "angle_deriv": self.angle_deriv_normaliser.scale_value(tmp_angle_change)
        }
        self.prev_angle_dt_t = self.angle_deriv_normaliser.scale_value(tmp_angle_change)

    def setPidGains(self, posP, posI, posD, velP, velI, velD):
        self.env.enable_pilot(False)
        rospy.set_param("/pilot/controller/pos_n/kp", float(posP))
        rospy.set_param("/pilot/controller/pos_n/ki", float(posI))
        rospy.set_param("/pilot/controller/pos_n/kd", float(posD))
        # rospy.set_param("/pilot/controller/vel_r/kp", float(velP))
        # rospy.set_param("/pilot/controller/vel_r/ki", float(velI))
        # rospy.set_param("/pilot/controller/vel_r/kd", float(velD))
        self.env.enable_pilot(True)
        self.env.enable_pilot(False)
        self.env.enable_pilot(True)

    def calculate_fitness(self, response):
        max_idx = response.shape[0]
        step_errors = []

        for idx in range(max_idx):
            r = response[idx, 1]
            b = self.baseline_response[idx, 1]
            step_error = 0.0
            if r > b:
                step_error = r - b
            else:
                step_error = b - r
            step_errors.append(step_error)

        fitness = sum(step_errors)  # / len(step_errors)
        return fitness

    def get_response(self, id, individual):

        # reset stuff for the run
        self.env.nav_reset()
        # Set usable gains for DoHover action to get to initial position again
        # position sim gains: { "kp": 0.35, "ki": 0.0, "kd": 0.0 }
        # velocity sim gains: { "kp": 35.0, "ki": 0.0, "kd": 0.0 }
        self.setPidGains(0.35, 0, 0, 35.0, 0, 0)
        rospy.sleep(1)
        self.env.reset(disable_behaviours=False)
        self.angle_dt_moving_window.reset()
        self.prev_angle_dt_t = 0.0
        self.prev_angle_dt_tp1 = 0.0

        # Set the gains to those of the individual/solution
        self.setPidGains(individual[0], individual[1], individual[2], 0, 0, 0)

        # create log file
        f_actions = open("{0}{1}".format(self.results_dir, "/actions{0}.csv".format(id)), "w", 1)

        start_time = time.time()
        end_time = start_time + CONFIG["run_time"]

        first_step = True
        response = np.zeros([350, 2])
        timestep = 0
        while time.time() < end_time and not rospy.is_shutdown():

            # send pilot request
            self.env.pilotPublishPositionRequest([0, 0, 0, 0, 0, 0])

            # perform a 'step'
            self.update_state_t()
            rospy.sleep(0.1)
            self.update_state_t_p1()

            # log the current state information
            if first_step:
                first_step = False
                state_keys = list(self.state_t.keys())
                state_keys.append("baseline_angle")
                state_keys.append("action")
                label_logging_format = "#{" + "}\t{".join(
                    [str(state_keys.index(el)) for el in state_keys]) + "}\n"
                f_actions.write(label_logging_format.format(*state_keys))

            logging_list = list(self.state_t.values())
            logging_list.append(self.baseline_response[timestep, 1])
            logging_list.append(self.pos_pid_output[5])
            action_logging_format = "{" + "}\t{".join(
                [str(logging_list.index(el)) for el in logging_list]) + "}\n"
            response[timestep, :] = [timestep, logging_list[0]]
            timestep += 1
            f_actions.write(action_logging_format.format(*logging_list))
        return response