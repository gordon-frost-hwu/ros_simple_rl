#! /usr/bin/python
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
import numpy as np

import sys
import time
import os

from copy import deepcopy

from srl.environments.ros_behaviour_interface import ROSBehaviourInterface
from srl.useful_classes.ros_environment_goals import EnvironmentInfo
from srl.useful_classes.ros_thruster_wrapper import Thrusters
from vehicle_interface.msg import FloatArray

from variable_normalizer import DynamicNormalizer
from moving_differentiator import SlidingWindow

CONFIG = {
    "num_runs": 20,
    "run_time": 30
}

class RepeatPilotRequest(object):
    def __init__(self):
        args = sys.argv
        if "-r" in args:
            self.results_dir_name = args[args.index("-r") + 1]
        else:
            self.results_dir_name = "repeat_pilot_request"

        self.position_normaliser = DynamicNormalizer([-2.4, 2.4], [-1.0, 1.0])
        self.position_deriv_normaliser = DynamicNormalizer([-1.75, 1.75], [-1.0, 1.0])
        self.angle_normaliser = DynamicNormalizer([-3.14, 3.14], [-1.0, 1.0])
        self.angle_deriv_normaliser = DynamicNormalizer([-0.02, 0.02], [-1.0, 1.0])

        self.angle_dt_moving_window = SlidingWindow(5)
        self.last_150_episode_returns = SlidingWindow(150)

        self.thrusters = Thrusters()
        self.env = ROSBehaviourInterface()
        self.environment_info = EnvironmentInfo()
        
        sub_pilot_position_controller_output = rospy.Subscriber("/pilot/position_pid_output", FloatArray, self.positionControllerCallback)

        self.prev_action = 0.0
        self.pos_pid_output = np.zeros(6)
    
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

    def run(self):
        results_dir = "/home/gordon/data/tmp/{0}{1}".format(self.results_dir_name, 0)
        
        for run in range(CONFIG["num_runs"]):
            
            # Create logging directory and files
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            filename = os.path.basename(sys.argv[0])
            os.system("cp {0} {1}".format(filename, results_dir))
            os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/utilities/repeat_pilot_request.py {0}".format(results_dir))
            
            # reset stuff for the run
            self.env.nav_reset()
            self.env.reset()
            self.angle_dt_moving_window.reset()
            self.prev_angle_dt_t = 0.0
            self.prev_angle_dt_tp1 = 0.0
            
            # create log file
            f_actions = open("{0}{1}".format(results_dir, "/actions{0}.csv".format(run)), "w", 1)
            
            start_time = time.time()
            end_time = start_time + CONFIG["run_time"]
            
            first_step = True
            
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
                    state_keys = self.state_t.keys()
                    state_keys.append("action")
                    label_logging_format = "#{" + "}\t{".join(
                        [str(state_keys.index(el)) for el in state_keys]) + "}\n"
                    f_actions.write(label_logging_format.format(*state_keys))

                logging_list = self.state_t.values()
                logging_list.append(self.pos_pid_output[5])
                action_logging_format = "{" + "}\t{".join(
                    [str(logging_list.index(el)) for el in logging_list]) + "}\n"
                f_actions.write(action_logging_format.format(*logging_list))
                


if __name__ == '__main__':
    rospy.init_node("repeat_pilot_request")

    pilot = RepeatPilotRequest()
    pilot.run()

        

        
