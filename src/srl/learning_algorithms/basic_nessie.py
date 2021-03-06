#! /usr/bin/python
import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
import numpy as np

import sys
import time
import os

from copy import deepcopy
from srl.useful_classes.angle_between_vectors import AngleBetweenVectors
from srl.basis_functions.simple_basis_functions import RBFBasisFunctions as BasisFunctions

from srl.approximators.linear_approximation import LinearApprox
from srl.approximators.ann_approximator_from_scratch import ANNApproximator
from srl.useful_classes.rl_traces import Traces, TrueTraces
from utilities.orstein_exploration import OUNoise

from srl.environments.ros_behaviour_interface import ROSBehaviourInterface
from srl.useful_classes.ros_environment_goals import EnvironmentInfo
from srl.useful_classes.ros_thruster_wrapper import Thrusters

from srl.learning_algorithms.true_online_td_lambda import TrueOnlineTDLambda
from srl.learning_algorithms.stober_td_learning import TDLinear
from utilities.variable_normalizer import DynamicNormalizer
from utilities.moving_differentiator import SlidingWindow

actor_config = {
    "approximator_name": "policy",
    "initial_value": 0.0,
    "alpha": 0.0005,
    "random_weights": False,
    "num_input_dims": 2,
    "num_hidden_units": 48
}

critic_config = {
    "approximator_name": "value-function",
    "initial_value": 0.0,
    "alpha": 0.0001,
    "random_weights": False,
    "num_input_dims": 2
}
CONFIG = {
    "test_policy": False,
    "generate_initial_weights": False,
    "log_actions": 1,
    "log_traces": False,
    "spin_rate": 10,
    "num_runs": 10,
    "num_episodes": 20,
    "max_num_steps": 350,
    "policy_type": "ann",
    "actor update rule": "cacla",
    "critic algorithm": "ann_true",
    "sparse reward": False,
    "gamma": 0.98,   # was 0.1
    "lambda": 0.93,  # was 0.93
    "alpha_decay": 0.0,  # was 0.00005
    "exploration_sigma": 0.1,
    "exploration_decay": 1.0, # was 0.975
    "min_trace_value": 0.1
}

TEST_CONFIG = {
    "folder": "NessieRepeatLearningNoDtReward",
    "run_numbers": [i for i in range(1)],
    "episode_numbers": [16, 17, 18, 19]    # episode number of 1000 means all episodes
}
# [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
# 20,21,22,23,24,25,26,27,28,29, 

class NessieRlSimulation(object):
    def __init__(self):
        args = sys.argv
        if "-r" in args:
            self.results_dir_name = args[args.index("-r") + 1]
        else:
            self.results_dir_name = "nessie_run"

        self.position_normaliser = DynamicNormalizer([-2.4, 2.4], [-1.0, 1.0])
        self.position_deriv_normaliser = DynamicNormalizer([-1.75, 1.75], [-1.0, 1.0])
        self.angle_normaliser = DynamicNormalizer([-3.14, 3.14], [-1.0, 1.0])
        self.angle_deriv_normaliser = DynamicNormalizer([-0.02, 0.02], [-1.0, 1.0])

        self.angle_dt_moving_window = SlidingWindow(5)
        self.last_150_episode_returns = SlidingWindow(150)

        self.thrusters = Thrusters()
        self.env = ROSBehaviourInterface()
        self.environment_info = EnvironmentInfo()

        self.ounoise = OUNoise()

        self.prev_action = 0.0

    def update_critic(self, reward, update):
        state_t_value = self.approx_critic.computeOutput(self.state_t.values())
        state_t_p1_value = self.approx_critic.computeOutput(self.state_t_plus_1.values())
        # print("state t: {0}".format(state_t_value))
        # print("state tp1: {0}".format(state_t_p1_value))

        if CONFIG["critic algorithm"] == "ann_trad":
            td_error = reward + (CONFIG["gamma"] * state_t_p1_value) - state_t_value
        elif CONFIG["critic algorithm"] == "ann_true":
            td_error = reward + (CONFIG["gamma"] * state_t_p1_value) - \
            self.approx_critic.computeOutputThetaMinusOne(self.state_t.values())
        prev_critic_weights = self.approx_critic.getParams()
        critic_gradient = self.approx_critic.calculateGradient(self.state_t.values())
        self.traces_policy.updateTrace(self.approx_policy.calculateGradient(self.state_t.values()), 1.0)

        if update:
            p = self.approx_critic.getParams()
            if CONFIG["critic algorithm"] == "ann_trad":
                self.traces_critic.updateTrace(critic_gradient, 1.0)  # for standard TD(lambda)
                X, T = self.traces_critic.getTraces()
                for x, trace in zip(X, T):
                    # print("updating critic using gradient vector: {0}\t{1}".format(x, trace))
                    p += critic_config["alpha"] * td_error * (x * trace)
                # self.approx_critic.setParams(prev_critic_weights + CONFIG["critic_config"]["alpha"] * td_error * critic_gradient)
            elif CONFIG["critic algorithm"] == "ann_true":
                # For True TD(lambda)
                #print("UPDATING ANN CRITC with TRUE TD(lambda)")
                self.traces_critic.updateTrace(critic_gradient)    # for True TD(lambda)
                part_1 = td_error * self.traces_critic.e
                part_2 = critic_config["alpha"] * \
                        np.dot((self.approx_critic.computeOutputThetaMinusOne(self.state_t.values()) - state_t_value), critic_gradient)
                p += part_1 + part_2
            
            self.approx_critic.setParams(p)
        return (td_error, critic_gradient, state_t_value, state_t_p1_value)

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


        # if (abs(angle_t)) > 0.5:
        #     if angle_t > 0 and angle_tp1 < 0:
        #         angle_change = (1.0 - angle_t) + (-1.0 - angle_tp1)
        #     elif angle_t < 0 and angle_tp1 > 0:
        #         angle_change = (1.0 - angle_tp1) + (-1.0 - angle_t)
        #     else:
        #         angle_change = angle_tp1 - angle_t
        # else:
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

        
    def update_policy(self, td_error, exploration):
        UPDATE_CONDITION = False
        if CONFIG["actor update rule"] == "cacla":
            if td_error > 0.0:
                UPDATE_CONDITION = True
            else:
                UPDATE_CONDITION = False
        elif CONFIG["actor update rule"] == "td lambda":
            UPDATE_CONDITION = True
        
        if UPDATE_CONDITION:
            # get original values
            params = self.approx_policy.getParams()
            old_action = self.approx_policy.computeOutput(self.state_t.values())
            policy_gradient = self.approx_policy.calculateGradient()

            # now update
            if CONFIG["actor update rule"] == "cacla":
                # policy.setParams(params + actor_config["alpha"] * (policy_gradient * exploration))
                X, T = self.traces_policy.getTraces()
                p = self.approx_policy.getParams()
                #print("Number of traces: {0}".format(len(T)))
                for x, trace in zip(X, T):
                    # print("updating critic using gradient vector: {0}\t{1}".format(x, trace))
                    p += actor_config["alpha"] * (x * trace) * exploration
                self.approx_policy.setParams(p)
            else:
                self.approx_policy.setParams(params + actor_config["alpha"] * (policy_gradient * td_error))

    def run(self):
        # Loop number of runs
        if CONFIG["test_policy"]:
            runs = TEST_CONFIG["run_numbers"]
        else:
            runs = range(CONFIG["num_runs"])

        for run in runs:
            if CONFIG["test_policy"]:
                self.results_dir_name = "nessie_validate_{0}".format(TEST_CONFIG["folder"])
                results_to_load_directory = "/tmp/{0}{1}".format(TEST_CONFIG["folder"], run)
            # Create logging directory and files
            results_dir = "/home/gordon/data/tmp/{0}{1}".format(self.results_dir_name, run)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            filename = os.path.basename(sys.argv[0])
            os.system("cp {0} {1}".format(filename, results_dir))
            os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/srl/environments/ros_behaviour_interface.py {0}".format(results_dir))
            os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/utilities/orstein_exploration.py {0}".format(results_dir))

            if CONFIG["test_policy"]:
                os.system("cp {0}/Epi* {1}/LearningEpisodeReturn.fso".format(results_to_load_directory, results_dir))
                os.system("cp {0}/basic* {1}/LearningMainScript.py".format(results_to_load_directory, results_dir))

            f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)

            # policies and critics
            self.approx_critic = ANNApproximator(actor_config["num_input_dims"],
                                            actor_config["num_hidden_units"], hlayer_activation_func="tanh")
            if not CONFIG["generate_initial_weights"]:
				critic_init = "/home/gordon/data/tmp/critic_params_48h.npy"
				self.approx_critic.setParams(list(np.load(critic_init)))

            self.approx_policy = ANNApproximator(actor_config["num_input_dims"], actor_config["num_hidden_units"], hlayer_activation_func="tanh")
            if not CONFIG["generate_initial_weights"]:
				policy_init = "/home/gordon/data/tmp/initial_2dim_48h_policy_params.npy"
				self.approx_policy.setParams(list(np.load(policy_init)))
            
            # if CONFIG["test_policy"] is True:
            #    if not os.path.exists("/tmp/{0}".format(results_to_validate)):
            #        continue
                #self.approx_policy.setParams()
            prev_critic_gradient = np.zeros(self.approx_critic.getParams().shape)

            # Set up trace objects
            if CONFIG["critic algorithm"] == "ann_trad":
                self.traces_critic = Traces(CONFIG["lambda"], CONFIG["min_trace_value"])
            elif CONFIG["critic algorithm"] == "ann_true":
                self.traces_critic = TrueTraces(critic_config["alpha"], CONFIG["gamma"], CONFIG["lambda"])
            self.traces_policy = Traces(CONFIG["lambda"], CONFIG["min_trace_value"])

            exploration_sigma = CONFIG["exploration_sigma"]

            for episode_number in range(CONFIG["num_episodes"]):
                
                if CONFIG["test_policy"] and episode_number not in TEST_CONFIG["episode_numbers"]:
                    # don't do anything for the episode number if we are testing policies and
                    # this episodes policy is not in the list to test
                    continue

                reward_cum = 0.0
                reward_cum_greedy = 0.0

                if episode_number % CONFIG["log_actions"] == 0:
                    f_actions = open("{0}{1}".format(results_dir, "/actions{0}.csv".format(episode_number)), "w", 1)
                
                # If testing a learnt policy, load it
                if CONFIG["test_policy"]:
                    policy_to_load = "{0}/policy_params{1}.npy".format(results_to_load_directory, episode_number)
                    critic_to_load = "{0}/critic_params{1}.npy".format(results_to_load_directory, episode_number)

                    print("policy_to_load: {0}".format(policy_to_load))
                    self.approx_policy.setParams(list(np.load(policy_to_load)))
                    self.approx_critic.setParams(list(np.load(critic_to_load)))

                # reset everything for the next episode
                self.traces_critic.reset()
                self.traces_policy.reset()

                # self.env.nav_reset()
                self.env.reset()
                self.ounoise.reset()

                self.angle_dt_moving_window.reset()

                episode_ended = False
                episode_ended_learning = False

                # if episode_number > 5 and exploration_sigma > 0.1:
                exploration_sigma *= CONFIG["exploration_decay"]

                self.prev_angle_dt_t = 0.0
                self.prev_angle_dt_tp1 = 0.0

                if CONFIG["generate_initial_weights"]:
                    self.approx_policy = ANNApproximator(actor_config["num_input_dims"],
                                                         actor_config["num_hidden_units"],
                                                         hlayer_activation_func="tanh")

                for step_number in range(CONFIG["max_num_steps"]):
                    # Update the state for timestep t
                    self.update_state_t()
                    
                    action_t_deterministic = self.approx_policy.computeOutput(self.state_t.values())

                    # if episode_number > 9:
                    #     control_rate = 0.5
                    # else:
                    control_rate = 3
                    if step_number % (control_rate * CONFIG["spin_rate"]) == 0:
                        # exploration = self.ounoise.get_action(action_t_deterministic)
                        # exploration = np.random.normal(0.0, exploration_sigma)
                        tmp_action = self.ounoise.get_action(action_t_deterministic)[0]
                        exploration = tmp_action - action_t_deterministic

                        # exploration = self.ounoise.function(action_t_deterministic, 0, 0.2, 0.1)[0]
                    # else:
                    #    action_t = deepcopy(self.prev_action)


                    # self.prev_action = deepcopy(action_t)

                    if not CONFIG["generate_initial_weights"] and not CONFIG["test_policy"]:
                        action_t = np.clip(action_t_deterministic + exploration, -10, 10)
                    else:
                        action_t = np.clip(action_t_deterministic, -10, 10)

                    # TODO - investigate what happens with the action!!!
                    self.env.performAction("gaussian_variance", action_t)

                    # TODO - time rather than rospy.sleep?!
                    time.sleep(1.0 / CONFIG["spin_rate"])

                    # Update the state for timestep t + 1, after action is performed
                    self.update_state_t_p1()

                    to_end = False
                    # if self.state_t["angle"] > 0.9:
                    #     reward = -10
                    #     to_end = True
                    # else:
                    reward = self.env.getReward(self.state_t_plus_1, action_t)
                    
                    if not episode_ended_learning:
                        if not CONFIG["generate_initial_weights"]:
                            # ---- Critic Update ----
                            (td_error, critic_gradient, state_t_value, state_tp1_value) = self.update_critic(reward, not CONFIG["test_policy"])

                            if episode_number % CONFIG["log_actions"] == 0:
                                if step_number == 0:
                                    state_keys = self.state_t.keys()
                                    state_keys.append("exploration")
                                    state_keys.append("reward")
                                    state_keys.append("tde")
                                    state_keys.append("st")
                                    state_keys.append("stp1")
                                    state_keys.append("explore_action")
                                    state_keys.append("action")
                                    label_logging_format = "#{" + "}\t{".join(
                                        [str(state_keys.index(el)) for el in state_keys]) + "}\n"
                                    f_actions.write(label_logging_format.format(*state_keys))

                                logging_list = self.state_t.values()
                                logging_list.append(exploration)
                                logging_list.append(reward)
                                logging_list.append(td_error)
                                logging_list.append(state_t_value)
                                logging_list.append(state_tp1_value)
                                logging_list.append(action_t)
                                logging_list.append(action_t_deterministic)
                                action_logging_format = "{" + "}\t{".join(
                                    [str(logging_list.index(el)) for el in logging_list]) + "}\n"
                                f_actions.write(action_logging_format.format(*logging_list))

                            if not CONFIG["test_policy"]:
                                # ---- Policy Update -------
                                self.update_policy(td_error, exploration)

                            prev_critic_gradient = deepcopy(critic_gradient)

                        reward_cum += reward
                        if to_end:
                            reward_cum = -3000 # for logging only
                            break
                    
                    # TODO - add check for if episode ended early. i.e. moving average
                    """ episode_ended_learning = self.env.episodeEnded()

                     if episode_ended_learning:
                        # episode complete, start a new one
                        break """
                # episode either ended early due to failure or completed max number of steps
                print("Episode ended - Learning {0} {1}".format(episode_number, reward_cum))

                f_returns.write("{0}\t{1}\n".format(episode_number, reward_cum))

                np.save("{0}/policy_params{1}".format(results_dir, episode_number), self.approx_policy.getParams())
                np.save("{0}/critic_params{1}".format(results_dir, episode_number), self.approx_critic.getParams())


if __name__ == '__main__':
    rospy.init_node("stober_cacla_nessie")

    agent = NessieRlSimulation()
    agent.run()

        

        
