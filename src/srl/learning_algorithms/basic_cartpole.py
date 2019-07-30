#! /usr/bin/python
import numpy as np

import sys
import time
import os

from copy import deepcopy
from srl.useful_classes.angle_between_vectors import AngleBetweenVectors
from srl.basis_functions.simple_basis_functions import RBFBasisFunctions as BasisFunctions

from srl.approximators.linear_approximation import LinearApprox
from srl.approximators.ann_approximator_from_scratch import ANNApproximator
from srl.useful_classes.rl_traces import self.traces, TrueTraces

from srl.environments.cartpole import CartPoleEnvironment

from srl.learning_algorithms.true_online_td_lambda import TrueOnlineTDLambda
from srl.learning_algorithms.stober_td_learning import TDLinear
from variable_normalizer import DynamicNormalizer
from moving_differentiator import SlidingWindow

CONFIG = {
    "log_actions": 100,
    "log_traces": False,
    "spin_rate": 200,
    "num_runs": 50,
    "num_episodes": 2500,
    "max_num_steps": 2000,
    "policy_type": "ann",
    "actor update rule": "cacla",
    "critic algorithm": "ann_true",
    "sparse reward": False,
    "gamma": 0.98,   # was 0.1
    "lambda": 0.0,  # was 0.0
    "alpha_decay": 0.0, # was 0.00005
    "exploration_sigma": 5.0,
    "exploration_decay": 1.0,
    "min_trace_value": 0.1
}

class CartPoleSimulation(object):
    def __init__(self):
        args = sys.argv
        if "-r" in args:
            self.results_dir_name = args[args.index("-r") + 1]
        else:
            self.results_dir_name = "cartpole_run"

        self.position_normaliser = DynamicNormalizer([-2.4, 2.4], [-1.0, 1.0])
        self.position_deriv_normaliser = DynamicNormalizer([-1.5, 1.5], [-1.0, 1.0])
        self.distance_normaliser = DynamicNormalizer([0.0, 25.0], [-1.0, 1.0])
        self.distance_reward_normaliser = DynamicNormalizer([0.0, 15.0], [0.0, 1.0])
        self.angle_normaliser = DynamicNormalizer([-0.20944, 0.20944], [-1.0, 1.0])
        self.angle_deriv_normaliser = DynamicNormalizer([-1.0, 1.0], [-1.0, 1.0])

        self.angle_dt_moving_window = SlidingWindow(5)

    def update_critic(self):
        if CONFIG["critic algorithm"] == "ann_trad":
            td_error = reward + (CONFIG["gamma"] * state_t_p1_value) - state_t_value
        elif CONFIG["critic algorithm"] == "ann_true":
            td_error = reward + (CONFIG["gamma"] * state_t_p1_value) - \
            self.approx_critic.computeOutputThetaMinusOne(state_t_raw.values())
        prev_critic_weights = self.approx_critic.getParams()
        critic_gradient = self.approx_critic.calculateGradient(state_t_raw.values())
        policy_traces.updateTrace(policy.calculateGradient(state_t_raw.values()), 1.0)

        p = self.approx_critic.getParams()
        if CONFIG["critic algorithm"] == "ann_trad":
            self.traces.updateTrace(critic_gradient, 1.0)  # for standard TD(lambda)
            X, T = self.traces.getTraces()
            for x, trace in zip(X, T):
                # print("updating critic using gradient vector: {0}\t{1}".format(x, trace))
                p += critic_config["alpha"] * td_error * (x * trace)
            # self.approx_critic.setParams(prev_critic_weights + CONFIG["critic_config"]["alpha"] * td_error * critic_gradient)
        elif CONFIG["critic algorithm"] == "ann_true":
            # For True TD(lambda)
            #print("UPDATING ANN CRITC with TRUE TD(lambda)")
            self.traces.updateTrace(critic_gradient)    # for True TD(lambda)
            part_1 = td_error * self.traces.e
            part_2 = CONFIG["critic_config"]["alpha"] * \
                    np.dot((self.approx_critic.computeOutputThetaMinusOne(self.state_t.values()) - state_t_value), critic_gradient)
            p += part_1 + part_2
        
        self.approx_critic.setParams(p)

    def update_state_t(self):
        sensors = env.getSensors()
        self.state_t = {"angle": self.angle_normaliser.scale_value(sensors[0]),
                        "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors[1]),
                        "position": self.position_normaliser.scale_value(sensors[2]),
                        "position_deriv": self.position_deriv_normaliser.scale_value(sensors[3])}

        sensors_greedy = env_greedy.getSensors()
        self.state_t_greedy = {"angle": self.angle_normaliser.scale_value(sensors_greedy[0]),
                                "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors_greedy[1]),
                                "position": self.position_normaliser.scale_value(sensors_greedy[2]),
                                "position_deriv": self.position_deriv_normaliser.scale_value(sensors_greedy[3])}

    def update_state_t_p1(self):
        sensors_t_plus_1 = env.getSensors()
        self.state_t_plus_1 = {"angle": self.angle_normaliser.scale_value(sensors_t_plus_1[0]),
                            "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors_t_plus_1[1]),
                            "position": self.position_normaliser.scale_value(sensors_t_plus_1[2]),
                            "position_deriv": self.position_deriv_normaliser.scale_value(sensors_t_plus_1[3])}

        sensors_t_plus_1_greedy = env_greedy.getSensors()
        self.state_t_plus_1_greedy = {"angle": self.angle_normaliser.scale_value(sensors_t_plus_1_greedy[0]),
                                    "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors_t_plus_1_greedy[1]),
                                    "position": self.position_normaliser.scale_value(sensors_t_plus_1_greedy[2]),
                                    "position_deriv": self.position_deriv_normaliser.scale_value(sensors_t_plus_1_greedy[3])}

    def run(self):
        # Loop number of runs
        for run in range(CONFIG["num_runs"]):
            # Create logging directory and files
            results_dir = "/tmp/{0}{1}".format(results_dir_name, run)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            filename = os.path.basename(sys.argv[0])
            os.system("cp {0} {1}".format(filename, results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/basis_functions/simple_basis_functions.py {0}".format(results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/environments/cartpole.py {0}".format(results_dir))

            f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)
            f_returns_ = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)

            env = CartPoleEnvironment()
            env_greedy = CartPoleEnvironment()

            # policies and critics
            self.approx_critic = ANNApproximator(CONFIG["actor_config"]["num_input_dims"],
                                            CONFIG["actor_config"]["num_hidden_units"], hlayer_activation_func="tanh")
            approx_policy = ANNApproximator(CONFIG["actor_config"]["num_input_dims"], CONFIG["actor_config"]["num_hidden_units"], hlayer_activation_func="tanh")
            prev_critic_gradient = np.zeros(self.approx_critic.getParams().shape)

            self.traces = self.traces(CONFIG["lambda"], CONFIG["min_trace_value"])
            traces_policy = self.traces(CONFIG["lambda"], CONFIG["min_trace_value"])

            for episode_number in range(CONFIG["num_episodes"]):
                reward_cum = 0.0
                reward_cum_greedy = 0.0

                # reset everything for the next episode
                self.traces.reset()
                traces_policy.reset()
                env.reset()
                env_greedy.reset()

                episode_ended = False
                episode_ended_greedy = False

                for step_number in range(CONFIG["max_num_steps"]):
                    self.update_state_t()
                    
                    action_t_greedy = approx_policy.computeOutput(self.state_t_greedy.values)
                    action_t_deterministic = approx_policy.computeOutput(self.state_t.values)
                    exploration = np.random.normal(0.0, CONFIG["exploration_sigma"])
                    action_t =  np.clip(action_t_deterministic + exploration, -10, 10)
                    
                    env.performAction(action_t)
                    env_greedy.performAction(action_t_greedy)

                    self.update_state_t_p1()

                    reward = env.getReward()

                    state_t_value = self.approx_critic.computeOutput(self.state_t.values)
                    state_t_p1_value = self.approx_critic.computeOutput(self.state_t_plus_1.values)

                    # QUESTIONABLE???!!!
                    # ---- Critic Update ----
                    self.update_critic()
                    # ---- End Critic Update ----

                    # ---- Policy Update -------
                    params = approx_policy.getParams()
                    # ---- End Policy Update -------

if __name__ == '__main__':
    agent = CartPoleSimulation()
    agent.run()

        

        
