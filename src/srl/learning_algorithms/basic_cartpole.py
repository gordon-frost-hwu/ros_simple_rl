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
from srl.useful_classes.rl_traces import Traces, TrueTraces

from srl.environments.cartpole import CartPoleEnvironment

from srl.learning_algorithms.true_online_td_lambda import TrueOnlineTDLambda
from srl.learning_algorithms.stober_td_learning import TDLinear
from variable_normalizer import DynamicNormalizer
from moving_differentiator import SlidingWindow

actor_config = {
    "approximator_name": "policy",
    "initial_value": 0.0,
    "alpha": 0.001,
    "random_weights": False,
    "minimise": False,
    "approximator_boundaries": [-200.0, 200.0],
    "num_input_dims": 4,
    "num_hidden_units": 12
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    # "basis_functions": BasisFunctions()
}

critic_config = {
    "approximator_name": "value-function",
    "initial_value": 0.0,
    "alpha": 0.002,
    "random_weights": False,
    "minimise": False,
    "approximator_boundaries": [-200.0, 200.0],
    "num_input_dims": 4,
    "rbf_basis_resolution": 20,
    "rbf_basis_scalar": 15.0,
    "number_of_dims_in_state": 2
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    # "basis_functions": BasisFunctions()
}
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
    "lambda": 0.9,  # was 0.0
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
        self.position_deriv_normaliser = DynamicNormalizer([-1.75, 1.75], [-1.0, 1.0])
        self.angle_normaliser = DynamicNormalizer([-0.25944, 0.25944], [-1.0, 1.0])
        self.angle_deriv_normaliser = DynamicNormalizer([-1.5, 1.5], [-1.0, 1.0])

        self.angle_dt_moving_window = SlidingWindow(5)

    def update_critic(self, reward):
        state_t_value = self.approx_critic.computeOutput(self.state_t.values())
        state_t_p1_value = self.approx_critic.computeOutput(self.state_t_plus_1.values())

        if CONFIG["critic algorithm"] == "ann_trad":
            td_error = reward + (CONFIG["gamma"] * state_t_p1_value) - state_t_value
        elif CONFIG["critic algorithm"] == "ann_true":
            td_error = reward + (CONFIG["gamma"] * state_t_p1_value) - \
            self.approx_critic.computeOutputThetaMinusOne(self.state_t.values())
        prev_critic_weights = self.approx_critic.getParams()
        critic_gradient = self.approx_critic.calculateGradient(self.state_t.values())
        self.traces_policy.updateTrace(self.approx_policy.calculateGradient(self.state_t.values()), 1.0)

        p = self.approx_critic.getParams()
        if CONFIG["critic algorithm"] == "ann_trad":
            self.traces_critic.updateTrace(critic_gradient, 1.0)  # for standard TD(lambda)
            X, T = self.traces.getTraces()
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
        return (td_error, critic_gradient)

    def update_state_t(self):
        sensors = self.env.getSensors()
        self.state_t = {"angle": self.angle_normaliser.scale_value(sensors[0]),
                        "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors[1]),
                        "position": self.position_normaliser.scale_value(sensors[2]),
                        "position_deriv": self.position_deriv_normaliser.scale_value(sensors[3])}

        sensors_greedy = self.env_greedy.getSensors()
        self.state_t_greedy = {"angle": self.angle_normaliser.scale_value(sensors_greedy[0]),
                                "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors_greedy[1]),
                                "position": self.position_normaliser.scale_value(sensors_greedy[2]),
                                "position_deriv": self.position_deriv_normaliser.scale_value(sensors_greedy[3])}

    def update_state_t_p1(self):
        sensors_t_plus_1 = self.env.getSensors()
        self.state_t_plus_1 = {"angle": self.angle_normaliser.scale_value(sensors_t_plus_1[0]),
                            "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors_t_plus_1[1]),
                            "position": self.position_normaliser.scale_value(sensors_t_plus_1[2]),
                            "position_deriv": self.position_deriv_normaliser.scale_value(sensors_t_plus_1[3])}

        sensors_t_plus_1_greedy = self.env_greedy.getSensors()
        self.state_t_plus_1_greedy = {"angle": self.angle_normaliser.scale_value(sensors_t_plus_1_greedy[0]),
                                    "angle_deriv": self.angle_deriv_normaliser.scale_value(sensors_t_plus_1_greedy[1]),
                                    "position": self.position_normaliser.scale_value(sensors_t_plus_1_greedy[2]),
                                    "position_deriv": self.position_deriv_normaliser.scale_value(sensors_t_plus_1_greedy[3])}

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
        for run in range(CONFIG["num_runs"]):
            # Create logging directory and files
            results_dir = "/tmp/{0}{1}".format(self.results_dir_name, run)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            filename = os.path.basename(sys.argv[0])
            os.system("cp {0} {1}".format(filename, results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/basis_functions/simple_basis_functions.py {0}".format(results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/environments/cartpole.py {0}".format(results_dir))

            f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)
            f_returns_greedy = open("{0}{1}".format(results_dir, "/GreedyEpisodeReturn.fso"), "w", 1)

            self.env = CartPoleEnvironment()
            self.env_greedy = CartPoleEnvironment()

            # policies and critics
            self.approx_critic = ANNApproximator(actor_config["num_input_dims"],
                                            actor_config["num_hidden_units"], hlayer_activation_func="tanh")
            self.approx_policy = ANNApproximator(actor_config["num_input_dims"], actor_config["num_hidden_units"], hlayer_activation_func="tanh")
            prev_critic_gradient = np.zeros(self.approx_critic.getParams().shape)

            # Set up trace objects
            if CONFIG["critic algorithm"] == "ann_trad":
                self.traces_critic = Traces(CONFIG["lambda"], CONFIG["min_trace_value"])
            elif CONFIG["critic algorithm"] == "ann_true":
                self.traces_critic = TrueTraces(critic_config["alpha"], CONFIG["gamma"], CONFIG["lambda"])
            self.traces_policy = Traces(CONFIG["lambda"], CONFIG["min_trace_value"])

            for episode_number in range(CONFIG["num_episodes"]):
                reward_cum = 0.0
                reward_cum_greedy = 0.0

                if episode_number % CONFIG["log_actions"] == 0:
                    f_actions = open("{0}{1}".format(results_dir, "/actions{0}.csv".format(episode_number)), "w", 1)
                    f_actions_greedy = open("{0}{1}".format(results_dir, "/greedy_actions{0}.csv".format(episode_number)), "w", 1)

                # reset everything for the next episode
                self.traces_critic.reset()
                self.traces_policy.reset()
                self.env.reset()
                self.env_greedy.reset()

                episode_ended = False
                episode_ended_learning = False
                episode_ended_greedy = False

                for step_number in range(CONFIG["max_num_steps"]):
                    # Update the state for timestep t
                    self.update_state_t()
                    
                    action_t_greedy = self.approx_policy.computeOutput(self.state_t_greedy.values())
                    action_t_deterministic = self.approx_policy.computeOutput(self.state_t.values())
                    exploration = np.random.normal(0.0, CONFIG["exploration_sigma"])
                    action_t =  np.clip(action_t_deterministic + exploration, -10, 10)
                    
                    if episode_number % CONFIG["log_actions"] == 0:
                        if step_number == 0:
                            state_keys = self.state_t.keys()
                            state_keys.append("action")
                            label_logging_format = "#{" + "}\t{".join(
                                [str(state_keys.index(el)) for el in state_keys]) + "}\n"
                            f_actions.write(label_logging_format.format(*state_keys))

                            state_keys = self.state_t_greedy.keys()
                            state_keys.append("action")
                            label_logging_format = "#{" + "}\t{".join(
                                [str(state_keys.index(el)) for el in state_keys]) + "}\n"
                            f_actions_greedy.write(label_logging_format.format(*state_keys))

                        logging_list = self.state_t.values()
                        logging_list.append(action_t_deterministic)

                        action_logging_format = "{" + "}\t{".join(
                            [str(logging_list.index(el)) for el in logging_list]) + "}\n"
                        f_actions.write(action_logging_format.format(*logging_list))

                        logging_list = self.state_t_greedy.values()
                        logging_list.append(action_t_greedy)

                        action_logging_format = "{" + "}\t{".join(
                            [str(logging_list.index(el)) for el in logging_list]) + "}\n"
                        f_actions_greedy.write(action_logging_format.format(*logging_list))


                    self.env.performAction(action_t)
                    self.env_greedy.performAction(action_t_greedy)

                    # Update the state for timestep t + 1, after action is performed
                    self.update_state_t_p1()

                    reward = self.env.getReward()
                    
                    if not episode_ended_learning:
                        # ---- Critic Update ----
                        (td_error, critic_gradient) = self.update_critic(reward)

                        # ---- Policy Update -------
                        self.update_policy(td_error, exploration)

                        prev_critic_gradient = deepcopy(critic_gradient)
                    
                        reward_cum += reward
                    if not episode_ended_greedy:
                        reward_cum_greedy += self.env_greedy.getReward()

                    episode_ended_learning = self.env.episodeEnded()
                    episode_ended_greedy = self.env_greedy.episodeEnded()

                    if episode_ended_learning and episode_ended_greedy:
                        
                        # episode complete, start a new one
                        break
                # episode either ended early due to failure or completed max number of steps
                print("Episode ended - Learning {0} {1}".format(episode_number, reward_cum))
                print("Episode ended - Greedy {0} {1}".format(episode_number, reward_cum_greedy))
                f_returns.write("{0}\t{1}\n".format(episode_number, reward_cum))
                f_returns_greedy.write("{0}\t{1}\n".format(episode_number, reward_cum_greedy))


if __name__ == '__main__':
    agent = CartPoleSimulation()
    agent.run()

        

        
