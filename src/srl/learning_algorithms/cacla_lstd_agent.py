#! /usr/bin/python
# Adaption of the previous LSTD NAC agent to the CACLA actor update methodology
import roslib; roslib.load_manifest("rl_pybrain")
import rospy

from srv_msgs.srv import LogNav

from scipy import random
from copy import deepcopy
import numpy as np

import sys
import time
import os

from srl.useful_classes.ros_environment_goals import EnvironmentInfo
from srl.useful_classes.ros_thruster_wrapper import Thrusters
from srl.useful_classes.angle_between_vectors import AngleBetweenVectors
from srl.useful_classes.rl_traces import Traces

from srl.basis_functions.simple_basis_functions import PolynomialBasisFunctions

from srl.approximators.linear_approximation import LinearApprox

from srl.environments.ros_behaviour_interface import ROSBehaviourInterface

from srl.learning_algorithms.td_lambda_traditional import Traditional_TD_LAMBDA

# from mmlf.resources.function_approximators.BasisFunction import BasisFunction
from utilities.nav_class import Nav
from utilities.variable_normalizer import DynamicNormalizer
from utilities.gaussian import gaussian1D, gaussian2D
# from utilities.body_velocity_calculator import calculate_body_velocity

from rospkg import RosPack
rospack = RosPack()


def gaussian1DHalfInverse(x, x0, mu, sigma):
    # make sure everything is of type float so that calculation is correct
    x = float(x); x0 = float(x0); mu = float(mu); sigma = float(sigma)
    x_term = (x - x0)**2 / (2 * sigma**2)
    if x < x0:
        return -(1.0 - (mu * np.exp(-x_term)))
    else:
        return 1.0 - (mu * np.exp(-x_term))

def noisy_variable(var, sig):
    return var + np.random.normal(0.0, sig)


def minmax(item, limit1, limit2):
        "Bounds item to between limit1 and limit2 (or -limit1)"
        return max(limit1, min(limit2, item))


actor_config = {
    "approximator_name": "policy",
    "initial_value": -0.2,
    "alpha": 0.00003,
    "random_weights": False,
    "minimise": False,
    "approximator_boundaries": [-200.0, 200.0],
    "num_input_dims": 2,
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    "basis_functions": PolynomialBasisFunctions()
}

critic_config = {
    "approximator_name": "value-function",
    "initial_value": 0.0,
    "alpha": 0.0003,
    "random_weights": False,
    "minimise": False,
    "approximator_boundaries": [-200.0, 200.0],
    "num_input_dims": 2,
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    "basis_functions": PolynomialBasisFunctions()
}

CONFIG = {
    "goal_position": [10.0, 0.0],
    "episode_fault_injected": 0,
    "num_episodes_not_learning": 0,
    "spin_rate": 5,
    "num_runs": 3,
    "num_episodes": 120,
    "max_num_steps": 300,
    "log_nav_every": 1,
    "gamma": 1.0,   # was 0.1
    "lambda": 0.8,  # was 0.0
    "lstd_decay_rate": 0.9,     # was 0.3
    "epsilon": 1.0,
    "alpha_decay": 0.0, # was 0.00005
    "exploration_sigma": 0.2,
    "exploration_decay": 0.005,
    "min_trace_value": 0.01
}


# This Service is global so that it can be called upon rospy shutdown within the on_rospyShutdown function
# otherwise, if this node is terminated during an episode, the nav will be logged to csv until this node is restarted
logNav = rospy.ServiceProxy("/nav/log_nav", LogNav)

def on_rospyShutdown():
    global episode_number, results_dir
    logNav(log_nav=False, dir_path=results_dir, file_name_descriptor=str(episode_number))

if __name__ == '__main__':
    args = sys.argv
    if "-r" in args:
        results_dir_name = args[args.index("-r") + 1]
    else:
        results_dir_name = "cacla_run"

    rospy.init_node("natural_actor_critic")

    navigation = Nav()
    environmental_data = EnvironmentInfo()
    angle_between_vectors = AngleBetweenVectors()
    ros_behaviour_interface = ROSBehaviourInterface()

    # Set ROS spin rate
    rate = rospy.Rate(CONFIG["spin_rate"])
    rospy.on_shutdown(on_rospyShutdown)

    # Set Thruster Status
    thrusters = Thrusters()

    # initialise some global variables/objects
    # global normalisers
    yaw_velocity_normaliser = DynamicNormalizer([-1.0, 1.0], [0.0, 1.0])
    surge_velocity_normaliser = DynamicNormalizer([-0.6, 0.6], [0.0, 1.0])
    codependant_normaliser = DynamicNormalizer([-0.4, 0.4], [0.0, 1.0])
    distance_normaliser = DynamicNormalizer([0.0, 12.0], [0.0, 1.0])
    angle_to_goal_normaliser = DynamicNormalizer([0.0, 3.14], [0.0, 1.0])
    angle_dt_normaliser = DynamicNormalizer([-0.1, 0.1], [0.0, 1.0])
    angle_and_velocity_normaliser = DynamicNormalizer([-1.6, 1.6], [0.0, 1.0])
    angles_normaliser = DynamicNormalizer([0.0, 3.14], [0.0, 1.0])
    north_normaliser = DynamicNormalizer([0.0, 14.0], [0.0, 1.0])
    east_normaliser = DynamicNormalizer([-5.0, 5.0], [0.0, 1.0])
    reward_normaliser = DynamicNormalizer([-3.14, 0.0], [-1.0, -0.1])

    # Set the global basis functions and approximator objects
    # basis_functions = RBFBasisAction()



    # Loop number of runs
    for run in range(CONFIG["num_runs"]):
        # Create logging directory and files
        results_dir = "/tmp/{0}{1}".format(results_dir_name, run)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        os.system("cp /home/gordon/software/mmlf-1.0/mmlf/agents/cacla_lstd_agent.py {0}".format(results_dir))

        f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)
        f_timings = open("{0}{1}".format(results_dir, "/AvgStepTime.fso"), "w", 1)

        # initialise policy and value functions
        policy = LinearApprox(actor_config)
        basis_functions = PolynomialBasisFunctions(idx=run)

        # traditional TD(lambda) learning algorithm for the critic
        td_lambda = Traditional_TD_LAMBDA(critic_config, CONFIG["lambda"], CONFIG["min_trace_value"])

        # Loop number of episodes
        for episode_number in range(CONFIG["num_episodes"]):
            global episode_number, results_dir
            # initialise state and internal variables
            f_actions = open("{0}{1}".format(results_dir, "/actions{0}.csv".format(episode_number)), "w", 1)
            f_rewards = open("{0}{1}".format(results_dir, "/RewardsEpisode{0}.fso".format(episode_number)), "w", 1)
            cum_reward = 0.0
            cum_step_time = 0.0
            angle_dt = 0.0
            prev_action = None
            critic_value_func_params = None

            if episode_number == CONFIG["episode_fault_injected"]:
                thrusters.inject_surge_thruster_fault(17, 1)

            td_lambda.episodeReset()

            # Reset the last action value
            # last_yaw_mean = yaw_ros_action("gaussian_variance", 0.7, False, 1.0)

            # Enable behaviours
            ros_behaviour_interface.disable_behaviours(disable=True)
            ros_behaviour_interface.nav_reset()
            random_yaw = 1.57 + 0.785#float(random.randrange(-314, 314, 5)) / 100
            print("Moving to Random Starting Yaw: {0}".format(random_yaw))
            hover_completed = ros_behaviour_interface.pilot([0, 0, 0, 0, 0,random_yaw])
            rospy.sleep(5)
            ros_behaviour_interface.disable_behaviours(disable=False)
            print(hover_completed)
            rospy.sleep(0.5)    # give the nav topic callback to be called before the next episode is started.
                                # otherwise the next episode may start in a episode termination condition

            logNav(log_nav=True, dir_path=results_dir, file_name_descriptor=str(episode_number))

            # Loop number of steps
            for step_number in range(CONFIG["max_num_steps"]):
                # --------- STATE AND TRACES ---------
                # get current state
                step_start_time = time.time()
                print("------ STEP {0} -------".format(step_number))
                state_t = {"angle": deepcopy(environmental_data.raw_angle_to_goal),
                           # "distance": gaussian1D(surge_velocity_normaliser.scale_value(
                           #     navigation._nav.body_velocity.x), 0.7, 0.5, 0.2) +
                           #                      gaussian1D(yaw_velocity_normaliser.scale_value(
                           #                          navigation._nav.orientation_rate.yaw), 0.5, 0.5, 0.3),
                           "angle_dt": deepcopy(angle_dt)}
                           # "combi": angles_normaliser.scale_value(deepcopy(distance_normaliser.scale_value(environmental_data.distance_to_goal)
                           #                             * abs(environmental_data.raw_angle_to_goal)))}
                           # "yaw_velocity": deepcopy(navigation._nav.orientation_rate.yaw)}
                td_lambda.updateTrace(state_t.values(), 1.0)

                print("State: {0}".format(state_t))
                print("Angle to Goal: {0}".format(environmental_data.raw_angle_to_goal))

                # --------- GET ACTION AND PERFORM IT ---------
                # Compute the deterministic (??? deterministic if its an approximation ???) action
                state_t_sub_features = basis_functions.computeFeatures(state_t.values())
                state_t_full_features = np.r_[state_t_sub_features,
                                     np.array([0.0 for i in range(len(state_t_sub_features))])]
                action_t_deterministic = policy.computeOutput(state_t_sub_features)
                # action_t_deterministic = policy.computeOutput(np.array(state_t.values()))
                # action_t_deterministic = policy.computeOutput(state_t_full_features)
                # Add noise/exploration to the action in the form of a Gaussian/Normal distribution
                # if state_t["yaw_velocity"] > 0.6 or state_t["yaw_velocity"] < 0.4:
                if step_number % (3*CONFIG["spin_rate"]) == 0:
                    if random.random() < CONFIG["epsilon"]:
                        exploration = np.random.normal(0.0, CONFIG["exploration_sigma"])
                        CONFIG["exploration_sigma"] = CONFIG["exploration_sigma"] - CONFIG["exploration_sigma"]
                    else:
                        exploration = 0.0
                # else:
                #     exploration = np.random.normal(0.0, 0.1)
                action_t = action_t_deterministic + exploration
                print("Deterministic Action: {0}".format(action_t_deterministic))
                print("Stochastic Action: {0}".format(action_t))
                if prev_action is None:
                    prev_action = deepcopy(action_t)

                # Log the deterministic action chosen for each state according to the policy LA
                # use the deterministic action just so that it is cleaner to look at for debugging
                logging_list = state_t.values()
                logging_list.append(action_t_deterministic)
                action_logging_format = "{"+"}\t{".join([str(logging_list.index(el)) for el in logging_list])+"}\n"
                f_actions.write(action_logging_format.format(*logging_list))

                # Perform the action
                if action_t > -100.0:
                    last_yaw_mean = ros_behaviour_interface.performAction("gaussian_variance", action_t, False, 1.0)
                else:
                    last_yaw_mean = ros_behaviour_interface.performAction("gaussian_variance", 0.1, False, 1.0)

                print("Sleeping for a bit ...")
                rate.sleep()

                # --------- GET NEW STATE ---------
                # observe new state --- which is dependant on whether it is the final goal state or not

                state_t_plus_1 = {"angle": deepcopy(environmental_data.raw_angle_to_goal),
                               #     "distance": gaussian1D(surge_velocity_normaliser.scale_value(
                               # navigation._nav.body_velocity.x), 0.7, 0.5, 0.2) +
                               #                  gaussian1D(yaw_velocity_normaliser.scale_value(
                               #                      navigation._nav.orientation_rate.yaw), 0.5, 0.5, 0.3),
                                   "angle_dt": deepcopy(angle_dt)}
                                # "combi": angles_normaliser.scale_value(deepcopy(distance_normaliser.scale_value(environmental_data.distance_to_goal)
                                #                        * abs(environmental_data.raw_angle_to_goal)))}
                                #    "yaw_velocity": deepcopy(navigation._nav.orientation_rate.yaw)}

                # check if it is small as otherwise if the angle we are differentiating crosses the -3.14/3.14 wrap,
                # the derivative will be massive which in turn screws the policy approximators weights.
                if abs(state_t_plus_1["angle"] - state_t["angle"]) < 0.2:
                    angle_dt = state_t_plus_1["angle"] - state_t["angle"]

                # state_t_plus_1 = {"angle": 0.5,
                #                    # "distance": angle_and_velocity_normaliser.scale_value(0.0),
                #                    "yaw_velocity": 0.5}
                print("State + 1: {0}".format(state_t_plus_1))

                # receive reward based on new state
                # CURRENT SIMULATION's REWARD FUNCTION
                # if abs(environmental_data.raw_angle_to_goal) < 0.1 and navigation._nav.body_velocity.x > 0.0:
                #     reward = 2.0
                # # elif navigation._nav.body_velocity.x < 0.0:
                # #     reward = -3.0
                # else:
                #     reward = -1.0
                # if not np.sqrt((CONFIG["goal_position"][0] - navigation._nav.position.north)**2 +
                #                (CONFIG["goal_position"][1] - navigation._nav.position.east)**2) < 1.0:
                # reward = -1.0 + gaussian2D(abs(environmental_data.raw_angle_to_goal),
                #                            navigation._nav.orientation_rate.yaw, 0.0, 0.0,
                #                            0.9, [0.1, 0.2])
                # Q = np.diag([1.1, 0.2])
                # state_for_reward = np.array(state_t_plus_1.values())
                # reward = - np.dot(state_for_reward.transpose(), np.dot(Q, state_for_reward))# - (action_t - prev_action)**2
                if environmental_data.distance_to_goal < 1.0:
                    reward = 0.0
                else:
                    reward = -1 + gaussian1D(environmental_data.raw_angle_to_goal, 0.0, 0.5, 1.2) + \
                             gaussian1D(environmental_data.distance_to_goal, 0.0, 0.5, 10.0)
                # if environmental_data.raw_angle_to_goal < 0.0:
                #     reward = reward_normaliser.scale_value(environmental_data.raw_angle_to_goal)
                # else:
                #     reward = -1.0
                # reward = -1.0 - gaussian1DHalfInverse(environmental_data.raw_angle_to_goal, 0.0, 1.0, 0.3)

                # if abs(environmental_data.raw_angle_to_goal) < 0.1 and navigation._nav.body_velocity.x > 0.0:
                #     reward = 1.0 #gaussian1D(environmental_data.distance_to_goal, 0.0, 1.0, 5.0)
                # # elif navigation._nav.body_velocity.x < 0.0 or abs(navigation._nav.body_velocity.y) > 0.2:
                # #     reward = 1.0
                # else:
                #     reward = -1.0

                # if action_t_deterministic > 1.0:
                #     # limit the actions max value
                #     reward = -1.0
                # elif action_t_deterministic < 0.0:
                #     reward = 1.0
                # else:
                #     reward = 0.0


                print("Distance from Goal: {0}".format(environmental_data.distance_to_goal))
                print("Distance from Goal (normalised reward): {0}".format(distance_normaliser.scale_value(environmental_data.distance_to_goal)))
                print("REWARD: {0}".format(reward))
                f_rewards.write("{0}\t{1}\n".format(episode_number, reward))

                # if not np.sqrt((CONFIG["goal_position"][0] - navigation._nav.position.north)**2 +
                #                (CONFIG["goal_position"][1] - navigation._nav.position.east)**2) < 1.0:
                state_t_sub_features = basis_functions.computeFeatures(state_t.values(), goalState=False)
                state_t_plus_1_sub_features = basis_functions.computeFeatures(state_t_plus_1.values(), goalState=False)

                state_t_value = td_lambda.getStateValue(state_t_sub_features)
                state_t_plus_1_value = td_lambda.getStateValue(state_t_plus_1_sub_features)

                td_error = td_lambda.computeTDError(reward, CONFIG["gamma"], state_t_plus_1_value, state_t_value)
                print("State t Value: {0}".format(state_t_value))
                print("State t+1 Value: {0}".format(state_t_plus_1_value))
                print("TD ERROR: {0}".format(td_error))


                td_lambda.updateWeights(td_error, basis_functions)
                # else:
                #     state_t_sub_features = basis_functions.computeFeatures(state_t.values(), goalState=True)
                #     state_t_plus_1_sub_features = basis_functions.computeFeatures(state_t_plus_1.values(), goalState=True)

                # From LSTD implementation
                # if critic_value_func_params is not None:
                #     state_t_value = np.dot(state_t_sub_features, critic_value_func_params)
                #     state_t_plus_1_value = np.dot(state_t_plus_1_sub_features, critic_value_func_params)
                #     print("State t Value: {0}".format(state_t_value))
                #     print("State t+1 Value: {0}".format(state_t_plus_1_value))
                #     td_error = reward + ((CONFIG["gamma"] * state_t_plus_1_value) - state_t_value)
                #     print("TD ERROR: {0}".format(td_error))

                # --------- CALCULATE VALUE FUNCTIONS FROM STATE TRANSITION ---------
                # characteristic eligibility for a Gaussian policy of form: a ~ N(u(s), sigma)
                # state_features = basis_functions.computeFeatures(state_t.values())
                # advantage_approximation_features = ((action_t - action_t_deterministic) * \
                #                                    state_t_sub_features) / CONFIG["exploration_sigma"]**2

                # phi_squiggle = np.array([state_t_plus_1_sub_features.transpose(),
                #                      np.array([0.0 for i in range(len(state_t_plus_1_sub_features))]).transpose()]).transpose()
                # phi_hat = np.array([state_t_sub_features.transpose(), advantage_approximation_features.transpose()]).transpose()
                # print("Phi-Hat shape: {0}".format(phi_hat.shape))
                # print("Phi-Squiggle shape: {0}".format(phi_squiggle.shape))
                # lstd.updateTraces(phi_hat)
                # lstd.updateA(phi_hat, CONFIG["gamma"] * phi_squiggle)
                # lstd.update_b(reward)

                # --------- GET REWARD, UPDATE VALUE FUNCTIONS AND POLICY ---------
                # Update value functions
                # calculate the target (lambda return): U_t = R_t+1 + gamma * V_approx(S_t+1)
                # where V_approx(S_t+1) = dot(theta_t, phi_t+1) as weights have not been updated yet

                # Update policy weights only after the first episode
                # if step_number % CONFIG["spin_rate"] == 0:
                # if episode_number > -1 and not CONFIG["episode_fault_injected"] <= episode_number < CONFIG["episode_fault_injected"] + CONFIG["num_episodes_not_learning"]:
                if td_error > 0.0:
                    # # policy.plotFeatures = True
                    # critic_gradient, critic_value_func_params = lstd.calculateBeta()
                    # print("critic_gradient: {0}".format(critic_gradient.shape))
                    # if "prev_critic_gradient" not in locals():
                    #     prev_critic_gradient = np.zeros(critic_gradient.shape)
                    # print("Angle between gradient vectors: {0}".format(angle_between_vectors.angle_between(critic_gradient, prev_critic_gradient)))
                    # if angle_between_vectors.angle_between(critic_gradient, prev_critic_gradient) < 0.5 and td_error < 0.0:
                    print("UPDATING THE POLICY WEIGHTS!!!! :O")
                    X, T = td_lambda.traces.getTraces()
                    for x, trace in zip(X, T):
                        state_for_trace = basis_functions.computeFeatures(x, goalState=False) * trace
                        policy.updateWeights(gradient_vector=state_for_trace * (action_t - action_t_deterministic))
                    # lstd.decayStatistics()
                    # prev_critic_gradient = deepcopy(critic_gradient)
                print("Number of Traces: {0}".format(len(td_lambda.traces._values)))

                print("Policy ALPHA: {0}".format(policy.alpha))

                step_time = time.time() - step_start_time
                # accumulate total reward
                cum_reward += reward
                cum_step_time += step_time

                prev_action = deepcopy(action_t)

                # Check for goal condition
                if np.sqrt((CONFIG["goal_position"][0] - navigation._nav.position.north)**2 +
                               (CONFIG["goal_position"][1] - navigation._nav.position.east)**2) < 1.0:

                    if episode_number >= 20:
                        policy.decayAlpha(CONFIG["alpha_decay"])

                    # Zero all LSTD statistics as LSTD is not really an episodic algorithm
                    # lstd.decayStatistics(decay=0.0)
                    # print("ZEROES LSTD STATISTICS")
                    break

            # Stop logging navigation as episode has terminated either from max num steps or goal condition reached
            logNav(log_nav=False, dir_path=results_dir, file_name_descriptor=str(episode_number))

            # Log cumulative reward for episode and number of steps
            f_returns.write("{0}\t{1}\n".format(episode_number, cum_reward))
            f_timings.write("{0}\t{1}\n".format(episode_number, cum_step_time / step_number))

            # Disable behaviours
            ros_behaviour_interface.disable_behaviours(disable=True)
            # Now start the next episode where the nav will be reset etc.
