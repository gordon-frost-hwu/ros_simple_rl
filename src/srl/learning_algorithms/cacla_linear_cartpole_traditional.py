#! /usr/bin/python
# Adaption of the previous LSTD NAC agent to the CACLA actor update methodology
import roslib; roslib.load_manifest("rl_pybrain")
import rospy
from std_msgs.msg import Empty

from scipy import random
from copy import deepcopy
import numpy as np

import sys
import time
import os
import yaml

from srl.useful_classes.angle_between_vectors import AngleBetweenVectors
from srl.useful_classes.rl_traces import Traces

# from srl.basis_functions.simple_basis_functions import PolynomialBasisFunctions as BasisFunctions
from srl.basis_functions.simple_basis_functions import RBFBasisFunctions as BasisFunctions
# import simple_basis_functions as cbasis
# BasisFunctions = cbasis.RBFBasisFunctions
# from srl.basis_functions.simple_basis_functions import TileCodingBasisFunctions as BasisFunctions

from srl.approximators.linear_approximation import LinearApprox
from srl.approximators.ann_approximator_from_scratch import ANNApproximator
# import linear_approximation as la

from srl.environments.cartpole import CartPoleEnvironment

from srl.learning_algorithms.td_lambda_traditional import Traditional_TD_LAMBDA
from srl.learning_algorithms.stober_td_learning import TDLinear
from srl.learning_algorithms.true_online_td_lambda import TrueOnlineTDLambda

from utilities.variable_normalizer import DynamicNormalizer

from rospkg import RosPack
rospack = RosPack()

def makeRange(values, num_iterations):
    res = []
    for value in values:
        for _ in range(num_iterations):
            res.append(value)
    return res

# controlled_var = makeRange([2, 4, 8, 16, 24, 32], 5)
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
    "alpha": 0.01,
    "random_weights": False,
    "minimise": False,
    "approximator_boundaries": [-200.0, 200.0],
    "num_input_dims": 4,
    "rbf_basis_resolution": 20,
    "rbf_basis_scalar": 3.0
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    # "basis_functions": BasisFunctions()
}

# controlled_var = makeRange([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95], 5)
controlled_var = makeRange([0.5, 0.6, 0.7, 0.8, 0.85, 0.95], 5)
CONFIG = {
    "actor_config": actor_config,
    "critic_config": critic_config,
    "goal_position": [10.0, 0.0],
    "episode_fault_injected": 0,
    "num_episodes_not_learning": 0,
    "log_actions": 50,
    "spin_rate": 40,
    "num_runs": len(controlled_var),
    "num_episodes": 300,
    "max_num_steps": 400,
    "linear policy": True,
    "sparse reward": False,
    "gamma": 0.9,   # was 0.1
    "lambda":controlled_var,  # was 0.0
    "alpha_decay": 0.0, # was 0.00005
    "exploration_sigma": 1.0,
    "exploration_decay": 0.005,
    "min_trace_value": 0.1
}

EndRun = False

if __name__ == '__main__':
    global EndRun
    args = sys.argv
    if "-r" in args:
        results_dir_name = args[args.index("-r") + 1]
    else:
        results_dir_name = "cartpole_run"

    rospy.init_node(results_dir_name.replace(".", ""))
    rospy.set_param("~NumRuns", CONFIG["num_runs"])
    rospy.set_param("~NumEpisodes", CONFIG["num_episodes"])
    rospy.set_param("~NumSteps", CONFIG["max_num_steps"])
    # sub = rospy.Subscriber("EndRun", Empty, manualEndRun)

    angle_between_vectors = AngleBetweenVectors()


    # initialise some global variables/objects
    # global normalisers
    position_normaliser = DynamicNormalizer([-2.4, 2.4], [-1.0, 1.0])
    position_deriv_normaliser = DynamicNormalizer([-3.0, 3.0], [-1.0, 1.0])
    angle_normaliser = DynamicNormalizer([-0.2, 0.2], [-1.0, 1.0])
    angle_deriv_normaliser = DynamicNormalizer([-2.0, 2.0], [-1.0, 1.0])

    while not rospy.is_shutdown():

        # Loop number of runs
        for run in range(CONFIG["num_runs"]):
            # Create logging directory and files
            results_dir = "/tmp/{0}{1}".format(results_dir_name, run)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            filename = os.path.basename(sys.argv[0])
            # Save the used python modules and a quick CONFIG reference guide
            os.system("cp {0} {1}".format(filename, results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/basis_functions/simple_basis_functions.py {0}".format(results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/environments/cartpole.py {0}".format(results_dir))
            yaml.dump(CONFIG, open("{0}/CONFIG.yaml".format(results_dir), "w"),  allow_unicode=True)

            f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)
            f_num_steps = open("{0}{1}".format(results_dir, "/NumSteps.fso"), "w", 1)
            f_timings = open("{0}{1}".format(results_dir, "/AvgStepTime.fso"), "w", 1)

            basis_functions = BasisFunctions(resolution=CONFIG["critic_config"]["rbf_basis_resolution"], scalar=CONFIG["critic_config"]["rbf_basis_scalar"])

            # initialise policy and value functions
            if CONFIG["linear policy"]:
                policy = LinearApprox(actor_config, basis_functions=basis_functions)
            else:
                policy = ANNApproximator(CONFIG["actor_config"]["num_input_dims"], CONFIG["actor_config"]["num_hidden_units"], hlayer_activation_func="tanh")
            # basis_functions = PolynomialBasisFunctions(idx=run)

            cartpole_environment = CartPoleEnvironment()

            # traditional TD(lambda) learning algorithm for the critic
            # td_lambda = Traditional_TD_LAMBDA(LinearApprox(critic_config), CONFIG["lambda"], CONFIG["min_trace_value"])
            td_lambda = TDLinear(len(basis_functions.computeFeatures([0.0 for _ in range(critic_config["num_input_dims"])])),
                                 critic_config["alpha"],
                                 CONFIG["gamma"],
                                 CONFIG["lambda"][run], init_val=CONFIG["critic_config"]["initial_value"])

            # Loop number of episodes
            for episode_number in range(CONFIG["num_episodes"]):
                global episode_number, results_dir
                # initialise state and internal variables
                if episode_number % CONFIG["log_actions"] == 0:
                    f_actions = open("{0}{1}".format(results_dir, "/actions{0}.csv".format(episode_number)), "w", 1)
                # f_rewards = open("{0}{1}".format(results_dir, "/RewardsEpisode{0}.fso".format(episode_number)), "w", 1)
                cum_reward = 0.0
                cum_step_time = 0.0
                angle_dt = 0.0
                prev_det_action = None
                critic_value_func_params = None

                td_lambda.episodeReset()
                # basis_functions.setSigma(0.25)
                # basis_functions.setSigma(0.5 / (episode_number + 1))
                EndEpisodeonStep = rospy.get_param("~NumSteps")


                # Loop number of steps
                for step_number in range(CONFIG["max_num_steps"]):
                    # --------- STATE AND TRACES ---------
                    # get current state
                    step_start_time = time.time()
                    print("------ STEP {0} -------".format(step_number))
                    sensors = cartpole_environment.getSensors()
                    state_t = {"angle": angle_normaliser.scale_value(deepcopy(sensors[0])),
                               "angle_deriv": angle_deriv_normaliser.scale_value(deepcopy(sensors[1])),
                               "position": position_normaliser.scale_value(deepcopy(sensors[2])),
                               "position_deriv": position_deriv_normaliser.scale_value(deepcopy(sensors[3]))}

                    # td_lambda.updateTrace(state_t.values(), 1.0)
                    # if td_lambda.stateprime is None:
                    #     td_lambda.start(state_t.values())

                    print("State: {0}".format(state_t))

                    # --------- GET ACTION AND PERFORM IT ---------
                    # Compute the deterministic (??? deterministic if its an approximation ???) action
                    state_t_sub_features = basis_functions.computeFeatures(state_t.values())
                    state_t_full_features = np.r_[state_t_sub_features,
                                         np.array([0.0 for i in range(len(state_t_sub_features))])]
                    if policy.name == "scratch_ann":
                        action_t_deterministic = policy.computeOutput(state_t.values())
                        # action_t_deterministic = policy.computeOutput(basis_functions.computeFeatures(state_t.values(), approx="policy"))
                    elif policy.name == "LinearApprox":
                        action_t_deterministic = policy.computeOutput(basis_functions.computeFeatures(state_t.values(), approx="policy"))

                    print("BASIS SIGMA: {0}".format(basis_functions.sigma))
                    exploration = np.random.normal(0.0, CONFIG["exploration_sigma"])

                    action_t = action_t_deterministic + exploration
                    print("Deterministic Action: {0}".format(action_t_deterministic))
                    print("Stochastic Action: {0}".format(action_t))
                    if prev_det_action is None:
                        prev_det_action = deepcopy(action_t_deterministic)

                    # Log the deterministic action chosen for each state according to the policy LA
                    # use the deterministic action just so that it is cleaner to look at for debugging
                    if episode_number % CONFIG["log_actions"] == 0:
                        logging_list = state_t.values()
                        logging_list.append(action_t_deterministic)
                        action_logging_format = "{"+"}\t{".join([str(logging_list.index(el)) for el in logging_list])+"}\n"
                        f_actions.write(action_logging_format.format(*logging_list))

                    # Perform the action
                    cartpole_environment.performAction(action_t)

                    print("Sleeping for a bit ...")
                    time.sleep(1.0 / CONFIG["spin_rate"])

                    # --------- GET NEW STATE ---------
                    # observe new state --- which is dependant on whether it is the final goal state or not

                    sensors = cartpole_environment.getSensors()
                    state_t_plus_1 = {"angle": angle_normaliser.scale_value(deepcopy(sensors[0])),
                               "angle_deriv": angle_deriv_normaliser.scale_value(deepcopy(sensors[1])),
                               "position": position_normaliser.scale_value(deepcopy(sensors[2])),
                               "position_deriv": position_deriv_normaliser.scale_value(deepcopy(sensors[3]))}

                    print("State + 1: {0}".format(state_t_plus_1))
                    print("Cart Position: {0}".format(cartpole_environment.getCartPosition()))
                    print("Pole Angles: {0}".format(cartpole_environment.getPoleAngles()))

                    if CONFIG["sparse reward"]:
                        reward = cartpole_environment.getReward()
                    else:
                        reward = 10.0 - (10 * (abs(state_t_plus_1["angle"]))**2) - \
                                        (0.1 * (action_t_deterministic - prev_det_action)**2)
                        # - (7 * (abs(state_t_plus_1["position"]))**2)
                    # if reward > 0.0:
                    #     print("Difference in ACTIONS: {0}".format(action_t_deterministic - prev_det_action))
                    #     reward = reward - (action_t_deterministic - prev_det_action)**2
                    print("REWARD: {0}".format(reward))
                    # f_rewards.write("{0}\t{1}\n".format(episode_number, reward))

                    # state_t_sub_features = basis_functions.computeFeatures(state_t.values(), goalState=False)
                    if cartpole_environment.episodeEnded():
                        state_t_plus_1_sub_features = np.zeros(state_t_sub_features.shape)
                    else:
                        state_t_plus_1_sub_features = basis_functions.computeFeatures(state_t_plus_1.values(), goalState=False)

                    state_t_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t.values()))
                    state_t_plus_1_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t_plus_1.values()))

                    # For traditional TD(lambda)
                    # terminalState = cartpole_environment.episodeEnded()
                    # td_error = td_lambda.computeTDError(reward, CONFIG["gamma"], state_t_plus_1_value, state_t_value, terminalState)

                    # For Stober TD(lambda)
                    # if cartpole_environment.episodeEnded():
                    td_error = td_lambda.train(basis_functions.computeFeatures(state_t.values()), reward,
                                               basis_functions.computeFeatures(state_t_plus_1.values()))
                    # else:
                    #     td_error = td_lambda.end(reward, basis_functions.computeFeatures(state_t.values()))

                    # For True Online TD(lambda)
                    # if not cartpole_environment.episodeEnded():
                    #     td_error = td_lambda.step(reward, state_t_plus_1.values())
                    # else:
                    #     td_error = td_lambda.end(reward)

                    print("State t Value: {0}".format(state_t_value))
                    print("State t+1 Value: {0}".format(state_t_plus_1_value))
                    print("TD ERROR: {0}".format(td_error))

                    # td_lambda.updateWeights(td_error, basis_functions, use_traces=True)
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
                        params = policy.getParams()

                        if policy.name == "scratch_ann":
                            old_action = policy.computeOutput(state_t.values())
                            # old_action = policy.computeOutput(basis_functions.computeFeatures(state_t.values(), approx="policy"))
                        elif policy.name == "LinearApprox":
                            old_action = policy.computeOutput(basis_functions.computeFeatures(state_t.values(), approx="policy"))
                        elif policy.name == "rbfn":
                            old_action = policy.computeOutput(state_t.values())
                        else:
                            print("Give approximator a name attribute!!!")
                            exit(0)
                        print("Action BEFORE actor update: {0}".format(old_action))

                        if policy.name == "scratch_ann":
                            gradient = policy.calculateGradient(state=state_t.values())
                        elif policy.name == "LinearApprox":
                            # gradient = policy.calculateGradient(state_t.values(), basis_functions)
                            gradient = policy.calculateGradient(state=basis_functions.computeFeatures(state_t.values(), approx="policy"))
                        elif policy.name == "rbfn":
                            gradient = policy.calculateGradient(state_t.values())
                        else:
                            print("Give approximator a name attribute!!!")
                            exit(0)

                        # print("GRADIENT: {0}".format(gradient))
                        policy.setParams(params + actor_config["alpha"] * (gradient * (action_t - action_t_deterministic)))

                        if policy.name == "scratch_ann":
                            new_action = policy.computeOutput(state_t.values())
                            # new_action = policy.computeOutput(basis_functions.computeFeatures(state_t.values(), approx="policy"))
                        elif policy.name == "LinearApprox":
                            new_action = policy.computeOutput(basis_functions.computeFeatures(state_t.values(), approx="policy"))
                        elif policy.name == "rbfn":
                            new_action = policy.computeOutput(state_t.values())
                        else:
                            print("Give approximator a name attribute!!!")
                            exit(0)
                        print("Action AFTER actor update: {0}".format(new_action))
                        # policy.updateWeights(gradient_vector=td_lambda.traces * (action_t - action_t_deterministic))

                    # print("Number of Traces: {0}".format(len(td_lambda.traces._values)))

                    # print("Policy ALPHA: {0}".format(policy.alpha))

                    step_time = time.time() - step_start_time
                    # accumulate total reward
                    cum_reward += reward
                    cum_step_time += step_time

                    prev_det_action = deepcopy(action_t_deterministic)

                    # Check for goal condition
                    if cartpole_environment.episodeEnded():
                        # Zero all LSTD statistics as LSTD is not really an episodic algorithm
                        # lstd.decayStatistics(decay=0.0)
                        # print("ZEROES LSTD STATISTICS")
                        break
                    if rospy.is_shutdown():
                        exit(0)
                    if step_number >= EndEpisodeonStep:
                        break

                if episode_number % CONFIG["log_actions"] == 0:
                    np.save("{0}/policy_params{1}".format(results_dir, episode_number), policy.getParams())
                    np.save("{0}/critic_params{1}".format(results_dir, episode_number), td_lambda.params)


                # Log cumulative reward for episode and number of steps
                f_returns.write("{0}\t{1}\n".format(episode_number, cum_reward))
                f_num_steps.write("{0}\t{1}\n".format(episode_number, step_number))
                f_timings.write("{0}\t{1}\n".format(episode_number, cum_step_time / step_number))
                cartpole_environment.reset()

                EndRunonEpisode = rospy.get_param("~NumEpisodes")
                if episode_number >= EndRunonEpisode:
                    break

            EndonRun = rospy.get_param("~NumRuns")
            if run >= EndonRun:
                break

        rospy.signal_shutdown("Simulation Finished ...")

    rospy.spin()
