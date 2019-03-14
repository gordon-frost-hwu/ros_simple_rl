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
# from srl.basis_functions.simple_basis_functions import TileCodingBasisFunctions as BasisFunctions

from srl.approximators.linear_approximation import LinearApprox
from srl.approximators.ann_approximator_from_scratch import ANNApproximator

from srl.environments.cartpole import CartPoleEnvironment

from srl.useful_classes.rl_traces import Traces, TrueTraces
from srl.learning_algorithms.stober_td_learning import TDLinear
from srl.learning_algorithms.true_online_td_lambda import TrueOnlineTDLambda
from srl.learning_algorithms.lstd_algorithm import LSTD

from utilities.variable_normalizer import DynamicNormalizer

from rospkg import RosPack
rospack = RosPack()

class PolynomialBasisFunctions(object):
    def __init__(self, idx=0):
        self.resolution = 10
    # def computeFeatures(self, state, goalState=False):
    #     return np.array(state)
    def computeFeatures(self, state, goalState=False):
        if not goalState:
            x1 = state[0]
            x2 = state[1]
            x3 = state[2]
            x4 = state[3]
            # if state[0] >= 0.0:
            #     x1 = state[0]
            # else:
            #     x1 = 0.0
            # x2 = state[1]
            # if state[0] < 0.0:
            #     x3 = state[0]
            # else:
            #     x3 = 0.0
            # poly_state = np.array([x1**2, x1**3, x1*x2, x2**2, x2**3, 0.1])
            poly_state = np.array([x1**2, x1**3, x1*x2, x1*x3, x1*x4, x2**2, x2**3, x2*x3, x2*x4, x3**2, x3**3, x3*x4,
                                   x4**2, x4**3, 0.1])
            self.feature_vector_size = max(poly_state.shape)
            return poly_state
        else:
            return np.array([0.0 for i in range(self.feature_vector_size)])

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
    "num_hidden_units": 24
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
    "rbf_basis_scalar": 3.0,
    "number_of_dims_in_state": 4
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    # "basis_functions": BasisFunctions()
}

controlled_var = makeRange([0.5, 0.6, 0.7, 0.8, 0.85, 0.95], 5)
# controlled_var = makeRange([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95], 5)
CONFIG = {
    "test_policy": False,
    "test_vf": False,
    "actor_config": actor_config,
    "critic_config": critic_config,
    "goal_position": [20.0, 0.0],
    "episode_fault_injected": 0,
    "num_episodes_not_learning": 0,
    "log_actions": 1,
    "log_traces": False,
    "spin_rate": 40,
    "num_runs": 5, #len(controlled_var),
    "num_episodes": 300,
    "max_num_steps": 400,
    "policy_type": "ann_trad",
    "actor update rule": "cacla",
    "critic algorithm": "true",
    "sparse reward": False,
    "gamma": 0.9,   # was 0.1
    "lambda": 0.0,  # was 0.0
    "policy_lambda": 0.5,  # was 0.0
    "alpha_decay": 0.0, # was 0.00005
    "exploration_sigma": 1.0,
    "exploration_decay": 1.0,
    "min_trace_value": 0.1
}

class SynthPolicy(object):
    def __init__(self):
        self.name = "synth_policy"
        self.init_value = 0.0
        self.step = 0.01
        self.offset = 0.0
        self.magnitude = 1.0
        self.alpha = "SYNTH POLICY - NO LEARNING"
    def computeOutput(self):
        out = np.sin(self.init_value + self.offset)
        self.init_value -= self.step
        return self.magnitude * out
    def getParams(self):
        return np.zeros([1, 4])
    def updateOffset(self, val):
        self.offset += val

EndRun = False
# def manualEndRun(msg):
#     global EndRun
#     EndRun = True

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
            os.system("cp {0} {1}".format(filename, results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/basis_functions/simple_basis_functions.py {0}".format(results_dir))
            os.system("cp /home/gordon/software/simple-rl/srl/environments/cartpole.py {0}".format(results_dir))
            yaml.dump(CONFIG, open("{0}/CONFIG.yaml".format(results_dir), "w"),  allow_unicode=True)

            f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)
            f_num_steps = open("{0}{1}".format(results_dir, "/NumSteps.fso"), "w", 1)
            f_timings = open("{0}{1}".format(results_dir, "/AvgStepTime.fso"), "w", 1)

            basis_functions = BasisFunctions(resolution=CONFIG["critic_config"]["rbf_basis_resolution"], scalar=CONFIG["critic_config"]["rbf_basis_scalar"],
                                         num_dims=CONFIG["critic_config"]["number_of_dims_in_state"])

            # initialise policy and value functions
            if CONFIG["policy_type"] == "linear":
                policy = LinearApprox(actor_config, basis_functions=basis_functions)
                # policy.setParams(np.linspace(-0.2, 0.2, CONFIG["critic_config"]["rbf_basis_resolution"]))
            elif "ann" in CONFIG["policy_type"]:
                policy = ANNApproximator(CONFIG["actor_config"]["num_input_dims"], CONFIG["actor_config"]["num_hidden_units"], hlayer_activation_func="tanh")
                # policy.setParams(list(np.load("/tmp/policy_params0.npy")))
            elif CONFIG["policy_type"] == "synth":
                policy = SynthPolicy()

            # Init the type of Critic Algorithm you wish according to CONFIG dict
            if CONFIG["critic algorithm"] == "trad":
                td_lambda = TDLinear(len(basis_functions.computeFeatures([0.0 for _ in range(critic_config["num_input_dims"])])),
                                         critic_config["alpha"],
                                         CONFIG["gamma"],
                                         CONFIG["lambda"], init_val=CONFIG["critic_config"]["initial_value"])
                # td_lambda = Traditional_TD_LAMBDA(LinearApprox(critic_config), CONFIG["lambda"], CONFIG["min_trace_value"])
            elif CONFIG["critic algorithm"] == "true":
                td_lambda = TrueOnlineTDLambda(basis_functions, critic_config, CONFIG["lambda"], CONFIG["gamma"], init_val=0.0)
            elif "ann" in CONFIG["critic algorithm"]:
                td_lambda = ANNApproximator(CONFIG["actor_config"]["num_input_dims"],
                                            CONFIG["actor_config"]["num_hidden_units"], hlayer_activation_func="tanh")
                tmp_params = td_lambda.getParams()
                td_lambda.setParams(tmp_params)
                # td_lambda.setParams(list(np.load("/tmp/critic_params0.npy")))
                # td_lambda.setParams(list(np.load("/tmp/critic_params0.npy")))
            elif CONFIG["critic algorithm"] == "nac":
                td_lambda = LSTD(lmbda=CONFIG["lambda"], gamma=CONFIG["gamma"])
            # td_lambda = RBFNApprox(3, 50, 1)
            if CONFIG["critic algorithm"] == "ann_trad":
                traces = Traces(CONFIG["lambda"], CONFIG["min_trace_value"])
            elif CONFIG["critic algorithm"] == "ann_true":
                traces = TrueTraces(CONFIG["critic_config"]["alpha"], CONFIG["gamma"], CONFIG["lambda"])
            policy_traces = Traces(CONFIG["policy_lambda"], CONFIG["min_trace_value"])
            # td_lambda = PyBrainANNApproximator(critic_config["alpha"])

            cartpole_environment = CartPoleEnvironment()
            exploration_sigma = deepcopy(CONFIG["exploration_sigma"])

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

                if CONFIG["policy_type"] == "synth" and episode_number > 0:
                    policy.updateOffset(0.2)

                if td_lambda.name == "true online" or td_lambda.name == "trad online":
                    td_lambda.episodeReset()
                if td_lambda.name == "nac":
                    f_gradient_vec_diff = open("{0}{1}".format(results_dir, "/GradientVecDiff{0}.fso".format(episode_number)), "w", 1)
                    if episode_number > 0:
                        td_lambda.decayStatistics(decay=0.0)
                # basis_functions.setSigma(0.25)
                # basis_functions.setSigma(0.5 / (episode_number + 1))
                EndEpisodeonStep = rospy.get_param("~NumSteps")
                if episode_number > 5 and exploration_sigma > 0.1:
                    exploration_sigma *= CONFIG["exploration_decay"]

                if "ann" in CONFIG["critic algorithm"]:
                    traces.reset()
                if "ann" in CONFIG["policy_type"]:
                    policy_traces.reset()

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
                    state_t_raw = {"angle": angle_normaliser.scale_value(deepcopy(sensors[0])),
                               "angle_deriv": angle_deriv_normaliser.scale_value(deepcopy(sensors[1])),
                               "position": position_normaliser.scale_value(deepcopy(sensors[2])),
                               "position_deriv": position_deriv_normaliser.scale_value(deepcopy(sensors[3]))}

                    # td_lambda.updateTrace(state_t.values(), 1.0)
                    if td_lambda.name == "true online":
                        if td_lambda.stateprime is None:
                            td_lambda.start(state_t.values())
                            # print("HEREERER FIRST: {0}".format(td_lambda.theta))

                    print("State: {0}".format(state_t))
                    # print("TD GAMMA: {0}".format(td_lambda.gamma))

                    # --------- GET ACTION AND PERFORM IT ---------
                    # Compute the deterministic (??? deterministic if its an approximation ???) action
                    state_t_sub_features = basis_functions.computeFeatures(state_t.values())
                    state_t_full_features = np.r_[state_t_sub_features,
                                         np.array([0.0 for i in range(len(state_t_sub_features))])]
                    if policy.name == "scratch_ann":
                        action_t_deterministic = policy.computeOutput(state_t_raw.values())
                    elif policy.name == "LinearApprox":
                        action_t_deterministic = policy.computeOutput(basis_functions.computeFeatures(state_t, approx="policy"))
                    elif policy.name == "rbfn":
                        action_t_deterministic = policy.computeOutput(state_t.values())
                    elif policy.name == "synth_policy":
                        action_t_deterministic = policy.computeOutput()
                    else:
                        print("Give approximator a name attribute!!!")
                        exit(0)

                    print("Deterministic Action: {0}".format(action_t_deterministic))
                    print("BASIS SIGMA: {0}".format(basis_functions.sigma))
                    # if step_number % (3 * CONFIG["spin_rate"]) == 0:
                    exploration_mean = 0.0#np.clip(state_t["angle"], -0.1, 0.1)
                    exploration = np.random.normal(exploration_mean, exploration_sigma)

                    action_t = float(action_t_deterministic) + exploration
                    print("Deterministic Action: {0}".format(action_t_deterministic))
                    print("Stochastic Action: {0}".format(action_t))
                    if prev_det_action is None:
                        prev_det_action = deepcopy(action_t_deterministic)

                    # Log the deterministic action chosen for each state according to the policy LA
                    # use the deterministic action just so that it is cleaner to look at for debugging
                    if episode_number % CONFIG["log_actions"] == 0:
                        if step_number == 0:
                            state_keys = state_t.keys()
                            state_keys.append("action")
                            label_logging_format = "#{"+"}\t{".join([str(state_keys.index(el)) for el in state_keys])+"}\n"
                            f_actions.write(label_logging_format.format(*state_keys))
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
                    state_t_plus_1_raw = {"angle": angle_normaliser.scale_value(deepcopy(sensors[0])),
                                   "angle_deriv": angle_deriv_normaliser.scale_value(deepcopy(sensors[1])),
                                   "position": position_normaliser.scale_value(deepcopy(sensors[2])),
                                   "position_deriv": position_deriv_normaliser.scale_value(deepcopy(sensors[3]))}
                    # state_t_plus_1_raw = {"angle": deepcopy(sensors[0]),
                    #                "angle_deriv": deepcopy(sensors[1]),
                    #                "position": deepcopy(sensors[2]),
                    #                "position_deriv": deepcopy(sensors[3])}

                    print("State + 1: {0}".format(state_t_plus_1))
                    print("Cart Position: {0}".format(cartpole_environment.getCartPosition()))
                    print("Pole Angles: {0}".format(cartpole_environment.getPoleAngles()))

                    if CONFIG["sparse reward"]:
                        reward = cartpole_environment.getReward()
                    else:
                        # reward = 0.0 + (10.0 * (1.0 - abs(state_t_plus_1["angle"])))
                        reward = 10.0 - (10 * (abs(state_t_plus_1["angle"]))**2) - \
                                        (0.1 * (action_t_deterministic - prev_det_action)**2)
                    # - (state_t_plus_1["position"]**2)
                    # if not cartpole_environment.episodeEnded():
                    # reward = 10.0 - (10 * (abs(state_t_plus_1["angle"]))**2) - (7 * (abs(state_t_plus_1["position"]))**2) - \
                    #             (0.1 * (action_t_deterministic - prev_det_action)**2)
                    #
                    # else:
                    #     reward = -10
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

                    if td_lambda.name == "true online":
                        print("True Online")
                        state_t_value = td_lambda.getStateValue(state_t.values())
                        state_t_plus_1_value = td_lambda.getStateValue(state_t_plus_1.values())
                    elif td_lambda.name == "trad online":
                        print("Trad Online")
                        state_t_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t.values()))
                        state_t_plus_1_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t_plus_1.values()))
                    elif td_lambda.name == "scratch_ann":
                        print("ANN")
                        state_t_value = td_lambda.computeOutput(state_t_raw.values())
                        state_t_plus_1_value = td_lambda.computeOutput(state_t_plus_1_raw.values())
                    elif td_lambda.name == "nac":
                        if critic_value_func_params is not None:
                            state_t_value = np.dot(state_t_sub_features, critic_value_func_params)
                            state_t_plus_1_value = np.dot(state_t_plus_1_sub_features, critic_value_func_params)
                        else:
                            state_t_value = 0.0
                            state_t_plus_1_value = 0.0


                    print("State_t value: {0}".format(state_t_value))
                    print("State_t+1 value: {0}".format(state_t_plus_1_value))

                    if td_lambda.name == "true online":
                        if not cartpole_environment.episodeEnded():
                            td_error = td_lambda.step(reward, state_t_plus_1)
                        else:
                            td_error = td_lambda.end(reward)
                        new_state_t_plus_1_value = td_lambda.getStateValue(state_t_plus_1)
                    elif td_lambda.name == "trad online":
                        # if not environment_info.distance_to_goal < 2.0:
                        td_error = td_lambda.train(basis_functions.computeFeatures(state_t.values()), reward,
                                               basis_functions.computeFeatures(state_t_plus_1.values()))
                        # else:
                        #     print("MWHAHAHAHAHAHAH")
                        #     td_error = td_lambda.end(basis_functions.computeFeatures(state_t), reward)
                        #     print("AHAHAHAHAHAHAH")
                        new_state_t_plus_1_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t_plus_1))
                    elif td_lambda.name == "scratch_ann":
                        # if step_number > 2: # due to indexes required for true TD lambda updates
                        if "prev_critic_gradient" not in locals():
                            prev_critic_gradient = np.zeros(td_lambda.getParams().shape)

                        # For ANN critic
                        if CONFIG["critic algorithm"] == "ann_trad":
                            td_error = reward + (CONFIG["gamma"] * state_t_plus_1_value) - state_t_value
                        elif CONFIG["critic algorithm"] == "ann_true":
                            td_error = reward + (CONFIG["gamma"] * state_t_plus_1_value) - \
                                       td_lambda.computeOutputThetaMinusOne(state_t_raw.values())
                        prev_critic_weights = td_lambda.getParams()
                        critic_gradient = td_lambda.calculateGradient(state_t_raw.values())

                        p = td_lambda.getParams()
                        if CONFIG["critic algorithm"] == "ann_trad":
                            traces.updateTrace(critic_gradient, 1.0)  # for standard TD(lambda)
                            X, T = traces.getTraces()
                            for x, trace in zip(X, T):
                                # print("updating critic using gradient vector: {0}\t{1}".format(x, trace))
                                p += critic_config["alpha"] * td_error * (x * trace)
                            # td_lambda.setParams(prev_critic_weights + CONFIG["critic_config"]["alpha"] * td_error * critic_gradient)
                        elif CONFIG["critic algorithm"] == "ann_true":
                            # For True TD(lambda)
                            print("UPDATING ANN CRITC with TRUE TD(lambda)")
                            traces.updateTrace(critic_gradient)    # for True TD(lambda)
                            part_1 = td_error * traces.e
                            part_2 = CONFIG["critic_config"]["alpha"] * \
                                    np.dot((td_lambda.computeOutputThetaMinusOne(state_t_raw.values()) - state_t_value), critic_gradient)
                            p += part_1 + part_2

                        td_lambda.setParams(p)
                        new_state_t_plus_1_value = td_lambda.computeOutput(state_t_plus_1_raw.values())
                    elif td_lambda.name == "nac":
                        # characteristic eligibility for a Gaussian policy of form: a ~ N(u(s), sigma)
                        advantage_approximation_features = (exploration * \
                                                           state_t_sub_features) / CONFIG["exploration_sigma"]**2
                        phi_squiggle = np.array([state_t_plus_1_sub_features.transpose(),
                                             np.array([0.0 for i in range(len(state_t_plus_1_sub_features))]).transpose()]).transpose()
                        phi_hat = np.array([state_t_sub_features.transpose(), advantage_approximation_features.transpose()]).transpose()
                        td_lambda.updateTraces(phi_hat)
                        td_lambda.updateA(phi_hat, CONFIG["gamma"] * phi_squiggle)
                        td_lambda.update_b(reward)
                        td_error = reward + CONFIG["gamma"] * state_t_plus_1_value - state_t_value
                        critic_value_func_params, critic_gradient = td_lambda.calculateBeta()
                        # np.save("{0}/critic_gradient{1}".format(results_dir, step_number), critic_gradient)
                        # print("critic_gradient: {0}".format(critic_gradient.shape))
                        if "prev_critic_gradient" not in locals():
                            prev_critic_gradient = np.zeros(critic_gradient.shape)
                        angle_between_gradient_vectors = angle_between_vectors.angle_between(critic_gradient, prev_critic_gradient)
                        print("Angle between gradient vectors: {0}".format(angle_between_gradient_vectors))
                        f_gradient_vec_diff.write("{0}\t{1}\n".format(episode_number, angle_between_gradient_vectors))

                        print("GRADIENT (CRITIC): {0}".format(critic_gradient))

                        print("State t Value: {0}".format(state_t_value))
                        print("State t+1 Value: {0}".format(state_t_plus_1_value))
                        print("TD ERROR: {0}".format(td_error))


                    policy_traces.updateTrace(policy.calculateGradient(state_t_raw.values()), 1.0)

                    # Update policy weights only after the first episode
                    params = policy.getParams()

                    ACTOR_UPDATE_CONDITION = False
                    if CONFIG["actor update rule"] == "cacla":
                        if td_error > 0.0:
                            ACTOR_UPDATE_CONDITION = True
                        else:
                            ACTOR_UPDATE_CONDITION = False
                    elif CONFIG["actor update rule"] == "td lambda":
                        ACTOR_UPDATE_CONDITION = True

                    vali_actor_update_condition = (not CONFIG["test_policy"] or
                                                    (CONFIG["test_policy"] and CONFIG["policy_test_type"] == "online_nessie"))
                    if td_lambda.name == "nac" and vali_actor_update_condition:
                        # if state_t_plus_1_value > 0.0:
                        print("angle_between_gradient_vectors: {0}".format(angle_between_gradient_vectors))
                        if abs(angle_between_gradient_vectors) < 0.01:
                            print("Angle Between Gradients is small --- update the ACTOR")
                            old_action = policy.computeOutput(basis_functions.computeFeatures(state_t, approx="policy"))
                            policy.setParams(params + actor_config["alpha"] * (critic_gradient * (action_t - action_t_deterministic)))
                            # policy.setParams(params - actor_config["alpha"] * critic_gradient)
                            new_action = policy.computeOutput(basis_functions.computeFeatures(state_t, approx="policy"))
                            print("Old Action: {0}".format(old_action))
                            print("New Action: {0}".format(new_action))
                            td_lambda.decayStatistics(decay=0.9)
                        else:
                            print("Angle between Gradients too Large, NOT UPDATING POLICY")

                    elif ACTOR_UPDATE_CONDITION and vali_actor_update_condition and CONFIG["policy_type"] != "synth":# and episode_number > 0:
                        # # policy.plotFeatures = True


                        print("UPDATING THE POLICY WEIGHTS!!!! :O")
                        # X, T = td_lambda.traces.getTraces()
                        # for x, trace in zip(X, T):
                        #     state_for_trace = basis_functions.computeFeatures(x, goalState=False) * trace
                        # desired_stepsized_output = (actor_config["alpha"] * (action_t - action_t_deterministic)) + action_t_deterministic
                        # policy.updateWeights(state_t.values(), action_t)
                        # Experimental weights updates
                        # tmp_params = deepcopy(policy.network.params)
                        # policy.updateWeights(state_t.values(), action_t)
                        # tmp_gradient = policy.trainer.gradient
                        # print(tmp_gradient)
                        # policy.network._params = tmp_params + actor_config["alpha"] * (action_t - action_t_deterministic) * tmp_gradient


                        if policy.name == "scratch_ann":
                            old_action = policy.computeOutput(state_t_raw.values())
                        elif policy.name == "LinearApprox":
                            old_action = policy.computeOutput(basis_functions.computeFeatures(state_t, approx="policy"))
                        elif policy.name == "rbfn":
                            old_action = policy.computeOutput(state_t.values())
                        else:
                            print("Give approximator a name attribute!!!")
                            exit(0)
                        print("Action BEFORE actor update: {0}".format(old_action))

                        if policy.name == "scratch_ann":
                            policy_gradient = policy.calculateGradient(state=state_t.values())
                        elif policy.name == "LinearApprox":
                            policy_gradient = policy.calculateGradient(state=basis_functions.computeFeatures(state_t, approx="policy"))
                            # policy_gradient = (exploration * \
                            #                                state_t_sub_features) / CONFIG["exploration_sigma"]**2
                        elif policy.name == "rbfn":
                            policy_gradient = policy.calculateGradient(state_t.values())
                        else:
                            print("Give approximator a name attribute!!!")
                            exit(0)

                        # print("GRADIENT (POLICY): {0}".format(policy_gradient))

                        # print("GRADIENT: {0}".format(policy_gradient))
                        # policy.setParams(params + actor_config["alpha"] * (policy_gradient * (action_t - action_t_deterministic)))
                        # policy.setParams(params + actor_config["alpha"] * (policy_gradient * td_error))

                        if policy.name == "scratch_ann":
                            if CONFIG["actor update rule"] == "cacla":
                                # policy.setParams(params + actor_config["alpha"] * (policy_gradient * exploration))
                                X, T = policy_traces.getTraces()
                                p = policy.getParams()
                                # policy.setParams(p + actor_config["alpha"] * (policy_gradient * (action_t - action_t_deterministic)))
                                print("Number of traces: {0}".format(len(T)))
                                for x, trace in zip(X, T):
                                    #print("updating critic using gradient vector: {0}\t{1}".format(x, trace))
                                    p += actor_config["alpha"] * (x * trace) * exploration
                                policy.setParams(p)
                            else:
                                policy.setParams(params + actor_config["alpha"] * (policy_gradient * td_error))
                        else:
                            if CONFIG["actor update rule"] == "td lambda":
                                # TD(Lambda) actor update
                                print("TD LAMBDA Actor udpate")
                                policy.setParams(params + actor_config["alpha"] * (policy_gradient * td_error * td_lambda.getTraces()))
                            elif CONFIG["actor update rule"] == "cacla":
                                # CACLA(Lambda) actor update
                                print("CACLA Actor update")
                                policy.setParams(params + actor_config["alpha"] * (policy_gradient * exploration * td_lambda.getTraces()))



                        if policy.name == "scratch_ann":
                            new_action = policy.computeOutput(state_t_raw.values())
                        elif policy.name == "LinearApprox":
                            new_action = policy.computeOutput(basis_functions.computeFeatures(state_t, approx="policy"))
                        elif policy.name == "rbfn":
                            new_action = policy.computeOutput(state_t.values())
                        else:
                            print("Give approximator a name attribute!!!")
                            exit(0)

                        print("Action AFTER actor update: {0}".format(new_action))

                    # print("Number of Traces: {0}".format(len(td_lambda.traces._values)))
                    # np.save("{0}/policy_params{1}".format(results_dir, step_number), policy.getParams())

                    if td_lambda.name == "nac" or td_lambda.name == "scratch_ann":
                        prev_critic_gradient = deepcopy(critic_gradient)
                    print("Policy ALPHA: {0}".format(policy.alpha))
                    print("GAMMA: {0}".format(CONFIG["gamma"]))
                    print("LAMBDA: {0}".format(CONFIG["lambda"]))

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
                    np.save("{0}/critic_params{1}".format(results_dir, episode_number), td_lambda.getParams())


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
