#! /usr/bin/python
# Adaption of the previous LSTD NAC agent to the CACLA actor update methodology
import roslib; roslib.load_manifest("rl_pybrain")
import rospy

from scipy import random
from copy import deepcopy
import numpy as np
from random import randrange
import sys
import time
import os
import yaml

# from srl.basis_functions.simple_basis_functions import PolynomialBasisFunctions as BasisFunctions
from srl.basis_functions.simple_basis_functions import RBFBasisFunctions as BasisFunctions
# from srl.basis_functions.simple_basis_functions import TileCodingBasisFunctions as BasisFunctions

from srl.approximators.linear_approximation import LinearApprox
from srl.approximators.ann_approximation import ANNApproximator as PyBrainANNApproximator
from srl.approximators.ann_approximator_from_scratch import ANNApproximator
from srl.approximators.rbfn_approximator import RBFNApprox
from srl.useful_classes.rl_traces import Traces
from srl.useful_classes.angle_between_vectors import AngleBetweenVectors

from srl.environments.ros_behaviour_interface import ROSBehaviourInterface
from srl.useful_classes.ros_environment_goals import EnvironmentInfo
from srl.useful_classes.ros_thruster_wrapper import Thrusters

from srl.learning_algorithms.stober_td_learning import TDLinear
from srl.learning_algorithms.true_online_td_lambda import TrueOnlineTDLambda
from srl.learning_algorithms.lstd_algorithm import LSTD

from utilities.variable_normalizer import DynamicNormalizer
from utilities.moving_differentiator import SlidingWindow

from rospkg import RosPack
rospack = RosPack()

def makeRange(values, num_iterations):
    res = []
    for value in values:
        for _ in range(num_iterations):
            res.append(value)
    return res

actor_config = {
    "approximator_name": "policy",
    "initial_value": 0.0,
    "alpha": 0.005,
    "random_weights": False,
    "minimise": False,
    "approximator_boundaries": [-200.0, 200.0],
    "num_input_dims": 2,
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
    "num_input_dims": 2,
    "rbf_basis_resolution": 20,
    "rbf_basis_scalar": 0.01,
    "number_of_dims_in_state": 1
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    # "basis_functions": BasisFunctions()
}

CONFIG = {
    "test_policy": True,
    "test_vf": False,
    "actor_config": actor_config,
    "critic_config": critic_config,
    "goal_position": [20.0, 0.0],
    "episode_fault_injected": 0,
    "num_episodes_not_learning": 0,
    "log_actions": 1,
    "log_traces": False,
    "spin_rate": 10,
    "num_runs": 5,
    "num_episodes": 100,
    "max_num_steps": 500,
    "policy_type": "linear",
    "actor update rule": "cacla",
    "critic algorithm": "nac",
    "sparse reward": False,
    "gamma": 0.95,   # was 0.1
    "lambda": 0.8,  # was 0.0
    "alpha_decay": 0.0, # was 0.00005
    "exploration_sigma": 0.1,
    "exploration_decay": 1.0,
    "min_trace_value": 0.1
}
results_to_validate = ["/tmp/nessie_no_surge_linear_1dim_true0"]
                       # "/tmp/repeat_nessie_linear_no_surge_two_dim_poly1",
                       # "/tmp/repeat_nessie_linear_no_surge_two_dim_poly2",
                       # "/tmp/repeat_nessie_linear_no_surge_two_dim_poly3"]
if CONFIG["test_policy"]:
    # CONFIG["policy_test_type"] = "vali"
    CONFIG["policy_test_type"] = "nessie"
    CONFIG["policy_test_index"] = 75
    CONFIG["num_runs"] = len(results_to_validate)
    CONFIG["dir_of_policies_to_load"] = results_to_validate[0]
    learning_config = yaml.load(open("{0}/CONFIG.yaml".format(CONFIG["dir_of_policies_to_load"]), "r"))
    CONFIG["num_episodes"] = learning_config["num_episodes"]
    CONFIG["gamma"] = learning_config["gamma"]
    CONFIG["max_num_steps"] = learning_config["max_num_steps"]
    CONFIG["lambda"] = learning_config["lambda"]
    CONFIG["policy_type"] = learning_config["policy_type"]
    CONFIG["critic_config"]["rbf_basis_scalar"] = learning_config["critic_config"]["rbf_basis_scalar"]
    CONFIG["critic_config"]["rbf_basis_resolution"] = learning_config["critic_config"]["rbf_basis_resolution"]
    CONFIG["critic_config"]["number_of_dims_in_state"] = learning_config["critic_config"]["number_of_dims_in_state"]
    # if CONFIG["policy_test_type"] == "nessie":
    #     CONFIG["max_num_steps"] = 1000000

controlled_var = makeRange(range(0, CONFIG["num_episodes"]), 1)

if CONFIG["test_vf"]:
    CONFIG["vf_params_to_load"] = "/tmp/critic_params25.npy"


def polynomialExpansion(state):
    return [state[0], state[0]**2, state[0]**3]
def planarReward(var1, var2):
    # res1 = 1.0 - ((var1 + var2) / 2.0)
    res1 = ((var2 - var1) / 2.0) + 1.0
    # res2 = 1.0 + ((var1 + var2) / 2.0)
    # side1 = 1 - ((var2 - var1) / 8.0)
    if res1 > 1.0:
        res1 = 1.0 - (res1 - 1.0)
    # if var1 < 0 and var2 > 0.0:
    #     return side1
    # else:
    return res1

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


if __name__ == '__main__':
    rospy.init_node("stober_cacla_nessie")
    rospy.set_param("~NumRuns", CONFIG["num_runs"])
    rospy.set_param("~NumEpisodes", CONFIG["num_episodes"])
    rospy.set_param("~NumSteps", CONFIG["max_num_steps"])
    args = sys.argv
    if "-r" in args:
        results_dir_name = args[args.index("-r") + 1]
    else:
        results_dir_name = "cartpole_run"

    # initialise some global variables/objects
    # global normalisers
    position_normaliser = DynamicNormalizer([-1.0, 1.0], [-1.0, 1.0])
    position_deriv_normaliser = DynamicNormalizer([-3.0, 3.0], [-1.0, 1.0])
    distance_normaliser = DynamicNormalizer([0.0, 25.0], [-1.0, 1.0])
    distance_reward_normaliser = DynamicNormalizer([0.0, 15.0], [0.0, 1.0])
    angle_normaliser = DynamicNormalizer([-3.14, 3.14], [-1.0, 1.0])
    angle_deriv_normaliser = DynamicNormalizer([-0.15, 0.15], [-1.0, 1.0])

    angle_between_vectors = AngleBetweenVectors()
    angle_dt_moving_window = SlidingWindow(5)

    thrusters = Thrusters()
    nessie = ROSBehaviourInterface()
    environment_info = EnvironmentInfo()

    # Loop number of runs
    for run in range(CONFIG["num_runs"]):
        if CONFIG["test_policy"]:
            CONFIG["dir_of_policies_to_load"] = results_to_validate[run]
            if CONFIG["policy_test_type"] == "vali":
                results_dir_name = "validate_{0}".format(CONFIG["dir_of_policies_to_load"].split("/")[-1])
            elif CONFIG["policy_test_type"] == "nessie":
                results_dir_name = "real_{0}".format(CONFIG["dir_of_policies_to_load"].split("/")[-1])
        # Create logging directory and files
        results_dir = "/tmp/{0}{1}".format(results_dir_name, run)
        print("results_dir: {0}".format(results_dir))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        filename = os.path.basename(sys.argv[0])
        os.system("cp {0} {1}".format(filename, results_dir))
        os.system("cp /home/gordon/software/simple-rl/srl/basis_functions/simple_basis_functions.py {0}".format(results_dir))
        # os.system("cp /home/gordon/software/simple-rl/srl/environments/cartpole.py {0}".format(results_dir))
        yaml.dump(CONFIG, open("{0}/CONFIG.yaml".format(results_dir), "w"),  allow_unicode=True)

        if CONFIG["test_policy"]:
            os.system("cp {0}/Num* {1}/LearningNumSteps.fso".format(CONFIG["dir_of_policies_to_load"], results_dir))
            os.system("cp {0}/Epi* {1}/LearningEpisodeReturn.fso".format(CONFIG["dir_of_policies_to_load"], results_dir))
            os.system("cp {0}/sto* {1}/LearningMainScript.py".format(CONFIG["dir_of_policies_to_load"], results_dir))
            os.system("cp {0}/sim* {1}/LearningBasisFunctions.py".format(CONFIG["dir_of_policies_to_load"], results_dir))

        f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)
        f_num_steps = open("{0}{1}".format(results_dir, "/NumSteps.fso"), "w", 1)
        # f_timings = open("{0}{1}".format(results_dir, "/AvgStepTime.fso"), "w", 1)

        basis_functions = BasisFunctions(resolution=CONFIG["critic_config"]["rbf_basis_resolution"], scalar=CONFIG["critic_config"]["rbf_basis_scalar"],
                                         num_dims=CONFIG["critic_config"]["number_of_dims_in_state"])

        # initialise policy and value functions
        # policy = PyBrainANNApproximator(actor_config["alpha"])
        if CONFIG["policy_type"] == "linear":
            policy = LinearApprox(actor_config, basis_functions=basis_functions)
        elif CONFIG["policy_type"] == "ann":
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
        elif CONFIG["critic algorithm"] == "ann":
            td_lambda = ANNApproximator(CONFIG["actor_config"]["num_input_dims"],
                                        CONFIG["actor_config"]["num_hidden_units"], hlayer_activation_func="tanh")
            # td_lambda.setParams(list(np.load("/tmp/critic_params0.npy")))
        elif CONFIG["critic algorithm"] == "nac":
            td_lambda = LSTD(lmbda=CONFIG["lambda"], gamma=CONFIG["gamma"])
        # td_lambda = RBFNApprox(3, 50, 1)
        # traces = Traces(CONFIG["lambda"], CONFIG["min_trace_value"])
        # td_lambda = PyBrainANNApproximator(critic_config["alpha"])

        exploration_sigma = deepcopy(CONFIG["exploration_sigma"])
        thrusters.inject_surge_thruster_fault(85, 1)

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
            prev_distance_to_goal = None
            critic_value_func_params = None

            if CONFIG["test_policy"]:
                if CONFIG["policy_test_type"] == "vali":
                    policy_to_load = "{0}/policy_params{1}.npy".format(CONFIG["dir_of_policies_to_load"], controlled_var[episode_number])
                elif CONFIG["policy_test_type"] == "nessie":
                    policy_to_load = "{0}/policy_params{1}.npy".format(CONFIG["dir_of_policies_to_load"],
                                                                       controlled_var[CONFIG["policy_test_index"]])
                print(policy_to_load)
                if policy.name == "LinearApprox":
                    policy.weights = np.load(policy_to_load)
                elif policy.name == "scratch_ann":
                    policy.setParams(list(np.load(policy_to_load)))
                try:
                    os.system("cp {0} {1}/tested_{2}".format(policy_to_load, results_dir, policy_to_load.split("/")[-1]))
                except:
                    os.system("touch {0}/FAILED_POLICY_COPY_{1}".format(results_dir, controlled_var[episode_number]))
            if CONFIG["test_vf"]:
                td_lambda.setParams(list(np.load(CONFIG["vf_params_to_load"])))


            if CONFIG["policy_type"] == "synth" and episode_number > 0:
                policy.updateOffset(0.2)

            if td_lambda.name == "true online" or td_lambda.name == "trad online":
                td_lambda.episodeReset()
            if td_lambda.name == "nac":
                f_gradient_vec_diff = open("{0}{1}".format(results_dir, "/GradientVecDiff{0}.fso".format(episode_number)), "w", 1)
                if episode_number > 0:
                    td_lambda.decayStatistics(decay=0.0)

            # traces.reset()
            nessie.disable_behaviours(True)
            nessie.nav_reset()
            random_yaw = float(randrange(-314, 314, 1)) / 100.0
            possible_starting_states = [-1.57, 1.57]
            # random_yaw = possible_starting_states[episode_number % 2]
            print("YAWING to: {0}".format(random_yaw))
            nessie.pilot([0, 0, 0, 0, 0, 1.57])
            rospy.sleep(2.0)
            nessie.disable_behaviours(False)

            if episode_number % CONFIG["log_actions"] == 0:
                nessie.logNav(log_nav=True, dir_path=results_dir, file_name_descriptor=str(episode_number))
            if episode_number > 5 and exploration_sigma > 0.1:
                exploration_sigma *= CONFIG["exploration_decay"]


            # Loop number of steps
            for step_number in range(CONFIG["max_num_steps"]):
                print("------ STEP {0} -------".format(step_number))
                # --------- STATE AND TRACES ---------
                # get current state
                step_start_time = time.time()
                print("prev distance 1: {0}".format(prev_distance_to_goal))
                distance_t = environment_info.distance_to_goal
                if prev_distance_to_goal is None:
                    distance_dt = 0.0   #distance_dt = distance_t - prev_distance_to_goal

                state_t = {"angle": angle_normaliser.scale_value(deepcopy(environment_info.raw_angle_to_goal)),
                           "distance": distance_normaliser.scale_value(distance_t),
                           "yaw_deriv": angle_deriv_normaliser.scale_value(angle_dt),
                           "reward": np.sign(environment_info.raw_angle_to_goal) * (abs(angle_normaliser.scale_value(deepcopy(environment_info.raw_angle_to_goal)) * (deepcopy(nessie._nav.orientation_rate.yaw) + 2.0)) / 3.0),
                           # "reward": planarReward(angle_normaliser.scale_value(deepcopy(environment_info.raw_angle_to_goal)), deepcopy(nessie._nav.orientation_rate.yaw)),
                           # "distance_dt": distance_dt,
                           "yaw_dt": deepcopy(nessie._nav.orientation_rate.yaw)}
                state_t_raw = {"angle": environment_info.raw_angle_to_goal,
                            "angle_deriv": angle_dt}
                           # "angle_deriv": angle_deriv_normaliser.scale_value(angle_dt),
                           # "yaw_dt": deepcopy(nessie._nav.orientation_rate.yaw)}
                           # "distance": distance_t}
                           # "distance": distance_normaliser.scale_value(deepcopy(environment_info.distance_to_goal))}
                # traces.updateTrace(state_t.values(), 1.0)

                # td_lambda.updateTrace(state_t.values(), 1.0)
                if td_lambda.name == "true online":
                    if td_lambda.stateprime is None:
                        td_lambda.start(state_t.values())
                # if td_lambda.name == "nac" and not td_lambda.initialised:
                #     td_lambda.initLSTD(basis_functions.computeFeatures(state_t))

                print("State: {0}".format(state_t))
                print("State: {0}".format(state_t_raw))

                # --------- GET ACTION AND PERFORM IT ---------
                # Compute the deterministic (??? deterministic if its an approximation ???) action
                state_t_sub_features = basis_functions.computeFeatures(state_t)
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

                if step_number % (3 * CONFIG["spin_rate"]) == 0:
                    exploration_mean = 0.0#np.clip(state_t["angle"], -0.1, 0.1)
                    exploration = np.random.normal(exploration_mean, exploration_sigma)
                if CONFIG["test_policy"] or CONFIG["policy_type"] == "synth":
                    exploration = 0.0
                # print("ANN param size {0}".format(policy.num_params))
                print("Distance to goal: {0}".format(environment_info.distance_to_goal))
                print("Exploration: {0}".format(exploration))
                print("Expl. SIGMA: {0}; Expl. MEAN: {1}".format(exploration_sigma, exploration_mean))

                # if step_number % 2 * CONFIG["spin_rate"] == 0.0:
                action_t = float(action_t_deterministic) + exploration
                print("Deterministic Action: {0}".format(action_t_deterministic))
                print("Stochastic Action: {0}".format(action_t))
                if prev_det_action is None:
                    prev_det_action = deepcopy(action_t_deterministic)

                print("prev distance 2: {0}".format(prev_distance_to_goal))


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
                nessie.performAction("gaussian_variance", action_t)

                print("Sleeping for a bit ...")
                time.sleep(1.0 / CONFIG["spin_rate"])

                # --------- GET NEW STATE ---------
                # observe new state --- which is dependant on whether it is the final goal state or not
                angle_change = state_t_raw["angle"] - environment_info.raw_angle_to_goal
                tmp_angle_change = sum(angle_dt_moving_window.getWindow(angle_change)) / 5.0
                # print(tmp_angle_change)
                # print(type(tmp_angle_change))
                distance_t_plus_1 = deepcopy(environment_info.distance_to_goal)
                distance_dt = distance_t_plus_1 - distance_t
                state_t_plus_1 = {"angle": angle_normaliser.scale_value(deepcopy(environment_info.raw_angle_to_goal)),
                           "distance": distance_normaliser.scale_value(distance_t),
                           "yaw_deriv": angle_deriv_normaliser.scale_value(tmp_angle_change),
                           "reward": np.sign(environment_info.raw_angle_to_goal) * (abs(angle_normaliser.scale_value(deepcopy(environment_info.raw_angle_to_goal)) * (deepcopy(nessie._nav.orientation_rate.yaw) + 2.0)) / 3.0),
                           # "reward": planarReward(angle_normaliser.scale_value(deepcopy(environment_info.raw_angle_to_goal)), deepcopy(nessie._nav.orientation_rate.yaw)),
                           # "distance_dt": distance_dt,
                           "yaw_dt": deepcopy(nessie._nav.orientation_rate.yaw)}
                state_t_plus_1_raw = {"angle": environment_info.raw_angle_to_goal,
                           "angle_deriv": angle_change}
                           # "distance": distance_t_plus_1}
                           # "distance": distance_normaliser.scale_value(deepcopy(environment_info.distance_to_goal))}
                angle_dt = deepcopy(tmp_angle_change)
                state_t_plus_1_sub_features = basis_functions.computeFeatures(state_t_plus_1)

                print("DISTANCE T1: {0}".format(distance_t_plus_1))
                prev_distance_to_goal = deepcopy(distance_t_plus_1)

                print("State + 1: {0}".format(state_t_plus_1))
                print("State + 1: {0}".format(state_t_plus_1_raw))

                failEpisode = False
                # if abs(state_t_plus_1_raw["angle"]) > 2.5:
                #     reward = 0.0
                # else:
                # if distance_t_plus_1 < 1.0 or step_number == CONFIG["max_num_steps"] - 1:
                #     reward = 10.0 - (10 * abs(state_t_plus_1["angle"])) + (10*(CONFIG["max_num_steps"] - step_number))
                #     # - (5 * distance_normaliser.scale_value(distance_t_plus_1))
                # else:
                #     reward = 10.0 - (10 * abs(state_t_plus_1["angle"]))

                # if distance_t_plus_1 < 2.0:
                #     reward = 200.0 - (200.0 * abs(state_t_plus_1["angle"]))
                # else:
                #     reward = 10.0 - (6.0 * abs(state_t_plus_1["angle"])) - (4.0 * distance_reward_normaliser.scale_value(distance_t_plus_1))

                # if distance_t_plus_1 < 2.0:
                #     reward = 0.0#-2.0 * abs(state_t_plus_1["angle"])
                # else:
                angular_vel_based_reward = planarReward(state_t_plus_1["angle"],
                                                        angle_deriv_normaliser.scale_value(nessie._nav.orientation_rate.yaw))

                reward = -10.0 + (10.0 * (1.0 - abs(state_t_plus_1["angle"]))) + \
                         (0.0 * (1.0 - angle_deriv_normaliser.scale_value(nessie._nav.orientation_rate.yaw))) + \
                         (0.0 * angular_vel_based_reward) + \
                         (0.0 * (1.0 - distance_reward_normaliser.scale_value(distance_t_plus_1)))
                print("Norm. Yaw Rate: {0}".format(angle_deriv_normaliser.scale_value(nessie._nav.orientation_rate.yaw)))
                print("REWARD ANGULAR: {0}".format(angular_vel_based_reward))

                # reward = -10 + (1.0 - abs(state_t_plus_1["angle"]))


                # reward = (10 * (1.0 - distance_normaliser.scale_value(distance_t_plus_1))**2) - \
                #                 (0.1 * (action_t_deterministic - prev_det_action)**2)
                # if environment_info.distance_to_goal < 1.0:# or step_number == CONFIG["max_num_steps"] - 1:
                #     reward = 0.0
                # # elif step_number == CONFIG["max_num_steps"] - 1:
                # #     reward = -10.0
                # elif failEpisode:
                #     reward = -10.0
                # else:
                #     reward = - (abs(angle_normaliser.scale_value(environment_info.raw_angle_to_goal))) - (action_t_deterministic - prev_det_action)**2
                                # distance_normaliser.scale_value(environment_info.distance_to_goal))

                    # reward = - (1.0 - gaussian1D(environment_info.raw_angle_to_goal, 0.0, 0.8, 0.8))
                                # gaussian1D(environment_info.distance_to_goal, 1.0, 0.3, 5.0)) - state_t_plus_1["yaw_deriv"]**2



                print("REWARD: {0}".format(reward))
                # f_rewards.write("{0}\t{1}\n".format(episode_number, reward))

                # state_t_sub_features = basis_functions.computeFeatures(state_t.values(), goalState=False)
                # if environment_info.distance_to_goal < 1.0:# or step_number == CONFIG["max_num_steps"] - 1:
                #     state_t_plus_1_sub_features = np.zeros(state_t_sub_features.shape)
                # else:
                #     state_t_plus_1_sub_features = basis_functions.computeFeatures(state_t_plus_1.values(), goalState=False)


                # Calculate the state values (before updating the critic)
                # state_t_value = td_lambda.getStateValue(state_t_sub_features)
                # state_t_plus_1_value = td_lambda.getStateValue(state_t_plus_1_sub_features)
                if td_lambda.name == "true online":
                    print("True Online")
                    state_t_value = td_lambda.getStateValue(state_t)
                    state_t_plus_1_value = td_lambda.getStateValue(state_t_plus_1)
                elif td_lambda.name == "trad online":
                    print("Trad Online")
                    state_t_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t))
                    state_t_plus_1_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t_plus_1))
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

                # Update the critic
                # terminalState = False#cartpole_environment.episodeEnded()
                # td_error = td_lambda.computeTDError(reward, CONFIG["gamma"], state_t_plus_1_value, state_t_value, terminalState)
                # if not CONFIG["test_vf"]:
                if td_lambda.name == "true online":
                    if not environment_info.distance_to_goal < 2.0:
                        td_error = td_lambda.step(reward, state_t_plus_1)
                    else:
                        td_error = td_lambda.end(reward)
                    new_state_t_plus_1_value = td_lambda.getStateValue(state_t_plus_1)
                elif td_lambda.name == "trad online":
                    # if not environment_info.distance_to_goal < 2.0:
                    td_error = td_lambda.train(basis_functions.computeFeatures(state_t), reward,
                                               basis_functions.computeFeatures(state_t_plus_1))
                    # else:
                    #     print("MWHAHAHAHAHAHAH")
                    #     td_error = td_lambda.end(basis_functions.computeFeatures(state_t), reward)
                    #     print("AHAHAHAHAHAHAH")
                    new_state_t_plus_1_value = td_lambda.getStateValue(basis_functions.computeFeatures(state_t_plus_1))
                elif td_lambda.name == "scratch_ann":
                    # For ANN critic
                    td_error = reward + (CONFIG["gamma"] * state_t_plus_1_value) - state_t_value
                    prev_critic_weights = td_lambda.getParams()
                    critic_gradient = td_lambda.calculateGradient(state_t_raw.values())
                    td_lambda.setParams(prev_critic_weights + CONFIG["critic_config"]["alpha"] * td_error * critic_gradient)
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
                    # angle_between_gradient_vectors = angle_between_vectors.angle_between(critic_gradient, prev_critic_gradient)
                    angle_between_gradient_vectors = sum(critic_gradient - prev_critic_gradient)
                    print("Angle between gradient vectors: {0}".format(angle_between_gradient_vectors))
                    f_gradient_vec_diff.write("{0}\t{1}\n".format(episode_number, angle_between_gradient_vectors))

                    print("GRADIENT (CRITIC): {0}".format(critic_gradient))
                # else:
                #     td_error = reward + CONFIG["gamma"] * state_t_plus_1_value - state_t_value

                # if cartpole_environment.episodeEnded():
                #     state_t_plus_1_value = 0.0

                # td_error = td_lambda.computeTDError(reward, CONFIG["gamma"], state_t_plus_1_value, state_t_value, False)
                # For Scratch ANN critic


                # For TDLinear
                # td_error = td_lambda.train(state_t_sub_features, reward, state_t_plus_1_sub_features)

                # td_error = reward + CONFIG["gamma"] * state_t_plus_1_value - state_t_value
                # critic_value_func_params = td_lambda.getParams()
                # _states, _traces = traces.getTraces()
                # for _s, _t in zip(_states, _traces):
                # critic_gradient = td_lambda.calculateGradient(state_t.values())
                # critic_value_func_params += critic_config["alpha"] * td_error * critic_gradient
                # td_lambda.setParams(critic_value_func_params)

                print("State t Value: {0}".format(state_t_value))
                print("State t+1 Value: {0}".format(state_t_plus_1_value))
                # print("New State t+1 Value: {0}".format(new_state_t_plus_1_value))
                print("TD ERROR: {0}".format(td_error))
                # print("Number of Traces: {0}".format(len(traces._values)))
                # X, T = traces.getTraces()
                # p = td_lambda.getParams()
                # for x, trace in zip(X, T):
                #     p += critic_config["alpha"] * td_error * (td_lambda.calculateGradient(x) * trace)
                # td_lambda.setParams(p)

                # For scratch ANN
                # critic_params = td_lambda.getParams() + critic_config["alpha"] * td_error * td_lambda.calculateGradient()
                # td_lambda.setParams(critic_params)

                # For Traditional TD(lambda) critic
                # td_lambda.updateWeights(td_error, basis_functions, use_traces=True, terminalState=False)


                # for ANN approximator
                # td_lambda.updateWeights(reward + CONFIG["gamma"] * state_t_plus_1_value, state_t.values())
                # else:
                #     state_t_sub_features = basis_functions.computeFeatures(state_t.values(), goalState=True)
                #     state_t_plus_1_sub_features = basis_functions.computeFeatures(state_t_plus_1.values(), goalState=True)
                params = policy.getParams()

                ACTOR_UPDATE_CONDITION = False
                if CONFIG["actor update rule"] == "cacla":
                    if td_error > 0.0:
                        ACTOR_UPDATE_CONDITION = True
                    else:
                        ACTOR_UPDATE_CONDITION = False
                elif CONFIG["actor update rule"] == "td lambda":
                    ACTOR_UPDATE_CONDITION = True

                # if state_t_plus_1_value > state_t_value:
                if td_lambda.name == "nac" and not CONFIG["test_policy"]:
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

                elif ACTOR_UPDATE_CONDITION and not CONFIG["test_policy"] and CONFIG["policy_type"] != "synth":# and episode_number > 0:
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
                        policy_gradient = policy.calculateGradient()
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
                            policy.setParams(params + actor_config["alpha"] * (policy_gradient * exploration))
                        else:
                            policy.setParams(params - actor_config["alpha"] * (policy_gradient * td_error))
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

                if td_lambda.name == "nac":
                    prev_critic_gradient = deepcopy(critic_gradient)
                print("Policy ALPHA: {0}".format(policy.alpha))
                print("GAMMA: {0}".format(CONFIG["gamma"]))
                print("LAMBDA: {0}".format(CONFIG["lambda"]))

                step_time = time.time() - step_start_time
                # accumulate total reward
                cum_reward += reward
                cum_step_time += step_time

                prev_det_action = deepcopy(action_t_deterministic)

                # Check for goal condition
                if environment_info.distance_to_goal < 2.0 or failEpisode:
                    # Zero all LSTD statistics as LSTD is not really an episodic algorithm
                    # if td_lambda.name == "nac":
                    #     td_lambda.decayStatistics(decay=0.0)
                    # print("ZEROES LSTD STATISTICS")
                    break

            if episode_number % CONFIG["log_actions"] == 0:
                np.save("{0}/policy_params{1}".format(results_dir, episode_number), policy.getParams())
                if td_lambda.name != "nac":
                    np.save("{0}/critic_params{1}".format(results_dir, episode_number), td_lambda.getParams())
                if CONFIG["log_traces"]:
                    if td_lambda.name == "trad online":
                        np.save("{0}/critic_trace{1}".format(results_dir, episode_number), td_lambda.getTraces())
                    elif td_lambda.name == "true online":
                        np.save("{0}/critic_trace{1}".format(results_dir, episode_number), td_lambda.traces)

            if nessie.ManualEndRun:
                # Break out of the Episode loop to get to the next run. Remember to reset ManualEndRun so as not
                # to just skip through all runs
                nessie.ManualEndRun = False
                break

            # if cum_reward < 50.0:
            #     actor_config["alpha"] = 0.001

            if episode_number % CONFIG["log_actions"] == 0:
                nessie.logNav(log_nav=False, dir_path=results_dir, file_name_descriptor=str(episode_number))

            # Log cumulative reward for episode and number of steps
            f_returns.write("{0}\t{1}\n".format(episode_number, cum_reward))
            f_num_steps.write("{0}\t{1}\n".format(episode_number, step_number))
            # f_timings.write("{0}\t{1}\n".format(episode_number, cum_step_time / step_number))
            nessie.disable_behaviours(True)
