#! /usr/bin/python
import roslib; roslib.load_manifest("rl_pybrain")
import rospy
from scipy import random
from copy import deepcopy
import numpy as np
import sys
import time
import os
sys.path.append("/home/gordon/software/mmlf-1.0")
from mmlf.resources.function_approximators.Fourier import Fourier
from pyrl.basis.tilecode import TileCodingBasis
from scipy import ones, dot, ravel
from scipy.linalg import pinv
from matplotlib.mlab import PCA
sin = np.sin
cos = np.cos

# from mmlf.resources.function_approximators.BasisFunction import BasisFunction
from utilities.nav_class import Nav
from utilities.cartesian_distance_error import CartesianDistanceError
from utilities.variable_normalizer import DynamicNormalizer
from utilities.gaussian import gaussian1D, gaussian2D
from utilities.body_velocity_calculator import calculate_body_velocity
from srv_msgs.srv import DisableOutput, LogNav, ChangeParam, DoHover
from my_msgs.msg import Goals
from vehicle_interface.msg import Vector6
from std_srvs.srv import Empty, EmptyResponse
from vehicle_interface.srv import DictionaryService
from diagnostic_msgs.msg import KeyValue
from rospkg import RosPack
rospack = RosPack()

# import matplotlib.pyplot as plt
# plt.ion()
# fig = plt.figure(1)
# fig, ((ax_sv_features, ax_sv_weights), (ax_adv_features, ax_adv_weights), (ax_pol_features, ax_pol_weights)) = \
#     plt.subplots(nrows=3, ncols=2)
# ax_adv_features.set_title("adv_features")
# ax_adv_weights.set_title("adv_weights")
# ax_sv_features.set_title("sv_features")
# ax_sv_weights.set_title("sv_weights")
# ax_pol_features.set_title("pol_features")
# ax_pol_weights.set_title("pol_weights")

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

class Thrusters(object):
    def __init__(self):
        self.srv_fault = rospy.ServiceProxy('/thrusters/faults', DictionaryService)

    def inject_surge_thruster_fault(self, new_surge_health, index_of_fault):
        fault_type = KeyValue(key='fault_type',value='hard_limit')
        print("Injecting Fault of {0} into thruster {1}".format(new_surge_health, index_of_fault))
        if index_of_fault == 0:
            th_min = KeyValue(key='th_min', value='-{0}, -85, -85, -85, -85, -85'.format(str(new_surge_health)))
            th_max = KeyValue(key='th_max', value='{0}, 85, 85, 85, 85, 85'.format(str(new_surge_health)))
        elif index_of_fault == 1:
            th_min = KeyValue(key='th_min', value='-85, -{0}, -85, -85, -85, -85'.format(str(new_surge_health)))
            th_max = KeyValue(key='th_max', value='85, {0}, 85, 85, 85, 85'.format(str(new_surge_health)))

        thruster_fault_response = self.srv_fault(request=[fault_type, th_min, th_max])

        last_thruster_failure_index = index_of_fault
        last_thruster_health = new_surge_health
        print("Thruster Health changed to:\n{0}".format(thruster_fault_response))
        print(thruster_fault_response.response[-1].value)
class EnvironmentInfo(object):
    def __init__(self):
        ros_env_sub = rospy.Subscriber('/goal_points_data', Goals, self.rosEnvCallback)
        self.raw_angle_to_goal = 0.0
        self.distance_to_goal = 0.0

    def rosEnvCallback(self, msg):
        """
        :param msg: Goals msg which contains the environment data such as angle to goal wrt. to vehicle heading
        """
        for point in msg.goals:
            # Get the angle angle to the point-of-interest; 1 = waypoint_goal, 2 = inspection_goal, 3 = fear_goal
            if point.point_type == 1.0:
                self.raw_angle_to_goal = point.angle_wrt_vehicle
                self.distance_to_goal = point.distance

class AngleBetweenVectors(object):
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.dot(v1_u, v2_u))
        if np.isnan(angle):
            if (v1_u == v2_u).all():
                return 0.0
            else:
                return np.pi
        return angle

# class RBFBasisState(object):
#     def __init__(self, idx=0):
#         self.resolution = 20
#         self.centres = np.linspace(0.0, 1.0, self.resolution) #[0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 1.0]
#         self.sigma = 1.0 / np.sqrt(2*self.resolution)    #0.2
#     def computeFeatures(self, state, goalState=False):
#         # state_enhanced = [state[0], state[1], state[2], state[1] * state[2]]
#         # state_reduced = [state[1], state[2]]
#         x1 = state[0]
#         x2 = state[1]
#         # x3 = state[2]
#         poly_state = np.array([x1**2, x1*x2, x2**2, 0.1])
#
#         rbfs = [[gaussian1D(s_dim, centre, 1.0, self.sigma) for centre in self.centres] for s_dim in poly_state]
#         rbfs = np.array(rbfs)
#
#         # Make sure we do not generalise across a state dimensions range that we shouldn't
#         # e.g. the best action is of opposite direction for -0.1 than 0.1 so the rbfs shouldn't generalise across
#         # the mid point of the state dimensions range
#         # half_way = self.resolution / 2
#         # if state[0] >= 0.5: # signifies change in direction of angle to goal - should not generalise across this line
#         #    rbfs[0][:half_way] = 0.0
#         # else:
#         #    rbfs[0][half_way:] = 0.0
#         # if state[0] >= 0.5: # signifies Zero in East direction - should not generalise across this line
#         #     rbfs[0][:half_way] = 0.0
#         # else:
#         #     rbfs[0][half_way:] = 0.0
#         # if state[1] >= 0.5: # signifies Zero in East direction - should not generalise across this line
#         #     rbfs[1][:half_way] = 0.0
#         # else:
#         #     rbfs[1][half_way:] = 0.0
#         return rbfs.flatten()

class RBFBasisState(object):
    def __init__(self, idx=0):
        self.num_features = [256, 128, 128]
        self.num_tilings = [80, 24, 24]
        self.feature_ranges = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        self.tile_coding = TileCodingBasis(len(self.feature_ranges), self.feature_ranges, self.num_tilings[idx], self.num_features[idx])
        self.resolution = self.num_features
    def computeFeatures(self, state, goalState=False):
        if not goalState:
            # TODO: Polynomial basis first??
            x1 = state[0]
            x2 = state[1]
            # x3 = state[2]
            poly_state = np.array([x1**2, x1**3, x1*x2, x2**2, x2**3, 0.1])
            # poly_state = np.array([x1**2, x1**3, x1*x2, x1*x3, x2**2, x2**3, x2*x3, x3**2, x3**3, 0.1])
            # return self.tile_coding.computeFeatures(poly_state)
            return poly_state
        else:
            return np.array([0.0 for i in range(self.num_features)])

# class RBFBasisState(object):
#     def __init__(self):
#         self.resolution = 10
#         self.centres = np.linspace(0.0, 1.0, self.resolution) #[0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 1.0]
#         self.sigma = 0.2
#     def computeFeatures(self, state, additional_features=None):
#         # state_enhanced = [state[0], state[1], state[2], state[1] * state[2]]
#         # state_reduced = [state[1], state[2]]
#         state_raw = [state[0], state[0]*state[1], state[1], state[0]**2, state[1]**2, state[2], state[2]**2, 1.0]
#         # if additional_features is None:
#         #     return np.r_[np.array(state_raw), np.array([0.0 for i in range(len(state_raw))])]
#         # else:
#         #     return np.r_[np.array(state_raw), additional_features]
#         return np.array(state_raw)


def minmax(item, limit1, limit2):
        "Bounds item to between limit1 and limit2 (or -limit1)"
        return max(limit1, min(limit2, item))

class LSTD(object):
    def __init__(self):
        self.weights = None
        self.A = None
        self.b = None
        self.traces_across_features = None
        self.lamb = CONFIG["lambda"]
        self.decay_rate = CONFIG["lstd_decay_rate"]
        self.initialised = False

    def initLSTD(self, phi_t):
        """
        :param phi_t: feature vector at time t
        :return: initialises the LSTD algorithm
        """
        feature_vector_length = max(phi_t.shape)
        self.A = np.zeros([feature_vector_length, feature_vector_length])
        self.b = np.zeros(phi_t.shape)
        self.traces_across_features = phi_t
        self.initialised = True

    def updateA(self, features_t, features_t_plus_1):
        # print("----updateA----")
        # outer product used in order to create NxN matrix
        A_update = np.dot(self.traces_across_features, (features_t - features_t_plus_1).transpose())
        print("A_update shape: {0}".format(A_update.shape))
        self.A = self.A + A_update

    def update_b(self, reward):
        # print("b: {0}".format(self.b))
        reward_across_features = self.traces_across_features * reward
        self.b += reward_across_features

    def updateTraces(self, features_t_plus_1):
        # print("updateTraces: {0}".format(self.traces_across_features))
        if not self.initialised:
            self.initLSTD(features_t_plus_1)
        self.traces_across_features *= self.lamb
        self.traces_across_features += features_t_plus_1

    def decayStatistics(self, decay=None):
        if decay is None:
            self.traces_across_features *= self.decay_rate
            self.A *= self.decay_rate
            self.b *= self.decay_rate
        else:
            self.traces_across_features *= decay
            self.A *= decay
            self.b *= decay

    def calculateBeta(self):
        # print("A list: {0}".format(self.A.tolist()))
        # print("self.b: {0}".format(self.b.tolist()))
        print("DET A: {0}".format(np.linalg.det(self.A)))
        # U, X, V = np.linalg.svd(self.A)
        # print("U: {0}".format(U.shape))
        # print("X: {0}".format(X.shape))
        # print("V: {0}".format(V.shape))
        # beta = np.dot(U, self.b.transpose()).transpose()    # did converge to a semi decent policy using the eigenvector as A_inverse
        # inv_diag = np.diag(1.0 / X)
        # print("invdiag: {0}".format(inv_diag))
        # A_inverse = np.dot(V, inv_diag).dot(U.transpose())
        A_inverse = np.linalg.pinv(self.A)
        # A_inverse = np.linalg.pinv(self.A)
        beta = np.dot(A_inverse, self.b)
        print("BETA: {0}".format(beta.shape))
        return beta.transpose()[0,:], beta.transpose()[1,:]
        # return np.diag(np.dot(U, V))[0]

class LinearApprox(object):
    def __init__(self, config):
        self.name = config["approximator_name"]
        self.alpha = config["alpha"]
        self._minimize = config["minimise"]
        self.compute_features = config["basis_functions"].computeFeatures
        self.num_features_per_dim = config["basis_functions"].resolution
        self.num_input_dims = config["num_input_dims"]
        self.initial_value = config["initial_value"]
        self.num_features = len(self.compute_features([0.5 for i in range(self.num_input_dims)]))
        self.lower_bound, self.upper_bound = config["approximator_boundaries"]
        self.last_features = None
        self.weights = None
        self.drawn_stuff_to_remove = []
        self.plotFeatures = False

    def polynomial_rbf_features(self, state):
        state = [1.0, state[0]**2, state[1]**2, state[0]*state[1], (state[0]*state[1])**2]
        return self.compute_features(state)

    def setWeights(self, weights_list):
        self.weights[:] = weights_list
        print("weights set to: {0}".format(weights_list))

    def decayAlpha(self, decay_by_amount):
        if self.alpha - decay_by_amount > 0.0:
            self.alpha -= decay_by_amount

    def computeOutput(self, state_features):
        # If the number of state dimensions given is less than the configuration dict says there should be, add some
        # some zeroes. This padding is to allow the same state and weight vector dimensions for state value and action
        # if len(state_raw) < self.num_input_dims:
        #     state_raw.extend([0.0 for i in range(self.num_input_dims - len(state_raw))])
        # Extend state_raw again if there are additional features to be used (e.g. for the advantage fnct approximation
        # global ax_sv_features, ax_sv_weights, ax_adv_features, ax_adv_weights
        # if self.drawn_stuff_to_remove is not None:
        #     if len(self.drawn_stuff_to_remove) == 1:
        #         for item in self.drawn_stuff_to_remove:
        #             item.remove()
        #         self.drawn_stuff_to_remove = []
        # drawn_feature, = ax_adv_features.plot(state_features, "k"); self.drawn_stuff_to_remove.append(drawn_feature)
        # plt.draw()

        # Initialise the weights of the approximator now that we know the size of the feature vector
        if self.weights is None:
            # _features = self.compute_features([0.5 for i in range(self.num_input_dims)])
            avg = self.initial_value / sum(state_features)
            print("avg for weights: {0}".format(avg))
            self.weights = np.array([avg for i in range(state_features.shape[0])])

        # print("{0}->feature size: {1}".format(self.name, state_features.shape))
        # print("{0}->weights size: {1}".format(self.name, self.weights.shape))

        # print("size of state_features: {0}".format(len(state_features)))
        # print("size of weights: {0}".format(len(self.weights)))
        self.last_features = deepcopy(state_features)
        return np.dot(state_features, self.weights)

    def updateWeights(self, gradient_vector=None):
        # Bound the TD Error and reverse it. Why it needs to be inverted, who knows!

        # If the number of state dimensions given is less than the configuration dict says there should be, add some
        # some zeroes. This padding is to allow the same state and weight vector dimensions for state value and action
        # if len(state_raw) < self.num_input_dims:
        #     state_raw.extend([0.0 for i in range(self.num_input_dims - len(state_raw))])

        print("Updating ACTOR --------------Mwhahahahahah")
        feature_vector = gradient_vector

        # if self.plotFeatures:
        #     global ax_pol_features, ax_pol_weights
        #     if self.drawn_stuff_to_remove is not None:
        #         for item in self.drawn_stuff_to_remove:
        #             item.remove()
        #         self.drawn_stuff_to_remove = []
        #     drawn_feature, = ax_pol_features.plot(gradient_vector, "k"); self.drawn_stuff_to_remove.append(drawn_feature)
        #     drawn_weights, = ax_pol_weights.plot(self.weights, "k"); self.drawn_stuff_to_remove.append(drawn_weights)
        #     plt.draw()

        # print("updateWeights-> shape of weights: {0}".format(self.weights.shape))
        # print("updateWeights-> shape of feature: {0}".format(feature_vector.shape))
        # check that the update requested does not put the Approximators output outside it's bounds
        old_weights = deepcopy(self.weights)

        if self._minimize:
            tmp_updated_weights = old_weights - self.alpha * feature_vector
        else:
            tmp_updated_weights = old_weights + self.alpha * feature_vector

        output_tmp = np.dot(tmp_updated_weights, self.last_features)
        output = np.dot(self.weights, self.last_features)
        # print("updateParameters: {0} before tmp update: {1}".format(self.name, output))
        # print("updateParameters: {0} after tmp update: {1}".format(self.name, output_tmp))

        # When using the natural gradient vector, you don't use the TD Error in the weight update
        if (output_tmp > output and output_tmp < self.upper_bound) or (output_tmp < output and output_tmp > self.lower_bound):
            # print("updating weights IDEAL")
            if self._minimize:
                self.weights -= self.alpha * feature_vector
            else:
                self.weights += self.alpha * feature_vector
        elif (output_tmp < self.lower_bound and output_tmp > output) or (output_tmp > self.upper_bound and output_tmp < output):
            if self._minimize:
                self.weights -= self.alpha * feature_vector
            else:
                self.weights += self.alpha * feature_vector
                # print("updating weights TOWARDS RANGE")
        else:
            # print("NOT UPDATING as out of output range")
            pass

actor_config = {
    "approximator_name": "policy",
    "initial_value": -0.2,
    "alpha": 0.00006,
    "random_weights": False,
    "minimise": False,
    "approximator_boundaries": [-200.0, 200.0],
    "num_input_dims": 2,
    # "basis_functions": Fourier(FourierDomain(), order=5)
    # "basis_functions": TileCodingBasis(2, [[-1.2, 0.6], [-0.07, 0.07]], 64, 128)
    "basis_functions": RBFBasisState()
}

CONFIG = {
    "goal_position": [10.0, 0.0],
    "episode_fault_injected": 1000,
    "num_episodes_not_learning": 0,
    "spin_rate": 5,
    "num_runs": 1,
    "num_episodes": 300,
    "max_num_steps": 300,
    "log_nav_every": 1,
    "gamma": 0.6,   # was 0.1
    "lambda": 0.95,  # was 0.0
    "lstd_decay_rate": 0.9,     # was 0.3
    "epsilon": 1.0,
    "alpha_decay": 0.0, # was 0.00005
    "exploration_sigma": 0.2,
    "min_trace_value": 0.01
}

# This Service is global so that it can be called upon rospy shutdown within the on_rospyShutdown function
# otherwise, if this node is terminated during an episode, the nav will be logged to csv until this node is restarted
logNav = rospy.ServiceProxy("/nav/log_nav", LogNav)

def on_rospyShutdown():
    global episode_number, results_dir
    logNav(log_nav=False, dir_path=results_dir, file_name_descriptor=str(episode_number))

if __name__ == '__main__':
    # agent = ActorCriticAgent(actor_config, critic_config)
    # approximator = LinearApprox(actor_config)
    args = sys.argv
    if "-r" in args:
        results_dir_name = args[args.index("-r") + 1]
    else:
        results_dir_name = "nac_agent"
    rospy.init_node("natural_actor_critic")

    navigation = Nav()
    environmental_data = EnvironmentInfo()

    angle_between_vectors = AngleBetweenVectors()

    # Services
    disable_behaviours = rospy.ServiceProxy('/behaviours/coordinator/disable_output', DisableOutput)
    pilot = rospy.ServiceProxy('/do_hover', DoHover)
    yaw_ros_action = rospy.ServiceProxy('/behaviours/yaw/params', ChangeParam)
    nav_reset = rospy.ServiceProxy("/nav/reset", Empty)

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
        f_returns = open("{0}{1}".format(results_dir, "/EpisodeReturn.fso"), "w", 1)
        f_timings = open("{0}{1}".format(results_dir, "/AvgStepTime.fso"), "w", 1)

        # initialise policy and value functions
        policy = LinearApprox(actor_config)
        rbf_state = RBFBasisState(idx=run)

        lstd = LSTD()

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

            if episode_number == CONFIG["episode_fault_injected"]:
                thrusters.inject_surge_thruster_fault(17, 1)

            # Reset the last action value
            # last_yaw_mean = yaw_ros_action("gaussian_variance", 0.7, False, 1.0)

            # Enable behaviours
            disable_behaviours(disable=True)
            nav_reset()
            random_yaw = 1.57 + 0.785#float(random.randrange(-314, 314, 5)) / 100
            print("Moving to Random Starting Yaw: {0}".format(random_yaw))
            hover_completed = pilot(Vector6(values=[0, 0, 0, 0, 0,random_yaw]))
            rospy.sleep(5)
            disable_behaviours(disable=False)
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
                           # "angle_dt": deepcopy(angle_dt),
                           # "combi": angles_normaliser.scale_value(deepcopy(distance_normaliser.scale_value(environmental_data.distance_to_goal)
                           #                             * abs(environmental_data.raw_angle_to_goal)))}
                           "yaw_velocity": deepcopy(navigation._nav.orientation_rate.yaw)}
                # if not lstd.initialised:
                #     # Init the statistics required for the LSTD algorithm
                #     lstd.initLSTD(rbf_state.computeFeatures(state_t.values(), additional_features=None))
                print("State: {0}".format(state_t))
                print("Angle to Goal: {0}".format(environmental_data.raw_angle_to_goal))

                # --------- GET ACTION AND PERFORM IT ---------
                # Compute the deterministic (??? deterministic if its an approximation ???) action
                state_t_sub_features = rbf_state.computeFeatures(state_t.values())
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
                    last_yaw_mean = yaw_ros_action("gaussian_variance", action_t, False, 1.0)
                else:
                    last_yaw_mean = yaw_ros_action("gaussian_variance", 0.1, False, 1.0)

                print("Sleeping for a bit ...")
                rate.sleep()

                # --------- GET NEW STATE ---------
                # observe new state --- which is dependant on whether it is the final goal state or not

                state_t_plus_1 = {"angle": deepcopy(environmental_data.raw_angle_to_goal),
                               #     "distance": gaussian1D(surge_velocity_normaliser.scale_value(
                               # navigation._nav.body_velocity.x), 0.7, 0.5, 0.2) +
                               #                  gaussian1D(yaw_velocity_normaliser.scale_value(
                               #                      navigation._nav.orientation_rate.yaw), 0.5, 0.5, 0.3),
                               #     "angle_dt": deepcopy(angle_dt),
                                # "combi": angles_normaliser.scale_value(deepcopy(distance_normaliser.scale_value(environmental_data.distance_to_goal)
                                #                        * abs(environmental_data.raw_angle_to_goal)))}
                                   "yaw_velocity": deepcopy(navigation._nav.orientation_rate.yaw)}

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
                state_t_sub_features = rbf_state.computeFeatures(state_t.values(), goalState=False)
                state_t_plus_1_sub_features = rbf_state.computeFeatures(state_t_plus_1.values(), goalState=False)
                # else:
                #     state_t_sub_features = rbf_state.computeFeatures(state_t.values(), goalState=True)
                #     state_t_plus_1_sub_features = rbf_state.computeFeatures(state_t_plus_1.values(), goalState=True)

                if episode_number > 1:
                    print("State Value: {0}".format(np.dot(state_t_sub_features, critic_value_func_params)))

                # --------- CALCULATE VALUE FUNCTIONS FROM STATE TRANSITION ---------
                # characteristic eligibility for a Gaussian policy of form: a ~ N(u(s), sigma)
                # state_features = rbf_state.computeFeatures(state_t.values())
                advantage_approximation_features = ((action_t - action_t_deterministic) * \
                                                   state_t_sub_features) / CONFIG["exploration_sigma"]**2

                phi_squiggle = np.array([state_t_plus_1_sub_features.transpose(),
                                     np.array([0.0 for i in range(len(state_t_plus_1_sub_features))]).transpose()]).transpose()
                phi_hat = np.array([state_t_sub_features.transpose(), advantage_approximation_features.transpose()]).transpose()
                print("Phi-Hat shape: {0}".format(phi_hat.shape))
                print("Phi-Squiggle shape: {0}".format(phi_squiggle.shape))

                lstd.updateTraces(phi_hat)
                lstd.updateA(phi_hat, CONFIG["gamma"] * phi_squiggle)
                lstd.update_b(reward)

                # --------- GET REWARD, UPDATE VALUE FUNCTIONS AND POLICY ---------
                # Update value functions
                # calculate the target (lambda return): U_t = R_t+1 + gamma * V_approx(S_t+1)
                # where V_approx(S_t+1) = dot(theta_t, phi_t+1) as weights have not been updated yet

                # Update policy weights only after the first episode
                # if step_number % CONFIG["spin_rate"] == 0:
                if episode_number > -1 and not CONFIG["episode_fault_injected"] <= episode_number < CONFIG["episode_fault_injected"] + CONFIG["num_episodes_not_learning"]:
                    # policy.plotFeatures = True
                    critic_gradient, critic_value_func_params = lstd.calculateBeta()
                    print("critic_gradient: {0}".format(critic_gradient.shape))
                    if "prev_critic_gradient" not in locals():
                        prev_critic_gradient = np.zeros(critic_gradient.shape)
                    print("Angle between gradient vectors: {0}".format(angle_between_vectors.angle_between(critic_gradient, prev_critic_gradient)))
                    if angle_between_vectors.angle_between(critic_gradient, prev_critic_gradient) < 0.06:
                        print("UPDATING THE POLICY WEIGHTS!!!! :O")
                        policy.updateWeights(gradient_vector=critic_gradient)
                        lstd.decayStatistics()
                    prev_critic_gradient = deepcopy(critic_gradient)

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
                    lstd.decayStatistics(decay=0.0)
                    print("ZEROES LSTD STATISTICS")
                    break

            # Stop logging navigation as episode has terminated either from max num steps or goal condition reached
            logNav(log_nav=False, dir_path=results_dir, file_name_descriptor=str(episode_number))

            # Log cumulative reward for episode and number of steps
            f_returns.write("{0}\t{1}\n".format(episode_number, cum_reward))
            f_timings.write("{0}\t{1}\n".format(episode_number, cum_step_time / step_number))

            # Disable behaviours
            disable_behaviours(disable=True)
            # Now start the next episode where the nav will be reset etc.
