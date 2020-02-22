#! /usr/bin/python
import roslib;

roslib.load_manifest("ros_simple_rl")
import rospy
import numpy as np
import pandas as pd
import sys
import time
import os

from copy import deepcopy

from srl.environments.ros_behaviour_interface import ROSBehaviourInterface
from srl.useful_classes.ros_environment_goals import EnvironmentInfo
from srl.useful_classes.ros_thruster_wrapper import Thrusters
from vehicle_interface.msg import FloatArray
from utilities.optimal_control_response import optimal_control_response
from variable_normalizer import DynamicNormalizer
from moving_differentiator import SlidingWindow
from ga import *

CONFIG = {
    "num_generations": 50,
    "run_time": 15,
    "sol_per_pop": 8,   # was 8
    "num_parents_mating": 4
}


class GAOptimizer(object):
    def __init__(self, process):
        self._process = process
        if not hasattr(process, "get_response"):
            print("process to be optimised does not have a get_response method")
        if not hasattr(process, "calculate_fitness"):
            print("process to be optimised does not have a calculate_fitness method")

        args = sys.argv
        if "-r" in args:
            results_dir_name = args[args.index("-r") + 1]
        else:
            results_dir_name = "ga_pid_tuning"
        self.results_dir = "/home/gordon/data/tmp/{0}{1}".format(results_dir_name, 0)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        filename = os.path.basename(sys.argv[0])
        os.system("cp {0} {1}".format(filename, self.results_dir))
        os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/ga_optimization/ga.py {0}".format(
            self.results_dir))

        # TODO - add DataFrame for evolution history and generation
        self.df_evolution_history = pd.DataFrame()
        self.f_evolution_history = open("{0}{1}".format(self.results_dir, "/evolution_history.csv"), "w", 1)

        self.position_normaliser = DynamicNormalizer([-2.4, 2.4], [-1.0, 1.0])
        self.position_deriv_normaliser = DynamicNormalizer([-1.75, 1.75], [-1.0, 1.0])
        self.angle_normaliser = DynamicNormalizer([-3.14, 3.14], [-1.0, 1.0])
        self.angle_deriv_normaliser = DynamicNormalizer([-0.02, 0.02], [-1.0, 1.0])

        self.angle_dt_moving_window = SlidingWindow(5)
        self.last_150_episode_returns = SlidingWindow(150)

        self.thrusters = Thrusters()
        self.env = ROSBehaviourInterface()
        self.environment_info = EnvironmentInfo()
        self.baseline_response = optimal_control_response()

        sub_pilot_position_controller_output = rospy.Subscriber("/pilot/position_pid_output", FloatArray,
                                                                self.positionControllerCallback)

        self.prev_action = 0.0
        self.pos_pid_output = np.zeros(6)

    def run(self):
        # Inputs of the equation.
        ga_state = [0.5, 0.0]

        # Preparing the population
        # Number of the weights we are looking to optimize.
        ga_num_weights = len(ga_state)

        pop_size = (CONFIG["sol_per_pop"], ga_num_weights)

        population = np.random.uniform(low=0.0, high=0.1, size=pop_size)
        f_generation_max = open("{0}{1}".format(self.results_dir, "/generation_history.csv"), "w", 1)
        global_individual_id = 0
        # TODO - loop generations
        for generation_id in range(CONFIG["num_generations"]):
            generation_df = pd.DataFrame()
            local_individual_id = 0
            print("------- New Generation ----------")
            # TODO - loop over population
            for individual in population:
                print("----Individual----")
                # TODO - Check for similar individual in history and if exists, just use it's fitness
                # individual_df = pd.DataFrame([tuple(individual)])
                # fudge the column names to be consistent with self.df_evolution_history
                # individual_df.rename(columns={0: 1, 1: 2}, inplace=True)

                run_response = True
                match_found = False
                if not self.df_evolution_history.empty:
                    match_found, first_occurance_idx, fitness = self.check_for_existing_individual(individual, ga_num_weights)
                    if match_found:
                        # print("Match found in history for individual, using it's previous fitness: {0}".format(fitness))
                        run_response = False

                if run_response:
                    response = self._process.get_response(global_individual_id, individual)
                    # print("response:")
                    # print_friendly_rounded = np.around(response[0:50, :], decimals=2)
                    # print(print_friendly_rounded)
                    individual_fitness = self._process.calculate_fitness(response)
                else:
                    individual_fitness = fitness

                individual_copy = deepcopy(individual)
                # gen_row2 = np.array([global_individual_id, individual_copy[0], individual_copy[1], individual_fitness])
                gen_row = pd.DataFrame()

                gen_row[0] = [global_individual_id] if not match_found else [first_occurance_idx]
                gen_row[1] = [individual_copy[0]]
                gen_row[2] = [individual_copy[1]]
                gen_row[3] = [individual_fitness]
                print("Row being added to generation_df: ")
                print(gen_row)
                generation_df = generation_df.append(gen_row, ignore_index=True)
                # if local_individual_id == 0:
                #     gen_df_last_index = generation_df.columns.array[-1]
                #     generation_df.rename(columns={gen_df_last_index: 'fitness'}, inplace=True)

                evolution_row = pd.DataFrame()
                evolution_row[0] = [global_individual_id]
                evolution_row[1] = [individual_copy[0]]
                evolution_row[2] = [individual_copy[1]]
                evolution_row[3] = [individual_fitness]
                self.df_evolution_history = self.df_evolution_history.append(evolution_row, ignore_index=True)
                # print("DataFrame Evolution History: {0}".format(global_individual_id))
                # print(self.df_evolution_history)
                # if global_individual_id == 0:
                #     self.df_evolution_history.rename(
                #                                 columns={self.df_evolution_history.columns.array[-1]: 'fitness'},
                #                                 inplace=True
                #     )
                self.log_entry(self.f_evolution_history, global_individual_id, individual, individual_fitness)

                global_individual_id += 1
                local_individual_id += 1

            # fitness now calculated for population
            # print("----------")
            # print(generation_df.head(8))
            # print("-----=======")
            # print(self.df_evolution_history)
            # print("============")
            # Log the max fitness for the current generation
            best_fitness_idx = generation_df[generation_df.columns[-1]].idxmin()
            print("generation row idx of min fitness: {0}".format(best_fitness_idx))
            generation_max_fitness = generation_df.iloc[best_fitness_idx, generation_df.columns[-1]]
            generation_max_global_id = generation_df.iloc[best_fitness_idx][0]
            generation_max_fitness_individual = generation_df.iloc[best_fitness_idx][1:ga_num_weights+1]
            self.log_entry(f_generation_max,
                           generation_max_global_id,
                           generation_max_fitness_individual,
                           generation_max_fitness)

            fitness = deepcopy(generation_df.iloc[:, -1].array)

            print("Fitness array:")
            print(fitness)
            # Selecting the best parents in the population for mating.
            parents = select_mating_pool(population, fitness, CONFIG["num_parents_mating"])
            print("parents (best {0}): ".format(CONFIG["num_parents_mating"]))
            print(parents)

            offspring_crossover = crossover(parents,
                                            offspring_size=(pop_size[0] - parents.shape[0], ga_num_weights))

            print("offspring_crossover: {0}".format(offspring_crossover))
            offspring_mutation = mutation(offspring_crossover, num_mutations=2)

            # Creating the new population based on the parents and offspring.
            population[0:parents.shape[0], :] = parents
            population[parents.shape[0]:, :] = offspring_mutation

    def check_for_existing_individual(self, individual, ga_num_weights):
        # print("check_for_existing_individual-> individual: {0}".format(individual))
        evolution_minus_fitness = self.df_evolution_history.iloc[:, 1:ga_num_weights + 1]
        # print("check_for_existing_individual-> evolution_minus_fitness: {0}".format(evolution_minus_fitness))

        match_found = False
        fitness = 9999999
        # TODO - remove loop and use more efficient numpy or pandas routine
        # individual_exists = evolution_minus_fitness[(evolution_minus_fitness == individual_df).all(axis=1)]
        for idx, row in evolution_minus_fitness.iterrows():
            is_close = np.allclose(individual, row.array, atol=0.01)
            if is_close:
                match_found = True
                history_entry = self.df_evolution_history.iloc[idx, :]
                print("match found: ")
                print(history_entry)
                fitness = history_entry.array[-1]
                break
        return match_found, idx, fitness

    def log_entry(self, file, id, individual, fitness):
        delimiter = "\t"
        ind_str = '\t'.join(map(str, individual))
        entry = "{0}{1}{2}{3}{4}\n".format(id, delimiter, ind_str, delimiter, fitness)
        file.write(entry)


class PilotPidProcess(object):
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
        # rospy.set_param("/pilot/controller/vel_r/kp", str(velP))
        # rospy.set_param("/pilot/controller/vel_r/ki", str(velI))
        # rospy.set_param("/pilot/controller/vel_r/kd", str(velD))
        self.env.enable_pilot(True)
    def calculate_fitness(self, response):
        diff = abs(response) - abs(self.baseline_response)
        max_idx = 100   # response.shape[0]
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
        self.setPidGains(0.35, 0, 0, 0, 0, 0)
        self.env.reset(disable_behaviours=False)
        self.angle_dt_moving_window.reset()
        self.prev_angle_dt_t = 0.0
        self.prev_angle_dt_tp1 = 0.0

        # Set the gains to those of the individual/solution
        self.setPidGains(individual[0], 0, individual[1], 0, 0, 0)

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
                state_keys = self.state_t.keys()
                state_keys.append("baseline_angle")
                state_keys.append("action")
                label_logging_format = "#{" + "}\t{".join(
                    [str(state_keys.index(el)) for el in state_keys]) + "}\n"
                f_actions.write(label_logging_format.format(*state_keys))

            logging_list = self.state_t.values()
            logging_list.append(self.baseline_response[timestep, 1])
            logging_list.append(self.pos_pid_output[5])
            action_logging_format = "{" + "}\t{".join(
                [str(logging_list.index(el)) for el in logging_list]) + "}\n"
            response[timestep, :] = [timestep, logging_list[0]]
            timestep += 1
            f_actions.write(action_logging_format.format(*logging_list))
        return response


if __name__ == '__main__':
    rospy.init_node("ga_pid_optimization")

    process = PilotPidProcess()
    pilot = GAOptimizer(process)
    pilot.run()
