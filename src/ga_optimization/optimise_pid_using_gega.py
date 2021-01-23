# pylint: disable=unused-import
import roslib; roslib.load_manifest("all_agents_bridge")
import rospy

import argparse
import os
import numpy as np
from copy import copy
import random
import allagents.presets as presets
from nessie_learn import MDP
from simple_logger import SimpleLogger, Level
from directory_manager import DirectoryManager

# Autonomous Learning Library imports
from all.logging import ExperimentWriter
from mdp_server import MDPToNessieServer

# GA imports
import gega

from processes.pilot_pid_process import PilotPidProcess

OPTIMISATION_DIR_BASENAME = "gega_pid_optimisation"
VALIDATION_DIR_BASENAME = "gega_pid_validation"
LEARNING_DIR_BASENAME = "pid_run"

class Optimise(object):
    def __init__(self, args, write_loss=False):
        self.args = args
        self._write_loss = write_loss

        self.logger = SimpleLogger(level=Level.INFO)
        self.directory_manager = DirectoryManager(self.logger, "runs")

        if not args.validate:
            result_dir = self.directory_manager.get_next_name("root", OPTIMISATION_DIR_BASENAME)
        else:
            result_dir = self.directory_manager.get_next_name("root", VALIDATION_DIR_BASENAME)

        self.directory_manager["optimisation"] = result_dir
        
        self.logger.initialise(self.directory_manager["optimisation"])
        
        self.process = PilotPidProcess(result_dir)

        self.ind_lookup = {}
        self.individual_id = 0
        self.run_count = 1


    def run_validation(self):
        all_fitness = []
        for i in range(5):
            individual = np.array([0.5, 0.9194994491098809, 0.46860466129221917])
            control_response = self.process.get_response(i, individual)
            avg_fitness = self.process.calculate_fitness(control_response)
            all_fitness.append(avg_fitness)
        print(all_fitness)

    def run_optimisation(self):
        num_generations = 200
        num_genes = 3
        gene_bounds = np.array([[0.1, 1.0], [0, 0.9], [0.0, 0.9]])
        gene_init_range = np.array([[0.1, 1.0], [0, 0.9], [0.0, 0.9]])
        gene_sigma = np.array([0.1 for gene in range(num_genes)])
        gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
        gene_mutation_type = ["log", "log", "log"]
        atol = np.array([0.05, 0.05, 0.05])
        # atol = np.array([1e-6 for gene in range(num_genes)])
        
        self.f_fitness_run_map = open("{0}{1}".format(self.directory_manager["optimisation"], "/fitness_map.csv"), "w", 1)

        # gene_bounds = np.array([[0, 10] for gene in range(num_genes)])
        # gene_init_range = np.array([[0, 10] for gene in range(num_genes)])
        # gene_sigma = np.array([0.5 for gene in range(num_genes)])
        # gene_mutation_probability = np.array([0.2 for gene in range(num_genes)])
        self.solution_description = gega.SolutionDescription(num_genes, gene_bounds,
                                                        gene_init_range, gene_sigma,
                                                        gene_mutation_probability,
                                                        gene_mutation_type,
                                                        atol)
        self.ga = gega.GeneticAlgorithm(self.directory_manager["optimisation"],
                                        self.solution_description,
                                        0.5, 15,
                                        population_size=8,
                                        generations=num_generations,
                                        skip_known_solutions=True,
                                        load_past_data=False)
        # setup fitness calculation
        self.ga.calculate_fitness = self.fitness

        self.ga.run()

    def fitness(self, individual):
        print("Running individual: {0}".format(individual))

        learning_run_name = self.directory_manager.get_next_name("optimisation", LEARNING_DIR_BASENAME)
        print("NEW LEARNING RUN NAME FROM MANAGER: {0}".format(learning_run_name))
        self.directory_manager["learning"] = learning_run_name
        
        control_response = self.process.get_response(self.individual_id, individual)
        avg_fitness = self.process.calculate_fitness(control_response)

        self.f_fitness_run_map.write("{0}\t{1}\t{2}\n".format(self.individual_id, self.individual_id, avg_fitness))
        self.f_fitness_run_map.flush()

        self.individual_id += 1
        return avg_fitness

    def create_result_dir(self, agent_name, env_name):
        idxs = [0]
        parent_folder = "{0}_{1}_optimisation".format(agent_name, env_name)
        dirs = os.listdir("runs")

        print("dirs: {0}".format(dirs))
        for s in dirs:
            if "optimisation" in s:
                idxs.append((int)(s.split("_")[-1].strip("optimisation")))
        maxDirectoryIndex = max(idxs)
        parent_folder = parent_folder + (str(maxDirectoryIndex + 1))

        result_name = os.path.join("runs", parent_folder)

        if os.path.exists(result_name):
            print("Result directory exists, aborting....")
            exit(0)

        os.mkdir(result_name)
        return result_name

if __name__ == "__main__":
    rospy.init_node("pid_optimise")

    parser = argparse.ArgumentParser(description="Run a continuous actions benchmark.")

    parser.add_argument(
        "--repeat", type=int, default=1, help="The number of times to repeat the optimisation"
    )
    parser.add_argument(
        "--validate", action='store_true', help="Run validation of individual"
    )

    args = parser.parse_args()

    for _ in range(args.repeat):
        optimiser = Optimise(args)
        if args.validate:
            optimiser.run_validation()
        else:
            optimiser.run_optimisation()

    rospy.spin()