#! /usr/bin/python
import numpy as np
import pandas as pd

from copy import deepcopy
from ga import *

class GAOptimizer(object):
    def __init__(self, config, result_dir, process):
        self.config = config
        self.results_dir = result_dir
        self._process = process
        if not hasattr(process, "get_response"):
            print("process to be optimised does not have a get_response method")
        if not hasattr(process, "calculate_fitness"):
            print("process to be optimised does not have a calculate_fitness method")

        # TODO - add DataFrame for evolution history and generation
        self.df_evolution_history = pd.DataFrame()
        self.f_evolution_history = open("{0}{1}".format(self.results_dir, "/evolution_history.csv"), "w", 1)

    def seed_population(self, solutions_per_population, num_weights):
        population_shape = (solutions_per_population, num_weights)
        # population = np.random.uniform(low=0.0, high=0.1, size=population_shape)
        position_weights = np.random.uniform(low=0.0, high=0.1, size=(solutions_per_population, 2))
        velocity_weights_p = np.random.uniform(low=20.0, high=25, size=(solutions_per_population, 1))
        velocity_weights_d = np.random.uniform(low=0.0, high=0.1, size=(solutions_per_population, 1))
        population = np.concatenate((position_weights, velocity_weights_p), axis=1)
        population = np.concatenate((population, velocity_weights_d), axis=1)
        return population

    def run(self):
        # Inputs of the equation.
        ga_state = [0.5, 0.0, 10.0, 0.0]

        # Preparing the population
        # Number of the weights we are looking to optimize.
        ga_num_weights = len(ga_state)

        population = self.seed_population(self.config["sol_per_pop"], ga_num_weights)

        f_generation_max = open("{0}{1}".format(self.results_dir, "/generation_history.csv"), "w", 1)
        global_individual_id = 0
        # TODO - loop generations
        for generation_id in range(self.config["num_generations"]):
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
                gen_row[3] = [individual_copy[2]]
                gen_row[4] = [individual_copy[3]]
                gen_row[5] = [individual_fitness]
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
                evolution_row[3] = [individual_copy[2]]
                evolution_row[4] = [individual_copy[3]]
                evolution_row[5] = [individual_fitness]
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
            parents = select_mating_pool(population, fitness, self.config["num_parents_mating"])
            print("parents (best {0}): ".format(self.config["num_parents_mating"]))
            print(parents)

            offspring_crossover = crossover(parents,
                                            offspring_size=(self.config["sol_per_pop"] - self.config["num_parents_mating"],
                                                            ga_num_weights))

            print("offspring_crossover: {0}".format(offspring_crossover))
            # offspring_mutation = mutation(offspring_crossover, num_mutations=2)
            offspring_mutation = mutGaussian(offspring_crossover, [0, 0, 0, 0],
                                                                 [0.1, 0.1, 5, 0.1],
                                                                 [0.5, 0.3, 0.5, 0.3])

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