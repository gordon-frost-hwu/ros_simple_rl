import numpy

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.amin(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = 99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    print("MUTATION FUNC - START")
    print("offspring_crossover shape: {0}".format(offspring_crossover.shape))
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    print("mutations_counter: {0}".format(mutations_counter))

    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.random.normal(scale=0.3)   # numpy.random.uniform(-1.0, 1.0, 1)
            print("mutating index: {0} {1}".format(idx, gene_idx))
            offspring_crossover[idx, gene_idx] = numpy.clip(offspring_crossover[idx, gene_idx] + random_value, 0, 5)
            gene_idx = gene_idx + mutations_counter

    print("MUTATION FUNC - End")

    return offspring_crossover

def mutGaussian(offspring, mu, sigma, indpb):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    print("mutGaussian: offspring type: {0}".format(type(offspring)))
    for individual in offspring:

        size = len(individual)

        if len(mu) < size:
            raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
        if len(sigma) < size:
            raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        for i, m, s in zip(range(size), mu, sigma):
            if numpy.random.random() < indpb:
                individual[i] = numpy.clip(individual[i] + numpy.random.normal(scale=s), 0, 50)

    return offspring
