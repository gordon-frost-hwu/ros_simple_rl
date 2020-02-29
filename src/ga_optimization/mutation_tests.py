#! /usr/bin/python

import numpy as np
import pandas as pd
import ga


def test_mutGaussian(population):
    print("population:")
    print(population)
    mutated = ga.mutGaussian(population, [0, 0, 0, 0],
                                        [0.1, 0.1, 5, 0.1],
                                        [0.5, 0.3, 0.5, 0.3])
    return mutated


if __name__ == '__main__':
    df = pd.read_csv('/tmp/evolution_history.csv', delimiter='\t', header=None)
    population = df.iloc[0:4, 1:5].values

    mutated = test_mutGaussian(population)

    print("mutated:")
    print(mutated)
