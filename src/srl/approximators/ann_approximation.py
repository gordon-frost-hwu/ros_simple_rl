#! /usr/bin/python

import numpy as np
from copy import deepcopy

# PyBrain imports
import sys; sys.path.append("/home/gordon/software/pybrain")
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

class ANNApproximator(object):
    def __init__(self, alpha):
        self.name = "ANNApprox"
        self.network = FeedForwardNetwork()
        inLayer = LinearLayer(4)
        hiddenLayer = SigmoidLayer(12)
        outLayer = LinearLayer(1)
        self.network.addInputModule(inLayer)
        self.network.addModule(hiddenLayer)
        self.network.addOutputModule(outLayer)
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        self.network.addConnection(in_to_hidden)
        self.network.addConnection(hidden_to_out)

        # Last step to make sure everything works in the connections
        self.network.sortModules()

        self.dataset = SupervisedDataSet(4, 1)
        self.trainer = BackpropTrainer(self.network, self.dataset, learningrate=alpha, momentum=0.0, verbose=True)

    def computeOutput(self, state_features):
        return self.network.activate(state_features)[0]

    def updateWeights(self, features, desired_output):
        print("updateWeights: features: {0}".format(features))
        print("updateWeights: value: {0}".format(desired_output))
        self.dataset.addSample(features, desired_output)
        # self.trainer.train()
        self.trainer.trainEpochs(10)
        self.dataset.clear()


if __name__ == '__main__':
    approximator = ANNApproximator()
    print(approximator.network)
    print("-------")
    approximator.train([0.0, 0.2, 1.0, -0.4], 0.2)