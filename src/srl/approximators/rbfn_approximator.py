#! /usr/bin/python
import pdb
from scipy import *
from scipy.linalg import norm, pinv
from copy import deepcopy
from numpy import array, zeros
import matplotlib.pyplot as plt

class RBFNApprox:

    def __init__(self, indim, numCenters, outdim):
        self.name = "rbfn"
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = 4.0
        self.W = zeros((self.numCenters, self.outdim))
        self.numParams = max(self.W.flatten().shape)
        self.h = 10.0

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        # print("X shape: {0}".format())
        X = X.reshape((1,self.indim))
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]

        print "center", self.centers
        # calculate activations of RBFs
        G = self._calcAct(X)
        print G

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

    # ------- Simple-rl API ----------
    def getParams(self):
        return deepcopy(self.W.reshape((self.outdim, self.numCenters))[0])
    def setParams(self, l):
        self.W = l.reshape((self.numCenters, self.outdim))

    def computeOutput(self, inpt):
        # print("rbfn->params: {0}".format(self.W))
        X = self._calcAct(array(inpt))
        return dot(X, self.W)[0][0]

    def calculateGradient(self, inpt):
        orig_parameter_vector = self.getParams()

        orig_output = self.computeOutput(inpt)
        gradient = []
        # pdb.set_trace()
        for idx in range(len(orig_parameter_vector)):
            new_param_vector = deepcopy(orig_parameter_vector)
            new_param_vector[idx] += self.h

            self.setParams(new_param_vector)

            new_output = self.computeOutput(inpt)
            gradient.append((new_output - orig_output) / self.h)

        # Reset the weights to what they were before the gradient calculation
        self.setParams(orig_parameter_vector)
        return array(gradient)


if __name__=='__main__':
    plt.ion()
    fig, ax = plt.subplots()

    inpt = [0.1, 0.0, -0.1, -1.0]

    approx = RBFNApprox(4, 100, 1)
    features = approx._calcAct(array(inpt))
    print(features)
    ax.plot(features[0])
    plt.draw()
    useless = raw_input("press Enter to continue ...")