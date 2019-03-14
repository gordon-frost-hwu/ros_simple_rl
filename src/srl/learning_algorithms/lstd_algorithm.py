#! /usr/bin/python

import numpy as np

class LSTD(object):
    def __init__(self, lmbda=0.9, gamma=0.9):
        self.name = "nac"
        self.weights = None
        self.A = None
        self.b = None
        self.traces_across_features = None
        self.lamb = lmbda
        self.decay_rate = gamma # Was config["lstd_decay_rate"]
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
        # if self.A is not None and self.b is not None and self.traces_across_features is not None:
        if decay is None:
            self.traces_across_features *= self.decay_rate
            self.A *= self.decay_rate
            self.b *= self.decay_rate
        else:
            self.traces_across_features *= decay
            self.A *= decay
            self.b *= decay
        # print("LSTD: type(self.traces_across..): {0}".format(type(self.traces_across_features)))

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
        print("calculateBeta->A.shape: {0}".format(self.A.shape))
        print("calculateBeta->b.shape: {0}".format(self.b.shape))
        beta = np.dot(A_inverse, self.b)
        print("BETA: {0}".format(beta.shape))
        return beta.transpose()[0,:], beta.transpose()[1,:]
        # return np.diag(np.dot(U, V))[0]