"""Fourier representation"""

from srl.basis_functions.mmlf_representation import Representation
from numpy import indices, pi, cos, dot
from numpy.linalg import norm
import numpy


__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class Fourier(Representation):
    """ Fourier representation.
    Represents the value function using a Fourier series of the specified
    order (eg 3rd order, 5th order, etc).
    See Konidaris, Osentoski, and Thomas, "Value Function Approximation in
    Reinforcement Learning using Fourier Basis" (2011).
    http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf

    """

    def __init__(self, domain, order=3, scaling=False):
        """
        :param domain: the problem :py:class:`~rlpy.Domains.Domain.Domain` to learn
        :param order: The degree of approximation to use in the Fourier series
            (eg 3rd order, 5th order, etc).  See reference paper in class API.

        """
        dims = domain.state_space_dims
        self.coeffs = indices((order,) * dims).reshape((dims, -1)).T
        self.features_num = self.coeffs.shape[0]

        if scaling:
            coeff_norms = numpy.array(map(norm, self.coeffs))
            coeff_norms[0] = 1.0
            self.alpha_scale = numpy.tile(1.0/coeff_norms, (domain.actions_num,))
        else:
            self.alpha_scale = 1.0

        super(Fourier, self).__init__(domain)

    def phi(self, s, terminal):
        """
        Returns :py:meth:`~rlpy.Representations.Representation.Representation.phi_nonTerminal`
        for a given representation, or a zero feature vector in a terminal state.

        :param s: The state for which to compute the feature vector

        :return: numpy array, the feature vector evaluted at state *s*.

        .. note::
            If state *s* is terminal the feature vector is returned as zeros!
            This prevents the learning algorithm from wrongfully associating
            the end of one episode with the start of the next (e.g., thinking
            that reaching the terminal state causes it to teleport back to the
            start state s0).
        """
        if terminal or self.features_num == 0:
            return numpy.zeros(self.features_num, 'bool')
        else:
            return self.phi_nonTerminal(s)

    def phi_nonTerminal(self, s):
        # normalize the state
        s_min, s_max = self.domain.statespace_limits.T
        # print("s_min, s_max : {0}\t{1}".format(s_min, s_max))
        # print("s : {0}".format(s))
        # print("type s: {0}".format(type(s)))
        # s_min, s_max = 0.0, 1.0
        # norm_state = (s - s_min) / (s_max - s_min)
        # print("norm_State : {0}".format(norm_state))
        # print("coeffs: {0}".format(self.coeffs.tolist()))
        return cos(pi * dot(self.coeffs, s))

    def featureType(self):
        return float

    def featureLearningRate(self):
        return self.alpha_scale
