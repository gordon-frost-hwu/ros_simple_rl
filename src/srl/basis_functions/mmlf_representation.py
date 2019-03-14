"""Representation base class."""

from copy import deepcopy
import scipy.sparse as sp
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Representation(object):

    """
    The Representation is the :py:class:`~rlpy.Agents.Agent.Agent`'s model of the
    value function associated with a :py:class:`~rlpy.Domains.Domain.Domain`.

    As the Agent interacts with the Domain, it receives updates in the form of
    state, action, reward, next state, next action. \n
    The Agent passes these quantities to its Representation, which is
    responsible for maintaining the value function usually in some
    lower-dimensional feature space.
    Agents can later query the Representation for the value of being in a state
    *V(s)* or the value of taking an action in a particular state
    ( known as the Q-function, *Q(s,a)* ).

    .. note::

        Throughout the framework, ``phi`` refers to the vector of features;
        ``phi`` or ``phi_s`` is thus the vector of feature functions evaluated
        at the state *s*.  phi_s_a appends \|A\|-1 copies of phi_s, such that
        \|phi_s_a\| = \|A\| * \|phi\|, where \|A\| is the size of the action
        space and \|phi\| is the number of features.  Each of these blocks
        corresponds to a state-action pair; all blocks except for the selected
        action ``a`` are set to 0.

    The Representation class is a base class that provides the basic framework
    for all representations. It provides the methods and attributes
    that allow child classes to interact with the Agent and Domain classes
    within the RLPy library. \n
    All new representation implementations should inherit from this class.

    .. note::
        At present, it is assumed that the Linear Function approximator
        family of representations is being used.

    """
    #: A numpy array of the Linear Weights, one for each feature (theta)
    weight_vec = None
    #: The Domain that this Representation is modeling
    domain = None
    #: Number of features in the representation
    features_num = 0
    #: Number of actions in the representation
    actions_num = 0
    # Number of bins used for discretization of each continuous dimension
    discretization = 20
    #: Number of possible states per dimension [1-by-dim]
    bins_per_dim = 0
    #: Width of bins in each dimension
    binWidth_per_dim = 0
    #: Number of aggregated states based on the discretization.
    #: If the represenation is adaptive, set to the best resolution possible
    agg_states_num = 0
    # A simple object that records the prints in a file
    logger = None
    # A seeded numpy random number generator
    random_state = None

    #: True if the number of features may change during execution.
    isDynamic = False
    #: A dictionary used to cache expected results of step(). Used for planning algorithms
    expectedStepCached = None

    def __init__(self, domain, discretization=20, seed=1):
        """
        :param domain: the problem :py:class:`~rlpy.Domains.Domain.Domain` to learn
        :param discretization: Number of bins used for each continuous dimension.
            For discrete dimensions, this parameter is ignored.
        """

        for v in ['features_num']:
            if getattr(self, v) is None:
                raise Exception('Missed domain initialization of ' + v)
        self.expectedStepCached = {}
        self.setBinsPerDimension(domain, discretization)
        self.domain = domain
        self.state_space_dims = domain.state_space_dims
        self.actions_num = domain.actions_num
        self.discretization = discretization
        try:
            self.weight_vec = np.zeros(self.features_num * self.actions_num)
        except MemoryError as m:
            print(
                "Unable to allocate weights of size: %d\n" %
                self.features_num *
                self.actions_num)
            raise m

        self._phi_sa_cache = np.empty(
            (self.actions_num, self.features_num))
        self._arange_cache = np.arange(self.features_num)
        self.agg_states_num = np.prod(self.bins_per_dim.astype('uint64'))

        # a new stream of random numbers for each representation
        self.random_state = np.random.RandomState(seed=seed)
        
    def init_randomization(self):
        """
        Any stochastic behavior in __init__() is broken out into this function
        so that if the random seed is later changed (eg, by the Experiment),
        other member variables and functions are updated accordingly.
        
        """
        pass

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
            return np.zeros(self.features_num, 'bool')
        else:
            return self.phi_nonTerminal(s)

    def phi_sa(self, s, terminal, a, phi_s=None, snippet=False):
        """
        Returns the feature vector corresponding to a state-action pair.
        We use the copy paste technique (Lagoudakis & Parr 2003).
        Essentially, we append the phi(s) vector to itself *|A|* times, where
        *|A|* is the size of the action space.
        We zero the feature values of all of these blocks except the one
        corresponding to the actionID *a*.

        When ``snippet == False`` we construct and return the full, sparse phi_sa.
        When ``snippet == True``, we return the tuple (phi_s, index1, index2)
        where index1 and index2 are the indices defining the ends of the phi_s
        block which WOULD be nonzero if we were to construct the full phi_sa.

        :param s: The queried state in the state-action pair.
        :param terminal: Whether or not *s* is a terminal state
        :param a: The queried action in the state-action pair.
        :param phi_s: (optional) The feature vector evaluated at state s.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.
        :param snippet: if ``True``, do not return a single phi_sa vector,
            but instead a tuple of the components needed to create it.
            See return value below.

        :return: If ``snippet==False``, return the enormous phi_sa vector
            constructed by the copy-paste method.
            If ``snippet==True``, do not construct phi_sa, only return
            a tuple (phi_s, index1, index2) as described above.

        """
        if phi_s is None:
            phi_s = self.phi(s, terminal)
        if snippet is True:
            return phi_s, a * self.features_num, (a + 1) * self.features_num

        phi_sa = np.zeros(
            (self.features_num * self.actions_num),
            dtype=phi_s.dtype)
        if self.features_num == 0:
            return phi_sa
        if len(self._arange_cache) != self.features_num:
            self._arange_cache = np.arange(
                a * self.features_num,
                (a + 1) * self.features_num)
        else:
            self._arange_cache += a * self.features_num - self._arange_cache[0]
        phi_sa[self._arange_cache] = phi_s
        # Slower alternatives
        # Alternative 1: Set only non_zeros (Very close on running time with the current solution. In fact it is sometimes better)
        #nnz_ind = phi_s.nonzero()
        #phi_sa[nnz_ind+a*self.features_num] = phi_s[nnz_ind]
        # Alternative 2: Use of Kron
        #A = zeros(self.actions_num)
        #A[a] = 1
        #F_sa = kron(A,F_s)
        return phi_sa

    def setBinsPerDimension(self, domain, discretization):
        """
        Set the number of bins for each dimension of the domain.
        Continuous spaces will be slices using the ``discretization`` parameter.
        :param domain: the problem :py:class:`~rlpy.Domains.Domain.Domain` to learn
        :param discretization: The number of bins a continuous domain should be sliced into.

        """
        self.bins_per_dim = np.zeros(domain.state_space_dims, np.uint16)
        self.binWidth_per_dim = np.zeros(domain.state_space_dims)
        for d in xrange(domain.state_space_dims):
            if d in domain.continuous_dims:
                self.bins_per_dim[d] = discretization
            else:
                self.bins_per_dim[d] = domain.statespace_limits[d, 1] - \
                    domain.statespace_limits[d, 0]
            self.binWidth_per_dim[d] = (domain.statespace_limits[d,1] - domain.statespace_limits[d, 0]) / (self.bins_per_dim[d] * 1.)

    def binState(self, s):
        """
        Returns a vector where each element is the zero-indexed bin number
        corresponding with the given state.
        (See :py:meth:`~rlpy.Representations.Representation.Representation.hashState`)
        Note that this vector will have the same dimensionality as *s*.

        (Note: This method is binary compact; the negative case of binary features is
        excluded from feature activation.
        For example, if the domain has a light and the light is off, no feature
        will be added. This is because the very *absence* of the feature
        itself corresponds to the light being off.
        """
        s = np.atleast_1d(s)
        limits = self.domain.statespace_limits
        assert (np.all(s >= limits[:, 0]))
        assert (np.all(s <= limits[:, 1]))
        width = limits[:, 1] - limits[:, 0]
        diff = s - limits[:, 0]
        bs = (diff * self.bins_per_dim / width).astype("uint32")
        m = bs == self.bins_per_dim
        bs[m] = self.bins_per_dim[m] - 1
        return bs

    def phi_nonTerminal(self, s):
        """ *Abstract Method* \n
        Returns the feature vector evaluated at state *s* for non-terminal
        states; see
        function :py:meth:`~rlpy.Representations.Representation.Representation.phi`
        for the general case.

        :param s: The given state

        :return: The feature vector evaluated at state *s*.
        """
        raise NotImplementedError

    def activeInitialFeatures(self, s):
        """
        Returns the index of active initial features based on bins in each
        dimension.
        :param s: The state

        :return: The active initial features of this representation
            (before expansion)
        """
        bs = self.binState(s)
        shifts = np.hstack((0, np.cumsum(self.bins_per_dim)[:-1]))
        index = bs + shifts
        return index.astype('uint32')

    def batchPhi_s_a(self, all_phi_s, all_actions,
                     all_phi_s_a=None, use_sparse=False):
        """
        Builds the feature vector for a series of state-action pairs (s,a)
        using the copy-paste method.

        .. note::
            See :py:meth:`~rlpy.Representations.Representation.Representation.phi_sa`
            for more information.

        :param all_phi_s: The feature vectors evaluated at a series of states.
            Has dimension *p* x *n*, where *p* is the number of states
            (indexed by row), and *n* is the number of features.
        :param all_actions: The set of actions corresponding to each feature.
            Dimension *p* x *1*, where *p* is the number of states included
            in this batch.
        :param all_phi_s_a: (Optional) Feature vector for a series of
            state-action pairs (s,a) using the copy-paste method.
            If the feature vector phi(s) has already been cached,
            pass it here as input so that it need not be computed again.
        :param use_sparse: Determines whether or not to use sparse matrix
            libraries provided with numpy.


        :return: all_phi_s_a (of dimension p x (s_a) )
        """
        p, n = all_phi_s.shape
        a_num = self.actions_num
        if use_sparse:
            phi_s_a = sp.lil_matrix(
                (p, n * a_num), dtype=all_phi_s.dtype)
        else:
            phi_s_a = np.zeros((p, n * a_num), dtype=all_phi_s.dtype)

        for i in xrange(a_num):
            rows = np.where(all_actions == i)[0]
            if len(rows):
                phi_s_a[rows, i * n:(i + 1) * n] = all_phi_s[rows,:]
        return phi_s_a

    def stateID2state(self, s_id):
        """
        Returns the state vector correponding to a state_id.
        If dimensions are continuous it returns the state representing the
        middle of the bin (each dimension is discretized according to
        ``representation.discretization``.

        :param s_id: The id of the state, often calculated using the
            ``state2bin`` function

        :return: The state *s* corresponding to the integer *s_id*.
        """

        # Find the bin number on each dimension
        s = np.array(id2vec(s_id, self.bins_per_dim))

        # Find the value corresponding to each bin number
        for d in xrange(self.domain.state_space_dims):
            s[d] = bin2state(s[d], self.bins_per_dim[d], self.domain.statespace_limits[d,:])

        if len(self.domain.continuous_dims) == 0:
            s = s.astype(int)
        return s

    def episodeTerminated(self):
        pass

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k is "logger":
                continue
            setattr(result, k, deepcopy(v, memo))
        return result
