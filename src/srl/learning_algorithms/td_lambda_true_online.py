from numpy import dot
from srl.approximators.linear_approximation import LinearApprox
from srl.useful_classes.rl_traces import Traces

class TrueOnline_TD_LAMBDA(object):
    """ Algorithm is an adaption of the TD(lambda) algorithm to follow a new forward view of TD learning. Idea is that
    this True Online TD(lambda) matches the forward view exactly. Taken from paper: Seijen and Sutton, "True Online
    TD(lambda), 2014"
    """
    def __init__(self, approximator_config, trace_decay_rate, min_trace_value):
        self.approximator = LinearApprox(approximator_config)
        self.traces = Traces(trace_decay_rate, min_trace_value)

    def updateTrace(self, key, trace_value):
        """
        :param key: raw state values at time step t. Raw state is used to minimize the size of the trace dict key
        :param trace_value: 1.0 for replacing traces
        :return: na
        """
        self.traces.updateTrace(key, trace_value)

    def getStateValue(self, state_features):
        """
        :param state_features: features for the state which you want the value of
        :return: state-value. i.e. real valued
        """
        return self.approximator.computeOutput(state_features)

    def computeTDError(self, reward, gamma, state_t_plus_1_value, state_t_value):
        """
        :param reward: real valued reward signal having made latest state transition
        :param gamma: discount factor
        :param state_t_plus_1_value: value of state t+1 (using weights of step t. i.e. before update)
        :param state_t_value: value of state t (using weights of step t. i.e. before update)
        :return:
        """
        return reward + ((gamma * state_t_plus_1_value) - state_t_value)

    def updateWeights(self, td_error, basis_functions):
        """
        :param: automatically loops over the stored traces to update all recent states with one method call
        :return: Update the weights of the approximator according to the TD-Lambda (Sutton et al '98) update rules
        """
        X, T = self.traces.getTraces()
        for x, trace in zip(X, T):
            state_features = basis_functions.computeFeatures(x) * trace
            self.approximator.updateWeights(gradient_vector=td_error*state_features)