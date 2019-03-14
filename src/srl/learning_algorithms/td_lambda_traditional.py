from numpy import dot
from srl.useful_classes.rl_traces import Traces

class Traditional_TD_LAMBDA(object):
    def __init__(self, approximator, trace_decay_rate, min_trace_value):
        self.approximator = approximator
        self.traces = Traces(trace_decay_rate, min_trace_value)

    def updateTrace(self, key, trace_value):
        """
        :param key: raw state values at time step t. Raw state is used to minimize the size of the trace dict key
        :param trace_value: 1.0 for replacing traces
        :return: na
        """
        self.traces.updateTrace(key, trace_value)

    def episodeReset(self):
        self.traces.reset()

    def getStateValue(self, state_features):
        """
        :param state_features: features for the state which you want the value of
        :return: state-value. i.e. real valued
        """
        return self.approximator.computeOutput(state_features)

    def computeTDError(self, reward, gamma, state_t_plus_1_value, state_t_value, terminalState=False):
        """
        :param reward: real valued reward signal having made latest state transition
        :param gamma: discount factor
        :param state_t_plus_1_value: value of state t+1 (using weights of step t. i.e. before update)
        :param state_t_value: value of state t (using weights of step t. i.e. before update)
        :return:
        """
        # target = reward + gamma * state_t_plus_1_value
        if not terminalState:
            return reward + gamma * state_t_plus_1_value - state_t_value
        else:
            return reward - state_t_value

    def updateWeights(self, td_error, basis_functions, use_traces=True, terminalState=False):
        """
        :param: automatically loops over the stored traces to update all recent states with one method call
        :return: Update the weights of the approximator according to the TD-Lambda (Sutton et al '98) update rules
        """
        if self.approximator.name == "LinearApprox":
            print("Updating the TD(lambda) critics weights ...")
            X, T = self.traces.getTraces()
            if use_traces:
                for x, trace in zip(X, T):
                    if terminalState and trace == 1.0:
                        state_features = basis_functions.computeFeatures(x, terminalState) * trace
                    else:
                        state_features = basis_functions.computeFeatures(x) * trace
                    self.approximator.updateWeights(gradient_vector=td_error*state_features)
            else:
                state_features = basis_functions.computeFeatures(X[-1], terminalState) * T[-1]
                self.approximator.updateWeights(gradient_vector=td_error*state_features)
        elif self.approximator.name == "ANNApprox":
            self.approximator.updateWeights(basis_functions, td_error)