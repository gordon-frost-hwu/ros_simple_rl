from Queue import Queue
from copy import deepcopy
from numpy import array, zeros, dot

class Traces(object):
    _keys = []
    _values = []

    def __init__(self, trace_decay_rate=0.9, min_trace_value=0.5):
        self._lambda = trace_decay_rate
        self._min_trace_value = min_trace_value

    def getTraces(self):
        """ Usage: first call: X, Y = traces.getTraces; then iterate usin:g for x, y in zip(X, Y):"""
        tmp_keys = deepcopy(self._keys); tmp_values = deepcopy(self._values)
        tmp_keys.reverse()
        tmp_values.reverse()
        return tmp_keys, tmp_values

    def decayTraces(self):
        self._values = list(array(self._values) * self._lambda)
        self._keys = self._keys
        self._values = self._values
        if any([item < self._min_trace_value for item in self._values]):
            while min(self._values) < self._min_trace_value:
                print("removing old trace (below threshold) ....")
                self._keys.remove(self._keys[self._values.index(min(self._values))])
                self._values.remove(min(self._values))

    def updateTrace(self, key, value):
        if len(self._values) > 0 and self._lambda != 0:
            self.decayTraces()
        else:
            self._keys = []
            self._values = []
        self._keys.append(key)
        self._values.append(value)

    def reset(self):
        self._keys = []
        self._values = []

class TrueTraces(object):
    def __init__(self, alpha, gamma, lmbda):
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.e = None
        self.e_minus_one = None

    def updateTrace(self, x_t_vector):
        if self.e_minus_one is None:
            self.e_minus_one = zeros(array(x_t_vector).shape)
            self.e = zeros(array(x_t_vector).shape)
        part_1 = self.gamma * self.lmbda * self.e
        part_2 = self.alpha * x_t_vector
        part_3 = self.alpha * self.gamma * self.lmbda * (dot(dot(self.e_minus_one, x_t_vector), x_t_vector))
        self.e = part_1 + part_2 - part_3
    def reset(self):
        pass