from micrograd import Value
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b')

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        act.label = 'act'
        out = act.tanh(); out.label = 'out'
        return out
