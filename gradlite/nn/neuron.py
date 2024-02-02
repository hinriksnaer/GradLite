import random

from gradlite.nn.module import Module
from gradlite.parameter import Parameter

class Neuron(Module):
    def __init__(self, n_dim, nonlin=True):
        self.w = [Parameter(random.uniform(-1,1)) for _ in range(n_dim)]
        self.b = Parameter(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f'{"ReLU" if self.nonlin else "Linear"}Neuron({len(self.w)})'

