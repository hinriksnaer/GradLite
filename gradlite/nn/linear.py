from gradlite.nn.module import Module
from gradlite.nn.neuron import Neuron

class Linear(Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        self.neurons = [Neuron(in_dim, **kwargs) for _ in range(out_dim)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f'Layer of [{", ".join(str(n) for n in self.neurons)}]'


