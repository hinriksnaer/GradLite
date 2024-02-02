from gradlite.nn.module import Module
from gradlite.nn.linear import Linear

class MLP(Module):

    def __init__(self, input_dim, output_dims):
        sz = [input_dim] + output_dims
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i!=len(output_dims)-1) for i in range(len(output_dims))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f'MLP of [{", ".join(str(layer) for layer in self.layers)}]'
