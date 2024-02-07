
class SGD:
    def __init__(self, parameters, learning_rate):
        self.lr = learning_rate
        self.params = parameters
        

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

