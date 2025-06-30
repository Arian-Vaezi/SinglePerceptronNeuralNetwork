import numpy as np
X=np.array([1,2])
W=np.array([0.5,-1])
b=1;
y=W@X+b
print(y)

class Perceptron:
    def __init__(self,n_inputs, learning_rate):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate

        self.weights=np.random.randn(n_inputs)
        self.bias=float(np.random.randn())
    def forward(self,X):
        return self.weights@X+self.bias
        
    def train(self, X,y,epochs):
        for epoch in range (epochs):
            y_pred=self.forward(X)