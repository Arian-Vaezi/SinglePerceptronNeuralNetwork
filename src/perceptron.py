import numpy as np

class Perceptron:
    def __init__(self,n_inputs, learning_rate):
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate

        self.weights=np.random.randn(n_inputs)
        self.bias=float(np.random.randn())

    def forward(self,X):
        return X@self.weights+self.bias
    
    def compute_loss(self,y_pred,y_true):
        errors=y_pred-y_true
        return np.mean(errors**2)
        
    def train(self, X,y,epochs):
        loss_history=[]
        N=X.shape[0]
        for epoch in range (epochs):
            y_pred=self.forward(X)
            loss=self.compute_loss(y_pred,y)
            loss_history.append(loss)
            errors=y_pred-y
            grad_w=(2/N)*(X.T@errors)
            grad_b=(2/N)*errors.sum()
            self.weights-=self.learning_rate*grad_w
            self.bias-=self.learning_rate*grad_b
        # print(loss_history)
        return loss_history
    
    def predict(self, X_new):

        return self.forward(X_new)
    



