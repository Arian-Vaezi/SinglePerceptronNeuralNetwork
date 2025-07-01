import numpy as np
import matplotlib.pyplot as plt

from src.perceptron import Perceptron

def main():
    data = np.loadtxt('data/simple_lr.csv', delimiter=',',skiprows=1)
    X=data[:,0].reshape(-1,1)
    y=data[:,1]

    p= Perceptron(n_inputs=1,learning_rate=0.1)
    loss_history=p.train(X,y,epochs=500)
    print(f"Learned weight:{p.weights[0]:.4f}   bias:{float(p.bias):.4f}")
    predict=p.predict(X)
    print(predict)

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve')
    plt.show()
if __name__ == '__main__':
    main()
