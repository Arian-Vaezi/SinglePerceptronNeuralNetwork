import numpy as np
import matplotlib.pyplot as plt

from src.perceptron import Perceptron

def main():
    data = np.loadtxt('data/simple_lr.csv', delimiter=',',skiprows=1)
    X=data[:,:2]
    y=data[:,2]

    p= Perceptron(n_inputs=2,learning_rate=0.15)
    loss_history=p.train(X,y,epochs=200)
    print(f"Learned weight:{p.weights}   bias:{p.bias}")
    predict=p.predict(X)
    print(predict)

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve')
    plt.show()

    
if __name__ == '__main__':
    main()
