from model import Model
from layer import Dense
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def mse(y, y_hat):
    return 0.5*np.square(y_hat - y)

def d_mse(y, y_hat):
    return y_hat - y

def main():
    model = Model(optimizer="BGD", loss=mse, d_loss=d_mse)
    model.add_layer(Dense((10,), (1,), sigmoid, d_sigmoid, method="normal", mu=0, sigma=0.05))
    model.summary()
    model.forward(np.random.uniform(-1,1,10))

if __name__ == "__main__":
    main()