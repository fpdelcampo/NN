from model import Model
from layer import Dense
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def main():
    model = Model()
    model.add_layer(Dense((10,), (1,), sigmoid, d_sigmoid, method="normal", mu=0, sigma=0.05))
    model.summary()
    model.forward(np.random.uniform(-1,1,10))


if __name__ == "__main__":
    main()