from model import Model
from layer import Dense
import numpy as np
from keras.datasets import mnist

def sigmoid(x):
    #if np.abs(x)>10:
        #return np.sign(x)
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp)

def d_softmax(x):
    softmax_x = softmax(x)
    jacobian = np.outer(softmax_x, -softmax_x)
    adj = np.eye(x.shape[0])*softmax_x
    jacobian += adj
    return jacobian.reshape((x.shape[0], x.shape[0]))

def relu(x):
    return np.maximum(np.zeros(x.shape), x)

def d_relu(x):
    zeros = np.zeros(x.shape)
    zeros[zeros>0] = 1
    return zeros    

def mse(y, y_hat):
    return 0.5*np.square(y_hat - y)

def d_mse(y, y_hat):
    return y_hat - y

def cross_entropy(y, y_hat): # Cross entropy loss adjusted for my implementation on mnist
    # y represents the true probabilities (since we've done one-hot encoding); y_hat represents the predicted probabilities
    value = np.log2(np.sum(y*y_hat)) # This finds the predicted probability corresponding to the correct class
    return value

def d_cross_entropy(y, y_hat):
    #return y_hat - y
    return -y/y_hat*np.log(2)

def one_hot_encoding(y, num_classes): # Assumes number of samples is on axis 0 of y
    final = []
    for i in range(y.shape[0]):
        z = np.zeros(num_classes)
        z[y[i]] = 1
        final.append(z)
    return np.array(final)

def main():
    model = Model(optimizer="BGD", loss=cross_entropy, d_loss=d_cross_entropy)
    model.add_layer(Dense((784,), (128,), relu, d_relu, method="normal", mu=0, sigma=0.05))
    model.add_layer(Dense((128,), (128,), relu, d_relu, method="normal", mu=0, sigma=0.05,))
    model.add_layer(Dense((128,), (128,), relu, d_relu, method="normal", mu=0, sigma=0.05))
    model.add_layer(Dense((128,), (10,), softmax, d_softmax, method="normal", mu=0, sigma=0.05))
    model.summary()

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(60000,784).T/255
    test_X = test_X.reshape(10000,784).T/255

    train_y = one_hot_encoding(train_y, 10).T
    test_y = one_hot_encoding(test_y, 10).T

    model.train(train_X, train_y, epochs=40, batch_size=5000)
    accuracy = model.get_accuracy_of_classifier(test_X, test_y)
    print(f"Accuracy: {accuracy}")

    sample = test_X[:,0]
    print(np.argmax(test_y[:,0]))
    print(model.forward(sample))

if __name__ == "__main__":
    main()