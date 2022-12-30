import numpy as np

class Dense:
    def __init__(self, input_shape, output_shape, activation, d_activation, input=None, output=None, **initialize) -> None:
        if not isinstance(input_shape, tuple) or not isinstance(output_shape, tuple):
            raise ValueError("Input and output shape must be tuple")
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.d_activation = d_activation
        self.gradient = np.zeros(output_shape+input_shape)
        self.input = input
        self.output = output
        print(output_shape+input_shape)
        if initialize['method'].lower() == "normal":
            if 'mu' not in initialize or 'sigma' not in initialize:
                raise ValueError(f"Normal initailization requires mu and sigma, received {initialize}")
            self.weights = self.normal_initialization(output_shape+input_shape, initialize['mu'], initialize['sigma'])
            print(self.weights.shape)
            self.bias = self.normal_initialization(output_shape, initialize['mu'], initialize['sigma'])
        elif initialize['method'].lower == "uniform":
            if 'left' not in initialize or 'right' not in initialize:
                raise ValueError(f"Uniform initailization requires left and right, received {initialize}")
            self.weights = self.normal_initialization(output_shape+input_shape, initialize['left'], initialize['right'])
            self.bias = self.normal_initialization(output_shape, initialize['left'], initialize['right'])
        else:
            raise ValueError(f"Initialization not understood, initialize params = {initialize} ")

    def normal_initialization(self, shape, mu=0, sigma=0.01):
        return np.random.normal(mu, sigma, size=shape)

    def uniform_initialization(self, shape, left=-0.01, right=0.01):
        return np.random.uniform(-left, right, size=shape)

    def forward(self, in_data):
        if in_data.shape != self.input_shape:
            raise ValueError(f"Layer accepts input of shape {self.input_shape} but was passed {in_data.shape}")
        print(self.weights.shape, in_data.shape)
        print('===========')
        self.input = in_data
        #print(self.input_shape, self.weights.shape)
        return self.activation(self.weights @ in_data + self.bias)

    def backward(self, in_grad, next_weights):
        grad = (self.d_activation(self.weights @ self.input + self.bias)*next_weights.T) @ in_grad
        return grad*self.input.T