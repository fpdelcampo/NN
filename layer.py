import numpy as np

class Dense:
    def __init__(self, input_shape, output_shape, activation, d_activation, learning_rate=0.0001, input=None, output=None, **initialize) -> None:
        if not isinstance(input_shape, tuple) or not isinstance(output_shape, tuple):
            raise ValueError("Input and output shape must be tuple")
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.d_activation = d_activation
        self.weights_gradient = np.zeros(output_shape+input_shape)
        self.bias_gradient = np.zeros(output_shape)
        self.backward_passes = 0
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        if initialize['method'].lower() == "normal":
            if 'mu' not in initialize or 'sigma' not in initialize:
                raise ValueError(f"Normal initailization requires mu and sigma, received {initialize}")
            self.weights = self.normal_initialization(output_shape+input_shape, initialize['mu'], initialize['sigma'])
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
        self.input = in_data
        return self.activation(self.weights @ in_data + self.bias)

    def backward(self, delta_l=None, weights_l=None, first=False, loss=None): # delta_l_1 = delta_l * d_activation(self.weights @ self.input + self.bias) * weights_l.T      

        # Read this article for issues with backward dimensionality. In particular, about how treating the gradient of something like softmax vs sigmoid differs:
        # https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html
        # http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf

        self.backward_passes +=1 
        z = self.weights @ self.input + self.bias
        d = self.d_activation(z)
        if first:
            if loss is None:
                raise ValueError("Must provide gradient of the loss function with respect to the model's outputs")
            if d.ndim == 1:
                delta_l = np.outer(d, loss)
            else:
                delta_l = d @ loss
            self.weights_gradient += self.learning_rate*np.outer(delta_l, self.input)
            self.bias_gradient += self.learning_rate*delta_l
            return delta_l, self.weights
        else:
            delta_l_1 = d * (weights_l.T @ delta_l)
            self.weights_gradient += self.learning_rate*np.outer(delta_l_1, self.input)
            self.bias_gradient += self.learning_rate*delta_l_1
            return delta_l_1, self.weights

    def update(self):
        print('called')
        self.weights -= self.weights_gradient/self.backward_passes
        self.bias -= self.bias_gradient/self.backward_passes
        self.weights_gradient = np.zeros(self.output_shape+self.input_shape)
        self.bias_gradient = np.zeros(self.output_shape)
        self.backward_passes = 0