import numpy as np

class Model:
    def __init__(self, layers=[], optimizer=None, loss=None, d_loss=None) -> None:
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.d_loss = d_loss

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        if self.layers == []:
            raise ValueError("Add layers")
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, prediction, target):
        if self.layers == []:
            raise ValueError("Add layers")
        if self.optimizer == None:
            raise ValueError("Add an optimizer")
        if self.d_ == None:
            raise ValueError("Add the loss function's derivative")

        l = self.d_loss(prediction, target)
        first = True
        delta = None
        weights = None
        for layer in self.layers:
            delta, weights = layer.backward(delta, weights, first, loss=l)
            first = False

    def update(self):
        if self.layers == []:
            raise ValueError("Add layers")
        for layer in self.layers:
            layer.update()

    def train(self, X, Y, epochs=20, batch_size=100): # Assumes that the X data if of shape [a, b, ... c, N] where N is the number of training samples; same assumption for Y
        if self.layers == []:
            raise ValueError("Add layers")
        if self.optimizer == None or self.optimizer not in ["GD", "SGD", "BGD"]:
            raise ValueError("Add an optimizer; only GD, SGD, and BGD are implemented")
        if self.loss == None:
            raise ValueError("Add a loss")
        if self.d_loss == None:
            raise ValueError("Add the loss function's derivative")
        if X.shape[:-1] != self.layers[0].input_shape:
            raise ValueError(f"X must be of shape (first_layers_input_shape, training_samples).  X.shape = {X.shape} and the input layer has shape {self.layers[0].input_shape}")
        if Y.shape[:-1] != self.layers[-1].output_shape:
            raise ValueError(f"Y must be of shape (last_layers_output_shape, training_samples).  Y.shape = {Y.shape} and the input layer has shape {self.layers[-1].output_shape}")

        
        if self.optimizer == "GD":
            for epoch in range(epochs):
                for element in range(X.shape[-1]):
                    prediction = self.forward(X[..., element])
                    target = Y[..., element]
                    self.backward(prediction, target)
                self.update()
        elif self.optimizer == "BGD":
            if batch_size < X.shape[-1]:
                raise ValueError(f"Batch size must be less than the number of training samples; batch_size = {batch_size} but there are {X.shape[-1]} training samples")
            for epoch in range(epochs):
                indices = np.random.randint(0, X.shape[-1], size=batch_size)
                for index in indices:
                    prediction = self.forward(X[..., index])
                    target = Y[..., index]
                    self.backward(prediction, target)
                self.update()
        else: # We do SGD
            for epoch in epochs:
                index = np.random.randint(0, X.shape[-1])
                prediction 
                prediction = self.forward(X[..., index])
                target = Y[..., index]
                self.backward(prediction, target)
                self.update()

    def summary(self):
        for layer in self.layers:
            print(f"{type(layer).__name__}: Input Shape: {layer.input_shape}, Output Shape: {layer.output_shape}, Activation: {layer.activation}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Loss: {self.loss}")