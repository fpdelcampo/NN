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
        if self.loss == None:
            raise ValueError("Add a loss")

        l = self.loss(prediction, target)

        

    def train(self, X, Y, learning_rate, epochs):
        if self.layers == []:
            raise ValueError("Add layers")
        if self.optimizer == None:
            raise ValueError("Add an optimizer")
        if self.loss == None:
            raise ValueError("Add a loss")

    def summary(self):
        for layer in self.layers:
            print(f"{type(layer).__name__}: Input Shape: {layer.input_shape}, Output Shape: {layer.output_shape}, Activation: {layer.activation}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Loss: {self.loss}")