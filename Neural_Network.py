import numpy as np

class Neural_Network(object):
    def __init__(self):
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        #weights
        self.W1 = np.random.randn(self.input_layer_size,self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size,self.output_layer_size)
    def forward(self,X):
        #propagate inputs through network
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.z2, self.W2)
        result = self.activation(self.z3)
        return result > .5
    def activation(self,z):
        return 1/(1+np.exp(-z))
