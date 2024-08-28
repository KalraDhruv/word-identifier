import numpy as np

def xavier_initialization(input_size, output_size):
    lower_limit = - np.sqrt(6/(input_size+output_size))
    upper_limit = - lower_limit
    return np.random.uniform(lower_limit, upper_limit, size=(output_size,input_size))

class Linear:
    def __init__(self,input_size:int, output_size:int, learning_rate:float):
        """
        :param input_size: Number of inputs from the previous layer
        :param output_size: Number of outputs for the next layer
        :param learning_rate: Learning Rate for processing Gradient Descent
        """
        weights = xavier_initialization(input_size, output_size)
        # Adding bias after initializing the weights on the 0 th column
        bias = np.zeros((output_size, 1))
        self.weights = np.append(bias,weights, axis=1)
        self.learning_rate = learning_rate

    def forward(self, x:np.ndarray)->np.ndarray:

    def backward(self, d_output:np.ndarray, learning_rate):
