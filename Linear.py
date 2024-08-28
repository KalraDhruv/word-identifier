import numpy as np

def xavier_initialization(n_in, n_out):
    lower_limit = - np.sqrt(6/(n_in+n_out))
    upper_limit = - lower_limit
    return np.random.uniform(lower_limit, upper_limit, size=(n_in,n_out))

class Linear:
    def __init__(self,input_size:int, output_size:int, learning_rate:float):
        """
        :param input_size: Number of inputs from the previous layer
        :param output_size: Number of outputs for the next layer
        :param learning_rate: Learning Rate for processing Gradient Descent
        """

    def forward(self, x:np.ndarray)->np.ndarray:

    def backward(self, d_output:np.ndarray, learning_rate):
