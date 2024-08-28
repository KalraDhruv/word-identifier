import numpy as np
class Sigmoid:
    def __init__(self):
        """
        Initialize state for sigmoid activation layer
        """

    def forward(self, x:np.ndarray) -> np.ndarray:
        def sigmoid(number):
            return 1/(1+np.exp(-number))
        sigmoid_applied = np.apply_along_axis(sigmoid,axis=0,arr=x)
        return sigmoid_applied

    def backward(self,d_output) -> np.ndarray:
        return NotImplementedError


