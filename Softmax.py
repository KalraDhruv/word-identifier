import numpy as np

class Softmax:
    def _softmax(self,z:np.ndarray) -> np.ndarray:
        def exponent(x):
            return np.exp(x)

        result = np.apply_along_axis(exponent,axis=0, arr=z)
        sum = np.sum(result)
        def softmax(x):
            return x/sum
        result_new = np.apply_along_axis(softmax,axis=0, arr=result)
        return result_new
