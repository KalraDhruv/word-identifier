import numpy as np
from typing import Tuple

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
    def _cross_entropy(self,y: int, y_hat: np.ndarray)->float:
        """
        Compute cross entropy loss for a single example
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        prob_true_class = y_hat[y]
        return -np.log(prob_true_class)

    def forward(self,z:np.ndarray,y:int)-> Tuple[np.ndarray,float]:
        """
        :param z: The input vector for which the classification must be done
                  using softmax layer
        :param y: The correct classification as given
        :return: output vector from softmax layer
        """
        predictions = self._softmax(z)
        loss = self._cross_entropy(y,predictions)
        return predictions, loss

    def backward(self, y:int, y_hat: np.ndarray) -> np.ndarray:
       """
       This function is for obtaining the gradient of the softmax output w.r.t. the softmax input
       :param y: Actual classification as given
       :param y_hat: vector consisting of softmax results
       :return: gradients
       """
       raise NotImplementedError
       