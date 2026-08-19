import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        #forward
        z1 = np.dot(x, np.transpose(W1)) + b1
        a1 = np.maximum(0, z1)

        y_pred = np.dot(a1, np.transpose(W2)) + b2

        ans = {}
        loss = 0.0
        
        n = len(y_pred)
        dz2 = np.zeros(n)
        for i in range(n):
            curr = (y_pred[i]-y_true[i])**2
            dz2[i] = 2*(y_pred[i]-y_true[i])/n
            loss+=curr

        loss/=n
        db2 = dz2
        dw2 = np.outer(dz2, a1)

        #relu grad
        da1 = np.dot(dz2, W2)
        mask = np.zeros(len(z1))

        for i in range(len(z1)):
            if z1[i]>0:
                mask[i] = 1.0

        dz1 = np.multiply(da1, mask)

        dw1 = np.outer(dz1,x)
        db1 = dz1 

        return {'loss': np.round(loss,5), 'dW1': np.round(dw1,5), 'db1': np.round(db1, 5), 'dW2': np.round(dw2, 5), 'db2': np.round(db2, 5)}






        

        

