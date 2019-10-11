# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w, costf):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e=y-tx.dot(w)
    N=len(y)
    
    if (costf=="MSE"):
        L=e.T.dot(e)/(2*N)
    else:
        if (costf=="MAE"):
            L=np.abs(np.sum(e))/N
        else:
            raise CostFunctionNotRecognisedError
    
    return L