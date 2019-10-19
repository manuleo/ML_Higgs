# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e=y-tx.dot(w)
    N=len(y)
    L=e.T.dot(e)/(2*N)
    
    return L

def compute_rmse(y, x, w):
    l = compute_loss(y, x, w)
    return np.math.sqrt(2*l)