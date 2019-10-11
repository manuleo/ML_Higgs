# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np
from costs import *


def compute_gradient(y, tx, w, costf="MSE"):
    """Compute the gradient."""
    e=y-tx.dot(w)
    N=len(y)
    if (costf=="MSE"):
        return -(np.transpose(tx).dot(e))/N
    else:
        if (costf=="MAE"):
            return -tx.T.dot(np.sign(e))/N
        else:
            raise CostFunctionNotRecognisedError


def gradient_descent(y, tx, initial_w, max_iters, gamma, costf="MSE"):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_gradient(y, tx, w, costf)
        loss = compute_loss(y, tx, w, costf)
        w = w - gamma*g;
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): ||gradient||={grad}, loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, grad=np.linalg.norm(g), l=loss, w0=w[0], w1=w[1]))

    return losses, ws