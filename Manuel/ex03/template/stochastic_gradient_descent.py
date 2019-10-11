# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from costs import *


def compute_stoch_gradient(y, tx, w, costf="MSE"):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e=y-tx.dot(w)
    N=len(y)
    if (costf=="MSE"):
        return -(np.transpose(tx).dot(e))/N
    else:
        if (costf=="MAE"):
            return -tx.T.dot(np.sign(e))/N
        else:
            raise CostFunctionNotRecognisedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, costf="MSE"):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for yn, xn in batch_iter(y, tx, batch_size):
            g = compute_stoch_gradient(yn, xn, w, costf)
            w = w - gamma*g;
            loss = compute_loss(y, tx, w, costf)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("SGD({bi}/{ti}): |gradient|={grad}, loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, grad=np.linalg.norm(g), l=loss, w0=w[0], w1=w[1]))

    return losses, ws