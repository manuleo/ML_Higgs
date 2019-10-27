# -*- coding: utf-8 -*-
import numpy as np

def compute_loss(y, tx, w):
    """
    Function used to compute the mean squared error, the loss of the least squares
    INPUTS: y
            X
            w
    OUTPUT: MSE
    
    """

    e=y-tx.dot(w)
    N=len(y)
    L=e.T.dot(e)/(2*N)
    
    return L

def compute_rmse(y, x, w):
    """
    Function that returns the root mean squared error
    INPUTS: y
            X
            w
    OUTPUT: RMSE

    """
    l = compute_loss(y, x, w)
    return np.math.sqrt(2*l)

def standardize(x):
    """
    Gives as OUTPUT the original dataset given as INPUT
    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generates a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def sigmoid(t):
    """
    Returns as OUTPUT the result of the sigmoid function applied to the INPUT
    """
    return 1/(1+np.exp(-t))


def compute_gradient(y, tx, w):
    """
    Compute the gradient of the loss function for least squares problem in a given point w
    INPUTS: y
            X
            w
    OUTPUT: gradient vector
    """
    e=y-tx.dot(w)
    N=len(y)
    return -(np.transpose(tx).dot(e))/N

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.
    INPUTS: x := the matrix
    OUTPUTS: the matrix with a new column for each degree and for each column of the old dataset
    """
    d = np.arange(1, degree+1).repeat(x.shape[1])
    psi = np.tile(x, degree)
    psi = np.power(psi, d)
    return psi

def calculate_loss(y, tx, w):
    """
    compute the negative log likelihood for the logistic regression.
    INPUTS: y
            X
            w
    OUTPUTS: negative log likelihood
    """
    N = tx.shape[0]
    first = np.sum(np.log(1+np.exp(tx.dot(w))))
    second = y.T.dot(tx.dot(w))
    #print(first, second)
    return first - second

def calculate_gradient(y, tx, w):
    """compute the gradient, in a given point w, of the loss for the logistic regression.
    INPUTS: y
            X
            w
    OUTPUTS: the gradient vector
    """
    return tx.T.dot(sigmoid(tx.dot(w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent for the logistic regression.
    Return the loss and the updated w.
    INPUTS: y
            X
            w
            gamma := the step size
    OUTPUTS:w
            loss
    """
    grad = calculate_gradient(y, tx, w)
    w = w - gamma * grad
    loss = calculate_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, w):
    """return the loss and the gradient for the logistic regression.
    INPUTS: y
            X
            w
    OUTPUTS: loss,
             gradient
    """
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and the gradient for the regularized logistic regression
    INPUTS: y
            X
            w
            lambda
    OUTPUTS: loss
             gradient
    """
    N = tx.shape[0]
    D = tx.shape[1]
    loss = calculate_loss(y, tx, w) + lambda_ / 2 * np.linalg.norm(w)
    gradient = calculate_gradient(y, tx, w) + lambda_ * w
    
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    INPUTS: y
            X
            w
            gamma
            lambda
    OUTPUTS: w
             loss
    """
    #loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return w, loss
