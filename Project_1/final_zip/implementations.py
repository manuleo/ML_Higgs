"""
    This file contains the 6 mandatory algorithm implementations for the project 1
"""

import numpy as np
from implementation_helpers import*


# LEAST SQUARES SGD

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """Gradient descent algorithm.
    
    INPUT:
        y : prediction
        x : samples
        initial_w : initial values for w
        max_inters : once reached this, the algo stops
        gamma : step size
    
    OUTPUT:
        w : best weights
        loss : minimun loss
    
    """
    # Define parameters to store w and loss
    
    ws = [initial_w]
    
    losses = []
    
    w = initial_w
    
    for n_iter in range(max_iters):
        
        # compute_grandient and compute_loss 
        g = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        # we upgrade w
        w = w - gamma*g;
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        if n_iter%10==0:
            print("Gradient Descent({bi}/{ti}): ||gradient||={grad}, loss={l}".format(
              bi=n_iter, ti=max_iters-1 , grad=np.linalg.norm(g), l=loss))

    return ws[-1], losses[-1] 


# LEAST SQUARES SGD

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    
    """Stochastic gradient descent algorithm.
    
    INPUT:
        y : prediction
        x : samples
        initial_w : initial values for w
        batch_size : standard mini-batch-size 1 (sample just one datapoint).
        max_inters : once reached this, the algo stops
        gamma : step size
    
    OUTPUT:
        w : best weights
        loss : minimun loss
    
    """
    ws = [initial_w]
    
    losses = []
    
    w = initial_w
    
    batch_size = 1
    
    for n_iter in range(max_iters):
        
        # we pick randmly one datapoint
        # batch_inter 
        for yn, xn in batch_iter(y, tx, batch_size):
            
            # compute_gradient
            g = compute_gradient(yn, xn, w)
        
        # we upgrade w by the stochastic gradient
        w = w - gamma*g
        
        # compute_loss 
        loss = compute_loss(y, tx, w)
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        if n_iter%10==0:
            print("Stochastic Gradient Descent({bi}/{ti}): ||gradient||={grad}, loss={l}".format(
              bi=n_iter, ti=max_iters-1 , grad=np.linalg.norm(g), l=loss))

    return  ws[-1], losses[-1]

# LEAST SQUARES

def least_squares(y, tx):
    """
    calculate the least squares solution using normal equations
    INPUT: y, X
    OUTPUT: w* (the optimal w) , loss(w*)
    
    """
    D = tx.shape[1]
    G = tx.T.dot(tx)
    if(np.linalg.matrix_rank(G)==D):
        w = np.linalg.inv(G).dot(tx.T).dot(y)
    else:
        w = np.linalg.solve(G,tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss



# RIDGE REGRESSION

def ridge_regression(y, tx, lambda_):
    """
    calculate the optimal w and the loss for the ridge_regression
    INPUT: y, X, lambda
    OUTPUT: w* 
    """
    N = len(y)
    G = tx.T.dot(tx)
    i = np.linalg.inv(G + 2*N*lambda_*np.eye(G.shape[0]))
    w_star = i.dot(tx.T).dot(y)
    loss = compute_loss(y, tx, w_star) + lambda_ * np.linalg.norm(w_star)**2
    return w_star, loss

# LOGISTIC REGRESSION

def logistic_regression(y, x, initial_w, max_iters, gamma):
    """
    Logistic regression via gradient descent
    INPUTS:
            y
            X
            initial_w := the initialization of w for the algorithm
            max_iters := max number of iterations
            gamma := step size gamma
            
    
    OUTPUTS:
            w*, loss


    """
    w = initial_w
    
    threshold = 1e-8

    # init parameters
    losses = []

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, x, w, gamma)
        # log info
        if iter % (max_iters/10)  == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    #print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss

# REGULARIZED LOGISTIC REGRESSION

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression via gradient descent
    INPUTS:
            y
            X
            max_iter := max number of iterations
            gamma := step size gamma
            lambda := penalizing factor lambda
            threshold := threshold for the update of the loss
            
    
    OUTPUTS:
            w*, loss


    """
    threshold = 1e-8

    # init parameters
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        #if iter % (max_iters/10) == 0:
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    #print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss
