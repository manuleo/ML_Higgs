# -*- coding: utf-8 -*-
"""my helper functions."""
import numpy as np

def split_data(y, x, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    train_size = round(x.shape[0] * ratio)
    indexes = np.arange(x.shape[0])
    np.random.shuffle(indexes)
    
    train_indexes = indexes[:train_size]
    test_indexes = np.setdiff1d(indexes, train_indexes)    
    x_train = x[train_indexes,:]
    y_train = y[train_indexes]
    
    x_test = x[test_indexes,:]
    y_test = y[test_indexes]
    
    return x_train, x_test, y_train, y_test

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    d = np.arange(1, degree+1).repeat(x.shape[1])
    psi = np.tile(x, degree)
    psi = np.power(psi, d)
    return psi

def get_subsample(y, x, sub_size, seed=1):
    np.random.seed(seed)
    indexes = np.arange(x.shape[0])
    np.random.shuffle(indexes)  
    sub_index = indexes[:sub_size] 
    x_sub = x[sub_index,:]
    y_sub = y[sub_index]
    return y_sub, x_sub


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=y-tx.dot(w)
    N=len(y)
    return -(np.transpose(tx).dot(e))/N

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e=y-tx.dot(w)
    N=len(y)
    return -(np.transpose(tx).dot(e))/N