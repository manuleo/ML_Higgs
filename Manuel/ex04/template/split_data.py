# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
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
    
    x_train = x[train_indexes]
    y_train = y[train_indexes]
    
    x_test = x[test_indexes]
    y_test = y[test_indexes]
    
    return x_train, x_test, y_train, y_test
