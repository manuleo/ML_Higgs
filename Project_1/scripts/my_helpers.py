# -*- coding: utf-8 -*-
"""my helper functions."""
import numpy as np
import pandas as pd
from lab_helpers import *
from proj1_helpers import *

def preprocessing(y, tX, test=False):
    tX_pd = pd.DataFrame(tX)
    tX_pds1 = []
    for jet in range(0, 4):
        tX_pds1.append(tX_pd[tX_pd[22]==jet])
    
    #dropping
    drops_0 = [4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29] # 29 all zeros
    drops_1 = [4, 5, 6, 12, 26, 27, 28]
    drop_22 = [22]
    tX_pds1[0].drop(drops_0, axis=1, inplace=True)
    tX_pds1[1].drop(drops_1, axis=1, inplace=True)
    for jet in range(0, 4):
        tX_pds1[jet].drop(drop_22, axis=1, inplace=True)
    
    tX_pds = []
    for jet in range(0, 4):
        indexes_nan = tX_pds1[jet][0] == -999
        indexes_not_nan = tX_pds1[jet][0] != -999
        tX_pds.append(tX_pds1[jet][indexes_nan])
        tX_pds.append(tX_pds1[jet][indexes_not_nan])
    
    drop_0 = [0]
    for jet in range(0, 8):
        if (jet%2==0):
            tX_pds[jet].drop(drop_0, axis=1, inplace=True)
    
    #for jet in range(0, 4):
     #   tX_pd[jet].where(tX_pd[jet]!=-999, inplace=True)
      #  tX_pd[jet].fillna(tX_pd[jet].median(), inplace=True)    
    
    #new datasets
    if test==False:
        y_new = []
        for jet in range(0, 8):
            y_new.append(y[tX_pds[jet].index.values])
    
    tX_new = []
    for jet in range(0, 8):
        tX_new.append(tX_pds[jet].values)
    
    ids_new = []
    for jet in range(0, 8):
        ids_new.append(tX_pds[jet].index.values)
        
    #normalize
    means, stds = [], []
    for jet in range (0, 8):
        tX_new[jet], mean, std = standardize(tX_new[jet])
        means.append(mean)
        stds.append(std)
    
    #new 1s column
    for jet in range(0,8):
        tX_new[jet] = np.c_[np.ones(tX_new[jet].shape[0]), tX_new[jet]]
        
    if test==False:
        return y_new, tX_new, ids_new, means, stds
    else:
        return tX_new, ids_new, means, stds
    
def y_for_logistic(y):
    y_new = np.where(y==-1, 0, y)
    return y_new

def build_predictions(tX, indexes, w, degrees=[], logistic=False):
    N = 0
    for jet in range(0, 8):
        N += tX[jet].shape[0]
    
    y_pred = np.zeros(N)
    for jet in range (0, 8):
        if (len(degrees) != 0):
            x = build_poly(tX[jet], degrees[jet])
        else:
            x = tX[jet]
        if (logistic==False):
            y_p = predict_labels(w[jet], x)
        else:
            y_p = predict_labels_logistic(w[jet], x)
        index = indexes[jet]
        y_pred[index] = y_p

    return y_pred

def accuracy(y_real, y_pred):
    N = len(y_real)
    if N!=len(y_pred):
        raise Exception('y_pred length wrong')
    correct = np.sum(y_real==y_pred)
    return correct/N
        

def split_data(y, x, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    N = x.shape[0]
    
    size_tr = round(N* ratio)
    index = np.arange(N)
    np.random.shuffle(index)
    
    ind_tr = index[:size_tr]
    ind_te= np.setdiff1d(index, ind_tr)   
    
    x_train = x[ind_tr]
    y_train = y[ind_tr]
    x_test = x[ind_te]
    y_test = y[ind_te]
    
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