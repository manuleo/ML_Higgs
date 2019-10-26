# -*- coding: utf-8 -*-
"""my helper functions."""
import numpy as np
import pandas as pd
from lab_helpers import *
from proj1_helpers import *

def preprocessing(y, tX, test=False):
    index = np.array(range(0,tX.shape[0])).reshape((tX.shape[0],1))
    tX = np.append(tX, index, axis=1)
    tX_pds1 = []
    for jet in range(0, 4):
        tX_pds1.append(tX[tX[:,22] == 0])
    
    #dropping
    # we drop the column 22 in each "jet"
    for jet in range(0, 4):
        tX_pds1[jet] = np.delete(tX_pds1[jet],22,1)
    drops_0 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28] # 29 all zeros
    drops_1 = [4, 5, 6, 12, 25, 26, 27]
    tX_pds1[0] = np.delete(tX_pds1[0],drops_0,1)
    tX_pds1[1] = np.delete(tX_pds1[1],drops_1, axis=1)
    
    tX_pds = []
    for jet in range(0, 4):
        indexes_nan = np.where(tX_pds1[jet][:,0] == -999)
        indexes_not_nan = np.where(tX_pds1[jet][:,0] != -999)
        tX_pds.append(tX_pds1[jet][indexes_nan])
        tX_pds.append(tX_pds1[jet][indexes_not_nan])
    
    for jet in range(0, 8):
        if (jet%2==0):
            tX_pds[jet] = np.delete(tX_pds[jet],0,1)
    
    #for jet in range(0, 4):
     #   tX_pd[jet].where(tX_pd[jet]!=-999, inplace=True)
      #  tX_pd[jet].fillna(tX_pd[jet].median(), inplace=True)    
    
    #new datasets
    if test==False:
        y_new = []
        for jet in range(0, 8):
            y_new.append(y[tX_pds[jet][:,tX_pds[jet].shape[1]-1].astype(int)])
    
    ids_new = []
    for jet in range(0, 8):
        ids_new.append(tX_pds[jet][:,tX_pds[jet].shape[1]-1].astype(int))
        
    for jet in range(0,8):
        tX_pds[jet] = np.delete(tX_pds[jet],tX_pds[jet].shape[1]-1,1)
        
    #normalize
    means, stds = [], []
    for jet in range (0, 8):
        tX_pds[jet], mean, std = standardize(tX_pds[jet])
        means.append(mean)
        stds.append(std)
    
    #new 1s column
    for jet in range(0,8):
        tX_pds[jet] = np.c_[np.ones(tX_pds[jet].shape[0]), tX_pds[jet]]
        
    if test==False:
        return y_new, tX_pds, ids_new, means, stds
    else:
        return tX_pds, ids_new, means, stds
    
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