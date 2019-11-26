# -*- coding: utf-8 -*-
"""my helper functions."""
import numpy as np
import pandas as pd

def preprocessing(y, tX, test=False):
    index = np.array(range(0,tX.shape[0])).reshape((tX.shape[0],1))
    tX = np.append(tX, index, axis=1)
    tX_pds1 = []
    for jet in range(0, 4):
        tX_pds1.append(tX[tX[:,22] == jet])
    
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
    
    #new 1s column
    for jet in range(0,8):
        tX_pds[jet] = np.c_[np.ones(tX_pds[jet].shape[0]), tX_pds[jet]]
        
    #new processing discovered after visualizing data
    
    cub0 = [1, 2, 4, 5, 6, 7, 8, 9, 12, 15, 17]
    tX_pds[0][:, cub0] = np.cbrt(tX_pds[0][:, cub0])
    tX_pds[0] = np.delete(tX_pds[0], 3, 1)
    
    cub1 = [1, 2, 3, 6, 7, 8, 9, 13, 16, 18]
    tX_pds[1][:, cub1] = np.cbrt(tX_pds[1][:, cub1])
    tX_pds[1] = np.delete(tX_pds[1], 4, 1)
    
    cub2 = [1,2,3,5,6,7,8,9,12,15,17,18]
    tX_pds[2][:, cub2] = np.cbrt(tX_pds[2][:, cub2])
    tX_pds[2] = np.delete(tX_pds[2], 21, 1)
    
    cub3 = [1,2,3,4,6,7,8,10,13,16,18,19,22]
    tX_pds[3][:, cub3] = np.cbrt(tX_pds[3][:, cub3])
    tX_pds[3] = np.delete(tX_pds[3], 22, 1)
    
    cub4 = [1,2,3,5,8,9,10,13,16,19,22,25,28]
    tX_pds[4][:, cub4] = np.cbrt(tX_pds[4][:, cub4])
    
    cub5 = [2,3,4,6,9,10,11,14,17,20,23,26,29]
    tX_pds[5][:, cub5] = np.cbrt(tX_pds[5][:, cub5])
    
    cub6 = [1,2,3,5,8,9,10,13,16,19,22,25,28]
    tX_pds[6][:, cub6] = np.cbrt(tX_pds[6][:, cub6])
    
    cub7 = [1,2,3,4,5,6,8,9,10,11,14,17,20,22,23,26,29]
    tX_pds[7][:, cub7] = np.cbrt(tX_pds[7][:, cub7])
    
    means, stds = [], []
    for jet in range (0, 8):
        tX_pds[jet], mean, std = standardize(tX_pds[jet])
        means.append(mean)
        stds.append(std)
        
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