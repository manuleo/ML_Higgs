# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def preprocessing(tX, test=False, y=np.array([])):
    """
    This function executes the various steps for the cleaning and preparation of the dataset Higgs' Boson.
    INPUTS: X
            y (not mandatory)
            test := is a boolean, False if the dataset is the training.

    
    if test == FALSE:
    OUTPUTS:y_new
            tX_pds
            ids_new
            means 
            stds
    
    if test == TRUE
    OUTPUTS:x_train 
            x_test 
            y_train 
            y_test

    """
    index = np.array(range(0,tX.shape[0])).reshape((tX.shape[0],1))
    tX = np.append(tX, index, axis=1)
    tX_pds1 = []
    for jet in range(0, 4):
        tX_pds1.append(tX[tX[:,22] == jet])
    
    # dropping
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
    """
    this function changes the -1 in 0 in the y vector to use it in the logistic regression
    """
    y_new = np.where(y==-1, 0, y)
    return y_new

def build_predictions(tX, indexes, w, degrees=[], logistic=False):
    """
    build the predictions for a given dataset
    INPUTS: X
            indexes := indexes for the association (from the preprocessing, useful to assign the data to the right jet)
            w
            degrees := list of polynomial degree per jet

    """
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
    """
    compute the accuracy: percentage of correct predictions over the total number of predictions
    INPUTS: y_real := the y vector of true labels
            y_pred := the y vector of the predicted labels
    OUTPUTS: accuracy
    """
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



def get_subsample(y, x, sub_size, seed=1):
    """
    this function gives a subsample for a given y and a given x
    INPUTS: y
            x
            sub_size := the size of the subsample
            seed := seed for reproducibility
    """
    np.random.seed(seed)
    indexes = np.arange(x.shape[0])
    np.random.shuffle(indexes)  
    sub_index = indexes[:sub_size] 
    x_sub = x[sub_index,:]
    y_sub = y[sub_index]
    return y_sub, x_sub




# -*- coding: utf-8 -*-
"""some practical helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

# def load_csv_data_logistic(data_path, sub_sample=False):
#     """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
#     y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
#     x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
#     ids = x[:, 0].astype(np.int)
#     input_data = x[:, 2:]

#     # convert class labels from strings to binary (-1,1)
#     yb = np.ones(len(y))
#     yb[np.where(y=='b')] = 0
    
#     # sub-sample
#     if sub_sample:
#         yb = yb[::50]
#         input_data = input_data[::50]
#         ids = ids[::50]

#     return yb, input_data, ids


def predict_labels(weights, data, logistic=False):
    """Generates predictions given weights, and a test data matrix for classification with least squares or logistic regression"""
    if logistic:
        
    else:
        y_pred = np.dot(data, weights)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
        
    return y_pred

def predict_labels_logistic(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def create_csv_submission(ids, y_pred, name, logistic=False):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    if (logistic==True):
        y_pred = np.where(y_pred==0, -1, y_pred)
    
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, dialect='unix')
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
